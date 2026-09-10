"""Frequency-Decoupled Prompt Tuning (FDPT).

FDPT uses frequency-specific prompt routing, low/high-frequency auxiliary
supervision, and reliability-aware high-frequency filtering.
"""

import json
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
USE_PROMPT_SPLIT = True
USE_ROUTING = True
USE_HIGH_FILTER = True
HIGH_FILTER_TAU = 1.0

GPT_PROMPT_FILES = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}

def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.FDPT.TEACHER_NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Return original CLIP model for generating frozen VL features
    design_details = {"trainer": 'IVLP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'FDPT',
                          "vision_depth": cfg.TRAINER.FDPT.PROMPT_DEPTH,
                          "language_depth": cfg.TRAINER.FDPT.PROMPT_DEPTH,
                          "vision_ctx": cfg.TRAINER.FDPT.N_CTX,
                          "language_ctx": cfg.TRAINER.FDPT.N_CTX}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, compound_prompts_deeper_text]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class FrequencyDecoupledPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        assert cfg.TRAINER.FDPT.PROMPT_DEPTH >= 1, "FDPT requires PROMPT_DEPTH >= 1"
        n_ctx = cfg.TRAINER.FDPT.N_CTX
        ctx_init = cfg.TRAINER.FDPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        visual_ctx_dim = clip_model.visual.conv1.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.compound_prompts_depth = cfg.TRAINER.FDPT.PROMPT_DEPTH
        self.use_prompt_split = getattr(cfg.TRAINER.FDPT, "USE_PROMPT_SPLIT", USE_PROMPT_SPLIT)
        self.n_ctx = n_ctx
        if self.use_prompt_split and n_ctx < 2:
            raise ValueError("FDPT prompt splitting requires N_CTX >= 2")
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print("FDPT: frequency-decoupled prompt tuning")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of FDPT context words (tokens): {n_ctx}")

        visual_ctx_vectors = torch.empty(n_ctx, visual_ctx_dim, dtype=dtype)
        nn.init.normal_(visual_ctx_vectors, std=0.02)

        if self.use_prompt_split:
            self.n_prompt_low = n_ctx // 2
            self.n_prompt_high = n_ctx - self.n_prompt_low

            self.text_prompt_low, self.text_prompt_high = self.split_prompt(ctx_vectors)
            self.visual_prompt_low, self.visual_prompt_high = self.split_prompt(visual_ctx_vectors)

            self.compound_prompts_text_low = nn.ParameterList()
            self.compound_prompts_text_high = nn.ParameterList()
            self.compound_prompts_visual_low = nn.ParameterList()
            self.compound_prompts_visual_high = nn.ParameterList()

            for _ in range(self.compound_prompts_depth - 1):
                text_prompt = torch.empty(n_ctx, ctx_dim)
                nn.init.normal_(text_prompt, std=0.02)
                text_prompt_low, text_prompt_high = self.split_prompt(text_prompt)
                self.compound_prompts_text_low.append(text_prompt_low)
                self.compound_prompts_text_high.append(text_prompt_high)

                visual_prompt = torch.empty(n_ctx, visual_ctx_dim, dtype=dtype)
                nn.init.normal_(visual_prompt, std=0.02)
                visual_prompt_low, visual_prompt_high = self.split_prompt(visual_prompt)
                self.compound_prompts_visual_low.append(visual_prompt_low)
                self.compound_prompts_visual_high.append(visual_prompt_high)

            print(
                "Prompt split enabled: "
                f"low={self.n_prompt_low}, high={self.n_prompt_high}, total={n_ctx}"
            )
        else:
            self.ctx = nn.Parameter(ctx_vectors)

            self.compound_prompts_text = nn.ParameterList([
                nn.Parameter(torch.empty(n_ctx, ctx_dim))
                for _ in range(self.compound_prompts_depth - 1)
            ])
            for single_para in self.compound_prompts_text:
                nn.init.normal_(single_para, std=0.02)

            self.visual_ctx = nn.Parameter(visual_ctx_vectors)
            self.compound_prompts_visual = nn.ParameterList([
                nn.Parameter(torch.empty(n_ctx, visual_ctx_dim, dtype=dtype))
                for _ in range(self.compound_prompts_depth - 1)
            ])
            for single_para in self.compound_prompts_visual:
                nn.init.normal_(single_para, std=0.02)

        ######## preparation for distillation ########
        # visual
        teacher_device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu"
        )
        clip_model_temp = load_clip_to_cpu(cfg, True).float().to(teacher_device)
        clip_model_temp_image = load_clip_to_cpu_teacher(cfg, True)
        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual
        # text
        with open(f"gpt_file/{GPT_PROMPT_FILES[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_temp, cfg.DATASET.NAME
        )
        # Recompute this class-dependent teacher target at load time instead of
        # storing it in checkpoints (base and novel class counts differ).
        self.register_buffer("fixed_embeddings", clip_weights, persistent=False)
        ######## preparation for distillation end ########

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # construct the text, a photo of a <class>.

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn): [n_cls, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_cls, n_tkn, n_dim]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names 
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor, [n_cls, 77]
        self.name_lens = name_lens

    def split_prompt(self, prompt):
        prompt_low = prompt[:self.n_prompt_low].clone()
        prompt_high = prompt[self.n_prompt_low:].clone()
        assert prompt_low.shape[0] + prompt_high.shape[0] == self.n_ctx
        return nn.Parameter(prompt_low), nn.Parameter(prompt_high)

    @staticmethod
    def merge_prompt(prompt_low, prompt_high, mode="full"):
        if mode == "full":
            # Raw branch: both low/high prompt groups receive gradients.
            prompt_parts = [prompt_low, prompt_high]
        elif mode == "low":
            # Low branch: high prompt is detached, so loss_low updates low prompts only.
            prompt_parts = [prompt_low, prompt_high.detach()]
        elif mode == "high":
            # High branch: low prompt is detached, so loss_high updates high prompts only.
            prompt_parts = [prompt_low.detach(), prompt_high]
        else:
            raise ValueError(f"Unsupported prompt mode: {mode}")

        return torch.cat(prompt_parts, dim=0)

    def merge_prompt_list(self, prompt_low_list, prompt_high_list, mode="full"):
        return [
            self.merge_prompt(prompt_low, prompt_high, mode)
            for prompt_low, prompt_high in zip(prompt_low_list, prompt_high_list)
        ]

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts

    def forward(self, mode="full"):
        if mode not in {"full", "low", "high"}:
            raise ValueError(f"Unsupported prompt mode: {mode}")

        if self.use_prompt_split:
            ctx = self.merge_prompt(self.text_prompt_low, self.text_prompt_high, mode)
            visual_ctx = self.merge_prompt(self.visual_prompt_low, self.visual_prompt_high, mode)
            compound_prompts_text = self.merge_prompt_list(
                self.compound_prompts_text_low, self.compound_prompts_text_high, mode
            )
            compound_prompts_visual = self.merge_prompt_list(
                self.compound_prompts_visual_low, self.compound_prompts_visual_high, mode
            )
        else:
            ctx = self.ctx
            visual_ctx = self.visual_ctx
            compound_prompts_text = self.compound_prompts_text
            compound_prompts_visual = self.compound_prompts_visual

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, 4, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        text_input = self.construct_prompts(ctx, prefix, suffix)  # [n_cls, 77, 512]

        return text_input, visual_ctx, compound_prompts_text, compound_prompts_visual


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = FrequencyDecoupledPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # Decompose a prompted intermediate feature map and route its two
        # frequency bands to dedicated prompt groups.
        self.freq_layer = getattr(cfg.TRAINER.FDPT, "FREQ_LAYER", 6)
        self.freq_kernel_size = getattr(cfg.TRAINER.FDPT, "FREQ_KERNEL", 5)
        self.lambda_low = getattr(cfg.TRAINER.FDPT, "LAMBDA_LOW", 0.1)
        self.lambda_high = getattr(cfg.TRAINER.FDPT, "LAMBDA_HIGH", 0.1)
        self.lambd = getattr(cfg.TRAINER.FDPT, "LAMBD", 0.0)
        self.use_routing = getattr(cfg.TRAINER.FDPT, "USE_ROUTING", USE_ROUTING)
        self.use_high_filter = getattr(cfg.TRAINER.FDPT, "USE_HIGH_FILTER", USE_HIGH_FILTER)
        self.high_filter_tau = getattr(cfg.TRAINER.FDPT, "HIGH_FILTER_TAU", HIGH_FILTER_TAU)
        if self.freq_kernel_size < 1 or self.freq_kernel_size % 2 == 0:
            raise ValueError("FREQ_KERNEL must be a positive odd integer")
        if self.use_high_filter and self.high_filter_tau <= 0:
            raise ValueError("HIGH_FILTER_TAU must be positive when filtering is enabled")

        visual_hidden_dim = self.image_encoder.conv1.out_channels
        embed_dim = clip_model.text_projection.shape[1]
        self.low_proj = nn.Linear(visual_hidden_dim, embed_dim)
        self.high_proj = nn.Linear(visual_hidden_dim, embed_dim)
        if self.dtype == torch.float16:
            self.low_proj.half()
            self.high_proj.half()

    @staticmethod
    def decompose_feature_map(feat, kernel_size=5):
        pad = kernel_size // 2
        feat_pad = F.pad(feat, (pad, pad, pad, pad), mode="reflect")
        feat_low = F.avg_pool2d(feat_pad, kernel_size=kernel_size, stride=1)
        feat_high = feat - feat_low
        return feat_low, feat_high

    def extract_prompted_patch_tokens(self, image, visual_ctx, deep_visual_prompts):
        visual = self.image_encoder
        target_layer = min(max(self.freq_layer, 1), len(visual.transformer.resblocks)) - 1

        x = visual.conv1(image.type(visual.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        num_patches = x.shape[1]
        x = torch.cat(
            [
                visual.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + visual.positional_embedding.to(x.dtype)
        if visual.VPT_shallow:
            visual_ctx = visual_ctx.expand(x.shape[0], -1, -1).to(dtype=x.dtype)
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert visual.prompt_till_layer_visual == 0

        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        block_state = [x, deep_visual_prompts]

        for layer_idx, block in enumerate(visual.transformer.resblocks):
            block_state = block(block_state)
            if layer_idx == target_layer:
                x = block_state[0]
                # Prompt tokens are appended after [CLS + image patches], so
                # keep only image patch tokens for feature-map decomposition.
                return x[1:1 + num_patches].permute(1, 0, 2).contiguous()

        raise RuntimeError("Failed to capture prompted intermediate patch tokens")

    def frequency_vectors(self, image, visual_ctx, deep_visual_prompts, dtype):
        patch_tokens = self.extract_prompted_patch_tokens(image, visual_ctx, deep_visual_prompts)
        batch_size, num_patches, hidden_dim = patch_tokens.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Cannot reshape {num_patches} patch tokens into a square feature map")

        feat = patch_tokens.transpose(1, 2).reshape(batch_size, hidden_dim, grid_size, grid_size)
        feat_low, feat_high = self.decompose_feature_map(feat, self.freq_kernel_size)
        low_vector = feat_low.mean(dim=(2, 3))
        high_vector = feat_high.abs().mean(dim=(2, 3))

        low_vector = self.low_proj(low_vector.to(self.low_proj.weight.dtype)).to(dtype)
        high_vector = self.high_proj(high_vector.to(self.high_proj.weight.dtype)).to(dtype)
        low_vector = F.normalize(low_vector, dim=-1)
        high_vector = F.normalize(high_vector, dim=-1)
        return low_vector, high_vector

    def text_features_from_prompts(self, text_input, deep_text_prompts):
        text_features = self.text_encoder(text_input, self.tokenized_prompts, deep_text_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_fixed = self.prompt_learner.fixed_embeddings.to(text_features.dtype)
        text_features = text_features + text_features_fixed
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features, text_features_fixed

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        with torch.no_grad():
            image_features_fixed = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
            image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        # Raw/full branch is also the inference path and the KD branch source.
        text_input, visual_ctx, deep_text_prompts, deep_visual_prompts = self.prompt_learner(mode="full")
        text_features, text_features_fixed = self.text_features_from_prompts(text_input, deep_text_prompts)
        image_features = self.image_encoder(image.type(self.dtype), visual_ctx, deep_visual_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features + image_features_fixed
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # prompted logits
        logits = logit_scale * image_features @ text_features.t()
        logits_raw = logits

        if self.prompt_learner.training:
            loss_cls = F.cross_entropy(logits_raw, label)
            if self.use_routing and self.prompt_learner.use_prompt_split:
                # Frequency routing gradient flow:
                # raw/full above updates both low and high prompts via loss_raw and KD;
                # low mode detaches high prompts, so loss_low does not update them;
                # high mode detaches low prompts, so loss_high does not update them.
                text_input_low, visual_ctx_low, deep_text_prompts_low, deep_visual_prompts_low = self.prompt_learner(
                    mode="low"
                )
                text_features_low, _ = self.text_features_from_prompts(text_input_low, deep_text_prompts_low)
                low_vector, _ = self.frequency_vectors(
                    image, visual_ctx_low, deep_visual_prompts_low, text_features_low.dtype
                )

                text_input_high, visual_ctx_high, deep_text_prompts_high, deep_visual_prompts_high = self.prompt_learner(
                    mode="high"
                )
                text_features_high, _ = self.text_features_from_prompts(text_input_high, deep_text_prompts_high)
                _, high_vector = self.frequency_vectors(
                    image, visual_ctx_high, deep_visual_prompts_high, text_features_high.dtype
                )
            else:
                # Step-3 ablation path: all branches use full prompts without detach.
                text_features_low = text_features
                text_features_high = text_features
                low_vector, high_vector = self.frequency_vectors(
                    image, visual_ctx, deep_visual_prompts, text_features.dtype
                )

            logits_low = logit_scale * low_vector @ text_features_low.t()
            logits_high = logit_scale * high_vector @ text_features_high.t()
            loss_low = F.cross_entropy(logits_low, label)
            loss_high_raw_ce = F.cross_entropy(logits_high, label)
            if self.use_high_filter:
                p_raw = F.softmax(logits_raw.detach(), dim=-1)
                log_p_high_for_gate = F.log_softmax(logits_high.detach(), dim=-1)
                kl_raw_high = F.kl_div(
                    log_p_high_for_gate,
                    p_raw,
                    reduction="none",
                ).sum(dim=-1)
                weight = torch.exp(-kl_raw_high / self.high_filter_tau).detach()
                ce_high_each = F.cross_entropy(logits_high, label, reduction="none")
                loss_high_filtered = (weight * ce_high_each).mean()
                loss_high = loss_high_filtered
            else:
                with torch.no_grad():
                    p_raw = F.softmax(logits_raw.detach(), dim=-1)
                    log_p_high_for_gate = F.log_softmax(logits_high.detach(), dim=-1)
                    kl_raw_high = F.kl_div(
                        log_p_high_for_gate,
                        p_raw,
                        reduction="none",
                    ).sum(dim=-1)
                    weight = torch.ones_like(kl_raw_high)
                loss_high_filtered = loss_high_raw_ce
                loss_high = loss_high_raw_ce
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            loss_distill_text = 1.0 - torch.mean(cos(text_features, text_features_fixed))
            loss_distill_image = 1.0 - torch.mean(cos(image_features, image_features_fixed))
            loss_distill = loss_distill_text + loss_distill_image
            loss = (
                loss_cls
                + self.lambda_low * loss_low
                + self.lambda_high * loss_high
                + self.lambd * loss_distill
            )
            return {
                "loss": loss,
                "loss_raw": loss_cls,
                "loss_low": loss_low,
                "loss_high": loss_high,
                "loss_high_raw_ce": loss_high_raw_ce.detach(),
                "loss_high_filtered": loss_high_filtered.detach(),
                "kl_raw_high_mean": kl_raw_high.detach().mean(),
                "high_filter_weight_mean": weight.detach().mean(),
                "high_filter_weight_min": weight.detach().min(),
                "high_filter_weight_max": weight.detach().max(),
                "acc_raw": (logits_raw.detach().argmax(dim=1) == label).float().mean(),
                "acc_low": (logits_low.detach().argmax(dim=1) == label).float().mean(),
                "acc_high": (logits_high.detach().argmax(dim=1) == label).float().mean(),
                "raw_high_agreement": (
                    logits_raw.detach().argmax(dim=1) == logits_high.detach().argmax(dim=1)
                ).float().mean(),
                "loss_distill": loss_distill,
                "loss_distill_text": loss_distill_text,
                "loss_distill_image": loss_distill_image,
            }
        return logits


def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):

    with torch.no_grad():
        clip_weights = []
        device = next(clip_model.parameters()).device
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            texts = texts.to(device)
            # prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=0)
    return clip_weights

@TRAINER_REGISTRY.register()
class FDPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FDPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FDPT.PREC == "fp32" or cfg.TRAINER.FDPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ("prompt_learner", "low_proj", "high_proj")

        for name, param in self.model.named_parameters():
            if "prompt_learner.ZS_image_encoder" in name:
                param.requires_grad_(False)
            elif not any(name_to_update in name for name_to_update in names_to_update):
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                param.requires_grad_(True)


        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FDPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.FDPT.PREC
        if prec == "amp":
            with autocast():
                loss_dict = model(image, label)
                loss = loss_dict["loss"]
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_dict = model(image, label)
            loss = loss_dict["loss"]
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item(),
            "loss_raw": loss_dict["loss_raw"].item(),
            "loss_low": loss_dict["loss_low"].item(),
            "loss_high": loss_dict["loss_high"].item(),
            "loss_high_raw_ce": loss_dict["loss_high_raw_ce"].item(),
            "loss_high_filtered": loss_dict["loss_high_filtered"].item(),
            "kl_raw_high_mean": loss_dict["kl_raw_high_mean"].item(),
            "high_filter_weight_mean": loss_dict["high_filter_weight_mean"].item(),
            "high_filter_weight_min": loss_dict["high_filter_weight_min"].item(),
            "high_filter_weight_max": loss_dict["high_filter_weight_max"].item(),
            "acc_raw": loss_dict["acc_raw"].item(),
            "acc_low": loss_dict["acc_low"].item(),
            "acc_high": loss_dict["acc_high"].item(),
            "raw_high_agreement": loss_dict["raw_high_agreement"].item(),
            "loss_distill": loss_dict["loss_distill"].item(),
            "loss_distill_text": loss_dict["loss_distill_text"].item(),
            "loss_distill_image": loss_dict["loss_distill_image"].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
