#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-/path/to/dataset/folder}
TRAINER=FDPT

DATASET=${1:?Usage: DATA=/path/to/data bash $0 DATASET SEED}
SEED=${2:?Usage: DATA=/path/to/data bash $0 DATASET SEED}
shift 2
EXTRA_OPTS=("$@")

CFG=vit_b16_c2_ep50_batch32_16ctx
SHOTS=16


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    SEED "${SEED}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES base \
    "${EXTRA_OPTS[@]}"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    SEED "${SEED}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES base \
    "${EXTRA_OPTS[@]}"
fi
