#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-/path/to/dataset/folder}
TRAINER=FDPT

DATASET=${1:?Usage: DATA=/path/to/data bash $0 DATASET SEED}
SEED=${2:?Usage: DATA=/path/to/data bash $0 DATASET SEED}

CFG=vit_b16_c2_ep5_batch32_2ctx_cross_datasets
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 5 \
    --eval-only
fi
