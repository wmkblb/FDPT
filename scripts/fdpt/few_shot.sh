#!/usr/bin/env bash
set -euo pipefail

# custom config
DATA=${DATA:-/path/to/dataset/folder}
TRAINER=FDPT

DATASET=${1:?Usage: DATA=/path/to/data bash $0 DATASET SHOTS SEED}
CFG=vit_b16_c2_ep50_batch32_16ctx_few_shot
SHOTS=${2:?Usage: DATA=/path/to/data bash $0 DATASET SHOTS SEED}
SEED=${3:?Usage: DATA=/path/to/data bash $0 DATASET SHOTS SEED}


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo " The results exist at ${DIR}"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
