#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
# SPTNet reuses the SimGCD + EAGC checkpoint as pretrained backbone/projector.
PRETRAINED_MODEL_PATH="/path/to/simgcd_eagc_best_seed0.pth"
SAVE_PATH="/path/to/save_dir"

mkdir -p log

CONFIG_VALUES=$(python -c "
from config1 import osr_split_dir, dino_pretrain_path, cifar_100_root
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--cifar_100_root={cifar_100_root}')
" | tr '\n' ' ')

EXP_NAME="cifar100_seed0"
LOG_FILE="log/${EXP_NAME}.log"

echo "Launching seed 0 on GPU 0, log -> ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=0 \
    python -m methods.SPTNet \
        --dataset_name 'cifar100' \
        --batch_size 128 \
        --grad_from_block 11 \
        --epochs 1000 \
        --num_workers 8 \
        --use_ssb_splits \
        --sup_weight 0.35 \
        --weight_decay 5e-4 \
        --transform 'imagenet' \
        --lr 5 \
        --lr2 0.003 \
        --prompt_size 1 \
        --freq_rep_learn 20 \
        --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
        --prompt_type 'all' \
        --eval_funcs 'v2' \
        --warmup_teacher_temp 0.07 \
        --teacher_temp 0.04 \
        --warmup_teacher_temp_epochs 10 \
        --memax_weight 1 \
        --exp_name "${EXP_NAME}" \
        --save_path "${SAVE_PATH}" \
        --seed 0 \
        --save_best \
        ${CONFIG_VALUES} \
    > "${LOG_FILE}" 2>&1
