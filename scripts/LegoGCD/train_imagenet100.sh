#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
# LegoGCD reuses the reference backbone checkpoint trained by SimGCD.
REFERENCE_MODEL_PATH="/path/to/simgcd_reference_model.pth"

mkdir -p log

CONFIG_VALUES=$(python -c "
from config1 import exp_root, osr_split_dir, dino_pretrain_path, imagenet_root
print(f'--exp_root={exp_root}')
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--imagenet_root={imagenet_root}')
" | tr '\n' ' ')

EXP_NAME="imagenet100_seed0"
LOG_FILE="log/imagenet100_seed0.log"

echo "Launching seed 0 on GPU 0, log -> ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=0 \
    python -m methods.LegoGCD \
        --dataset_name 'imagenet_100' \
        --batch_size 128 \
        --grad_from_block 11 \
        --epochs 200 \
        --num_workers 8 \
        --use_ssb_splits \
        --sup_weight 0.35 \
        --weight_decay 5e-5 \
        --transform 'imagenet' \
        --eval_funcs 'v2' \
        --warmup_teacher_temp 0.07 \
        --teacher_temp 0.04 \
        --warmup_teacher_temp_epochs 30 \
        --memax_weight 1 \
        --thr 0.8 \
        --exp_name "${EXP_NAME}" \
        --lr_backbone 0.1 \
        --lr 1.0 \
        --projection_head_nlayers 1 \
        --save_path "/path/to/save_dir" \
        --reference_model_path "${REFERENCE_MODEL_PATH}" \
        --aga_weight 0.7 \
        --aperture 2 \
        --eep_weight 0.5 \
        --seed 0 \
        --save_best \
        ${CONFIG_VALUES} \
    > "${LOG_FILE}" 2>&1
