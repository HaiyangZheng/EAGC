#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
# SPTNet reuses the SimGCD + EAGC checkpoint as pretrained backbone/projector.
PRETRAINED_MODEL_PATH="/path/to/simgcd_eagc_best_seed0.pth"
SAVE_PATH="/path/to/save_dir"

CONFIG_VALUES=$(python -c "
from config1 import osr_split_dir, dino_pretrain_path, cars_root
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--cars_root={cars_root}')
" | tr '\n' ' ')

python -m methods.SPTNet \
    --dataset_name 'scars' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 1000 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-4 \
    --transform 'imagenet' \
    --lr 10 \
    --lr2 0.05 \
    --prompt_size 1 \
    --freq_rep_learn 20 \
    --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    --prompt_type 'all' \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 10 \
    --memax_weight 1 \
    --save_path "${SAVE_PATH}" \
    --exp_name "scars_seed0" \
    $CONFIG_VALUES \
    --seed 0 \
    --save_best
