#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

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
    python -m methods.SelEx \
        --dataset_name imagenet_100 \
        --batch_size 128 \
        --epochs 200 \
        --base_model vit_dino \
        --num_workers 4 \
        --use_ssb_splits True \
        --sup_con_weight 0.35 \
        --weight_decay 5e-5 \
        --contrast_unlabel_only False \
        --transform imagenet \
        --seed 0 \
        --eval_funcs v2 \
        --unsupervised_smoothing 1 \
        --grad_from_block 10 \
        --exp_name "imagenet100_seed0" \
        --lr_backbone 0.1 \
        --lr 1.0 \
        --projection_head_nlayers 2 \
        --reference_epochs 30 \
        --reference_batch_size 128 \
        --reference_lr 0.02 \
        --save_path "/path/to/save_dir" \
        --reference_model_path "" \
        ${CONFIG_VALUES} \
        --aga_weight 0.7 \
        --aperture 2.0 \
        --eep_weight 0.5 \
        --save_best \
    > "${LOG_FILE}" 2>&1
