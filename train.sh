#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/experiment_sign_semi \
    --summary_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/experiment_sign_semi/log/ \
    --mode train \
    --train_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/signatures/full_org \
    --val_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/signatures/val \
    --val_dataset_name kaggle_signature \
    --learning_rate 0.0001 \
    --loss semi-hard \
    --decay_step 50000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
    --loss_margin 0.5 \
    --max_iter 200000
