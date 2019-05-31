#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/experiment_sign \
    --summary_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/experiment_sign/log/ \
    --mode train \
    --train_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/signatures/full_org \
    --val_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/SigComp2009-training/NISDCC-offline-all-001-051-6g \
    --learning_rate 0.0001 \
    --decay_step 100000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
    --loss_margin 10.0 \
    --max_iter 200000
