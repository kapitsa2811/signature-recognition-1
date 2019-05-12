#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/experiment_1 \
    --summary_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/experiment_1/log/ \
    --mode train \
    --train_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/iam-handwriting-top50/data_subset_train \
    --val_dir /mnt/069A453E9A452B8D/Ram/handwritten-data/iam-handwriting-top50/data_subset_val \
    --learning_rate 0.001 \
    --decay_step 100000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
    --max_iter 200000
