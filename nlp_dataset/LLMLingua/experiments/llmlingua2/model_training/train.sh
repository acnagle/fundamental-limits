#!/usr/env/bin bash

export CUDA_VISIBLE_DEVICES=0

python train_roberta.py --data_path ../data_collection/dolly-filtered/annotation_kept_cs384_dolly-filtered_train_formatted.pt --save_path ./models/xlm_roberta_large_dolly-filtered_kept_cs384_fp32
