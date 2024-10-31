#!/usr/env/bin bash

#export HF_TOKEN=
export OPENAI_API_KEY=XXXXX

python format_data.py

python compress.py --load_origin_from ./dolly-filtered/dolly-filtered_train_formatted.json \
    --compressor gpt4 \
    --model_name gpt-4o \
    --prompt_id 5 \
    --chunk_size 384 \
    --save_path ./dolly-filtered/compression_cs384_dolly-filtered_train_formatted.json

python label_word.py --load_prompt_from ./dolly-filtered/compression_cs384_dolly-filtered_train_formatted.json \
    --window_size 400 \
    --save_path ./dolly-filtered/annotation_cs384_dolly-filtered_train_formatted.json

python filter.py --load_path ./dolly-filtered/annotation_cs384_dolly-filtered_train_formatted.pt \
    --save_path ./dolly-filtered/annotation_kept_cs384_dolly-filtered_train_formatted.pt
