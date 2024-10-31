#!/usr/env/bin bash

device=1

ratios=(0.04 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.96 0.99 1.0)
metrics=("rougeL" "bertscore")

# Inference (no compression)
CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --distortion rougeL --mode inference
CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --distortion bertscore --mode inference

# Beam search ("optimality")
CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --distortion rougeL --mode beam_search --beam_search_budget 4000 --num_beams 5
# NOTE: we don't report beam search results on BertScore in the paper
#CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --distortion bertscore --mode beam_search --beam_search_budget 4000 --num_beams 5

for ratio in ${ratios[@]}; do
    for metric in ${metrics[@]}; do
      # LLMLingua
      CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --distortion $metric --mode llmlingua --iter_size 100 --ratio $ratio

      # LLMLingua Query
      CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --distortion $metric --mode llmlingua_query --iter_size 100 --ratio $ratio

      # LLMLingua-2
      CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id microsoft/llmlingua-2-xlm-roberta-large-meetingbank --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --distortion $metric --mode llmlingua2 --ratio $ratio

      # QuerySelect (Ours)
      CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id ./LLMLingua/experiments/llmlingua2/model_training/models/xlm_roberta_large_dolly-filtered2_kept_cs384_fp32 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype fp32 --distortion $metric --mode query_select --ratio $ratio

      # Adaptive QuerySelect (Ours)
      CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id ./LLMLingua/experiments/llmlingua2/model_training/models/xlm_roberta_large_dolly-filtered2_kept_cs384_fp32 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype fp32 --distortion $metric --mode adaptive_query_select --ratio $ratio

      # Selective
      CUDA_VISIBLE_DEVICES=$device python narrativeqa.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --distortion $metric --mode selective --ratio $ratio
    done
done
