#!/usr/env/bin bash

device=0

ratios=(0.04 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.96 0.99 1.0)
metrics=("log_loss" "accuracy")

for metric in ${metrics[@]}; do
    # Optimal
    CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.02 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode optimal --config_name config0 --force_tokenization
    CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode optimal --config_name config0

    # Inference (no compression)
    CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode inference --config_name config0 --force_tokenization
    CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id mistralai/Mistral-7B-Instruct-v0.2 --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode inference --config_name config0
done

for ratio in ${ratios[@]}; do
    for metric in ${metrics[@]}; do
      # Selective context with finetune
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id EleutherAI/pythia-1b-deduped --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode selective --ratio $ratio --config_name config0
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id EleutherAI/pythia-1b-deduped --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode selective --ratio $ratio --config_name config0 --force_tokenization

      # LLMLingua Query
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id EleutherAI/pythia-1b-deduped --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode llmlingua_query --iter_size 2 --ratio $ratio --config_name config0
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id EleutherAI/pythia-1b-deduped --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode llmlingua_query --iter_size 2 --ratio $ratio --config_name config0 --force_tokenization

      # LLMLingua
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id EleutherAI/pythia-1b-deduped --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode llmlingua --iter_size 2 --ratio $ratio --config_name config0
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id EleutherAI/pythia-1b-deduped --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode llmlingua --iter_size 2 --ratio $ratio --config_name config0 --force_tokenization

      # LLMLingua 2
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id FacebookAI/roberta-base --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode llmlingua2 --ratio $ratio --config_name config0
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id FacebookAI/roberta-base --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode llmlingua2 --ratio $ratio --config_name config0 --force_tokenization

      # QuerySelect (Ours)
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id FacebookAI/roberta-base --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode query_select --ratio $ratio --config_name config0
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id FacebookAI/roberta-base --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode query_select --ratio $ratio --config_name config0 --force_tokenization

      # Adaptive QuerySelect (Ours)
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id FacebookAI/roberta-base --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode adaptive_query_select --ratio $ratio --config_name config0
      CUDA_VISIBLE_DEVICES=$device python inference.py --enc_model_id FacebookAI/roberta-base --dec_model_id mistralai/Mistral-7B-Instruct-v0.2 --from_finetuned --dtype bf16 --data_path ./mc_data --distortion $metric --mode adaptive_query_select --ratio $ratio --config_name config0 --force_tokenization
    done
done
