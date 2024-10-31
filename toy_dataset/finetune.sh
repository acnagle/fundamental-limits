#!/usr/env/bin bash

device=0

# finetune black-box LLM
CUDA_VISIBLE_DEVICES=$device python finetune.py --model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --epochs 4 --lr 5e-5 --lora_rank 16 --lora_alpha 16 --batch_size 16 --train_path ./mc_data/dataset/config0/train_test_set.jsonl --test_path ./mc_data/dataset/config0/val_set.jsonl --config_name config0 --ift --save --wandb --wandb_project prompt-comp
CUDA_VISIBLE_DEVICES=$device python finetune.py --model_id mistralai/Mistral-7B-Instruct-v0.2 --dtype bf16 --epochs 4 --lr 5e-6 --lora_rank 16 --lora_alpha 64 --batch_size 16 --train_path ./mc_data/dataset/config0/train_test_set.jsonl --test_path ./mc_data/dataset/config0/val_set.jsonl --config_name config0 --ift --force_tokenization --save --wandb --wandb_project prompt-comp

# finetune encoder for Selective, LLMLingua
CUDA_VISIBLE_DEVICES=$device python finetune.py --model_id EleutherAI/pythia-1b-deduped --dtype bf16 --epochs 1 --lr 5e-5 --lora_rank 32 --lora_alpha 32 --batch_size 16 --train_path ./mc_data/dataset/config0/train_test_set.jsonl --test_path ./mc_data/dataset/config0/val_set.jsonl --config_name config0 --save --wandb --wandb_project prompt-comp
CUDA_VISIBLE_DEVICES=$device python finetune.py --model_id EleutherAI/pythia-1b-deduped --dtype bf16 --epochs 1 --lr 5e-5 --lora_rank 128 --lora_alpha 64 --batch_size 16 --train_path ./mc_data/dataset/config0/train_test_set.jsonl --test_path ./mc_data/dataset/config0/val_set.jsonl --config_name config0 --force_tokenization --save --wandb --wandb_project prompt-comp

# finetune encoder for LLMLingua Query
CUDA_VISIBLE_DEVICES=$device python finetune.py --model_id EleutherAI/pythia-1b-deduped --dtype bf16 --epochs 4 --lr 1e-4 --lora_rank 128 --lora_alpha 128 --batch_size 32 --train_path ./mc_data/dataset/config0/train_test_set.jsonl --test_path ./mc_data/dataset/config0/val_set.jsonl --config_name config0 --ift --query_first --wandb --wandb_project prompt-comp --save
CUDA_VISIBLE_DEVICES=$device python finetune.py --model_id EleutherAI/pythia-1b-deduped --dtype bf16 --epochs 4 --lr 1e-4 --lora_rank 64 --lora_alpha 128 --batch_size 16 --train_path ./mc_data/dataset/config0/train_test_set.jsonl --test_path ./mc_data/dataset/config0/val_set.jsonl --config_name config0 --ift --query_first --force_tokenization --wandb --wandb_project prompt-comp --save

# finetune encoder for LLMLingua-2
CUDA_VISIBLE_DEVICES=$device python train_llmlingua2.py --model_id FacebookAI/roberta-base --dtype bf16 --epochs 12 --lr 1e-4 --lora_rank 128 --lora_alpha 128 --batch_size 32 --train_path ./mc_data/dataset/config0/train_test_set_labels.jsonl --test_path ./mc_data/dataset/config0/val_set_labels.jsonl --config_name config0 --save --wandb --wandb_project prompt-comp
CUDA_VISIBLE_DEVICES=$device python train_llmlingua2.py --model_id FacebookAI/roberta-base --dtype bf16 --epochs 12 --lr 1e-4 --lora_rank 64 --lora_alpha 128 --batch_size 32 --train_path ./mc_data/dataset/config0/train_test_set_labels_forced.jsonl --test_path ./mc_data/dataset/config0/val_set_labels_forced.jsonl --config_name config0 --save --wandb --wandb_project prompt-comp

# finetune encoder for QuerySelect, Adaptive QuerySelect
CUDA_VISIBLE_DEVICES=$device python train_llmlingua2.py --model_id FacebookAI/roberta-base --dtype bf16 --epochs 12 --lr 1e-4 --lora_rank 128 --lora_alpha 128 --batch_size 32 --train_path ./mc_data/dataset/config0/train_test_set_labels_query.jsonl --test_path ./mc_data/dataset/config0/val_set_labels_query.jsonl --config_name config0 --save --wandb --wandb_project prompt-comp
CUDA_VISIBLE_DEVICES=$device python train_llmlingua2.py --model_id FacebookAI/roberta-base --dtype bf16 --epochs 12 --lr 1e-4 --lora_rank 64 --lora_alpha 128 --batch_size 32 --train_path ./mc_data/dataset/config0/train_test_set_labels_forced_query.jsonl --test_path ./mc_data/dataset/config0/val_set_labels_query_forced.jsonl --config_name config0 --save --wandb --wandb_project prompt-comp
