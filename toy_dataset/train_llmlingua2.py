import os
import argparse
import random
import json
import itertools
import time

import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForTokenClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from data.data import LLMLingua2Dataset


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attn_mask, labels in tqdm(data_loader):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            total_loss += outputs.loss.item()
    model.train()
    return total_loss / len(data_loader)

def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = RobertaForTokenClassification.from_pretrained(args.model_id, torch_dtype=torch_dtype, num_labels=2).to(device)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=True,
        init_lora_weights='Gaussian',
        bias="none",
        modules_to_save=['classifier']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = LLMLingua2Dataset(args.train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    test_dataset = LLMLingua2Dataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epochs * len(train_loader))

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

    model.train()
    start_time = time.time()
    avg_loss = 0
    num_batches = 0
    for epoch in range(1, args.epochs+1):
        for batch_idx, (input_ids, attn_mask, labels) in enumerate(tqdm(train_loader)):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            avg_loss += loss.item()
            num_batches += 1

            if args.wandb and (batch_idx % args.log_interval == 0):
                val_loss = evaluate(model, test_loader, device)
                log_entry = {
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "avg_train_loss": avg_loss / num_batches,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                wandb.log(log_entry)
                avg_loss = 0
                num_batches = 0

    end_time = time.time()
    print(f"Total time to train: {end_time - start_time:.2f}s")

    if args.save:
        save_dir = os.path.join('./train', args.config_name, args.model_id.split('/')[-1] + "_llmlingua2" + ("_query" if "query" in args.train_path else "") + ("_forced" if "forced" in args.train_path else "") + "_encoder")
        os.makedirs(save_dir, exist_ok=True)

        # merge and save the model
        print("Merging and saving the model")
        merged_model = model.merge_and_unload(progressbar=True)
        merged_model.save_pretrained(save_dir)

        # save the args
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on our dataset')
    parser.add_argument('--model_id', type=str, required=True, help='Hugging Face model ID to use')
    parser.add_argument('--train_path', type=str, default='./data/train_set.jsonl', help='path to the training dataset')
    parser.add_argument('--test_path', type=str, default='./data/test_set.jsonl', help='path to the testing dataset')
    parser.add_argument('--config_name', type=str, required=True, help='config name for the transition matrix that was used to generate the data')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--save', action='store_true', help='save the model')
    parser.add_argument('--dtype', type=str, default='fp16', help='data type for the model')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--lora_rank', type=int, default=16, help='rank of LoRA')
    parser.add_argument('--lora_alpha', type=int, default=16, help='alpha of LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='dropout of LoRA')
    parser.add_argument('--log_interval', type=int, default=100, help='number of batches between logging')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--wandb_name', type=str, default=None, help='name of the wandb run')
    parser.add_argument('--wandb_project', type=str, default='train-llmlingua2', help='wandb project name')

    args = parser.parse_args()

    if args.wandb and args.wandb_name is None:
        model_name = args.model_id.split("/")[-1]
        args.wandb_name = f'{model_name}_' + time.strftime('%m-%d-%Y_%H:%M:%S')
        print(f'Note: logging with wandb but no name specified. Using "{args.wandb_name}" as the run name.')

    print(args)
    main(args)
