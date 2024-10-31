# NOTE: this script loads in a pre-trained LLM and runs inference over binary dataset in one of two ways: either compute the distortion of dataset with respect to LLM over all possible compresssed methods, or uses an off-the-shelf prompt compression algorithm to compress the context and then compute the distortion over the performance of the black-box LLM.

import os
import sys
import argparse
import random
import json
import itertools
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaForTokenClassification

from data.data import BinaryDataset, LLMLingua2Dataset
from utils import get_input
from LLMLingua.llmlingua import PromptCompressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Selective_Context.selective_context import SelectiveContext


MAX_LENGTH = 128    # maximum length of input to LLM
MAX_STRING_LEN = 10

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def get_loss(model, input_ids, attn_mask, labels, metric='log_loss', tokenizer=None):
    if metric == 'log_loss':
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        return outputs.loss.item(), outputs.logits
    elif metric == 'accuracy':
        assert input_ids.shape[0] == 1, "Batch size must be 1 for args.distortion=accuracy"
        gt = labels[:, -1].item()
        input_ids = input_ids[:, :-1]
        attn_mask = attn_mask[:, :-1]
        outputs = model(input_ids, attention_mask=attn_mask)
        pred = outputs.logits.argmax(-1)[:, -1].item()
        return 1 - int(pred == gt), outputs.logits      # 1 if incorrect (high distortion), 0 if correct (low distortion)

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

    enc_model_name = args.enc_model_id.split("/")[-1]
    dec_model_name = args.dec_model_id.split("/")[-1]
    filename = os.path.join(args.save_dir, (f"{enc_model_name}_" if args.mode not in ["inference", "optimal"] else "") + f"{dec_model_name}" + (f"_iter_size={args.iter_size}" if (args.mode == "llmlingua" or args.mode == "llmlingua_query") else "") + (f"_ratio={args.ratio}" if args.mode in ["llmlingua", "selective", "llmlingua_query", "llmlingua2", "query_select", "adaptive_query_select"] else "") + f"_{args.distortion}" + f"_{args.mode}" + ("_forced" if args.force_tokenization else "") + ("_ft" if args.from_finetuned else "") + ".jsonl")
    print(f"Saving results to {filename}")

    # check if filename exists. if it does, delete it
    if os.path.exists(filename):
        print(f"File {filename} already exists. Deleting it.")
        os.remove(filename)

    if args.from_finetuned:
        dec_model_id = os.path.join('./train', args.config_name, dec_model_name + ("_forced" if args.force_tokenization else "") + "_decoder")
    else:
        dec_model_id = args.dec_model_id

    dec_model = AutoModelForCausalLM.from_pretrained(dec_model_id, torch_dtype=torch_dtype).to(device)
    dec_model.eval()

    enc_model_id = os.path.join('./train', args.config_name, enc_model_name + ("_llmlingua2" if args.mode in ["llmlingua2", "query_select", "adaptive_query_select"] else "") + ("_query" if ("query" in args.mode or "dynamic" in args.mode) else "") + ("_forced" if args.force_tokenization else "") + "_encoder")  # always use the finetuned model for the encoder

    if args.mode == "inference":
        tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
    elif args.mode in ["llmlingua", "llmlingua_query"]:
        tokenizer = AutoTokenizer.from_pretrained(args.enc_model_id)
        dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        compressor = PromptCompressor(enc_model_id, device_map=device, force_tokenization=args.force_tokenization, torch_dtype=torch_dtype)
    elif args.mode in ["llmlingua2", "query_select", "adaptive_query_select"]:
        tokenizer = AutoTokenizer.from_pretrained(args.enc_model_id)
        dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        enc_model = RobertaForTokenClassification.from_pretrained(enc_model_id, torch_dtype=torch_dtype).to(device).eval()
    elif args.mode == "selective":
        tokenizer = AutoTokenizer.from_pretrained(args.enc_model_id)
        dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        compressor = SelectiveContext(enc_model_id, load_local=args.from_finetuned, force_tokenization=args.force_tokenization)
        enc_model = compressor.model
    elif args.mode == "optimal":
        tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        with open("./data/binary_strings.json", "r") as f:
            binary_strings = json.load(f)

    if args.mode in ["llmlingua2", "query_select", "adaptive_query_select"]:
        dataset = LLMLingua2Dataset(args.data_path, return_strings=True)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    else:
        dataset = BinaryDataset(tokenizer, args.force_tokenization, args.data_path, training=False, with_instructions=True)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    start_time = time.time()
    for item_idx, input_ids, attn_mask, labels, context, query, answer in tqdm(data_loader):
        item_idx = item_idx.item()
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)
        context = context[0]
        query = query[0]
        answer = answer[0]

        if args.mode == "inference":
            loss, _ = get_loss(dec_model, input_ids, attn_mask, labels, args.distortion)
            loss_dict = {"idx": item_idx, "context": context, "query": query, "answer": answer, "loss": loss}
        elif args.mode in ["llmlingua", "llmlingua_query"]:
            if args.mode == "llmlingua_query":
                num_tokens = len(tokenizer(context, add_special_tokens=False)['input_ids'])
                if num_tokens > 1:
                    compressed_prompt = compressor.compress_prompt(context, instruction="", question=query, rate=args.ratio, iterative_size=args.iter_size, concate_question=False, condition_compare=True, use_context_level_filter=False)['compressed_prompt']  # NOTE: condition_in_question is ignored for token-level compression
                else:
                    compressed_prompt = context     # if there is only 1 token in the context, llmlingua_query will throw an error, so we just use the context as is
            else:
                compressed_prompt = compressor.compress_prompt(context, instruction="", question="", rate=args.ratio, iterative_size=args.iter_size, use_context_level_filter=False)['compressed_prompt']

            if args.force_tokenization:
                ratio = len(compressed_prompt) / len(context)
            else:
                ratio = len(tokenizer(compressed_prompt, add_special_tokens=False)['input_ids']) / len(tokenizer(context, add_special_tokens=False)['input_ids']) # NOTE: "- 1" is not needed for pythia tokenizer

            input_ids2, attn_mask2, labels2 = get_input(compressed_prompt, query, answer, dec_tokenizer, force_tokenization=args.force_tokenization, with_instructions=True, query_first=False, device=device)      # NOTE: query_first is false, even for llmlingua_query, because the decoder LLM is trained on the context first

            loss, _ = get_loss(dec_model, input_ids2, attn_mask2, labels2, args.distortion)
            loss_dict = {"idx": item_idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "loss": loss, "ratio": ratio}
        elif args.mode in ["llmlingua2", "query_select", "adaptive_query_select"]:
            with torch.no_grad():
                logits = enc_model(input_ids, attn_mask).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs[labels > -100]        # filter to keep only the predictions for the context
            pos_probs = probs[:, 1]      # get the probabilities for predicting the positive label only
            if args.mode == "adaptive_query_select":
                keep_idx = pos_probs > args.ratio
                if torch.all(~keep_idx):
                    # if all tokens are removed, always choose to keep the final token
                    keep_idx[-1] = True
                top_idx = [i for i in range(len(keep_idx)) if keep_idx[i]]
            else:
                topk = max(1, round(args.ratio * len(pos_probs)))       # always keep at least one token in the compressed context
                top_probs, top_idx = torch.topk(pos_probs, k=topk, largest=True, sorted=False)
                top_idx, _ = torch.sort(top_idx, descending=False)  # NOTE: top_idx does not maintain its order in the above line, so we need to sort

            keep_tokens = input_ids[0][top_idx]
            ratio = len(keep_tokens) / len(pos_probs)

            compressed_prompt = tokenizer.decode(keep_tokens)
            input_ids2, attn_mask2, labels2 = get_input(compressed_prompt, query, answer, dec_tokenizer, force_tokenization=args.force_tokenization, with_instructions=True, device=device)

            loss, _ = get_loss(dec_model, input_ids2, attn_mask2, labels2, args.distortion)
            loss_dict = {"idx": item_idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "loss": loss, "ratio": ratio}
        elif args.mode == "selective":
            compressed_prompt, _ = compressor(context, reduce_ratio=args.ratio, reduce_level='token')

            if args.force_tokenization:
                ratio = len(compressed_prompt) / len(context)
            else:
                ratio = len(tokenizer(compressed_prompt, add_special_tokens=False)['input_ids']) / len(tokenizer(context, add_special_tokens=False)['input_ids'])

            input_ids2, attn_mask2, labels2 = get_input(compressed_prompt, query, answer, dec_tokenizer, force_tokenization=args.force_tokenization, with_instructions=True, device=device)

            loss, _ = get_loss(dec_model, input_ids2, attn_mask2, labels2, args.distortion)
            loss_dict = {"idx": item_idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "loss": loss, "ratio": ratio}
        elif args.mode == "optimal":
            context_dict = {}
            for length in tqdm(range(1, len(context))):
                all_contexts = binary_strings[str(length)]     # get all possible binary contexts of given length
                for ctx in all_contexts:
                    input_ids2, attn_mask2, labels2 = get_input(ctx, query, answer, tokenizer, force_tokenization=args.force_tokenization, with_instructions=True, device=device)

                    loss, _ = get_loss(dec_model, input_ids2, attn_mask2, labels2, args.distortion)
                    context_dict[ctx] = loss

            # get loss for the original context
            loss, _ = get_loss(dec_model, input_ids, attn_mask, labels, args.distortion)
            loss_dict = {"idx": item_idx, "context": context, "query": query, "answer": answer, "loss": loss, "context_dict": context_dict}
        else:
            raise ValueError("Invalid mode")

        with open(filename, "a") as f:
            f.write(json.dumps(loss_dict) + "\n")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on our dataset')
    parser.add_argument('--dec_model_id', type=str, required=True, help='Hugging Face model ID to use')
    parser.add_argument('--enc_model_id', type=str, required=True, help='Hugging Face model ID to use')
    parser.add_argument('--data_path', type=str, default='./data/val_set.jsonl', help='path to the dataset')
    parser.add_argument('--save_dir', type=str, default='./out', help='path to save the results')
    parser.add_argument('--config_name', type=str, required=True, help='config name for the transition matrix that was used to generate the data')
    parser.add_argument('--mode', type=str, required=True, help='operating mode for this script. Must be one of "inference", "llmlingua", "llmlingua_query", "selective", "query_select", "adaptive_query_select", "optimal"')
    parser.add_argument('--distortion', type=str, default='log_loss', help='distortion metric to use. Must be one of "log_loss", "accuracy"')
    parser.add_argument('--force_tokenization', action='store_true', help='whether to force tokenization of each symbol individually in the context')
    parser.add_argument('--from_finetuned', action='store_true', help='load a finetuned model. --model_id should be the Hugging Face ID of the base model that was finetuned')
    parser.add_argument('--dtype', type=str, default='fp16', help='data type for the model')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--ratio', type=float, default=0.5, help='compression ratio')
    parser.add_argument('--iter_size', type=int, default=4, help='iterative_size for LLMLingua algorithm')

    args = parser.parse_args()
    print(args)

    # if save_dir does not exist, create it
    args.save_dir = os.path.join(args.save_dir, args.config_name, args.mode)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if 'jsonl' not in args.data_path:
        # if data_path is a directory, use config_name to find the correct file
        if args.mode in ['llmlingua2', 'query_select', 'adaptive_query_select']:
            if args.mode in ['query_select', 'adaptive_query_select']:
                args.data_path = os.path.join(args.data_path, 'dataset', args.config_name, 'val_set_labels_query' + ('_forced' if args.force_tokenization else '') + '.jsonl')
            else:
                args.data_path = os.path.join(args.data_path, 'dataset', args.config_name, 'val_set_labels' + ('_forced' if args.force_tokenization else '') + '.jsonl')
        else:
            args.data_path = os.path.join(args.data_path, 'dataset', args.config_name, 'val_set.jsonl')
        print(f'Loading data from {args.data_path}')

    main(args)
