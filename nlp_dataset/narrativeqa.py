import os
import sys
import argparse
import random
import json
import time
from math import floor
from typing import Tuple
from dataclasses import dataclass, field

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from LLMLingua.llmlingua import PromptCompressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Selective_Context.selective_context import SelectiveContext


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def get_loss(model, input_ids, attn_mask, labels, metric='log_loss', tokenizer=None, helper_model=None):
    if metric == 'log_loss':
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        return outputs.loss.item(), outputs.logits
    elif metric == 'generation':
        assert helper_model is not None, "helper_model must be provided for args.distortion = generation"
        outputs = model.generate(input_ids, attention_mask=attn_mask, max_length=input_ids.shape[1] + 32, num_beams=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
        outputs = outputs[:, input_ids.shape[1]:]       # only keep the generated part of the output
        output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ground_truth = tokenizer.decode(labels[0], skip_special_tokens=True)
        embeddings = helper_model.encode([output_str, ground_truth])
        similarity = pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        similarity = (similarity + 1) / 2   # normalize similarity to [0, 1]
        return 1 - similarity, output_str      # 1 - similarity because we want smaller distortion for higher similarity
    elif metric in ['rougeL', 'bertscore']:
        assert helper_model is not None, "helper_model must be provided for args.distortion = rougeL or bertscore"
        outputs = model.generate(input_ids, attention_mask=attn_mask, max_length=input_ids.shape[1] + 32, num_beams=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
        outputs = outputs[:, input_ids.shape[1]:]       # only keep the generated part of the output
        output_str = [tokenizer.decode(outputs[0], skip_special_tokens=True)]
        ground_truth = [tokenizer.decode(labels[0], skip_special_tokens=True)]
        if metric == 'rougeL':
            return 1 - helper_model.compute(predictions=output_str, references=ground_truth)['rougeL'], output_str[0]
        elif metric == 'bertscore':
            return 1 - helper_model.compute(predictions=output_str, references=ground_truth, lang='en')['f1'][0], output_str[0]

def get_input(context, query, answer, tokenizer, metric, device='cpu'):
    if metric == "log_loss":
        input_str = f"Instruction: Generate an answer based on the context and query provided. If the context does not provide enough information to answer the query, or if you don't know the answer, just reply with \"I don't know.\" Please give answers that are succinct and give a simple answer to the query without further elaboration.\n\nContext: {context}\nQuery: {query}\nAnswer: {answer}"
        input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"].to(device)
        answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        labels = input_ids.clone()
        labels[:, :-answer_ids.shape[1]] = -100
        attn_mask = torch.ones_like(input_ids)
    elif metric in ["generation", "rougeL", "bertscore"]:
        input_str = f"Instruction: Generate an answer based on the context and query provided. If the context does not provide enough information to answer the query, or if you don't know the answer, just reply with \"I don't know.\" Please give answers that are succinct and give a simple answer to the query without further elaboration.\n\nContext: {context}\nQuery: {query}\nAnswer: "
        input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"].to(device)
        labels = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        attn_mask = torch.ones_like(input_ids)
    else:
        raise ValueError(f"Invalid metric {metric}")

    return input_ids, attn_mask, labels

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
    if enc_model_name == dec_model_name:        # NOTE: this will not work for two different finetunings of the same model
        model_name = enc_model_name
    else:
        model_name = None

    if args.part_idx is not None:
        part_ext = f"_part_{args.part_idx}_of_{args.num_parts - 1}"
    else:
        part_ext = ""

    filename = os.path.join(args.save_dir, (f"{enc_model_name}_" if (args.mode not in ["inference", "beam_search"] and model_name is not None) else "") + f"{dec_model_name}" + (f"_iter_size={args.iter_size}" if (args.mode == "llmlingua" or args.mode == "llmlingua_query") else "") + (f"_ratio={args.ratio}" if args.mode in ["llmlingua", "selective", "llmlingua_query", "llmlingua2", "query_select",  "adaptive_query_select"] else "") + f"_{args.distortion}" + f"_{args.mode}" + f"_narrativeqa" + part_ext + ".jsonl")
    print(f"Saving results to {filename}")

    # check if filename exists. if it does, delete it
    if os.path.exists(filename) and not args.resume:
        print(f"File {filename} already exists. Deleting it.")
        os.remove(filename)
    elif os.path.exists(filename) and args.resume:
        print(f"Resuming from {filename}")

    if args.mode == "inference":
        dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        dec_model = AutoModelForCausalLM.from_pretrained(args.dec_model_id, torch_dtype=torch_dtype).to(device)
        dec_model.eval()
    elif args.mode in ["llmlingua", "llmlingua_query"]:
        enc_tokenizer = AutoTokenizer.from_pretrained(args.enc_model_id)
        dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        compressor = PromptCompressor(args.enc_model_id, device_map=device, torch_dtype=torch_dtype)
        compressor.model.eval()
        if model_name:
            dec_model = compressor.model
        else:
            dec_model = AutoModelForCausalLM.from_pretrained(args.dec_model_id, torch_dtype=torch_dtype).to(device).eval()
    elif args.mode in ["llmlingua2", "query_select", "adaptive_query_select"]:
        if args.mode == "llmlingua2":
            assert args.enc_model_id == "microsoft/llmlingua-2-xlm-roberta-large-meetingbank", "args.enc_model_id must be microsoft/llmlingua-2-xlm-roberta-large-meetingbank for llmlingua2"
            compressor = PromptCompressor(args.enc_model_id, device_map=device, torch_dtype=torch.float32, use_llmlingua2=True)
            enc_tokenizer = AutoTokenizer.from_pretrained(args.enc_model_id)
            dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        else:
            compressor = PromptCompressor(args.enc_model_id, device_map=device, torch_dtype=torch_dtype, use_llmlingua2=True)
            enc_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
            dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        compressor.model.eval()
        dec_model = AutoModelForCausalLM.from_pretrained(args.dec_model_id, torch_dtype=torch_dtype).to(device).eval()
    elif args.mode == "selective":
        enc_tokenizer = AutoTokenizer.from_pretrained(args.enc_model_id)
        dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        compressor = SelectiveContext(args.enc_model_id, torch_dtype=torch_dtype)
        dec_model = compressor.model.eval()
    elif args.mode == "beam_search":
        dec_tokenizer = AutoTokenizer.from_pretrained(args.dec_model_id)
        dec_model = AutoModelForCausalLM.from_pretrained(args.dec_model_id, torch_dtype=torch_dtype).to(device).eval()
    else:
        raise ValueError(f"Invalid mode {args.mode}")

    data = load_dataset("deepmind/narrativeqa", split="validation")

    if args.part_idx is not None:
        num_examples = len(data)
        part_size = num_examples // args.num_parts
        end_idx = (args.part_idx + 1) * part_size
        if args.resume:
            # read in the last line of the file
            with open(filename, "r") as f:
                jsonlines = f.readlines()
                if len(jsonlines) > 0:
                    last_line = jsonlines[-1]
                    last_line = json.loads(last_line)
                    start_idx = last_line["idx"] + 1
            data = data.select(range(start_idx, end_idx))
        else:
            start_idx = args.part_idx * part_size
            data = data.select(range(start_idx, end_idx))
    else:
        start_idx = 0

    if args.distortion == "generation":
        helper_model = SentenceTransformer("all-mpnet-base-v2", device=device)
        dec_model.generation_config.pad_token_ids = dec_tokenizer.pad_token_id
        with open("./data/examples.json", "r") as f:
            few_shot_examples = json.load(f)
        template = "Instruction: Generate an answer based on the context and query provided.\n\n"
        template += "\n\n".join(f"Context: {info['context']}\nQuery: {info['query']}\nAnswer: {info['answer']}" for info in few_shot_examples.values())
        template += "\n\nContext: {}\nQuery: {}\nAnswer:"
    elif args.distortion == "rougeL":
        helper_model = evaluate.load("rouge", experiment_id=args.part_idx)
    elif args.distortion == "bertscore":
        helper_model = evaluate.load("bertscore", experiment_id=args.part_idx)
    elif args.distortion == "log_loss":
        helper_model = None
    else:
        raise ValueError(f"Invalid distortion metric {args.distortion}")

    total_time = time.time()
    for idx in tqdm(range(len(data))):
        context = data[idx]["document"]["summary"]["text"]
        query = data[idx]["question"]["text"]
        answers = data[idx]["answers"]
        answer = random.choice(answers)["text"]     # randomly select an answer

        start_time = time.time()
        if args.mode == "inference":
            end_time = time.time()
            input_ids, attn_mask, labels = get_input(context, query, answer, dec_tokenizer, metric=args.distortion, device=device)
            loss, output = get_loss(dec_model, input_ids, attn_mask, labels, args.distortion, tokenizer=dec_tokenizer, helper_model=helper_model)
            if args.distortion == "log_loss":
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "loss": loss, "time": end_time - start_time}
            else:
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "response": output, "loss": loss, "time": end_time - start_time}
        elif args.mode in ["llmlingua", "llmlingua_query"]:
            if args.mode == "llmlingua_query":
                compressed_prompt = compressor.compress_prompt(
                    context,
                    instruction="",
                    question=query,
                    rate=args.ratio,
                    iterative_size=args.iter_size,
                    concate_question=False,
                    condition_compare=True,
                    use_context_level_filter=False
                )['compressed_prompt']  # NOTE: condition_in_question is ignored for token-level compression
            else:
                compressed_prompt = compressor.compress_prompt(
                    context,
                    instruction="",
                    question="",
                    rate=args.ratio,
                    iterative_size=args.iter_size,
                    use_context_level_filter=False
                )['compressed_prompt']
            end_time = time.time()

            ratio = len(enc_tokenizer(compressed_prompt, add_special_tokens=False)['input_ids']) / len(enc_tokenizer(context, add_special_tokens=False)['input_ids'])

            input_ids, attn_mask, labels = get_input(compressed_prompt, query, answer, dec_tokenizer, metric=args.distortion, device=device)

            loss, output = get_loss(dec_model, input_ids, attn_mask, labels, args.distortion, tokenizer=dec_tokenizer, helper_model=helper_model)
            if args.distortion == "log_loss":
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "loss": loss, "ratio": ratio, "time": end_time - start_time}
            else:
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "response": output, "loss": loss, "ratio": ratio, "time": end_time - start_time}
        elif args.mode == "llmlingua2":
            compressed_prompt = compressor(context, rate=args.ratio)['compressed_prompt']
            end_time = time.time()

            ratio = len(enc_tokenizer(compressed_prompt, add_special_tokens=False)['input_ids']) / len(enc_tokenizer(context, add_special_tokens=False)['input_ids'])

            input_ids, attn_mask, labels = get_input(compressed_prompt, query, answer, dec_tokenizer, metric=args.distortion, device=device)

            loss, output = get_loss(dec_model, input_ids, attn_mask, labels, args.distortion, tokenizer=dec_tokenizer, helper_model=helper_model)
            if args.distortion == "log_loss":
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "loss": loss, "ratio": ratio, "time": end_time - start_time}
            else:
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "response": output, "loss": loss, "ratio": ratio, "time": end_time - start_time}
        elif args.mode in ["query_select", "adaptive_query_select"]:
            compressed_prompt = compressor(context, question=query, rate=args.ratio, adaptive_query_select=(args.mode == "adaptive_query_select"))['compressed_prompt']
            end_time = time.time()

            ratio = len(enc_tokenizer(compressed_prompt, add_special_tokens=False)['input_ids']) / len(enc_tokenizer(context, add_special_tokens=False)['input_ids'])

            input_ids, attn_mask, labels = get_input(compressed_prompt, query, answer, dec_tokenizer, metric=args.distortion, device=device)

            loss, output = get_loss(dec_model, input_ids, attn_mask, labels, args.distortion, tokenizer=dec_tokenizer, helper_model=helper_model)
            if args.distortion == "log_loss":
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "loss": loss, "ratio": ratio, "time": end_time - start_time}
            else:
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "response": output, "loss": loss, "ratio": ratio, "time": end_time - start_time}
        elif args.mode == "selective":
            compressed_prompt, _ = compressor(
                context,
                reduce_ratio=args.ratio,
                reduce_level='token'
            )
            end_time = time.time()

            ratio = len(enc_tokenizer(compressed_prompt, add_special_tokens=False)['input_ids']) / len(enc_tokenizer(context, add_special_tokens=False)['input_ids'])

            input_ids, attn_mask, labels = get_input(compressed_prompt, query, answer, dec_tokenizer, metric=args.distortion, device=device)

            loss, output = get_loss(dec_model, input_ids, attn_mask, labels, args.distortion, tokenizer=dec_tokenizer, helper_model=helper_model)
            if args.distortion == "log_loss":
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "loss": loss, "ratio": ratio, "time": end_time - start_time}
            else:
                loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "compressed_context": compressed_prompt, "response": output, "loss": loss, "ratio": ratio, "time": end_time - start_time}
        elif args.mode == "beam_search":
            num_chunks = int(np.ceil(-1/2 + (1/2) * np.sqrt(1 + 8 * args.beam_search_budget / args.num_beams)))
            num_chunks = max(num_chunks, 1)     # there should be at least one chunk

            @dataclass(order=True)
            class MaskCandidate:
                score: float
                mask: Tuple[int, ...] = field(compare=False)
                generated_output: str = field(compare=False)

            init_input_ids = dec_tokenizer(context, return_tensors="pt")['input_ids'].to(device)
            # we shouldn't have more chunks than there are tokens in the input sequence. If there are, we clip the number of tokens to the length of the input and allocate the remainder of the budget to the number of beams
            if num_chunks > init_input_ids.shape[1]:
                num_chunks = init_input_ids.shape[1]
                if args.fix_beams:
                    num_beams = args.num_beams
                else:
                    num_beams = int((2 * args.beam_search_budget) / (num_chunks * (num_chunks + 1)))
                    print(f"Number of chunks exceeds the number of tokens in the input sequence. Setting number of chunks to {num_chunks} and allocating the remainder of the budget to the number of beams.")
                    print(f"Number of beams: {num_beams}")
            else:
                num_beams = args.num_beams

            chunk_size = init_input_ids.shape[1] / num_chunks     # size of each chunk
            chunk_start_idx = [floor(i * chunk_size) for i in range(num_chunks)] + [init_input_ids.shape[1]]        # start index for each chunk. the last element is the end of the init_input_ids
            init_mask = tuple([1] * init_input_ids.shape[1])     # initialize the mask to all 1s (i.e., no tokens are masked)

            full_input_ids, init_attn_mask, init_labels = get_input(context, query, answer, dec_tokenizer, metric=args.distortion, device=device)

            init_loss, init_output = get_loss(dec_model, full_input_ids, init_attn_mask, init_labels, args.distortion, tokenizer=dec_tokenizer, helper_model=helper_model)
            init_candidate = MaskCandidate(init_loss, init_mask, init_output)
            beam = [init_candidate]

            context_dict = {}

            for step_idx in tqdm(range(num_chunks)):
                curr_candidates = []
                seen_masks = set()
                for candidate in beam:
                    # the children are generated by flipping the mask of a chunk and then flattening the mask
                    for chunk_idx in range(num_chunks):
                        if candidate.mask[chunk_start_idx[chunk_idx]] == 1:         # check if the chunk is unmasked
                            new_mask = list(candidate.mask)
                            new_mask[chunk_start_idx[chunk_idx]:chunk_start_idx[chunk_idx + 1]] = [0] * (chunk_start_idx[chunk_idx + 1] - chunk_start_idx[chunk_idx])   # mask the chunk
                            new_mask_tuple = tuple(new_mask)
                            assert len(new_mask) == len(init_mask), "Mask length mismatch"

                            # avoid duplicate candidates
                            if new_mask_tuple in seen_masks:
                                continue
                            seen_masks.add(new_mask_tuple)

                            kept_mask = torch.tensor(new_mask_tuple, dtype=torch.bool, device=device)
                            kept_ids = init_input_ids.masked_select(kept_mask)
                            compressed_prompt = dec_tokenizer.decode(kept_ids, skip_special_tokens=True)
                            input_ids, attn_mask, labels = get_input(compressed_prompt, query, answer, dec_tokenizer, metric=args.distortion, device=device)
                            loss, output = get_loss(dec_model, input_ids, attn_mask, labels, args.distortion, tokenizer=dec_tokenizer, helper_model=helper_model)
                            ratio = len(kept_ids) / len(init_input_ids[0])
                            new_candidate = MaskCandidate(loss, new_mask, output)
                            curr_candidates.append(new_candidate)

                            context_dict[compressed_prompt] = (output, loss, ratio)

                curr_candidates = sorted(curr_candidates)     # sorted in ascending order. lower score/loss is better
                beam = curr_candidates[:args.num_beams]       # keep the top candidates

            loss_dict = {"idx": start_idx + idx, "context": context, "query": query, "answer": answer, "context_dict": context_dict}

            #best_candidate = max(beam, key=lambda x: x.score)
            # NOTE: the above line is not needed since we are writing to file at each step. we do not require that the best candidate is found in the beam at the end of the loop. In fact, as more tokens are removed, we suspect that the distortion values may increase, so the best candidate may have been removed from the beam
        else:
            raise ValueError("Invalid mode")

        with open(filename, "a") as f:
            f.write(json.dumps(loss_dict) + "\n")

    end_time = time.time()
    print(f"Total time taken: {end_time - total_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on our dataset')
    parser.add_argument('--dec_model_id', type=str, required=True, help='Hugging Face model ID to use')
    parser.add_argument('--enc_model_id', type=str, required=True, help='Hugging Face model ID to use')
    parser.add_argument('--save_dir', type=str, default='./out/narrativeqa', help='path to save the results')
    parser.add_argument('--mode', type=str, required=True, help='operating mode for this script. Must be one of "inference", "llmlingua", "llmlingua_query", "selective", "query_select", "adaptive_query_select", "beam_search"')
    parser.add_argument('--beam_search_budget', type=int, default=16000, help='maximum number of inference calls to make for beam search')
    parser.add_argument('--num_beams', type=int, default=5, help='number of beams to use for beam search')
    parser.add_argument('--fix_beams', action='store_true', help='whether to fix the number of beams to use for beam search, or adaptively increase the number of beams if the budget allows')
    parser.add_argument('--distortion', type=str, default='log_loss', help='distortion metric to use. Must be one of "log_loss", "rougeL", "bertscore", "generation"')
    parser.add_argument('--dtype', type=str, default='fp16', help='data type for the model')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--ratio', type=float, default=0.5, help='compression ratio')
    parser.add_argument('--iter_size', type=int, default=4, help='iterative_size for LLMLingua algorithm')
    parser.add_argument('--num_parts', type=int, default=None, help='number of partitions the dataset will be split into')
    parser.add_argument('--part_idx', type=int, default=None, help='index of the partition to run inference on')
    parser.add_argument('--resume', action='store_true', help='resume compression from where it left off')

    args = parser.parse_args()
    print(args)

    if args.num_parts is not None and args.part_idx is not None:
        assert args.part_idx < args.num_parts, "part_idx must be less than num_parts. part_idx starts from 0."

    if (args.num_parts is not None and args.part_idx is None) or (args.num_parts is None and args.part_idx is not None):
        raise ValueError("Both num_parts and part_idx must be provided if either one is provided")

    # if save_dir does not exist, create it
    args.save_dir = os.path.join(args.save_dir, args.mode)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(args)
