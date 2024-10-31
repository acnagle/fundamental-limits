import json
import argparse
from itertools import combinations
from transformers import AutoTokenizer
from multiprocessing import Pool

def tokenize_context(context, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokens = tokenizer.tokenize(context)
    return tokens

def generate_combinations(tokens):
    all_combinations = []
    n = len(tokens)
    for k in range(1, n + 1):
        for comb in combinations(tokens, k):
            all_combinations.append(comb)
    return all_combinations

def process_prompt(item):
    prompt_name, context, tokenizer = item
    tokens = tokenizer.tokenize(context)
    #tokens = tokenize_context(context, model_id)
    combinations = generate_combinations(tokens)
    return (prompt_name, combinations)

def main(args):
    # Load data
    with open('data.json', 'r') as f:
        data = json.load(f)

    # Extract prompts and contexts
    extracted_data = {key: item['context'] for key, item in data.items()}

    # Prepare data for multiprocessing
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tasks = [(prompt, context, tokenizer) for prompt, context in extracted_data.items()]

    # Use Pool for multiprocessing
    num_workers = len(tasks)
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_prompt, tasks)

    # Convert results to a dictionary
    output = {prompt: comb_list for prompt, comb_list in results}

    # Save to JSON
    with open('compressed_contexts.json', 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate token combinations from prompts')
    parser.add_argument('--model_id', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='Model ID for the tokenizer')
    args = parser.parse_args()
    main(args)

