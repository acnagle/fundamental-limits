import json
import argparse
import itertools
from itertools import groupby

from tqdm import tqdm
from transformers import AutoTokenizer
import torch

# Set up argument parser
parser = argparse.ArgumentParser(description="Process JSONL files for BERT-style training.")
parser.add_argument('--model_id', type=str, default='FacebookAI/roberta-base', help="Model ID for loading the tokenizer")
parser.add_argument('--force_tokenization', action='store_true', help="Force the toeknizer to encode each bit of the context separately")
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

# Function to load JSONL data
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Function to tokenize and assign labels based on query
def process_entry(entry):
    if args.force_tokenization:
        context_tokens = [tokenizer.tokenize(item)[0] for item in entry['context']]
    else:
        context_tokens = tokenizer.tokenize(entry['context'])

    query_tokens = tokenizer.tokenize(entry['query'])
    combined_tokens = context_tokens + query_tokens
    combined_ids = tokenizer.convert_tokens_to_ids(combined_tokens)
    attention_mask = [1] * len(combined_ids)

    context_tokens = [token.replace('\u2581', '') for token in context_tokens]  # remove whitespace in context tokens

    # Initialize labels with -100 (ignore label)
    labels = [-100] * len(combined_tokens)

    # Define a function to find the longest subsequence
    def find_longest_subsequence(binary_string):
        max_count = 0
        max_char = ''
        current_count = 1
        current_char = binary_string[0]

        # Variables to store the starting index and length of the longest subsequence
        max_start = 0
        max_length = 1

        # Traverse the binary string to find the longest subsequence
        for i in range(1, len(binary_string)):
            if binary_string[i] == binary_string[i - 1]:
                current_count += 1
            else:
                if current_count > max_count:
                    max_count = current_count
                    max_char = current_char
                    max_start = i - current_count
                    max_length = current_count
                current_char = binary_string[i]
                current_count = 1

        # Check the last run
        if current_count > max_count:
            max_count = current_count
            max_char = current_char
            max_start = len(binary_string) - current_count
            max_length = current_count

        return max_start, max_length, max_char

    # Define function to identify transitions
    def find_transitions(binary_string):
        transitions = []
        for i in range(1, len(binary_string)):
            if binary_string[i] != binary_string[i - 1]:  # Detect transition
                transitions.append(i)
        return transitions

    has_positive_label = False      # Flag to indicate if a positive label is present. Each entry should have at least one positive label
    transitions = find_transitions(entry['context'])
    token_start_index = 0
    for i, token in enumerate(context_tokens):
        token_end_index = token_start_index + len(token)
        # Check if the token overlaps with any transition index
        if any(token_start_index <= t < token_end_index for t in transitions):
            labels[i] = 1
            has_positive_label = True
        else:
            labels[i] = 0
        token_start_index = token_end_index

    # If the first included token does not have both 0 and 1, include the first non '' or '_' token
    first_has_both = False
    for i, token in enumerate(context_tokens):
        if labels[i] == 1:
            if '0' in token and '1' in token:
                first_has_both = True
            break

    if not first_has_both:
        if context_tokens[0] == '':
            labels[1] = 1   # always include the first non '' or '_' token
        else:
            labels[0] = 1

    if not has_positive_label:
        # Assume last token is to be predicted
        last_index = len(context_tokens) - 1
        labels[last_index] = 1
        labels[:last_index] = [0] * last_index

    # Pad the tokens and labels to the maximum length of 128
    while len(combined_ids) < 128:
        combined_ids.append(tokenizer.pad_token_id)
        labels.append(-100)
        attention_mask.append(0)

    item = {
        'idx': entry['idx'],
        'context': entry['context'],
        'query': entry['query'],
        'answer': entry['answer'],
        'input_ids': combined_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }

    return item

# Main function to process datasets
def process_datasets(file_names, data_dir):
    for file_name in tqdm(file_names):
        path = f"{data_dir}/{file_name}"
        data = load_jsonl(path)
        processed_data = [process_entry(entry) for entry in data]
        # Save processed data
        save_path = f"{data_dir}/{file_name.replace('.jsonl', '_labels') + ('_forced' if args.force_tokenization else '') + '.jsonl'}"
        with open(save_path, 'w') as f:
            for entry in processed_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Processed {file_name}")

# List of dataset filenames
datasets = ['train_set.jsonl', 'test_set.jsonl', 'train_test_set.jsonl', 'val_set.jsonl']

data_dir = './dataset/config0'

# Process each dataset
process_datasets(datasets, data_dir)

