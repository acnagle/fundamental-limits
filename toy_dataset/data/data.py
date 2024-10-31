import os
import json

import torch
from torch.utils.data import Dataset

from utils import get_input


class BinaryDataset(Dataset):
    def __init__(self, tokenizer, force_tokenization, filepath="binary_dataset.jsonl", training=False, with_instructions=True, query_first=False):
        # Check if the dataset file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist. Please run generate_binary_dataset.py to create the dataset.")

        self.data = []
        with open(filepath, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))

        self.max_length = 128   # max length of the input sequence; 128 is plenty for this task
        self.tokenizer = tokenizer
        self.force_tokenization = force_tokenization
        self.training = training
        self.with_instructions = with_instructions
        self.query_first = query_first

    def __len__(self):
        # Return the number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve an item by its index
        if idx >= len(self.data):
            raise IndexError("Index out of range")

        item = self.data[idx]
        item_idx = item['idx']      # unique identifier for the item
        context = item['context']
        query = item['query']
        answer = item['answer']

        input_ids, attention_mask, labels = get_input(context, query, answer, self.tokenizer, self.force_tokenization, with_instructions=self.with_instructions, query_first=self.query_first, get_item=True)

        if self.training:
            # pad the input and labels to the same length
            input_ids = torch.nn.functional.pad(input_ids, (0, self.max_length - input_ids.shape[0]), value=self.tokenizer.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, self.max_length - attention_mask.shape[0]), value=0)
            labels = torch.nn.functional.pad(labels, (0, self.max_length - labels.shape[0]), value=-100)
            return input_ids, attention_mask, labels
        else:
            return item_idx, input_ids, attention_mask, labels, context, query, answer


class LLMLingua2Dataset(Dataset):
    def __init__(self, file_path, return_strings=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist. Please run generate_binary_dataset.py to create the dataset.")

        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))

        self.max_length = 128   # max length of the input sequence; 128 is plenty for this task
        self.return_strings = return_strings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range")

        item = self.data[idx]

        input_ids = torch.tensor(item['input_ids'])
        attn_mask = torch.tensor(item['attention_mask'])
        labels = torch.tensor(item['labels'])

        if self.return_strings:
            item_idx = item['idx']
            context = item['context']
            query = item['query']
            answer = item['answer']
            return item_idx, input_ids, attn_mask, labels, context, query, answer
        else:
            return input_ids, attn_mask, labels
