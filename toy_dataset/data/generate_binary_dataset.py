import json

import torch
from torch.nn.functional import softmax
from tqdm import tqdm
import random

MAX_STRING_LEN = 10
MIN_STRING_LEN = 4
NUM_SAMPLES_PER_QUERY = 200
SEED = 1337

random.seed(SEED)
torch.manual_seed(SEED)


def generate_binary_string(length, transition_matrix=None):
    if transition_matrix is not None:
        binary_string = ''
        current_symbol = random.choice('01')
        for _ in range(length):
            binary_string += current_symbol
            current_symbol = '0' if random.random() < transition_matrix[int(current_symbol), 0] else '1'
        return binary_string
    else:
        return ''.join(random.choice('01') for _ in range(length))

def generate_transition_matrix(num_states):
    transition_matrix = torch.rand(num_states, num_states)
    transition_matrix = transition_matrix / transition_matrix.sum(dim=1, keepdim=True)
    return transition_matrix

def query_count_ones(string):
    return string.count('1')

def query_count_zeros(string):
    return string.count('0')

def query_predict_next_symbol(string, transition_matrix):
    current_symbol = string[-1]
    next_symbol = '0' if random.random() < transition_matrix[int(current_symbol), 0] else '1'
    return next_symbol

def query_compute_parity(string):
    count_ones = string.count('1')
    return count_ones % 2

def query_longest_consecutive_subsequence(string):
    max_consecutive = 0
    consecutive = 0
    current_symbol = None
    for symbol in string:
        if symbol == current_symbol:
            consecutive += 1
        else:
            consecutive = 1
            current_symbol = symbol

        max_consecutive = max(max_consecutive, consecutive)
    return max_consecutive

def query_palindrome_check(string):
    return 'Yes' if string == string[::-1] else 'No'

def query_count_transitions(string):
    transitions = sum(1 for i in range(1, len(string)) if string[i] != string[i-1])
    return transitions

def generate_dataset():
    queries = [
        ("Count the number of 1s.", query_count_ones),
        ("Count the number of 0s.", query_count_zeros),
        #("Predict the next symbol, either a 1 or a 0.", query_predict_next_symbol),
        ("Compute the parity.", query_compute_parity),
        ("What is the length of the longest subsequence of 0s or 1s?", query_longest_consecutive_subsequence),
        ("Is the binary string a palindrome?", query_palindrome_check),
        ("Count the number of transitions from 0 to 1 and 1 to 0.", query_count_transitions)
    ]

    dataset = []
    idx = 0
    for query, query_function in tqdm(queries):
        for _ in range(NUM_SAMPLES_PER_QUERY):
            if query_function == query_palindrome_check:
                if random.random() < 0.5:
                    # Make the string a palindrome with 50% probability for palindrome-related queries
                    length = random.randint(MIN_STRING_LEN // 2, MAX_STRING_LEN // 2)
                    binary_string = generate_binary_string(length)
                    binary_string = binary_string + binary_string[::-1]
            else:
                length = random.randint(MIN_STRING_LEN, MAX_STRING_LEN)

            if query_function == query_predict_next_symbol:
                transition_matrix = generate_transition_matrix(num_states=2)
                binary_string = generate_binary_string(length, transition_matrix)
                answer = query_function(binary_string, transition_matrix)
            else:
                binary_string = generate_binary_string(length)
                answer = query_function(binary_string)

            sample = {'idx': idx, 'context': binary_string, 'query': query, 'answer': str(answer)}
            dataset.append(sample)
            idx += 1

    # get query strings and turn them into a dictionary
    query_strings = [query[0] for query in queries]
    query_dict = {idx: query for idx, query in enumerate(query_strings)}

    return dataset, query_dict

if __name__ == "__main__":
    dataset, query_dict = generate_dataset()
    #print(dataset)

    with open('binary_dataset.jsonl', 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

    with open('queries.json', 'w') as f:
        json.dump(query_dict, f)
