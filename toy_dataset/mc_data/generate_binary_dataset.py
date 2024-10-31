import os
import json
import itertools
import random
from tqdm import tqdm

MIN_VAL_STRING_LEN = 4
MAX_VAL_STRING_LEN = 10
NUM_SAMPLES_PER_QUERY_VAL = 200

MIN_TRAIN_STRING_LEN = 2
MAX_TRAIN_STRING_LEN = 32
NUM_SAMPLES_PER_QUERY_TRAIN = 2000

def load_transition_matrix(filepath):
    with open(filepath, 'r') as file:
        config = json.load(file)
    return config["transition_matrix"]

def generate_markov_binary_sequence(transition_matrix, length):
    state = random.choice([0, 1])
    sequence = [str(state)]
    for _ in range(1, length):
        state = random.choices([0, 1], weights=transition_matrix[state])[0]
        sequence.append(str(state))
    return ''.join(sequence)

def query_count_ones(string):
    return string.count('1')

def query_count_zeros(string):
    return string.count('0')

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

def predict_next_bit(sequence, matrix):
    last_bit = int(sequence[-1])
    next_bit = random.choices([0, 1], weights=matrix[last_bit])[0]
    return str(next_bit)

def generate_dataset(min_length, max_length, num_samples_per_query, transition_matrix):
    queries = [
        ("Count the number of 1s.", query_count_ones),
        ("Count the number of 0s.", query_count_zeros),
        ("Compute the parity.", query_compute_parity),
        ("What is the length of the longest subsequence of 0s or 1s?", query_longest_consecutive_subsequence),
        ("Is the binary string a palindrome?", query_palindrome_check),
        ("Count the number of transitions from 0 to 1 and 1 to 0.", query_count_transitions),
        ("Predict the next bit.", lambda x: predict_next_bit(x, transition_matrix)),
    ]

    dataset = []
    idx = 0
    for query_text, query_function in tqdm(queries):
        for _ in range(num_samples_per_query):
            length = random.randint(min_length, max_length)
            if query_text == 'Is the binary string a palindrome?' and random.random() < 0.5:
                # Make the string a palindrome with 50% probability for palindrome-related queries
                length = max(1, length // 2)        # ensure that the length is at least 1
                binary_string = generate_markov_binary_sequence(transition_matrix, length)
                binary_string = binary_string + binary_string[::-1]
            else:
                binary_string = generate_markov_binary_sequence(transition_matrix, length)

            answer = query_function(binary_string)

            sample = {'idx': idx, 'context': binary_string, 'query': query_text, 'answer': str(answer)}
            dataset.append(sample)
            idx += 1

    return dataset

if __name__ == "__main__":
    transition_matrix_file_path = './matrix/config0.json'
    transition_matrix = load_transition_matrix(transition_matrix_file_path)
    train_dataset = generate_dataset(MIN_TRAIN_STRING_LEN, MAX_TRAIN_STRING_LEN, NUM_SAMPLES_PER_QUERY_TRAIN, transition_matrix)
    test_dataset = generate_dataset(MIN_VAL_STRING_LEN, MAX_VAL_STRING_LEN, NUM_SAMPLES_PER_QUERY_VAL, transition_matrix)
    val_dataset = generate_dataset(MIN_VAL_STRING_LEN, MAX_VAL_STRING_LEN, NUM_SAMPLES_PER_QUERY_VAL, transition_matrix)

    config_name = transition_matrix_file_path.split('/')[-1].split('.')[0]
    data_path = os.path.join('./dataset', config_name)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Save to separate JSONL files
    with open(os.path.join(data_path, 'val_set.jsonl'), 'w') as f:
        for item in val_dataset:
            f.write(json.dumps(item) + '\n')

    with open(os.path.join(data_path, 'test_set.jsonl'), 'w') as f:
        for item in test_dataset:
            f.write(json.dumps(item) + '\n')

    with open(os.path.join(data_path, 'train_set.jsonl'), 'w') as f:
        for item in train_dataset:
            f.write(json.dumps(item) + '\n')

    with open(os.path.join(data_path, 'train_test_set.jsonl'), 'w') as f:
        for item in test_dataset + train_dataset:
            f.write(json.dumps(item) + '\n')

