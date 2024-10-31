import json
import itertools
import random

MIN_STRING_LEN = 4
MAX_STRING_LEN = 10
NUM_SAMPLES_PER_QUERY_VAL = 200
NUM_SAMPLES_PER_QUERY_TEST = 200

def generate_binary_strings(min_len, max_len):
    for length in range(min_len, max_len + 1):
        for string in itertools.product('01', repeat=length):
            yield ''.join(string)

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

def generate_dataset():
    queries = [
        ("Count the number of 1s.", query_count_ones),
        ("Count the number of 0s.", query_count_zeros),
        ("Compute the parity.", query_compute_parity),
        ("What is the length of the longest subsequence of 0s or 1s?", query_longest_consecutive_subsequence),
        ("Is the binary string a palindrome?", query_palindrome_check),
        ("Count the number of transitions from 0 to 1 and 1 to 0.", query_count_transitions)
    ]

    dataset = []
    idx = 0
    for binary_string in generate_binary_strings(MIN_STRING_LEN, MAX_STRING_LEN):
        for query_text, query_function in queries:
            answer = query_function(binary_string)
            sample = {'idx': idx, 'context': binary_string, 'query': query_text, 'answer': str(answer)}
            dataset.append(sample)
            idx += 1

    return dataset, queries

def partition_dataset(dataset, queries):
    # Initialize dictionaries to hold the partitions
    val_samples = []
    test_samples = []
    train_samples = []

    # Group dataset by query
    query_groups = {query[0]: [] for query in queries}
    for sample in dataset:
        query_groups[sample['query']].append(sample)

    # Sample for each query type
    for query, samples in query_groups.items():
        random.shuffle(samples)
        val_samples.extend(samples[:NUM_SAMPLES_PER_QUERY_VAL])
        test_samples.extend(samples[NUM_SAMPLES_PER_QUERY_VAL:NUM_SAMPLES_PER_QUERY_VAL+NUM_SAMPLES_PER_QUERY_TEST])
        train_samples.extend(samples[NUM_SAMPLES_PER_QUERY_VAL+NUM_SAMPLES_PER_QUERY_TEST:])

    return val_samples, test_samples, train_samples

if __name__ == "__main__":
    dataset, queries = generate_dataset()
    val_set, test_set, train_set = partition_dataset(dataset, queries)

    # Save to separate JSONL files
    with open('val_set.jsonl', 'w') as f:
        for item in val_set:
            f.write(json.dumps(item) + '\n')

    with open('test_set.jsonl', 'w') as f:
        for item in test_set:
            f.write(json.dumps(item) + '\n')

    with open('train_set.jsonl', 'w') as f:
        for item in train_set:
            f.write(json.dumps(item) + '\n')

    with open('train_test_set.jsonl', 'w') as f:
        for item in test_set + train_set:
            f.write(json.dumps(item) + '\n')

