import numpy as np
from datasets import load_dataset

def main():
    # Load the dataset from Hugging Face
    print("Loading dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k")

    # print the number of examples in the dataset
    print(f"Number of examples in the dataset: {dataset.num_rows['train']}")

    # Filter the dataset to remove entries with empty 'context'
    print("Filtering dataset...")
    filtered_dataset = dataset.filter(lambda x: len(x['context'].strip()) > 50 and len(x['context'].strip()) < 5000 and len(x['instruction']) < 360)

    # print the number of examples in the filtered dataset
    print(f"Number of examples in the filtered dataset: {filtered_dataset.num_rows['train']}")

    # Extract the 'context' field from the dataset and compute its length
    context_lengths = [len(entry['context']) for entry in filtered_dataset['train']]
    query_lengths = [len(entry['instruction']) for entry in filtered_dataset['train']]

    # Calculate basic statistics
    min_length = np.min(context_lengths)
    max_length = np.max(context_lengths)
    avg_length = np.mean(context_lengths)
    median_length = np.median(context_lengths)

    # Print the statistics
    print("#" * 32)
    print("Statistics of context lengths:")
    print(f"Minimum length: {min_length}")
    print(f"Maximum length: {max_length}")
    print(f"Average length: {avg_length:.2f}")
    print(f"Median length: {median_length}")

    # Calculate basic statistics
    min_length = np.min(query_lengths)
    max_length = np.max(query_lengths)
    avg_length = np.mean(query_lengths)
    median_length = np.median(query_lengths)

    # Print the statistics
    print("#" * 32)
    print("Statistics of query lengths:")
    print(f"Minimum length: {min_length}")
    print(f"Maximum length: {max_length}")
    print(f"Average length: {avg_length:.2f}")
    print(f"Median length: {median_length}")
    print("#" * 32)

    # Push the filtered dataset to Hugging Face as a private dataset
    print("Uploading the filtered dataset to Hugging Face Hub...")
    filtered_dataset.push_to_hub("dolly-filtered", private=False)
    print("Dataset uploaded successfully.")

if __name__ == '__main__':
    main()

