import itertools
import json
from multiprocessing import Pool
from tqdm import tqdm

MAX_STRING_LEN = 20

def generate_bit_strings(length):
    """Generate all bit strings for a specific length."""
    return [''.join(bits) for bits in itertools.product('01', repeat=length)]

def worker(length):
    """Worker function to generate bit strings and wrap them with their length."""
    return length, generate_bit_strings(length)

def main(filename, num_workers=None):
    # Use a multiprocessing Pool to parallelize bit string generation
    with Pool(processes=num_workers) as pool:
        #results = pool.map(worker, range(1, MAX_STRING_LEN + 1))
        results = list(tqdm(pool.imap(worker, range(1, MAX_STRING_LEN)), total=MAX_STRING_LEN-1, desc="Generating Bit Strings"))

    # Convert list of tuples to a dictionary
    binary_strings = {str(length): bits for length, bits in results}

    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(binary_strings, f, indent=4)

if __name__ == "__main__":
    filename = "binary_strings.json"  # Output JSON file name
    num_workers = 12  # Set the desired number of worker processes
    main(filename, num_workers)
    print(f"Bit strings up to length {MAX_STRING_LEN} saved to {filename}")

