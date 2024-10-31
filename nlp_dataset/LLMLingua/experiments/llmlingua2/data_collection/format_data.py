import json
import os

from datasets import load_dataset

dataset = load_dataset("acnagle/dolly-filtered", split="train")
data = []
for idx, instance in enumerate(dataset):
    temp = {}
    temp["idx"] = idx
    temp["prompt"] = instance["context"]
    temp["query"] = instance["instruction"]
    data.append(temp)
os.makedirs("./dolly-filtered", exist_ok=True)
json.dump(
    data,
    open("./dolly-filtered/dolly-filtered_train_formatted.json", "w"),
    indent=4,
)
