import json

from datasets import load_dataset

# the `cache_dir` is optional
dataset = load_dataset("Open-Orca/OpenOrca", split="train")
with open("/root/datasets/OpenOrca/data.jsonl", "w+") as f:
    for document in dataset:
        if document["id"].startswith("cot."):
            f.write(json.dumps(document) + "\n")
