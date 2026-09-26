import json
import os
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

# 1. Download TinyStories for pre-training
print("Downloading TinyStories dataset (pre-training)...")
ds_base = load_dataset("roneneldan/TinyStories", split="train[:300000]")

raw_train_file = os.path.join("data", "raw", "train.txt")
with open(raw_train_file, "w", encoding="utf-8", newline="\n") as f:
    for item in ds_base:
        f.write(item["text"].strip() + "\n<|endoftext|>\n")
print(f"Pre-training data saved to {raw_train_file} ({len(ds_base):,} stories)")

# 2. Download TinyStories-Instruct for Q&A seeds
print("Downloading TinyStories-Instruct dataset (alignment seeds)...")
ds_instruct_raw = load_dataset("roneneldan/TinyStories-Instruct")

# Explicitly select the 'train' split if it exists
ds_instruct = ds_instruct_raw["train"] if "train" in ds_instruct_raw else ds_instruct_raw

instruct_file = os.path.join("data", "raw", "TinyStories-Instruct.json")
pairs = []

for item in ds_instruct:
    # Safely handle if the dataset yields raw strings instead of dictionaries
    if isinstance(item, str):
        pairs.append({
            "story": item,
            "prompt": "",
            "response": ""
        })
    else:
        pairs.append({
            "story": item.get("story", item.get("text", "")),
            "prompt": item.get("instruction", item.get("question", "")),
            "response": item.get("output", item.get("response", ""))
        })

with open(instruct_file, "w", encoding="utf-8") as f:
    json.dump(pairs, f, indent=2)
print(f"Instruction seeds saved to {instruct_file}")