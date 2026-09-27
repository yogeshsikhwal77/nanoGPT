import json
import random

# Set a fixed seed so the shuffle order is reproducible
random.seed(42)

# Load your 1-question dataset
with open("data/sft/aligned_pairs.json", "r", encoding="utf-8") as f:
    data_1q = json.load(f)

# Load your 3-question dataset
with open("data/sft/aligned_pairs_1000.json", "r", encoding="utf-8") as f:
    data_3q = json.load(f)

# Combine and shuffle
merged_data = data_1q + data_3q
random.shuffle(merged_data)

# Save the randomized dataset
with open("data/sft/merged_pairs.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2)

print(f"Merged and shuffled {len(data_1q)} single-Q and {len(data_3q)} triple-Q pairs.")
print(f"Saved {len(merged_data)} total randomized pairs to data/sft/merged_pairs.json")