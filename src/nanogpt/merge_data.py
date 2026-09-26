import json

# Load your large 1-question dataset
with open("data/sft/aligned_pairs.json", "r") as f:
    data_1q = json.load(f)

# Load your new 3-question dataset (replace with your exact filename)
with open("data/sft/aligned_pairs_1000.json", "r") as f:
    data_3q = json.load(f)

# Combine them
merged_data = data_1q + data_3q

# Save the combined dataset
with open("data/sft/merged_pairs.json", "w") as f:
    json.dump(merged_data, f, indent=2)

print(f"Merged {len(data_1q)} and {len(data_3q)} pairs into {len(merged_data)} total pairs.")