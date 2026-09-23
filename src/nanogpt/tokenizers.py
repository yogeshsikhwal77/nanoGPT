import os 
import glob
import argparse
import numpy as np
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,processors

SPECIAL_TOKENS = ["<|unk|>", "<|pad|>", "<|eos|>", "<|story|>", "<|question|>", "<|answer|>"]

def train(vocab_size: int, data_path: str, save_path: str = "tokenizer.json"):
    """trains a BPE tokenizer on the raw text files"""

    print(f"Training tokenizer with vocab size {vocab_size} on {data_path}...")

    # 1. intilaize a bype pair encoding
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    #2 satandard bytelevel pre tokenizers
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3.configure the trainer with your specific vocab size and special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS
    )

    # 4.find all text files in the raw data dir
    files = glob.glob(os.path.join(data_path, "**/*.txt"),recursive=True)
    if not files:
        raise ValueError(f'NO .txt files found in {data_path}')

    tokenizer.train(files,trainer)
    tokenizer.save(save_path)

    print(f"Tokenizer saved to {save_path}")


def encode(input_dir: str, output_dir: str, tokenizer_path: str = "tokenizer.json"):
    """Encodes raw text files into memory-mapped .bin arrays."""

    print(f"Encoding data from {input_dir} to {output_dir}...")
    os.makedirs(output_dir,exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|eos|>")

    files = glob.glob(os.path.join(input_dir, "**/*.txt"), recursive=True)
    if not files:
        raise ValueError(f"No .txt files found in {input_dir}")

    total_tokens = 0

    for i,file_path in enumerate(files):
        tokens = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens.extend(tokenizer.encode(line).ids)

        tokens.append(eos_id)
        # Vocab size is 10,000, so uint16 (max 65,535) is perfectly sized and saves memory
        arr = np.array(tokens,dtype = np.uint16)

        out_name = f"shard_{i:04d}.bin"
        out_path = os.path.join(output_dir, out_name)
        arr.tofile(out_path)
        
        total_tokens += len(tokens)
        print(f"Saved {out_name} ({len(tokens)} tokens)")

    print(f"Done! Encoded {total_tokens} total tokens across {len(files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoGPT Tokenizer Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    parser_train = subparsers.add_parser("train", help="Train the tokenizer")
    parser_train.add_argument("--vocab-size", type=int, default=10000)
    parser_train.add_argument("--data-path", type=str, required=True)
    
    # Encode command
    parser_encode = subparsers.add_parser("encode", help="Tokenize data into .bin shards")
    parser_encode.add_argument("--input-dir", type=str, required=True)
    parser_encode.add_argument("--output-dir", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(vocab_size=args.vocab_size, data_path=args.data_path)
    elif args.command == "encode":
        encode(input_dir=args.input_dir, output_dir=args.output_dir)