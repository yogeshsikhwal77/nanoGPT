import argparse
import math
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from nanogpt.model import GPT
from nanogpt.dataset import MemmapPretrainDataset


@torch.no_grad()
def estimate_val_loss(model: GPT, val_ds: MemmapPretrainDataset, batch_size: int, device: str, iters: int = 50) -> float:
    """Computes average cross-entropy loss over validation shards."""
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = val_ds.get_batch(batch_size, device=device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def benchmark_latency(model: GPT, tokenizer: Tokenizer, prompt: str, device: str, max_new_tokens: int = 50, runs: int = 10) -> tuple[float, float, str]:
    """Measures end-to-end inference latency and per-token generation latency."""
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("<|eos|>")

    # Warmup runs
    for _ in range(2):
        temp_idx = idx.clone()
        for _ in range(10):
            logits, _ = model(temp_idx[:, -model.cfg.context_len:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            temp_idx = torch.cat((temp_idx, next_token), dim=1)

    latencies = []
    generated_text = ""

    for _ in range(runs):
        curr_idx = idx.clone()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        tokens_generated = 0
        for _ in range(max_new_tokens):
            logits, _ = model(curr_idx[:, -model.cfg.context_len:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_idx = torch.cat((curr_idx, next_token), dim=1)
            tokens_generated += 1
            if eos_id is not None and next_token.item() == eos_id:
                break

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        if not generated_text:
            out_ids = curr_idx[0, len(input_ids):].tolist()
            generated_text = tokenizer.decode(out_ids).replace("<|eos|>", "").strip()

    avg_total_ms = sum(latencies) / len(latencies)
    avg_per_token_ms = avg_total_ms / max(tokens_generated, 1)
    return avg_total_ms, avg_per_token_ms, generated_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base or SFT NanoGPT models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/tokenized", help="Path to tokenized .bin shards")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--iters", type=int, default=50, help="Validation sampling iterations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating checkpoint: {args.checkpoint} on {device}")

    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.decoder = ByteLevel()

    # 2. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = ckpt["config"]
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3. Calculate Validation Loss & Perplexity
    val_ds = MemmapPretrainDataset(args.data_dir, mcfg.context_len, split="val")
    val_loss = estimate_val_loss(model, val_ds, args.batch_size, device, iters=args.iters)
    perplexity = math.exp(val_loss)

    # 4. Measure Inference Latency & Sample Generation
    sample_prompt = (
        "<|story|> Tim lost his red ball in the garden. He searched behind the shed and found it under the oak tree. "
        "<|question|> Where did Tim find his ball? <|answer|>"
    )
    total_ms, per_token_ms, sample_ans = benchmark_latency(model, tokenizer, sample_prompt, device)

    # 5. Output Summary
    print("\n" + "=" * 50)
    print("                 EVALUATION RESULTS               ")
    print("=" * 50)
    print(f"Parameters:         {model.num_params():,}")
    print(f"Context Window:     {mcfg.context_len} tokens")
    print(f"Validation Loss:    {val_loss:.4f}")
    print(f"Perplexity:         {perplexity:.2f}")
    print(f"Inference Latency:  {total_ms:.2f} ms ({per_token_ms:.2f} ms/token)")
    print("-" * 50)
    print(f"Prompt Q&A Test:")
    print(f"Generated Answer:   {sample_ans}")
    print("=" * 50)


if __name__ == "__main__":
    main()