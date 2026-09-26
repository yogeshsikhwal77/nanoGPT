import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from nanogpt.model import GPT

class DPODataset(Dataset):
    def __init__(self, data_path: str, context_len: int, tokenizer_path: str = "tokenizer.json"):
        self.context_len = context_len
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

    def __len__(self):
        return len(self.data)

    def _encode_sequence(self, prompt_text: str, response_text: str):
        p_ids = self.tokenizer.encode(prompt_text).ids
        r_ids = self.tokenizer.encode(" " + response_text).ids + [self.eos_id]

        total = len(p_ids) + len(r_ids)
        if total > self.context_len:
            p_ids = p_ids[total - self.context_len:]

        x_ids = p_ids + r_ids
        # Mask out prompt tokens so loss is only calculated over response tokens
        mask = [0] * (len(p_ids) - 1) + [1] * len(r_ids)

        pad_amt = self.context_len - len(x_ids)
        if pad_amt > 0:
            x_ids.extend([self.pad_id] * pad_amt)
            mask.extend([0] * pad_amt)

        return (
            torch.tensor(x_ids[:-1], dtype=torch.long),
            torch.tensor(x_ids[1:], dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32)
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"<|story|> {item['story']} <|question|> {item['question']} <|answer|>"
        cx, cy, cmask = self._encode_sequence(prompt, item["chosen"])
        rx, ry, rmask = self._encode_sequence(prompt, item["rejected"])
        return cx, cy, cmask, rx, ry, rmask

def compute_sequence_logps(model, x, y, mask):
    logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=y.unsqueeze(2)).squeeze(2)
    return (per_token_logps * mask).sum(dim=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", type=str, default="checkpoints/sft_model_33M.pt")
    parser.add_argument("--dpo-data", type=str, default="data/dpo/dpo_pairs.json")
    parser.add_argument("--save-path", type=str, default="checkpoints/dpo_model_33M.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running DPO on {device}...")

    # Load policy and reference models
    ckpt = torch.load(args.sft_checkpoint, map_location=device, weights_only=False)
    mcfg = ckpt["config"]

    policy_model = GPT(mcfg).to(device)
    policy_model.load_state_dict(ckpt["model"])
    policy_model.train()

    ref_model = GPT(mcfg).to(device)
    ref_model.load_state_dict(ckpt["model"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    dataset = DPODataset(args.dpo_data, mcfg.context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            cx, cy, cmask, rx, ry, rmask = [t.to(device) for t in batch]

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                # Policy log probabilities
                pi_chosen = compute_sequence_logps(policy_model, cx, cy, cmask)
                pi_rejected = compute_sequence_logps(policy_model, rx, ry, rmask)

                # Reference log probabilities
                with torch.no_grad():
                    ref_chosen = compute_sequence_logps(ref_model, cx, cy, cmask)
                    ref_rejected = compute_sequence_logps(ref_model, rx, ry, rmask)

                # DPO loss calculation
                pi_ratios = pi_chosen - pi_rejected
                ref_ratios = ref_chosen - ref_rejected
                logits = pi_ratios - ref_ratios
                loss = -F.logsigmoid(args.beta * logits).mean()

            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 5 == 0:
                print(f"Epoch {epoch+1} | Step {step} | DPO Loss: {loss.item() * args.grad_accum:.4f}")
            step += 1

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        "model": policy_model.state_dict(),
        "config": mcfg,
        "step": step,
    }, args.save_path)
    print(f"DPO alignment complete! Checkpoint saved to {args.save_path}")

if __name__ == "__main__":
    main()