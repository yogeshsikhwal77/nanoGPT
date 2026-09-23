import os
import glob
import time
import math
import torch

from nanogpt.config import load_base_config
from nanogpt.model import GPT
from nanogpt.dataset import MemmapPretrainDataset

def get_lr(step, max_steps, base_lr, warmup_steps, min_lr_ratio=0.1):
    """Linear warmup, then cosine decay down to min_lr_ratio * base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * min_lr_ratio + coeff * base_lr * (1 - min_lr_ratio)

def estimate_total_tokens(data_dir, val_fraction=0.1):
    """Rough token count for the train split, from raw file sizes (uint16 = 2 bytes/token)."""
    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(data_dir, "shard_*.bin")))
    return int((total // 2) * (1 - val_fraction))

@torch.no_grad()
def estimate_val_loss(model, val_ds, batch_size, device, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = val_ds.get_batch(batch_size, device=device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main():
    cfg = load_base_config("configs/base.yaml")
    mcfg, tcfg = cfg.model, cfg.train

    torch.manual_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA device found — training on CPU will be extremely slow.")

    model = GPT(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, betas=(0.9, 0.95), weight_decay=0.1)

    train_ds = MemmapPretrainDataset(tcfg.data_dir, mcfg.context_len, split="train")
    val_ds = MemmapPretrainDataset(tcfg.data_dir, mcfg.context_len, split="val")

    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.context_len
    total_train_tokens = estimate_total_tokens(tcfg.data_dir)
    steps_per_epoch = max(1, total_train_tokens // tokens_per_step)
    max_steps = steps_per_epoch * tcfg.epochs
    warmup_steps = max(1, min(100, max_steps // 20))

    print(f"device={device}  effective batch={tcfg.batch_size * tcfg.grad_accum} seqs "
          f"({tokens_per_step:,} tokens/step)")
    print(f"~{total_train_tokens:,} train tokens -> {steps_per_epoch:,} steps/epoch, "
          f"{max_steps:,} total steps, {warmup_steps} warmup steps")

    os.makedirs(os.path.dirname(tcfg.save_path), exist_ok=True)

    t0 = time.time()
    best_val = float("inf")

    for step in range(max_steps):
        lr = get_lr(step, max_steps, tcfg.lr, warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = train_ds.get_batch(tcfg.batch_size, device=device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / tcfg.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % 50 == 0 or step == max_steps - 1:
            elapsed = time.time() - t0
            print(f"step {step:6d}/{max_steps}  lr {lr:.2e}  train_loss {accum_loss:.4f}  "
                  f"elapsed {elapsed/60:.1f}m")

        if step % 500 == 0 or step == max_steps - 1:
            val_loss = estimate_val_loss(model, val_ds, tcfg.batch_size, device)
            print(f"          val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "config": mcfg,
                    "step": step,
                    "val_loss": val_loss,
                }, tcfg.save_path)
                print(f"          saved checkpoint -> {tcfg.save_path}")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()