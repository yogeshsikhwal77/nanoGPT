import argparse
import math
import os
import torch
from torch.utils.data import DataLoader, random_split
from nanogpt.config import load_sft_config
from nanogpt.model import GPT
from nanogpt.dataset import SFTDataset


def get_lr(step, max_steps, base_lr, warmup_steps, min_lr_ratio):
    """Linear warmup, then cosine decay down to min_lr_ratio * base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * min_lr_ratio + coeff * base_lr * (1 - min_lr_ratio)


@torch.no_grad()
def estimate_val_loss(model, val_dl, device, max_batches=50):
    """Average loss over the held-out split. Leaves model in the mode it found it in."""
    was_training = model.training
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_dl):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(model, mcfg, step, val_loss, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": mcfg, "step": step, "val_loss": val_loss},
        save_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/sft.yaml", help="Path to SFT YAML config")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Fraction of aligned_pairs held out for validation")
    parser.add_argument("--eval-every", type=int, default=100, help="Optimizer steps between validation passes")
    args = parser.parse_args()

    cfg = load_sft_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {cfg.model_path} via {args.config} on {device}...")

    ckpt = torch.load(cfg.model_path, map_location=device, weights_only=False)
    mcfg = ckpt["config"]
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    full_ds = SFTDataset(cfg.data_path, mcfg.context_len)
    val_size = max(1, int(len(full_ds) * args.val_fraction))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)

    warmup_steps = getattr(cfg, "warmup_steps", 50)
    min_lr_ratio = getattr(cfg, "min_lr_ratio", 0.1)
    opt_steps_per_epoch = max(1, len(dl) // cfg.grad_accum)
    max_opt_steps = opt_steps_per_epoch * cfg.epochs

    print(f"Starting SFT: {len(train_ds)} train / {len(val_ds)} val samples, {cfg.epochs} epochs")
    print(f"~{max_opt_steps} optimizer steps total, {warmup_steps} warmup steps")

    step = 0
    opt_step = 0
    best_val = float("inf")
    grad_pending = False  # True whenever backward() ran since the last opt.step()

    for epoch in range(cfg.epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)

            loss = loss / cfg.grad_accum
            loss.backward()
            grad_pending = True

            if (step + 1) % cfg.grad_accum == 0:
                lr = get_lr(opt_step, max_opt_steps, cfg.lr, warmup_steps, min_lr_ratio)
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                grad_pending = False
                opt_step += 1

                if opt_step % args.eval_every == 0:
                    val_loss = estimate_val_loss(model, val_dl, device)
                    print(f"          [opt step {opt_step}] val_loss {val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(model, mcfg, step, val_loss, cfg.save_path)
                        print(f"          saved checkpoint -> {cfg.save_path} (new best val_loss)")

            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs} | Step {step} | Loss: {loss.item() * cfg.grad_accum:.4f}")
            step += 1

        # Previously the final partial accumulation group of an epoch was silently
        # dropped if the batch count wasn't a multiple of grad_accum. Flush it here.
        if grad_pending:
            lr = get_lr(opt_step, max_opt_steps, cfg.lr, warmup_steps, min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            grad_pending = False
            opt_step += 1

        val_loss = estimate_val_loss(model, val_dl, device)
        print(f"Epoch {epoch+1} complete | val_loss {val_loss:.4f} (best so far: {best_val:.4f})")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, mcfg, step, val_loss, cfg.save_path)
            print(f"          saved checkpoint -> {cfg.save_path} (new best val_loss)")

    print(f"SFT complete! Best val_loss={best_val:.4f}, checkpoint saved to {cfg.save_path}")


if __name__ == "__main__":
    main()