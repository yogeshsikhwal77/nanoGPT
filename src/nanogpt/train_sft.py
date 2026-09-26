import argparse
import os
import torch
from torch.utils.data import DataLoader
from nanogpt.config import load_sft_config
from nanogpt.model import GPT
from nanogpt.dataset import SFTDataset

def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/sft.yaml", help="Path to SFT YAML config")
    args = parser.parse_args()

    cfg = load_sft_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {cfg.model_path} via {args.config} on {device}...")
    
    ckpt = torch.load(cfg.model_path, map_location=device, weights_only=False)
    mcfg = ckpt["config"]
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()
    
    ds = SFTDataset(cfg.data_path, mcfg.context_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    print(f"Starting SFT: {len(ds)} samples, {cfg.epochs} epochs")
    
    step = 0
    for epoch in range(cfg.epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
                
            loss = loss / cfg.grad_accum
            loss.backward()
            
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs} | Step {step} | Loss: {loss.item() * cfg.grad_accum:.4f}")
            step += 1

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": mcfg,
    }, cfg.save_path)
    
    print(f"SFT complete! Checkpoint saved to {cfg.save_path}")

if __name__ == "__main__":
    main()