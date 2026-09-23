import sys
import torch

sys.path.insert(0, "src")
from nanogpt.config import load_base_config
from nanogpt.model import GPT
from nanogpt.dataset import TinyOverfitDataset


def main():
    cfg = load_base_config("configs/base.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(cfg.model).to(device)
    ds = TinyOverfitDataset(cfg.train.data_dir, cfg.model.context_len, num_samples=100)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"Overfitting on {ds.x.shape[0]} fixed samples, device={device}")
    for step in range(300):
        x, y = ds.get_batch(16, device=device)
        logits, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == 299:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    print("\nDone. Loss should approach 0. If it plateaus above ~1.0, something in the data or model is wrong.")


if __name__ == "__main__":
    main()