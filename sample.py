import torch
from tokenizers import Tokenizer
from nanogpt.config import load_base_config
from nanogpt.model import GPT
from tokenizers import decoders
import torch.nn.functional as F

temperature = 0.8
top_k = 40

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = load_base_config("configs/base.yaml").model

# Load model checkpoint
ckpt = torch.load("checkpoints/base_model_33M.pt", map_location=device, weights_only=False)
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = Tokenizer.from_file("tokenizer.json")
prompt = "Once upon a time, there was a little bird named"
tokens = tokenizer.encode(prompt).ids
idx = torch.tensor([tokens], dtype=torch.long, device=device)

for _ in range(120):
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            logits, _ = model(idx[:, -cfg.context_len:])
        
        # Scale logits by temperature
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        top_k_val = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k_val)
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Sample from probability distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, next_token), dim=1)


print("\n--- GENERATED STORY ---")
tokenizer.decoder = decoders.ByteLevel()
print(tokenizer.decode(idx[0].tolist()))