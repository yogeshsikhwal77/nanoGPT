import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from nanogpt.model import GPT


class StoryQAInference:
    def __init__(self, checkpoint_path: str = "checkpoints/sft_model_33M.pt", tokenizer_path: str = "tokenizer.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.decoder = ByteLevel()
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        
        # 2. Load Checkpoint & Model
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.cfg = ckpt["config"]
        self.model = GPT(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def generate(self, story: str, question: str, temperature: float = 0.2, max_tokens: int = 50) -> tuple[str, int, float]:
        prompt = f"<|story|> {story} <|question|> {question} <|answer|>"
        input_ids = self.tokenizer.encode(prompt).ids
        
        # Truncate context from the start if it exceeds model limits
        if len(input_ids) > self.cfg.context_len:
            input_ids = input_ids[-self.cfg.context_len:]
            
        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        tokens_generated = 0
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.cfg.context_len:]
            logits, _ = self.model(idx_cond)
            
            if temperature == 0.0:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            idx = torch.cat((idx, next_token), dim=1)
            tokens_generated += 1
            
            if self.eos_id is not None and next_token.item() == self.eos_id:
                break
                
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        output_tokens = idx[0, len(input_ids):].tolist()
        raw_text = self.tokenizer.decode(output_tokens)
        cleaned_answer = raw_text.replace("<|eos|>", "").replace("<|answer|>", "").strip()
        
        return cleaned_answer, tokens_generated, elapsed_ms