import argparse
import json
import os
import torch
import ollama
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from nanogpt.model import GPT

@torch.no_grad()
def generate_candidate(model, tokenizer, prompt_str, device, max_tokens=30, temp=0.8):
    model.eval()
    input_ids = tokenizer.encode(prompt_str).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("<|eos|>")
    
    for _ in range(max_tokens):
        idx_cond = idx[:, -model.cfg.context_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temp
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_tok), dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break
            
    out_ids = idx[0, len(input_ids):].tolist()
    return tokenizer.decode(out_ids).replace("<|eos|>", "").strip()

def judge_with_qwen(story: str, question: str, cand_a: str, cand_b: str) -> str:
    system_prompt = (
        "You are an impartial evaluator assessing short answers for reading comprehension. "
        "Choose the answer that is more factually accurate, grammatically coherent, and directly grounded in the story. "
        "Respond ONLY with valid JSON in this exact structure: {\"winner\": \"A\"} or {\"winner\": \"B\"}."
    )
    user_prompt = (
        f"STORY: {story}\n"
        f"QUESTION: {question}\n"
        f"CANDIDATE A: {cand_a}\n"
        f"CANDIDATE B: {cand_b}\n"
        f"Which candidate is better?"
    )
    try:
        res = ollama.chat(
            model="qwen2.5:3b",
            format="json",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}
        )
        data = json.loads(res["message"]["content"])
        return data.get("winner", "").strip().upper()
    except Exception:
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", type=str, default="checkpoints/sft_model_33M.pt")
    parser.add_argument("--seed-data", type=str, default="data/sft/aligned_pairs.json")
    parser.add_argument("--output", type=str, default="data/dpo/dpo_pairs.json")
    parser.add_argument("--max-pairs", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer.decoder = ByteLevel()

    ckpt = torch.load(args.sft_checkpoint, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(args.seed_data, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    dpo_dataset = []
    print(f"Generating preference pairs via LLM-as-a-Judge for {min(len(qa_data), args.max_pairs)} items...")

    for i, item in enumerate(qa_data[:args.max_pairs]):
        story, question = item["story"], item["question"]
        prompt = f"<|story|> {story} <|question|> {question} <|answer|>"

        cand_a = generate_candidate(model, tokenizer, prompt, device)
        cand_b = generate_candidate(model, tokenizer, prompt, device)

        if not cand_a or not cand_b or cand_a == cand_b:
            continue

        winner = judge_with_qwen(story, question, cand_a, cand_b)
        if winner == "A":
            chosen, rejected = cand_a, cand_b
        elif winner == "B":
            chosen, rejected = cand_b, cand_a
        else:
            continue

        dpo_dataset.append({
            "story": story,
            "question": question,
            "chosen": chosen,
            "rejected": rejected
        })
        print(f"[{len(dpo_dataset)}/{args.max_pairs}] Pair recorded. Winner: {winner}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dpo_dataset, f, indent=2)
    print(f"Saved {len(dpo_dataset)} preference pairs to {args.output}")

if __name__ == "__main__":
    main()