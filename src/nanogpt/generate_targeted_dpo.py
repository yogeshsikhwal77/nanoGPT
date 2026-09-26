import argparse
import json
import os
import torch
import ollama
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from nanogpt.model import GPT

@torch.no_grad()
def get_model_prediction(model, tokenizer, prompt_str, device, max_tokens=25):
    model.eval()
    input_ids = tokenizer.encode(prompt_str).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("<|eos|>")

    for _ in range(max_tokens):
        idx_cond = idx[:, -model.cfg.context_len:]
        logits, _ = model(idx_cond)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, next_tok), dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break

    out_ids = idx[0, len(input_ids):].tolist()
    return tokenizer.decode(out_ids).replace("<|eos|>", "").strip()

def generate_targeted_qa(story: str) -> list:
    system_prompt = (
        "You are a reading comprehension expert. Read the story and generate 2 specific questions:\n"
        "1. A TEMPORAL question using words like 'after', 'before', 'then', or 'what did they do first?'.\n"
        "2. A MULTI-ITEM or LIST question (e.g., 'What two things...', 'Name both items...').\n"
        "Provide direct, concise answers (under 8 words each) grounded strictly in the story.\n"
        "Respond ONLY with valid JSON in this exact structure:\n"
        '{"qa_pairs": [{"question": "...", "answer": "..."}, {"question": "...", "answer": "..."}]}'
    )
    try:
        res = ollama.chat(
            model="qwen2.5:3b",
            format="json",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"STORY: {story}"}
            ],
            options={"temperature": 0.3}
        )
        return json.loads(res["message"]["content"]).get("qa_pairs", [])
    except Exception:
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/dpo_model_33M.pt")
    parser.add_argument("--seed-data", type=str, default="data/sft/aligned_pairs.json")
    parser.add_argument("--output", type=str, default="data/dpo/targeted_dpo_pairs.json")
    parser.add_argument("--max-stories", type=int, default=150)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer.decoder = ByteLevel()

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])

    with open(args.seed_data, "r", encoding="utf-8") as f:
        seed_items = json.load(f)

    # Extract unique stories
    stories = list({item["story"]: None for item in seed_items if len(item.get("story", "")) > 60}.keys())[:args.max_stories]

    targeted_pairs = []
    print(f"Generating targeted temporal & list pairs across {len(stories)} stories...")

    for i, story in enumerate(stories):
        qa_pairs = generate_targeted_qa(story)
        for qa in qa_pairs:
            q, true_ans = qa.get("question", "").strip(), qa.get("answer", "").strip()
            if not q or not true_ans:
                continue

            prompt = f"<|story|> {story} <|question|> {q} <|answer|>"
            model_ans = get_model_prediction(model, tokenizer, prompt, device)

            # Keep only pairs where the 33M model failed or gave a substantially different answer
            if model_ans.lower() != true_ans.lower() and len(model_ans) > 1:
                targeted_pairs.append({
                    "story": story,
                    "question": q,
                    "chosen": true_ans,
                    "rejected": model_ans
                })
                print(f"[{len(targeted_pairs)}] Trapped mistake on Q: {q}")
                print(f"    Chosen:   {true_ans}")
                print(f"    Rejected: {model_ans}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(targeted_pairs, f, indent=2)

    print(f"\nSaved {len(targeted_pairs)} targeted contrastive pairs to {args.output}")

if __name__ == "__main__":
    main()