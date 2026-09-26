import argparse
import json
import os
import ollama
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from nanogpt.model import GPT


@torch.no_grad()
def generate_answer(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    context_len: int = 512,
) -> str:
    """Generates an answer from the base model autoregressively until <|eos|>."""
    eos_id = tokenizer.token_to_id("<|eos|>")
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop to model context length
        idx_cond = idx[:, -context_len:]
        logits, _ = model(idx_cond)
        
        # Pull logits for the last token and scale by temperature
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    # Decode only the generated completion
    gen_tokens = idx[0, len(input_ids):].tolist()
    text = tokenizer.decode(gen_tokens)
    return text.replace("<|eos|>", "").replace("<|answer|>", "").strip()


def evaluate_with_llama(story: str, question: str, candidate: str, reference: str = "") -> dict:
    """Uses local Llama 3.2 1B (Ollama) at temperature 0 to fact-check the candidate."""
    system_instruction = (
        "You are an impartial judge fact-checking answers to children's stories. "
        "Determine if the CANDIDATE ANSWER accurately answers the QUESTION based ONLY on the STORY. "
        "Respond ONLY with valid JSON in this exact structure: "
        '{"acceptable": true, "reason": "brief reason"}'
    )

    user_content = f"STORY: {story}\nQUESTION: {question}\nCANDIDATE ANSWER: {candidate}"
    if reference:
        user_content += f"\nREFERENCE ANSWER: {reference}"

    try:
        response = ollama.chat(
            model="llama3.2:1b",
            format="json",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            options={"temperature": 0.0},
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        return {"acceptable": False, "reason": str(e)}


def load_instruct_data(data_path: str):
    """Loads seeds and stitches them back together if they were shattered line-by-line."""
    with open(data_path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
        except json.JSONDecodeError:
            f.seek(0)
            raw_data = [json.loads(line) for line in f if line.strip()]

    stitched_seeds = []
    current_summary = ""
    current_story = []
    
    for item in raw_data:
        # Extract the text, regardless of whether the parser put it in 'story' or 'text'
        text = str(item.get("story", "") or item.get("text", "")).strip()
        
        # When a new story block begins, save the previous one if it's complete
        if text.startswith("Features:") or text.startswith("Words:"):
            if current_summary and current_story:
                stitched_seeds.append({
                    "story": " ".join(current_story).strip(),
                    "instruction": "Can you summarize this story?",
                    "output": current_summary
                })
            current_summary = ""
            current_story = []
            
        elif text.startswith("Summary:"):
            current_summary = text.replace("Summary:", "").strip()
            
        elif text.startswith("Story:"):
            parsed_story = text.replace("Story:", "").strip()
            if parsed_story:
                current_story.append(parsed_story)
                
        elif current_summary:
            # Any remaining text after a summary is part of the story
            if text and not text.startswith("Features:") and not text.startswith("Words:"):
                current_story.append(text)

    # Catch the final story in the file
    if current_summary and current_story:
        stitched_seeds.append({
            "story": " ".join(current_story).strip(),
            "instruction": "Can you summarize this story?",
            "output": current_summary
        })

    # Fallback if the data wasn't shattered to begin with
    if not stitched_seeds and raw_data and (raw_data[0].get("prompt") or raw_data[0].get("question")):
        return raw_data
        
    return stitched_seeds

def main():
    parser = argparse.ArgumentParser(description="Best-of-k Rejection Sampling Judge")
    parser.add_argument("--model-path", type=str, required=True, help="Path to base model .pt checkpoint")
    parser.add_argument("--instruct-data", type=str, required=True, help="Path to TinyStories-Instruct data")
    parser.add_argument("--output", type=str, default="data/sft/aligned_pairs.json", help="Output path")
    parser.add_argument("--candidates", type=int, default=3, help="Number of candidate answers per prompt (k)")
    parser.add_argument("--max-samples", type=int, default=300, help="Max seed stories to process")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json", help="Path to trained tokenizer")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load trained Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.decoder = ByteLevel()

    # 2. Load Checkpoint and Initialize Model
    print(f"Loading checkpoint from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location=device,weights_only=False)
    cfg = ckpt["config"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3. Load Q&A Seeds
    print(f"Loading seed instructions from {args.instruct_data}...")
    raw_seeds = load_instruct_data(args.instruct_data)[:args.max_samples]

    temperatures = [0.7, 0.8, 0.9]
    aligned_dataset = []

    print(f"Running rejection sampling (k={args.candidates}) on {len(raw_seeds)} stories...")
    for i, item in enumerate(raw_seeds):
        # Support common key variations in TinyStories-Instruct
        story = item.get("story") or item.get("context", "")
        question = item.get("instruction") or item.get("question") or item.get("prompt", "")
        ref_answer = item.get("output") or item.get("summary") or item.get("answer", "")

        if not story or not question:
            continue

        prompt = f"<|story|> {story} <|question|> {question} <|answer|>"
        chosen_answer = None

        for k in range(min(args.candidates, len(temperatures))):
            temp = temperatures[k]
            candidate = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=40,
                temperature=temp,
                context_len=cfg.context_len,
            )

            judgment = evaluate_with_llama(story, question, candidate, ref_answer)
            if judgment.get("acceptable") is True:
                chosen_answer = candidate
                print(f"[{i + 1}/{len(raw_seeds)}] Accepted candidate at temp {temp}: \"{candidate}\"")
                break
            else:
                print(f"[{i + 1}/{len(raw_seeds)}] Rejected at temp {temp}: {judgment.get('reason')}")

        if chosen_answer:
            aligned_dataset.append({
                "story": story,
                "question": question,
                "answer": chosen_answer
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aligned_dataset, f, indent=2)

    accept_rate = (len(aligned_dataset) / len(raw_seeds)) * 100 if raw_seeds else 0
    print(f"\nCompleted! Saved {len(aligned_dataset)} aligned pairs to {args.output}")
    print(f"Acceptance rate: {accept_rate:.1f}%")


if __name__ == "__main__":
    main()