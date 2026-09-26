import argparse
import json
import os
import ollama

def extract_stories(data_path: str):
    """Extracts raw story text from the shattered TinyStories-Instruct JSON."""
    with open(data_path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
        except json.JSONDecodeError:
            f.seek(0)
            raw_data = [json.loads(line) for line in f if line.strip()]

    stories = []
    current_story = []
    
    for item in raw_data:
        text = str(item.get("story", "") or item.get("text", "")).strip()
        
        if text.startswith("Features:") or text.startswith("Words:") or text.startswith("Summary:"):
            if current_story:
                stories.append(" ".join(current_story).strip())
                current_story = []
        elif text.startswith("Story:"):
            parsed = text.replace("Story:", "").strip()
            if parsed:
                current_story.append(parsed)
        elif text and not text.startswith("<|endoftext|>"):
            current_story.append(text)

    if current_story:
        stories.append(" ".join(current_story).strip())
        
    return [s for s in stories if len(s) > 50]

def generate_qa_pair(story: str) -> dict:
    system_instruction = (
        "You are a teacher creating a reading comprehension test. "
        "Read the STORY carefully and generate ONE simple, factual question "
        "that can ONLY be answered using information stated in THIS story, "
        "and a short, direct answer copied or closely paraphrased from THIS story. "
        "Do not invent details not present in the story. "
        "Respond ONLY with valid JSON in the form: "
        '{"question": "<your question>", "answer": "<your answer>"}'
    )
    # no concrete filled-in example — nothing for a weak model to fall back on

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            format="json",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"STORY: {story}"},
            ],
            options={"temperature": 0.3},
        )
        result = json.loads(response["message"]["content"])
        question, answer = result.get("question", "").strip(), result.get("answer", "").strip()

        if not question or not answer:
            return None

        # Grounding check: reject if answer shares no vocabulary with the story
        # (catches hallucinations and copied boilerplate like the old "Under the bed." example)
        story_words = set(w.lower().strip(".,!?\"'") for w in story.split())
        answer_words = set(w.lower().strip(".,!?\"'") for w in answer.split())
        meaningful_answer_words = {w for w in answer_words if len(w) > 3}
        if meaningful_answer_words and not (meaningful_answer_words & story_words):
            return None  # answer has no grounding in the story at all

        return {"story": story, "question": question, "answer": answer}
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Teacher Distillation Dataset Generator")
    parser.add_argument("--instruct-data", type=str, required=True, help="Path to TinyStories-Instruct data")
    parser.add_argument("--output", type=str, default="data/sft/aligned_pairs.json", help="Output path")
    parser.add_argument("--max-samples", type=int, default=500, help="Max seed stories to process")
    
    # Keep args for compatibility with your existing Makefile/README commands
    parser.add_argument("--model-path", type=str, default="", help="Ignored")
    parser.add_argument("--candidates", type=int, default=3, help="Ignored")
    args = parser.parse_args()

    print(f"Loading stories from {args.instruct_data}...")
    stories = extract_stories(args.instruct_data)[:args.max_samples]

    aligned_dataset = []

    print(f"Generating Q&A pairs using Ollama (Teacher Distillation) for {len(stories)} stories...")
    for i, story in enumerate(stories):
        pair = generate_qa_pair(story)
        if pair and pair["question"] and pair["answer"]:
            aligned_dataset.append(pair)
            print(f"[{i + 1}/{len(stories)}] Q: {pair['question']} | A: {pair['answer']}")
        else:
            print(f"[{i + 1}/{len(stories)}] Failed to generate pair.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aligned_dataset, f, indent=2)

    print(f"\nCompleted! Saved {len(aligned_dataset)} synthetic Q&A pairs to {args.output}")

if __name__ == "__main__":
    main()