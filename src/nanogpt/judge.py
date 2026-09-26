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
    """Uses Llama 3.2 1B as a Teacher to generate synthetic Q&A pairs."""
    system_instruction = (
        "You are a teacher creating a reading comprehension test. "
        "Read the STORY and generate ONE simple, factual question about it, and a short, direct answer. "
        "You MUST respond ONLY with valid JSON. "
        "Example output: {\"question\": \"Where did the dog hide?\", \"answer\": \"Under the bed.\"}"
    )

    try:
        response = ollama.chat(
            model="llama3.2:1b",
            format="json",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"STORY: {story}"},
            ],
            options={"temperature": 0.3},
        )
        result = json.loads(response["message"]["content"])
        return {
            "story": story,
            "question": result.get("question", ""),
            "answer": result.get("answer", "")
        }
    except Exception as e:
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