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

def generate_qa_pairs(story: str) -> list:
    """Generates MULTIPLE distinct Q&A pairs for a single story."""
    system_instruction = (
        "You are a teacher creating a reading comprehension test. "
        "Read the STORY carefully and generate THREE distinct, factual questions "
        "that can ONLY be answered using information stated in THIS story. "
        "Try to ask about different things (e.g., who, what color, where, how many). "
        "Provide a short, direct answer for each, copied or closely paraphrased from the story. "
        "Do not invent details. "
        "Respond ONLY with valid JSON in this exact format: "
        '{"qa_pairs": [{"question": "...", "answer": "..."}, {"question": "...", "answer": "..."}, {"question": "...", "answer": "..."}]}'
    )

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            format="json",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"STORY: {story}"},
            ],
            # Bumped temp slightly to encourage diverse question types
            options={"temperature": 0.5},
        )
        result = json.loads(response["message"]["content"])
        pairs = result.get("qa_pairs", [])
        
        if not isinstance(pairs, list):
            return []

        valid_pairs = []
        story_words = set(w.lower().strip(".,!?\"'") for w in story.split())

        for qa in pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()

            if not question or not answer:
                continue

            # Grounding check: reject if answer shares no vocabulary with the story
            answer_words = set(w.lower().strip(".,!?\"'") for w in answer.split())
            meaningful_answer_words = {w for w in answer_words if len(w) > 3}
            if meaningful_answer_words and not (meaningful_answer_words & story_words):
                continue  # Skip this specific pair, answer has no grounding
                
            valid_pairs.append({
                "story": story, 
                "question": question, 
                "answer": answer
            })

        return valid_pairs
    except Exception as e:
        return []

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
        pairs = generate_qa_pairs(story)
        if pairs:
            for p_idx, pair in enumerate(pairs):
                aligned_dataset.append(pair)
                print(f"[{i + 1}/{len(stories)}] Q{p_idx + 1}: {pair['question']} | A: {pair['answer']}")
        else:
            print(f"[{i + 1}/{len(stories)}] Failed to generate pairs.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aligned_dataset, f, indent=2)

    print(f"\nCompleted! Generated {len(aligned_dataset)} synthetic Q&A pairs from {len(stories)} stories.")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()