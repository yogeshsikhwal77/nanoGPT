import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from nanogpt.model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer.decoder = ByteLevel()

ckpt = torch.load("checkpoints/dpo_model_33M.pt", map_location=device, weights_only=False)
model = GPT(ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

story = (
    "Jack and his dad went for a ride in the park. In their car, dad gave Jack a good lecture about why it's important to be safe. "
    "They drove past a big lake, and Jack wanted to go for a ride in a boat. Dad said no, so they looked at the lake instead. "
    "Then they rode to the park. They saw a playground with swings and slides. Jack wanted to go play, but Dad said it was time for a picnic. "
    "Dad set up a blanket and Jack ate a good lunch. After lunch, Jack saw a tire swing. He asked dad if he could give it a ride. Dad said yes, "
    "and Jack had so much fun swinging around. He asked for another ride and Dad said it was time to go home. Jack was sad, but Dad said they "
    "would come back again soon. On their way home, Jack thought of all the good times he had with his Dad."
)

eos_id = tokenizer.token_to_id("<|eos|>")

@torch.no_grad()
def generate(prompt: str, max_tokens: int = 30) -> str:
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    for _ in range(max_tokens):
        idx_cond = idx[:, -model.cfg.context_len:]
        logits, _ = model(idx_cond)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, next_tok), dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break
            
    return tokenizer.decode(idx[0, len(input_ids):].tolist()).replace("<|eos|>", "").strip()

print("=" * 60)
print("TEST 1: ANSWER PREFIX PRIMING (Forces attention past distractors)")
print("=" * 60)

prefix_tests = [
    {
        "question": "What did Jack ride after he finished his lunch?",
        "prefix": "After lunch, he rode a"
    },
    {
        "question": "What two things did Jack see at the playground?",
        "prefix": "At the playground, he saw"
    },
    {
        "question": "Did Jack get to ride in a boat on the lake?",
        "prefix": "No, because"
    }
]

for item in prefix_tests:
    prompt = f"<|story|> {story} <|question|> {item['question']} <|answer|> {item['prefix']}"
    output = generate(prompt, max_tokens=15)
    print(f"Q: {item['question']}")
    print(f"Completion: {item['prefix']} {output}\n")

print("=" * 60)
print("TEST 2: 1-SHOT CHRONOLOGICAL SCRATCHPAD")
print("=" * 60)

# Provide one exemplar showing timeline extraction before final answer
exemplar_story = "Mia ate breakfast. After breakfast, Mia went outside and rode her red bicycle."
exemplar_q = "What did Mia ride after breakfast?"
exemplar_a = "Timeline: Mia ate breakfast, then went outside. Answer: a red bicycle."

cot_prompt = (
    f"<|story|> {exemplar_story} <|question|> {exemplar_q} <|answer|> {exemplar_a} "
    f"<|story|> {story} <|question|> What did Jack ride after he finished his lunch? <|answer|> Timeline:"
)

cot_output = generate(cot_prompt, max_tokens=35)
print("Q: What did Jack ride after he finished his lunch?")
print(f"Model Reasoning + Output:\nTimeline: {cot_output}\n")