import os
import re
import time
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.inference import StoryQAInference

app = FastAPI(title="NanoGPT Multi-Tier Story Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engines: dict[str, StoryQAInference] = {}


def get_dynamic_prefix(question: str) -> str:
    """
    Directs small model (15M-33M) attention using natural linguistic tokens.
    Prevents out-of-distribution shocks and literal repetition traps.
    """
    q = question.lower().strip()

    # 1. Multi-item queries: Natural indefinite article primes a noun phrase
    if any(k in q for k in ["what two", "name two", "which two", "list two", "what both"]):
        return "A"

    # 2. Causal queries: Direct model to the reasoning clause
    if q.startswith("why"):
        return "Because"

    # 3. Temporal queries: Direct attention to time anchors
    if q.startswith("when"):
        return "When"

    # 4. Destination queries: Prime actor + directional verb phrase
    if "where did" in q and "go" in q:
        match = re.search(r"where did ([a-zA-Z]+) go", q)
        if match:
            actor = match.group(1).capitalize()
            return f"{actor} went to the"

    # 5. Temporal sequence queries: Mirror anchor clause to prevent backward action drift
    match_after = re.search(r"what did ([a-zA-Z]+) do after ([^?]+)", q)
    if match_after:
        actor = match_after.group(1).capitalize()
        event = match_after.group(2).strip()
        return f"After {event}, {actor}"

    # Fallback for simple 'first' or 'before' queries
    match_gen = re.search(r"what did ([a-zA-Z]+) do (before|first)", q)
    if match_gen:
        return match_gen.group(1).capitalize()

    return ""


@torch.no_grad()
def _run_autoregressive_decode(
    engine: StoryQAInference,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 50,
    min_tokens_before_eos: int = 0,
) -> tuple[str, int, float]:
    """Decoding loop with autocast acceleration and conditional EOS suppression."""
    input_ids = engine.tokenizer.encode(prompt).ids

    # Truncate context from the start if it exceeds model limits
    if len(input_ids) > engine.cfg.context_len:
        input_ids = input_ids[-engine.cfg.context_len :]

    idx = torch.tensor([input_ids], dtype=torch.long, device=engine.device)

    if engine.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    tokens_generated = 0
    for _ in range(max_tokens):
        idx_cond = idx[:, -engine.cfg.context_len :]
        with torch.autocast(device_type="cuda" if engine.device == "cuda" else "cpu", dtype=torch.bfloat16):
            logits, _ = engine.model(idx_cond)

        # Force model past single-entity answers on list queries
        if tokens_generated < min_tokens_before_eos and engine.eos_id is not None:
            logits[:, -1, engine.eos_id] = -float("inf")

        if temperature == 0.0:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        else:
            scaled_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)
        tokens_generated += 1

        if engine.eos_id is not None and next_token.item() == engine.eos_id:
            break

    if engine.device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    output_tokens = idx[0, len(input_ids) :].tolist()
    raw_text = engine.tokenizer.decode(output_tokens)
    cleaned = (
        raw_text.replace("<|eos|>", "")
        .replace("<|answer|>", "")
        .strip()
    )

    return cleaned, tokens_generated, elapsed_ms


def generate_guided(
    engine: StoryQAInference,
    story: str,
    question: str,
    temperature: float = 0.2,
    max_tokens: int = 50,
) -> tuple[str, int, float]:
    """Generation pipeline with dynamic prefix priming, EOS thresholding, and fallback."""
    prefix = get_dynamic_prefix(question)
    base_prompt = f"<|story|> {story} <|question|> {question} <|answer|>"

    # Ensure list queries generate at least 6 tokens to capture both entities
    q_lower = question.lower()
    is_list_query = any(k in q_lower for k in ["what two", "name two", "which two", "list two", "what both"])
    min_tokens = 6 if is_list_query else 0

    if prefix:
        primed_prompt = f"{base_prompt} {prefix}"
        continuation, tokens_gen, latency_ms = _run_autoregressive_decode(
            engine, primed_prompt, temperature, max_tokens, min_tokens_before_eos=min_tokens
        )

        # Fallback safeguard: If prefix produces nothing, fallback to unprimed decode
        if not continuation or len(continuation.strip()) == 0:
            raw_ans, tokens_gen_raw, latency_raw = _run_autoregressive_decode(
                engine, base_prompt, temperature, max_tokens, min_tokens_before_eos=min_tokens
            )
            return raw_ans, tokens_gen_raw, latency_ms + latency_raw

        final_answer = f"{prefix} {continuation}".strip()
        return final_answer, tokens_gen, latency_ms

    return _run_autoregressive_decode(
        engine, base_prompt, temperature, max_tokens, min_tokens_before_eos=min_tokens
    )


@app.on_event("startup")
def startup_event():
    global engines

    # MEMORY FIX: Only load the 'Max' models to stay under Render's 512MB Free Tier limit.
    models_to_load = {
        "fire-max": "checkpoints/dpo_model_33M_v2.pt",
        "fiko-max": "checkpoints/dpo_model_15M_v2.pt",
    }

    for name, path in models_to_load.items():
        if os.path.exists(path):
            print(f"Loading '{name}' engine from {path}...")
            engine = StoryQAInference(checkpoint_path=path)
            try:
                _ = engine.generate(story="A warm garden.", question="What garden?", max_tokens=2)
            except Exception:
                pass
            engines[name] = engine
            
    # Map all requests to the MAX models
    engines["fire"] = engines.get("fire-max")
    engines["fire-medium"] = engines.get("fire-max")
    engines["fire-high"] = engines.get("fire-max")
    
    engines["fiko"] = engines.get("fiko-max")
    engines["fiko-medium"] = engines.get("fiko-max")
    engines["fiko-high"] = engines.get("fiko-max")

@app.get("/")
def read_root():
    index_path = os.path.join("app", "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found in app/static")


@app.get("/health", response_model=HealthResponse)
def health():
    if not engines:
        raise HTTPException(status_code=503, detail="No models loaded yet")

    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    return HealthResponse(
        status="ok",
        device="cuda" if torch.cuda.is_available() else "cpu",
        vram_used_mb=round(vram_mb, 2),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.model_name not in engines:
        available = [k for k in engines.keys() if "-" in k]
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model_name}' is not loaded. Available engines: {available}",
        )

    engine = engines[req.model_name]

    answer, tokens_gen, latency_ms = generate_guided(
        engine=engine,
        story=req.story,
        question=req.question,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    return ChatResponse(
        answer=answer,
        tokens_generated=tokens_gen,
        inference_time_ms=round(latency_ms, 2),
    )


os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")