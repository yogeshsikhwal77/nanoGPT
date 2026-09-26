import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.inference import StoryQAInference

app = FastAPI(title="NanoGPT Story Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to hold the dual-model system
engines: dict[str, StoryQAInference] = {}


@app.on_event("startup")
def startup_event():
    global engines
    
    # Map your custom model names to the checkpoint files
    models_to_load = {
        "fiko": "checkpoints/sft_model_15M.pt",
        "fire": "checkpoints/sft_model_33M.pt"
    }
    
    for name, path in models_to_load.items():
        if os.path.exists(path):
            print(f"Loading '{name}' model from {path}...")
            engines[name] = StoryQAInference(checkpoint_path=path)
        else:
            print(f"Warning: Checkpoint for '{name}' not found at {path}")


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
        vram_used_mb=round(vram_mb, 2)
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.model_name not in engines:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not loaded.")
        
    engine = engines[req.model_name]
        
    answer, tokens_gen, latency_ms = engine.generate(
        story=req.story,
        question=req.question,
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    
    return ChatResponse(
        answer=answer,
        tokens_generated=tokens_gen,
        inference_time_ms=round(latency_ms, 2)
    )


os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")