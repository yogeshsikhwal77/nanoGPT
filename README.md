# NanoGPT — Fiko & Fire: Story Q&A Language Models

Two decoder-only Transformers trained **from scratch** and aligned with a local, offline RLAIF-style pipeline (hybrid SFT + Direct Preference Optimization), judged end-to-end by a local Qwen 2.5 (3B) model via Ollama — zero paid APIs, zero human labeling, tuned to fit inside **6 GB of VRAM**.

![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-BF16-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-000000?style=flat-square&logo=ollama&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

**🔗 Live demo:** **https://nanogpt-ic7p.onrender.com/**

> **Goal:** build small, understand every layer, and prove that a 15–33M parameter model can be turned into a reliable, hallucination-resistant fact-extractor using nothing but consumer hardware and open local models.

## Contents

* [Why This Project Matters](#why-this-project-matters)
* [Models](#models)
* [Architecture](#architecture)
* [The Pipeline](#the-pipeline)
* [A Recorded Training Run](#a-recorded-training-run)
* [Quick Start](#quick-start)
* [Project Layout](#project-layout)
* [Documentation](#documentation)
* [API](#api)
* [Known Gaps](#known-gaps)
* [Acknowledgments](#acknowledgments)
* [License](#license)

---

## Why This Project Matters

* **Automated synthetic dataset creation** — a local LLM (Qwen 2.5 3B via Ollama) both writes the SFT question/answer data and judges DPO preference pairs, with zero paid API calls and zero human labeling.
* **Small models can be taught not to hallucinate** — pre-training alone leaves a 15–33M parameter model prone to fabricating details or losing track of chronology. A subsequent DPO pass, driven entirely by a local judge, measurably corrects these failure modes without touching model size.
* **Runs end-to-end on a consumer laptop GPU** — tokenization → pre-training → synthetic data generation → SFT → DPO is all tuned to fit inside 6 GB of VRAM (production inference doesn't even need a GPU — see Quick Start).

## Models

The chat UI lets you pick between two model families, each with three tiers (Medium = SFT, High = baseline DPO, Max = targeted-DPO):

| Codename | Params | Context | Tiers exposed in the UI |
| :--- | :---: | :---: | :--- |
| **Fire** | ~33M | 512 | Medium (SFT) · High (DPO v1) · **Max (DPO v2)** |
| **Fiko** | ~15M | 256 | Medium (SFT) · High (DPO v1) · **Max (DPO v2)** |

> **Current status:** to keep memory under Render's free-tier limit, the server only ever loads each family's **Max** checkpoint — a "Medium"/"High" selection is transparently served by that same Max engine. The **Fire** checkpoints (`base_model_33M.pt`, `sft_model_33M.pt`, `dpo_model_33M_v1.pt`, `dpo_model_33M_v2.pt`) are committed to this repo via **Git LFS**. There is currently **no `dpo_model_15M_v2.pt` in the repo**, so a Fiko selection will 404 (`Model 'fiko-max' is not loaded`) until that checkpoint is trained and added — see [`docs/03-serving-and-reference.md`](docs/03-serving-and-reference.md#deployment-notes).

## Architecture

Both models share the same decoder block (pre-norm, SDPA causal attention, tied embeddings) and differ only in width/depth/context:

| Hyperparameter | Fiko | Fire |
| :--- | :---: | :---: |
| Layers | 6 | 8 |
| Attention heads | 6 | 8 |
| Hidden size | 384 | 512 |
| Context window | 256 | 512 |
| Vocabulary size | 10,000 | 10,000 |
| Precision | bfloat16 | bfloat16 |

Full hyperparameter tables, the decoder-block diagram, and exact parameter-counting instructions are in [`docs/01-architecture-and-pipeline.md`](docs/01-architecture-and-pipeline.md#architecture).

## The Pipeline

Tokenizer → base pre-training → Qwen-judged SFT data (single + multi-question) → SFT → Qwen-judged DPO preference pairs → DPO (v1) → targeted adversarial DPO (v2 / "Max" tier). Full flowchart with every script name: [`docs/01-architecture-and-pipeline.md`](docs/01-architecture-and-pipeline.md#the-rlaif-pipeline).

## A Recorded Training Run

From an actual run of the Fire (33M) pipeline, end to end:

| Stage | Result |
| :--- | :--- |
| Base pre-training | val_loss **1.4520** in 71.4 minutes |
| SFT (2 epochs) | val_loss **1.0540** (best) |
| DPO v1 (baseline) | loss stayed near `ln(2) ≈ 0.693` |
| DPO v2 (targeted) | loss dropped to **~0.667** |

Full breakdown, including data-generation timings and a note on a candidate-ordering bias observed in the judge: [`docs/01-architecture-and-pipeline.md`](docs/01-architecture-and-pipeline.md#a-recorded-training-run).

## Quick Start

```bash
git lfs install                 # required — checkpoints are stored via Git LFS
git clone https://github.com/yogeshsikhwal77/nanoGPT.git
cd nanoGPT

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1

# Serving-only dependencies (CPU torch — what the deployed app actually uses)
pip install -r requirements.txt

# For training instead, install a CUDA build of torch and a few extra packages
# (datasets, ollama) not declared in requirements.txt — see docs/02-training-guide.md
```

Run the chat UI locally:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Prerequisites: Python 3.10+, Git LFS, and — for training only — an NVIDIA GPU with Ampere/Ada Lovelace support for bfloat16, plus [Ollama](https://ollama.com) with `qwen2.5:3b` pulled for the data-generation phases.

## Project Layout

```
nanoGPT/
├── app/                      # FastAPI backend (main.py, inference.py, schemas.py) + static/ chat UI
├── src/nanogpt/               # tokenizer, model, training, judge & DPO scripts
├── configs/                   # base.yaml, sft_15M.yaml, sft_33M.yaml
├── scripts/                   # download_data.py, overfit_test.py
├── checkpoints/                # base_/sft_/dpo_ .pt files — tracked via Git LFS, not git-ignored
├── tests/                      # test_model.py
├── docs/                       # full documentation (3 parts, see below)
├── Dockerfile, Makefile, pyproject.toml, requirements.txt
├── sample.py, test_cot.py      # standalone generation scripts
├── tokenizer.json
└── README.md                   # you are here
```

`data/` (raw / tokenized / sft / dpo) is git-ignored — you regenerate it locally per [`docs/02-training-guide.md`](docs/02-training-guide.md).

## Documentation

Full technical documentation is split into three parts:

1. **[Architecture & Pipeline](docs/01-architecture-and-pipeline.md)** — model architecture, the RLAIF pipeline diagram, project structure, and a real recorded training run.
2. **[Training Guide](docs/02-training-guide.md)** — setup, config reference, and the verified phase-by-phase commands (tokenizer → base → SFT → DPO → targeted DPO) for both Fiko and Fire.
3. **[Serving & Reference](docs/03-serving-and-reference.md)** — deployment/Docker notes, the multi-engine serving logic, full API reference, hardware budget, testing, troubleshooting, limitations, and roadmap.

## API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "story": "Tim lost his red ball in the garden. He searched behind the shed and found it under the oak tree.",
    "question": "Where did Tim find his ball?",
    "model_name": "fire-max"
  }'
```

Response is `{"answer": ..., "tokens_generated": ..., "inference_time_ms": ...}`. Full request/response schema and error codes: [`docs/03-serving-and-reference.md`](docs/03-serving-and-reference.md#api-reference).

## Known Gaps

A few things worth knowing before you dig in — all covered in depth in the docs:

* Only the **Fire (33M)** checkpoint family is committed; **Fiko (15M)** has no checkpoint in the repo yet, so it 404s until trained.
* `train_base.py` ignores its `--config` flag and always reads `configs/base.yaml` — training Fiko means editing that file in place.

Full list: [`docs/03-serving-and-reference.md`](docs/03-serving-and-reference.md#troubleshooting).

## Acknowledgments

[TinyStories / TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStories) for the pre-training corpus and Q&A seed data, and [Qwen 2.5](https://huggingface.co/Qwen) (3B) via [Ollama](https://ollama.com) as the local synthetic-data writer and preference judge throughout the pipeline.

## License

MIT — see [`LICENSE`](LICENSE).