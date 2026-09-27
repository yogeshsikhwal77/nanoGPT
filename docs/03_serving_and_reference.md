# Part 3 — Serving & Reference

_Part of the full documentation for [nanoGPT](https://github.com/yogeshsikhwal77/nanoGPT). See also: [Part 1 — Architecture & Pipeline](01-architecture-and-pipeline.md) · [Part 2 — Training Guide](02-training-guide.md)._

## Contents

* [Deployment Notes](#deployment-notes)
* [API Reference](#api-reference)
* [Hardware and Memory Budget](#hardware-and-memory-budget)
* [Testing](#testing)
* [Troubleshooting](#troubleshooting)
* [Limitations](#limitations)
* [Roadmap](#roadmap)
* [License](#license)

---

## Deployment Notes

The live demo (**https://nanogpt-ic7p.onrender.com/**) runs the same `app/main.py` FastAPI app you'd run locally, built from the repo's `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Hugging Face Spaces routes traffic to port 7860 by default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Two things worth knowing:

* The port comment references **Hugging Face Spaces** conventions, but the current live deployment is on **Render** — the image itself is portable to either.
* `requirements.txt` installs PyTorch from the **CPU** wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`), so the production container does inference on CPU, not GPU.

### How model selection actually works

`app/schemas.py`'s `ChatRequest.model_name` accepts six values: `fire-medium`, `fire-high`, `fire-max`, `fiko-medium`, `fiko-high`, `fiko-max` — and the chat UI (`app/static/index.html`) presents all six as separate dropdown options. In production, though, `app/main.py`'s startup routine only ever loads **two actual model instances**:

```python
models_to_load = {
    "fire-max": "checkpoints/dpo_model_33M_v2.pt",
    "fiko-max": "checkpoints/dpo_model_15M_v2.pt",
}
# each family's "medium" and "high" selections are aliased to that family's "max" engine:
engines["fire"] = engines["fire-medium"] = engines["fire-high"] = engines.get("fire-max")
engines["fiko"] = engines["fiko-medium"] = engines["fiko-high"] = engines.get("fiko-max")
```

This is a deliberate memory-saving measure — the comment in the source calls it out explicitly: _"Only load the 'Max' models to stay under Render's 512MB Free Tier limit."_ Practically, this means:

* Selecting `fire-medium` or `fire-high` in the UI does **not** serve the actual SFT or DPO-v1 checkpoint — it silently serves `fire-max`'s output instead.
* Same for `fiko-medium` / `fiko-high` → `fiko-max`.
* Each engine only loads `if os.path.exists(path)` — **as of the current commit, `checkpoints/` in this repository contains only the Fire (33M) family** (`base_model_33M.pt`, `sft_model_33M.pt`, `dpo_model_33M_v1.pt`, `dpo_model_33M_v2.pt`), tracked via Git LFS. There is no `dpo_model_15M_v2.pt` in the tree, so `fiko-max` (and therefore every `fiko-*` selection) has no engine to route to and any `/chat` call with a `fiko-*` model name returns `404 Model 'fiko-max' is not loaded`. Whether the live Render deployment has a Fiko checkpoint added outside of this repo isn't something this documentation can confirm from the source alone — if you want Fiko to actually work (locally or in production), train it per [Part 2](02-training-guide.md#phase-7-dpo-alignment-high--max-tiers) and commit `checkpoints/dpo_model_15M_v2.pt` via Git LFS.

### Generation behavior in production

Every `/chat` request goes through `app/main.py`'s `generate_guided()`, which is a superset of the "Answer Prefix Priming" idea from `test_cot.py`: `get_dynamic_prefix()` inspects the question text and, for several question patterns (`what two…`/`name two…` list queries, `why…`, `when…`, `where did X go`, `what did X do after/before Y`), primes the model's answer with a short natural-language prefix before decoding — with a fallback to unprimed decoding if the primed attempt produces nothing. List-style questions also force at least 6 tokens of generation before EOS is allowed, so the model can't stop after naming only one of two requested items.

---

## API Reference

Schema is defined in `app/schemas.py` — the fields below are exactly what's implemented (no more, no less).

### `POST /chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "story": "Tim lost his red ball in the garden. He searched behind the shed and found it under the oak tree.",
    "question": "Where did Tim find his ball?",
    "temperature": 0.2,
    "max_tokens": 60,
    "model_name": "fire-max"
  }'
```

**Request fields**

| Field | Type | Default | Notes |
| :--- | :--- | :--- | :--- |
| `story` | string | required | Truncated from the start if it exceeds the model's context window |
| `question` | string | required | |
| `temperature` | float | `0.2` | Range `0.0`–`2.0`; `0` means greedy (argmax) decoding |
| `max_tokens` | int | `50` | Range `1`–`128` |
| `model_name` | string | `"fire-max"` | One of `fire-medium`, `fire-high`, `fire-max`, `fiko-medium`, `fiko-high`, `fiko-max` — see [Deployment Notes](#deployment-notes) for which actually differ |

**Response**

```json
{
  "answer": "Under the oak tree.",
  "tokens_generated": 6,
  "inference_time_ms": 27.4
}
```

(`ChatResponse` has exactly these three fields — there's no `model_stage` field in the implementation.)

**Errors**

| Status | When |
| :--- | :--- |
| `422` | Missing/invalid field (standard FastAPI/pydantic validation) |
| `404` | `model_name` isn't a key in the loaded `engines` dict — the response body includes the list of currently-available engine names |

### `GET /health`

```json
{
  "status": "ok",
  "device": "cpu",
  "vram_used_mb": 0.0
}
```

(`HealthResponse` has exactly these three fields — there's no `loaded_checkpoint` field in the implementation.) Returns `503 No models loaded yet` if the `engines` dict is empty (e.g. neither checkpoint file was found at startup).

### `GET /`

Serves `app/static/index.html` (the chat UI) via `FileResponse`; returns `404` if that file is missing.

CORS is wide open (`allow_origins=["*"]`) on every route.

---

## Hardware and Memory Budget

**Training** is engineered to fit inside a 6 GB VRAM budget, across every phase including DPO:

* **bfloat16 autocast** — cuts activation memory roughly in half vs. float32, without the underflow/overflow issues of full fp16.
* **FlashAttention (SDPA)** — `F.scaled_dot_product_attention` replaces materializing the O(T²) attention matrix in VRAM with tile-based fused kernels.
* **Memory-mapped pre-tokenization** — the training dataset is accessed directly from disk via `np.memmap(dtype=np.uint16)`; training uses well under 300 MB of system RAM regardless of dataset size.
* **Tied embeddings** — sharing weights between `tok_emb` and `lm_head` saves parameters and checkpoint size.
* **Frozen reference policy for DPO** — the DPO reference model runs inference-only (no gradients), so the extra memory cost over standard SFT is one extra forward pass, not a second trainable copy.
* **Local judge instead of a hosted API** — Qwen 2.5 3B runs through Ollama on the same machine, so data generation and judging never leave your GPU/CPU.

If you hit CUDA out-of-memory during training, lower `batch_size` and raise `grad_accum` to keep the effective batch size constant.

**Serving** runs on CPU by design (see [Deployment Notes](#deployment-notes)) and only two ~33M/15M-parameter checkpoints are ever resident in memory at once — the explicit reason the app only loads the Max tier of each family on Render's free tier.

---

## Testing

```bash
pytest -q
```

| Test file | Checks |
| :--- | :--- |
| `tests/test_model.py` | `test_shapes` (forward-pass tensor shapes), `test_weight_tying` (`lm_head.weight is tok_emb.weight`), `test_causality` (token *t* never attends to *t+1*), `test_ignore_index` (loss ignores `-100`-masked targets) |

This is currently the only test file in the repo — there is no tokenizer round-trip test or SFT-masking-specific test file, despite those being natural additions given the pipeline (see [Roadmap](#roadmap)).

---

## Troubleshooting

| Issue | Root cause | Solution |
| :--- | :--- | :--- |
| Checkpoints are ~130 bytes of text, not real weights | Cloned without Git LFS | Run `git lfs install` before cloning, or `git lfs pull` in an existing clone |
| `train_base.py` trains the same architecture no matter what `--config` you pass | The script hardcodes `load_base_config("configs/base.yaml")` and ignores CLI args entirely | Edit `configs/base.yaml` directly before running, or swap files in place |
| `judge1.py`'s output overwrites `judge.py`'s output | Both scripts default `--output` to `data/sft/aligned_pairs.json` | Always pass an explicit, different `--output` to `judge1.py` (e.g. `data/sft/aligned_pairs_1000.json`) |
| `merge_data.py` can't find your files | It has no CLI args — hardcoded to read `data/sft/aligned_pairs.json` and `data/sft/aligned_pairs_1000.json` | Name your judge outputs to match, or edit the two paths in `merge_data.py` |
| `generate_targeted_dpo.py --model-path` default doesn't exist | Its default is `checkpoints/dpo_model_33M.pt`, but the pipeline only ever produces `..._v1.pt` / `..._v2.pt` | Always pass `--model-path` explicitly |
| `test_cot.py` fails to find its checkpoint | Hardcodes `checkpoints/dpo_model_33M.pt` (no version suffix) | Copy/rename a checkpoint to that filename, or edit the path in the script |
| `ModuleNotFoundError: No module named 'datasets'` / `'ollama'` | Not declared in `requirements.txt` or `pyproject.toml` | `pip install datasets ollama` |
| `ollama pull llama3.2:1b` (from `make setup`) but judge scripts fail to find a model | The Makefile pulls a different model than the judge/DPO scripts actually call (`qwen2.5:3b`, hardcoded in `judge.py`/`judge1.py`/`judge_dpo.py`/`generate_targeted_dpo.py`) | `ollama pull qwen2.5:3b` |
| `RuntimeError: Error(s) in loading state_dict for GPT` | Architecture mismatch between the config used to load a checkpoint and the one it was trained with | Verify `dim`, `layers`, `heads`, and `context_len` match the values used to train the checkpoint |
| DPO loss stays near `ln(2) ≈ 0.693` for the whole run | Expected for a v1/baseline pass on a small, low-signal preference set — see the [recorded run](01-architecture-and-pipeline.md#a-recorded-training-run) | If it never moves even slightly, check `beta` isn't set too high, and confirm the reference model was loaded with `requires_grad=False` |
| Qwen judge returns inconsistent verdicts, or is biased toward one candidate slot | Non-zero temperature on judge calls, or positional bias in the judge prompt | Use temperature `0`; if one candidate slot (e.g. "B") wins suspiciously often, spot-check a sample manually — see the note in [Part 1](01-architecture-and-pipeline.md#a-recorded-training-run) |
| `torch.cuda.is_available()` is `False` during training | PyTorch installed without CUDA runtime dependencies, or you're using the CPU wheel from `requirements.txt` | Reinstall with CUDA wheels: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| ByteLevel special characters (`Ġ`, `Ċ`) in generated text | Tokenizer missing the ByteLevel decoding pipeline | Set `tokenizer.decoder = ByteLevel()` before calling `.decode()` (already done in `app/inference.py`, but easy to miss in a standalone script) |
| `fiko-*` requests 404 on `/chat` | `checkpoints/dpo_model_15M_v2.pt` isn't present | See [Deployment Notes](#deployment-notes) |

---

## Limitations

* Domain-bound to simple children's stories; no broad world knowledge.
* Weak at multi-step arithmetic even after DPO; alignment mainly corrects extraction/ordering errors, not reasoning depth.
* Context window (256 tokens for Fiko, 512 for Fire) still truncates longer stories.
* SFT and DPO data quality depend on a 3B local judge, smaller and noisier than frontier judge models — spot-checking accepted pairs is recommended, especially given the candidate-ordering bias noted in [Part 1](01-architecture-and-pipeline.md#a-recorded-training-run).
* "Medium" and "High" tiers are not actually served in production for either model family — only "Max" is live (see [Deployment Notes](#deployment-notes)).
* Fiko has no committed checkpoints in this repo as of now, so it isn't runnable out of the box.
* Much smaller than modern general-purpose LLMs; intended for learning and experimentation, not production use.

---

## Roadmap

* [ ] Commit (or otherwise make available) `checkpoints/dpo_model_15M_v2.pt` so `fiko-max` actually loads
* [ ] Either wire up real Medium/High engines per family, or drop those options from the UI to match what's actually served
* [ ] Give `train_base.py` a `--config` flag instead of a hardcoded path, and add a `base_15M.yaml`
* [ ] Add a `configs/dpo.yaml` so DPO hyperparameters don't have to be re-typed on the command line every run
* [ ] Fix the default paths in `judge1.py` and `generate_targeted_dpo.py` so they don't collide with / miss real outputs
* [ ] Publish full DPO win-rate ablations (base vs. SFT vs. DPO v1 vs. DPO v2) for both Fiko and Fire
* [ ] Add tokenizer round-trip and SFT-masking tests alongside `test_model.py`
* [ ] KV-cache for faster generation
* [ ] Streaming responses in the chat UI
* [ ] Swap in a larger local judge (Qwen 2.5 7B) as an ablation, and check whether it reduces the candidate-ordering bias observed with the 3B judge

---

## License

MIT — see [`LICENSE`](../LICENSE). Copyright (c) 2026 Yogesh Sikhwal.