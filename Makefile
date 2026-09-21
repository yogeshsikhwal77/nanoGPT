.PHONY: setup tokenize train judge sft serve test all

setup:
	pip install torch --index-url https://download.pytorch.org/whl/cu121
	pip install -r requirements.txt
	pip install -e .
	ollama pull llama3.2:1b

tokenize:
	python -m nanogpt.tokenizers encode --input-dir data/raw/ --output-dir data/tokenized/

train:
	python -m nanogpt.train_base --config configs/base.yaml

judge:
	python -m nanogpt.judge --model-path checkpoints/base_model_15M.pt --instruct-data data/raw/TinyStories-Instruct.json --output data/sft/aligned_pairs.json --candidates 3

sft:
	python -m nanogpt.train_sft --config configs/sft.yaml

serve:
	uvicorn app.main:app --host 127.0.0.1 --port 8000

test:
	pytest -q

all: setup tokenize train judge sft serve