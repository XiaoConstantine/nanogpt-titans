.PHONY: help install install-dev prepare-data prepare-openwebtext prepare-needle train sample eval lint format typecheck test clean

# Default target
help:
	@echo "nanoGPT-Titans - Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install dependencies"
	@echo "    make install-dev      Install dev dependencies"
	@echo ""
	@echo "  Data:"
	@echo "    make prepare-data     Prepare Shakespeare dataset"
	@echo "    make prepare-openwebtext  Prepare OpenWebText dataset"
	@echo "    make prepare-needle   Prepare needle-in-haystack eval data"
	@echo ""
	@echo "  Training:"
	@echo "    make train            Train on Shakespeare (default config)"
	@echo "    make train ARGS='--max_iters=1000'  Train with custom args"
	@echo ""
	@echo "  Inference:"
	@echo "    make sample           Sample from trained model"
	@echo "    make sample ARGS='--prompt=\"To be or not\"'  Sample with prompt"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make eval             Run needle-in-haystack evaluation"
	@echo ""
	@echo "  Development:"
	@echo "    make lint             Run ruff linter"
	@echo "    make format           Format code with ruff"
	@echo "    make typecheck        Run type checker (ty)"
	@echo "    make test             Run tests"
	@echo "    make check            Run lint + typecheck"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean            Remove generated files"

# Setup
install:
	uv sync

install-dev:
	uv sync --all-extras

# Data preparation
prepare-data:
	uv run python -m nanogpt_titans.prepare_data shakespeare

prepare-openwebtext:
	uv run python -m nanogpt_titans.prepare_data openwebtext

prepare-needle:
	uv run python -m nanogpt_titans.prepare_data needle

# Training
TRAIN_ARGS ?= --dataset=shakespeare --max_iters=5000
train:
	uv run python -m nanogpt_titans.train $(TRAIN_ARGS) $(ARGS)

# Sampling
SAMPLE_ARGS ?= --checkpoint=out-titans/ckpt.pt --max_new_tokens=200
sample:
	uv run python -m nanogpt_titans.sample $(SAMPLE_ARGS) $(ARGS)

# Evaluation
EVAL_ARGS ?= --checkpoint=out-titans/ckpt.pt
eval:
	uv run python -m nanogpt_titans.eval_needle $(EVAL_ARGS) $(ARGS)

# Development
lint:
	uv run ruff check src/

format:
	uv run ruff format src/

typecheck:
	uv run ty check src/

test:
	uv run pytest

check: lint typecheck

# Cleanup
clean:
	rm -rf out-titans/
	rm -rf data/shakespeare/
	rm -rf data/openwebtext/
	rm -rf data/needle/
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/
	rm -rf src/nanogpt_titans/__pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Package for Kaggle upload
kaggle-package:
	@echo "Creating Kaggle package..."
	rm -rf kaggle_package kaggle_titans.zip
	mkdir -p kaggle_package/src/nanogpt_titans
	mkdir -p kaggle_package/scripts
	cp -r src/nanogpt_titans/*.py kaggle_package/src/nanogpt_titans/
	cp -r src/nanogpt_titans/qwen_titans kaggle_package/src/nanogpt_titans/
	cp -r scripts/*.sh kaggle_package/scripts/ 2>/dev/null || true
	find kaggle_package -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find kaggle_package -type f -name "*.pyc" -delete 2>/dev/null || true
	cp pyproject.toml kaggle_package/
	cp README.md kaggle_package/ 2>/dev/null || true
	@echo "Creating setup script..."
	@echo '#!/bin/bash' > kaggle_package/setup.sh
	@echo 'pip install tiktoken datasets tqdm transformers accelerate' >> kaggle_package/setup.sh
	@echo 'export PYTHONPATH=$${PYTHONPATH}:$$(pwd)/src' >> kaggle_package/setup.sh
	cd kaggle_package && zip -r ../kaggle_titans.zip .
	rm -rf kaggle_package
	@echo ""
	@echo "Created: kaggle_titans.zip"
	@echo "Upload this to Kaggle as a dataset, then in your notebook:"
	@echo "  !unzip /kaggle/input/your-dataset-name/kaggle_titans.zip -d /kaggle/working/"
	@echo "  !bash /kaggle/working/setup.sh"
