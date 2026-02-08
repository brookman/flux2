#!/usr/bin/env bash

export KLEIN_4B_MODEL_PATH="./models/flux-2-klein-4b.safetensors"
export KLEIN_9B_MODEL_PATH="./models/flux-2-klein-base-9b-nvfp4.safetensors"
export AE_MODEL_PATH="./models/ae.safetensors"

# Memory optimization for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the CLI
PYTHONPATH=src python scripts/cli.py
