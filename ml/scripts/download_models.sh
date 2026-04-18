#!/usr/bin/env bash
# Pull the two base models for aura-ml.
#
# Disk: ~58 GB total. Run on a fast NVMe with HF_HOME pointed at it.
#   export HF_HOME=/path/to/big/ssd/hf
#
# Auth: Qwen models are public. No token required.

set -euo pipefail

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found. Install with: uv pip install huggingface_hub[cli]"
    exit 1
fi

DEST="${HF_HOME:-$HOME/.cache/huggingface}"
echo "HF cache: $DEST"
echo

free_gb=$(df -Pg "$DEST" 2>/dev/null | awk 'NR==2 {print $4}' || echo "?")
echo "Free space at cache: ${free_gb} GB (need ~65 GB)"
echo

echo "==> Qwen-Image-Edit-2509 (~40 GB)"
huggingface-cli download Qwen/Qwen-Image-Edit-2509 --resume-download

echo
echo "==> Qwen3.5-9B (~18 GB)"
huggingface-cli download Qwen/Qwen3.5-9B --resume-download

echo
echo "Done. Both models cached under $DEST"
