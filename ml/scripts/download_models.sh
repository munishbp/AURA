#!/usr/bin/env bash
# Pull the two base models for aura-ml.
#
# Disk: ~62 GB total. Run on a fast NVMe with HF_HOME pointed at it.
#   export HF_HOME=/path/to/big/ssd/hf
#
# Auth: Qwen models are public. No token required.

set -euo pipefail

HF=hf
if ! command -v hf >/dev/null 2>&1; then
    if command -v huggingface-cli >/dev/null 2>&1; then
        HF=huggingface-cli
    elif command -v uvx >/dev/null 2>&1; then
        HF="uvx --from huggingface_hub[cli] hf"
    else
        echo "hf CLI not found. Install with: uv pip install 'huggingface_hub[cli]'"
        exit 1
    fi
fi

DEST="${HF_HOME:-$HOME/.cache/huggingface}"
echo "HF cache: $DEST"
echo

free_gb=$(df -Pg "$DEST" 2>/dev/null | awk 'NR==2 {print $4}' || echo "?")
echo "Free space at cache: ${free_gb} GB (need ~70 GB)"
echo

echo "==> Qwen-Image-Edit-2511 (~40 GB) — the image editor"
$HF download Qwen/Qwen-Image-Edit-2511

echo
echo "==> Qwen3.5-9B (~22 GB) — the prompt expander"
$HF download Qwen/Qwen3.5-9B

echo
echo "Done. Both models cached under $DEST"
