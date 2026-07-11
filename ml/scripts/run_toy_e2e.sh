#!/usr/bin/env bash
# End-to-end toy-task validation: synthetic pairs -> validate -> QLoRA train.
#
# Proves the whole training loop on the 5090 with the "add glasses" task —
# high source/target divergence, so identity collapse fails loudly on the
# eval canary. Run with the API server STOPPED (needs the whole GPU).
#
#   bash scripts/run_toy_e2e.sh [n_faces]

set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-12}

# Metric models on CPU — GPU belongs to the editor + expander during
# candidate generation, and to the transformer during training.
export AURA_METRICS_DEVICE=cpu

echo "=== 1/3 synthetic toy pairs (first $N faces) ==="
mkdir -p /tmp/toy_faces_subset
i=0
for f in data/raw/faces/face_*.jpg; do
    [ $i -ge "$N" ] && break
    cp "$f" /tmp/toy_faces_subset/
    i=$((i+1))
done
uv run python -m aura_ml.data.synthetic_pairs \
    /tmp/toy_faces_subset data/pairs/toy_glasses \
    --procedure toy_glasses \
    --instructions data/instructions/toy_glasses.txt \
    --samples-per-source 1 --num-steps 8 --seed 0

echo "=== 2/3 validate dataset (structure + static-pair canary) ==="
uv run python -m aura_ml.data.pair_loader data/pairs/toy_glasses --check-edit-magnitude

echo "=== 2.5/3 toy eval holdout (held-out faces + glasses instructions) ==="
# The toy LoRA must be judged on ITS task: glasses instructions over faces
# the training set never saw (faces after the first N).
HOLD=data/pairs/toy_glasses_holdout
mkdir -p $HOLD/control $HOLD/prompts
j=0; k=0
for f in data/raw/faces/face_*.jpg; do
    j=$((j+1))
    [ $j -le "$N" ] && continue        # skip training faces
    k=$((k+1)); [ $k -gt 6 ] && break  # 6 holdout faces
    id=$(printf '%05d' $k)
    cp "$f" $HOLD/control/$id.jpg
    sed -n "$(( (k-1) % 3 + 1 ))p" data/instructions/toy_glasses.txt > $HOLD/prompts/$id.txt
done
echo "holdout: $(ls $HOLD/control | wc -l) faces"

echo "=== 3/3 QLoRA training (toy config) ==="
uv run python -m aura_ml.training.train --config configs/train_qwen_toy.yaml

echo "=== done — checkpoints in outputs/toy_glasses/ ==="
