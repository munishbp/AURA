# aura-ml

The Aura ML pipeline: photo + instruction → realistic surgical-outcome preview.

```
 face photo + "make the nose smaller"
        │
        ▼
 Qwen3.5-9B prompt expander (4-bit, ~7 GB)
   raw shorthand → precise, conservative, anatomically grounded edit
   instruction, with out-of-scope guard + deterministic sanitizer
        │
        ▼
 Qwen-Image-Edit-2511 (selective NF4, ~19 GB)
   + Lightning 8-step LoRA (serving default: ~10 s/edit, cfg 1.0)
   + optional procedure LoRA (rhinoplasty / facelift / blepharoplasty)
   + optional identity LoRA, composed via set_adapters
        │
        ▼
 edited face image → eval metrics (edit-magnitude canary, ArcFace, LPIPS, CLIP)
```

Everything fits and runs on a single RTX 5090 (32 GB).

### Quantization + speed recipe (measured on the 5090, seed-42 benchmark face)

| recipe | latency | edit magnitude | ArcFace identity |
|---|---|---|---|
| NF4, 40 steps, cfg 4.0 (model-card recipe) | 95 s | 0.191 | 0.553 |
| NF4 + **Lightning 8-step**, cfg 1.0 ← default | **10 s** | 0.174 | **0.689** |
| NF4 + Lightning 4-step, cfg 1.0 | 5 s | 0.138 | 0.784 |

NF4 is *selective*: first/last transformer blocks + in/out projections stay
bf16 (quantization noise compounds across denoise steps and surfaces as
grain). Fewer steps also means less noise accumulation — that's why Lightning
improves identity too, not just speed. torchao FP8 was evaluated and
rejected: `float8dq` OOMs at 32 GB (and can't CPU-offload), and `float8wo`
silently degraded to returning the input unchanged — edit magnitude 0.0125,
**caught by the static-image canary**, which is exactly the failure mode it
was built to catch.

## Setup

```bash
uv sync --extra eval --extra demo        # installs into .venv (Python 3.12)
bash scripts/download_models.sh          # ~62 GB: Qwen-Image-Edit-2511 + Qwen3.5-9B
uv run python scripts/verify_env.py      # sanity-check torch/CUDA/bitsandbytes
```

## Run the demo

```bash
uv run python -m app.demo                # http://localhost:7860
uv run python -m app.demo --host 0.0.0.0 # reachable from your phone on the same Wi-Fi
uv run python -m app.demo --share        # temporary public URL (Gradio tunnel)
uv run python -m app.demo --no-expander  # save ~7 GB VRAM (rule-based expansion)
```

Generation uses the Lightning 8-step recipe by default (~10 s/edit). For the
40-step model-card recipe, set `QwenEditConfig(lightning="off")` (slower, and
on the NF4 base measurably *worse* identity — see the table above).

Upload a face photo, pick a procedure, write the instruction. "Expand only"
shows (and lets you edit) the exact prompt sent to the diffusion model. Each
generation is scored live: the **canary** flags outputs that are secretly
identical to the input — the failure mode that killed the hackathon build.

## REST API

```bash
uv sync --extra serve
uv run python -m aura_ml.server --host 0.0.0.0 --port 8000        # docs at /docs
uv run python -m aura_ml.server --ui                              # + Gradio at /ui, same pipeline
```

| endpoint | what |
|---|---|
| `GET /v1/health` | GPU, VRAM, what's loaded |
| `GET /v1/procedures` | supported procedures + example instructions |
| `POST /v1/expand` | photo + instruction → expanded prompt (~3 s) |
| `POST /v1/generate` | photo + instruction → edited image, sync (~15 s warm w/ Lightning) |
| `POST /v1/jobs/generate` | same, async — returns a job id immediately |
| `GET /v1/jobs/{id}` / `…/image` | poll status / fetch the PNG |
| `POST /v1/metrics` | score any (before, after, instruction) triple |

All generation funnels through one GPU worker + a shared lock, so concurrent
requests queue instead of OOMing the card. Out-of-scope instructions come
back as HTTP 422 with the expander's reason.

```bash
# Expand only
curl -s -X POST localhost:8000/v1/expand \
  -F image=@face.jpg -F procedure=rhinoplasty \
  -F "instruction=make the nose smaller" | jq .prompt

# Async generation
JOB=$(curl -s -X POST localhost:8000/v1/jobs/generate \
  -F image=@face.jpg -F procedure=rhinoplasty \
  -F "instruction=reduce the dorsal hump" | jq -r .job_id)
curl -s localhost:8000/v1/jobs/$JOB | jq .status
curl -s localhost:8000/v1/jobs/$JOB/image -o preview.png
```

## Eval harness

```bash
# Build a holdout. With the HDA database on disk:
uv run python scripts/build_eval_holdout.py --seed 0
# Without it (synthetic faces, validates the pipeline only):
uv run python scripts/fetch_test_faces.py --n 16 --out data/raw/faces
uv run python scripts/build_eval_holdout.py --faces-dir data/raw/faces --out eval_holdout

# Prove the harness detects both failure and success:
uv run python scripts/eval_smoke.py --holdout eval_holdout

# Score a model (zero-shot baseline or LoRA checkpoint) against the holdout:
uv run python -m aura_ml.eval.grid --holdout eval_holdout --checkpoint outputs/rhino/best --out runs/rhino.html
uv run python -m aura_ml.eval.grid --holdout eval_holdout --outputs-dir runs/baseline/outputs --out runs/baseline.html
```

Metrics per (source, output, instruction) triple:

| metric | meaning | healthy |
|---|---|---|
| edit_magnitude | 1 − cos(DINOv2) — the static-image canary | > 0.05 |
| arcface_cosine | identity preservation | ≥ 0.6 |
| lpips | perceptual distance sanity check | 0.05–0.45 |
| clip_score | instruction followed | ↑ vs zero-shot baseline |

## Training a LoRA

```bash
# 1. Build a paired dataset (control/target/prompts — see src/aura_ml/data/SCHEMA.md).
#    From unpaired photos, bootstrap + curate synthetically:
uv run python -m aura_ml.data.synthetic_pairs data/raw/faces data/pairs/toy_glasses \
    --procedure toy_glasses --instructions data/instructions/toy_glasses.txt

# 2. Validate it (structure + static-pair canary):
uv run python -m aura_ml.data.pair_loader data/pairs/toy_glasses --check-edit-magnitude

# 3. Train (QLoRA on the NF4 transformer, flow-matching loss):
uv run python -m aura_ml.training.train --config configs/train_qwen_toy.yaml

# 4. Every eval interval the trainer scores the holdout and quarantines
#    checkpoints that trip the static-image canary.
```

The toy task ("add glasses") is deliberately high-divergence: if the training
loop is broken in the copy-the-input way, it fails loudly on the first eval.

## Layout

```
ml/
├── app/demo.py                    Gradio UI
├── configs/train_qwen_*.yaml      per-procedure training configs
├── data/instructions/*.txt        instruction variants for synthetic pairing
├── scripts/                       env check, model download, holdout build, smoke test
└── src/aura_ml/
    ├── inference/qwen_edit.py     Qwen-Image-Edit-2511 wrapper (NF4, LoRA mgmt)
    ├── inference/pipeline.py      expander → editor → LoRA composition
    ├── prompt_expander/qwen35.py  Qwen3.5-9B expander + sanitizer + fallback
    ├── training/train.py          QLoRA flow-matching trainer w/ canary quarantine
    ├── data/                      pair dataset, validator, synthetic pair curation
    └── eval/                      metrics + HTML grid + canary flagger
```
