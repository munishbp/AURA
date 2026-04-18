# Dataset schema

aura-ml's training data follows the **ai-toolkit `control / target / test`
folder convention** for paired-edit LoRA training on Qwen-Image-Edit-2509.

## On-disk layout

```
data/pairs/<procedure>/
├── control/        ← source images (the "before" / input to the edit)
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── target/         ← target images (the "after" / desired output)
│   ├── 00001.jpg     ← MUST match the same basename as control/00001.jpg
│   ├── 00002.jpg
│   └── ...
├── prompts/        ← one .txt per pair, plain instruction text
│   ├── 00001.txt
│   ├── 00002.txt
│   └── ...
└── test/           ← held-out unpaired controls used for visual sampling
    │                  during training (not used to compute loss)
    ├── 00001.jpg
    └── ...
```

### File naming
- Same numeric basename links control / target / prompts entries.
- Prefer 5-digit zero-padded names so lexical sort matches numeric order.
- Image format: JPEG, sRGB, ≥ 768 px on the long edge.
- Aspect ratio: arbitrary; `pair_loader.py` resizes + center-crops to the
  training resolution.

### prompts/<id>.txt format
A single instruction sentence, in the format the prompt expander produces
(detailed, anatomically grounded). Example:

```
Subtle dorsal hump reduction with refined nasal bridge and slightly narrowed
tip projection at 95 degrees. Maintain alar base width, bilateral symmetry,
and identity.
```

For workstream 4's toy task ("add round glasses"), keep it simple:

```
Add round wire-frame eyeglasses. Preserve all other facial features.
```

### Optional metadata sidecar
If you want to track provenance (e.g. "this pair was synthetically generated
from a single source via workstream 6 strategy #2"), add a sibling file:

```
data/pairs/<procedure>/meta/<id>.json
```

```json
{
  "source_type": "synthetic" | "real_curated" | "hand_aligned",
  "source_url": "...",
  "synthetic_seed": 42,
  "vlm_critic_score": 0.81,
  "notes": "optional human comment"
}
```

The dataloader ignores `meta/` — it's purely for human auditing.

## What goes in each procedure folder

| Procedure | Folder name |
|---|---|
| Rhinoplasty (nose) | `data/pairs/rhinoplasty/` |
| Rhytidectomy (facelift) | `data/pairs/facelift/` |
| Blepharoplasty (eyelid) | `data/pairs/blepharoplasty/` |
| Toy task (workstream 4) | `data/pairs/toy_glasses/` |
| Identity / per-subject | `data/pairs/identity_<subject>/` |

One LoRA per folder. Adapter name on disk should match folder name.

## Counts you should aim for

| Phase | Pairs needed |
|---|---|
| Workstream 4 toy task | ≥ 20 |
| Workstream 6 first procedure | ≥ 100, ideally 200+ |
| Workstream 7 identity LoRA | 5–10 photos of one subject |
| Workstream 8 each remaining procedure | ≥ 100 |

The original hackathon used ~200 synthetic pairs per procedure and the result
was the static-image collapse. More pairs help only if they have real
source-target divergence. See workstream 6 strategy notes in the plan / README.

## What's NOT in this schema

- **3D / depth data** — the original iOS app uploaded LiDAR depth frames.
  We don't use them in the rebuild. Photo input only.
- **Per-pixel masks** — Qwen-Image-Edit handles where-to-edit implicitly via
  the instruction text. No explicit segmentation required.
- **Multiple reference images per pair** — DreamOmni2's multi-ref pattern is
  not part of this schema. One control, one target.

## Validation

Run a sanity check on a folder before training:

```python
from aura_ml.data.pair_loader import PairDataset, validate_dataset
validate_dataset("data/pairs/toy_glasses")
```

This errors loudly on:
- Missing target for a control basename (or vice versa)
- Missing prompts/<id>.txt
- Non-image files in control/ or target/
- Suspiciously low edit-magnitude on the source/target pair (the
  static-image canary applied at dataset-load time — see eval/metrics.py)
