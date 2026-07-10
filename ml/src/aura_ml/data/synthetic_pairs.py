"""Bootstrap a paired training set from unpaired source photos.

Pattern: for each source image, run zero-shot Qwen-Image-Edit with a
procedure prompt to generate a candidate "after" image. Filter candidates
with the numeric eval metrics, then have Qwen3.5-9B judge the survivors.
Keep pairs that pass both.

Important caveat: this is the same general pattern that produced the bad
hackathon dataset. The difference is the aggressive curation — every
candidate must clear the edit-magnitude floor (not static), the ArcFace
floor (same person), the LPIPS ceiling (not a different photo), AND a VLM
judgment before it is written to disk. Reject reasons are tallied in the
report so a poisoned batch is visible immediately.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from aura_ml.inference.qwen_edit import QwenImageEditPipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class SyntheticPairConfig:
    procedure: str
    instructions: list[str]  # variants sampled round-robin per source image
    samples_per_source: int = 2  # candidates per source (different seeds)
    num_steps: int = 40
    min_edit_magnitude: float = 0.02  # below: candidate is "static" — drop
    min_arcface_cosine: float = 0.6   # below: identity not preserved — drop
    max_lpips: float = 0.45           # above: change too dramatic — drop
    min_critic_score: float = 0.6     # VLM judgment floor
    use_vlm_critic: bool = True
    seed: int = 0


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidate(
    source: Image.Image,
    instruction: str,
    edit_pipeline: QwenImageEditPipeline,
    seed: int,
    num_steps: int = 40,
) -> Image.Image:
    """Single zero-shot edit. Returns a candidate "after" image."""
    return edit_pipeline.generate(source, instruction, num_steps=num_steps, seed=seed)


# ---------------------------------------------------------------------------
# VLM critic
# ---------------------------------------------------------------------------

_CRITIC_PROMPT = """You are judging a training pair for a surgical-visualization \
image editor. The first image is the BEFORE photo, the second is a candidate \
AFTER photo generated for this instruction:

"{instruction}"

Score the candidate on three criteria:
- followed: the requested change is visibly present in the AFTER image
- realistic: the AFTER image looks like an anatomically plausible surgical \
outcome (no artifacts, warping, or impossible anatomy)
- identity: the AFTER image is clearly the same person (same skin, eyes, \
hair, expression, pose, lighting, background)

Reply with ONLY a JSON object, no other text:
{{"followed": <0.0-1.0>, "realistic": <0.0-1.0>, "identity": <0.0-1.0>, "reason": "<one sentence>"}}"""


def critique_pair(
    source: Image.Image,
    candidate: Image.Image,
    instruction: str,
    expander,  # Qwen35PromptExpander (typed loosely to avoid import cycle)
) -> dict[str, Any]:
    """Use the VLM as a judge. Returns {"score": float in [0,1], "reason": str,
    "followed": ..., "realistic": ..., "identity": ...}.

    The numeric metrics are computed by the caller; this adds the qualitative
    judgment. Score = min of the three criteria (a pair must pass ALL)."""
    if expander._model is None:
        expander.load()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source.convert("RGB")},
                {"type": "image", "image": candidate.convert("RGB")},
                {"type": "text", "text": _CRITIC_PROMPT.format(instruction=instruction)},
            ],
        },
    ]
    inputs = expander._processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(expander._model.device)

    with torch.inference_mode():
        out = expander._model.generate(
            **inputs, max_new_tokens=160, do_sample=False,
        )
    text = expander._processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"score": 0.0, "reason": f"unparseable critic output: {text[:120]}"}
    try:
        obj = json.loads(m.group(0))
        parts = [float(obj.get(k, 0.0)) for k in ("followed", "realistic", "identity")]
        return {
            "score": min(parts),
            "followed": parts[0],
            "realistic": parts[1],
            "identity": parts[2],
            "reason": str(obj.get("reason", "")),
        }
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return {"score": 0.0, "reason": f"critic parse error: {e}"}


# ---------------------------------------------------------------------------
# Curation loop
# ---------------------------------------------------------------------------


@dataclass
class _Tally:
    considered: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def curate_synthetic_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    config: SyntheticPairConfig,
    edit_pipeline: QwenImageEditPipeline,
    expander=None,  # Qwen35PromptExpander | None
) -> dict[str, Any]:
    """Walk source_dir, generate candidates, filter, write to output_dir
    in the SCHEMA.md layout. Returns a report dict.

    Keeps at most ONE candidate per source image (the highest critic score
    among survivors) so a single easy source can't dominate the dataset.
    """
    from aura_ml.eval.metrics import arcface_cosine, edit_magnitude, lpips_score

    src = Path(source_dir)
    out = Path(output_dir)
    for sub in ("control", "target", "prompts", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not sources:
        raise FileNotFoundError(f"no images in {src}")

    tally = _Tally()
    next_id = 1

    for src_idx, src_path in enumerate(sources):
        source = Image.open(src_path).convert("RGB")
        instruction = config.instructions[src_idx % len(config.instructions)]

        best: dict[str, Any] | None = None
        for k in range(config.samples_per_source):
            tally.considered += 1
            seed = config.seed + src_idx * 1000 + k
            candidate = generate_candidate(
                source, instruction, edit_pipeline, seed=seed, num_steps=config.num_steps
            )

            em = edit_magnitude(source, candidate)
            if em < config.min_edit_magnitude:
                tally.drop("static (edit_magnitude below floor)")
                continue
            af = arcface_cosine(source, candidate)
            if math.isnan(af):
                tally.drop("no face detected")
                continue
            if af < config.min_arcface_cosine:
                tally.drop("identity lost (arcface below floor)")
                continue
            lp = lpips_score(source, candidate)
            if lp > config.max_lpips:
                tally.drop("change too large (lpips above ceiling)")
                continue

            critic = {"score": 1.0, "reason": "critic disabled"}
            if config.use_vlm_critic and expander is not None:
                critic = critique_pair(source, candidate, instruction, expander)
                if critic["score"] < config.min_critic_score:
                    tally.drop("vlm critic rejected")
                    continue

            record = {
                "candidate": candidate,
                "seed": seed,
                "metrics": {"edit_magnitude": em, "arcface_cosine": af, "lpips": lp},
                "critic": critic,
            }
            if best is None or critic["score"] > best["critic"]["score"]:
                best = record

        if best is None:
            continue

        pair_id = f"{next_id:05d}"
        next_id += 1
        tally.kept += 1
        source.save(out / "control" / f"{pair_id}.jpg", quality=95)
        best["candidate"].save(out / "target" / f"{pair_id}.jpg", quality=95)
        (out / "prompts" / f"{pair_id}.txt").write_text(instruction + "\n", encoding="utf-8")
        (out / "meta" / f"{pair_id}.json").write_text(
            json.dumps(
                {
                    "source_type": "synthetic",
                    "source_file": src_path.name,
                    "procedure": config.procedure,
                    "synthetic_seed": best["seed"],
                    "metrics": {k: round(v, 4) for k, v in best["metrics"].items()},
                    "vlm_critic": best["critic"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[curate] kept {pair_id} <- {src_path.name} "
              f"(em={best['metrics']['edit_magnitude']:.3f} "
              f"af={best['metrics']['arcface_cosine']:.3f} "
              f"critic={best['critic']['score']:.2f})")

    report = {
        "sources": len(sources),
        "candidates_considered": tally.considered,
        "pairs_kept": tally.kept,
        "dropped": tally.dropped,
        "output_dir": str(out),
    }
    (out / "curation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_dir")
    p.add_argument("output_dir")
    p.add_argument(
        "--procedure",
        required=True,
        help="procedure label written to meta/ (or 'toy_glasses' etc.)",
    )
    p.add_argument(
        "--instructions",
        required=True,
        help="path to a .txt file with one instruction variant per line",
    )
    p.add_argument("--samples-per-source", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--no-critic", action="store_true", help="skip the VLM judge (metrics only)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    instructions = [
        line.strip()
        for line in Path(args.instructions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cfg = SyntheticPairConfig(
        procedure=args.procedure,
        instructions=instructions,
        samples_per_source=args.samples_per_source,
        num_steps=args.num_steps,
        use_vlm_critic=not args.no_critic,
        seed=args.seed,
    )

    edit = QwenImageEditPipeline()
    expander = None
    if cfg.use_vlm_critic:
        from aura_ml.prompt_expander.qwen35 import Qwen35PromptExpander

        expander = Qwen35PromptExpander()

    report = curate_synthetic_dataset(args.source_dir, args.output_dir, cfg, edit, expander)
    print(json.dumps(report, indent=2))
