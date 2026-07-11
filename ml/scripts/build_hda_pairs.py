"""Build real (control, target, instruction) training triples from the HDA
Facial Plastic Surgery Database.

HDA license (readme.txt): research use only, no commercial use/distribution;
results using it must cite Rathgeb et al., "Plastic Surgery: An Obstacle for
Deep Face Recognition?", CVPRW 2020. The output directory is gitignored —
never commit this imagery.

Leakage guard: any HDA stem already used by the eval holdout (recorded in
eval_holdout/meta/*.json by build_eval_holdout.py) is EXCLUDED from training.
Build the holdout first.

Instructions: by default each pair gets a procedure instruction variant
(round-robin from data/instructions/<procedure>.txt). With --vlm-instructions,
Qwen3.5-9B looks at each before/after pair and writes a per-pair instruction
describing the actual visible change (sanitized by the expander's guardrails;
falls back to the round-robin variant if the output fails validation).

Run from ml/ (GPU needed only with --vlm-instructions):
    uv run python scripts/build_hda_pairs.py --hda "<path>" --procedure rhinoplasty
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ML / "src"))

PROCEDURE_TO_HDA = {
    "rhinoplasty": "Nose",
    "facelift": "Facelift",
    "blepharoplasty": "Eyelid",
}

_DESCRIBE_PROMPT = """The first image is a patient BEFORE {procedure}; the \
second is the SAME patient AFTER the surgery. Write ONE imperative edit \
instruction (25-60 words) that would transform the before-photo into the \
after-photo: name the changed region in plain visual terms anchored to \
anatomy, describe the visible geometric change with a conservative magnitude \
word, and end by stating that identity, skin texture, expression, pose, \
lighting, and background stay unchanged. Describe ONLY changes from the \
surgery — ignore lighting, hair, or camera differences. No preamble, no \
quotes: output the instruction only."""


def _load_holdout_stems(holdout_dir: Path, hda_subdir: str) -> set[str]:
    """Stems (e.g. 'Nose/012') already consumed by the eval holdout."""
    stems: set[str] = set()
    meta_dir = holdout_dir / "meta"
    if not meta_dir.is_dir():
        return stems
    for p in meta_dir.glob("*.json"):
        meta = json.loads(p.read_text(encoding="utf-8"))
        src = meta.get("hda_source", "")
        if src.startswith(f"{hda_subdir}/"):
            stems.add(src.split("/", 1)[1])
    return stems


def _find_pair_stems(hda_subdir: Path) -> list[str]:
    stems = []
    for p in hda_subdir.iterdir():
        if p.suffix.lower() == ".jpg" and p.stem.endswith("_b"):
            stem = p.stem[:-2]
            if (hda_subdir / f"{stem}_a.jpg").exists():
                stems.append(stem)
    return sorted(stems)


def _describe_edit(expander, before, after, procedure: str) -> str | None:
    """Ask the VLM for a per-pair instruction. Returns None if unusable."""
    import torch

    from aura_ml.prompt_expander.qwen35 import _sanitize

    if expander._model is None:
        expander.load()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before.convert("RGB")},
                {"type": "image", "image": after.convert("RGB")},
                {"type": "text", "text": _DESCRIBE_PROMPT.format(procedure=procedure)},
            ],
        },
    ]
    inputs = expander._processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(expander._model.device)
    with torch.inference_mode():
        out = expander._model.generate(
            **inputs, max_new_tokens=160, do_sample=True,
            temperature=0.6, top_p=0.8, top_k=20,
        )
    text = expander._processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    if re.search(r"OUT_OF_SCOPE", text):
        return None
    return _sanitize(text, expander.config)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hda", required=True, help="HDA database root")
    p.add_argument("--procedure", required=True, choices=sorted(PROCEDURE_TO_HDA))
    p.add_argument("--holdout", default=str(REPO_ML / "eval_holdout"),
                   help="eval holdout dir (its HDA stems are excluded from training)")
    p.add_argument("--out", default=None,
                   help="output dir (default data/pairs/<procedure>)")
    p.add_argument("--vlm-instructions", action="store_true",
                   help="per-pair instructions from Qwen3.5-9B (needs GPU, ~4 s/pair)")
    p.add_argument("--limit", type=int, default=0, help="cap pairs (0 = all)")
    args = p.parse_args()

    from PIL import Image

    hda_subdir_name = PROCEDURE_TO_HDA[args.procedure]
    hda_subdir = Path(args.hda) / hda_subdir_name
    if not hda_subdir.is_dir():
        print(f"[fatal] {hda_subdir} not found", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else REPO_ML / "data" / "pairs" / args.procedure
    for sub in ("control", "target", "prompts", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    reserved = _load_holdout_stems(Path(args.holdout), hda_subdir_name)
    stems = [s for s in _find_pair_stems(hda_subdir) if s not in reserved]
    if args.limit:
        stems = stems[: args.limit]
    print(f"[hda] {len(stems)} training pairs "
          f"({len(reserved)} reserved by the eval holdout)")

    variants = [
        line.strip()
        for line in (REPO_ML / "data" / "instructions" / f"{args.procedure}.txt")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    expander = None
    if args.vlm_instructions:
        from aura_ml.prompt_expander.qwen35 import Qwen35PromptExpander

        expander = Qwen35PromptExpander()

    n_vlm, n_fallback = 0, 0
    for i, stem in enumerate(stems):
        pair_id = f"{i + 1:05d}"
        before = Image.open(hda_subdir / f"{stem}_b.jpg").convert("RGB")
        after = Image.open(hda_subdir / f"{stem}_a.jpg").convert("RGB")
        before.save(out / "control" / f"{pair_id}.jpg", quality=95)
        after.save(out / "target" / f"{pair_id}.jpg", quality=95)

        instruction = None
        if expander is not None:
            instruction = _describe_edit(expander, before, after, args.procedure)
        if instruction:
            n_vlm += 1
        else:
            instruction = variants[i % len(variants)]
            n_fallback += 1
        (out / "prompts" / f"{pair_id}.txt").write_text(instruction + "\n", encoding="utf-8")
        (out / "meta" / f"{pair_id}.json").write_text(
            json.dumps(
                {
                    "source_type": "real_hda",
                    "hda_source": f"{hda_subdir_name}/{stem}",
                    "instruction_source": "vlm" if n_vlm and instruction else "variant",
                    "license": "HDA research-only, non-commercial — do not distribute",
                    "citation": "Rathgeb et al., CVPRW 2020",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if (i + 1) % 25 == 0:
            print(f"[hda] {i + 1}/{len(stems)} pairs written")

    print(f"[hda] done: {len(stems)} pairs -> {out} "
          f"(instructions: {n_vlm} VLM, {n_fallback} variant)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
