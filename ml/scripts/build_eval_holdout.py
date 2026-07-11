"""Build the eval holdout.

Two modes:

1. HDA mode (default) — from the HDA Facial Plastic Surgery Database.
   Convention: `_b.jpg` = before surgery (source for our editor), `_a.jpg` =
   after surgery (visual ground-truth reference). License: research-only,
   non-commercial. Default split: 40 nose / 30 facelift / 30 eyelid.

2. Synthetic mode (`--faces-dir`) — from any directory of face photos (e.g.
   data/raw/faces produced by fetch_test_faces.py). No reference images;
   procedure prompts are assigned round-robin. Validates the *pipeline*
   when the HDA database isn't available on this machine.

The output directory is gitignored either way.

Run from the ml/ directory:
    uv run python scripts/build_eval_holdout.py --seed 0
    uv run python scripts/build_eval_holdout.py --faces-dir data/raw/faces --per-procedure 4
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HDA = REPO_ROOT / "HDA Facial Plastic Surgery Database"
DEFAULT_OUT = REPO_ROOT / "ml" / "eval_holdout"


@dataclass
class ProcedureSpec:
    label: str  # canonical aura name
    hda_subdir: str
    count: int
    id_start: int  # inclusive
    prompts: list[str]


SPECS: list[ProcedureSpec] = [
    ProcedureSpec(
        label="rhinoplasty",
        hda_subdir="Nose",
        count=40,
        id_start=1,
        prompts=[
            "Subtle dorsal hump reduction with refined nasal tip and preserved alar base width. Maintain bilateral symmetry and identity.",
            "Slightly narrow the nasal bridge and refine the tip projection. Preserve alar flare, skin texture, and identity.",
            "Moderate tip refinement with a gentle supratip break. Preserve dorsal aesthetic line and identity.",
            "Reduce a mild dorsal hump and rotate the tip upward by a small angle. Preserve columella, alar base, and identity.",
            "Refine the nasal tip and slightly narrow the alar base. Maintain overall facial proportion and identity.",
        ],
    ),
    ProcedureSpec(
        label="facelift",
        hda_subdir="Facelift",
        count=30,
        id_start=41,
        prompts=[
            "Tighten the lower-face jawline and reduce the nasolabial fold. Preserve midface volume, skin texture, and identity.",
            "Soften jowl prominence and refine the mandibular border. Maintain neck contour and identity.",
            "Reduce marionette lines and gently lift the midface. Preserve eye region, skin texture, and identity.",
            "Refine the jawline and reduce submental fullness. Preserve facial proportions and identity.",
            "Lift the cheek subtly and reduce nasolabial fold depth. Maintain identity and skin texture.",
        ],
    ),
    ProcedureSpec(
        label="blepharoplasty",
        hda_subdir="Eyelid",
        count=30,
        id_start=71,
        prompts=[
            "Reduce upper-lid skin redundancy and refine the supratarsal crease. Preserve eye shape and identity.",
            "Remove lower-lid bags and smooth the periorbital hollow. Preserve eye color, lash line, and identity.",
            "Refine the upper-lid fold and lift the lateral brow slightly. Preserve canthal tilt and identity.",
            "Reduce dermatochalasis on the upper lids and refresh the periorbital area. Preserve identity.",
            "Smooth the tear trough and reduce lower-lid puffiness. Preserve eye shape and identity.",
        ],
    ),
]


def _find_pair_stems(hda_subdir: Path) -> list[str]:
    """Return stems (e.g. '01', '100') for which both _b.jpg and _a.jpg exist."""
    stems = []
    for p in hda_subdir.iterdir():
        if p.suffix.lower() != ".jpg":
            continue
        name = p.stem
        if not name.endswith("_b"):
            continue
        stem = name[:-2]
        if (hda_subdir / f"{stem}_a.jpg").exists():
            stems.append(stem)
    return sorted(stems)


def build_holdout(hda_root: Path, out_root: Path, seed: int) -> dict:
    rng = random.Random(seed)
    if not hda_root.is_dir():
        raise FileNotFoundError(f"HDA database not found at {hda_root}")

    for sub in ("control", "reference", "prompts", "meta"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    report = {"total": 0, "per_procedure": {}}

    for spec in SPECS:
        hda_sub = hda_root / spec.hda_subdir
        stems = _find_pair_stems(hda_sub)
        if len(stems) < spec.count:
            raise RuntimeError(
                f"need {spec.count} pairs from {hda_sub} but only found {len(stems)}"
            )
        chosen = rng.sample(stems, spec.count)
        for i, src_stem in enumerate(chosen):
            holdout_id = f"{spec.id_start + i:05d}"
            ctrl_src = hda_sub / f"{src_stem}_b.jpg"
            ref_src = hda_sub / f"{src_stem}_a.jpg"
            shutil.copyfile(ctrl_src, out_root / "control" / f"{holdout_id}.jpg")
            shutil.copyfile(ref_src, out_root / "reference" / f"{holdout_id}.jpg")
            prompt = spec.prompts[i % len(spec.prompts)]
            (out_root / "prompts" / f"{holdout_id}.txt").write_text(
                prompt + "\n", encoding="utf-8"
            )
            (out_root / "meta" / f"{holdout_id}.json").write_text(
                json.dumps(
                    {
                        "procedure": spec.label,
                        "hda_source": f"{spec.hda_subdir}/{src_stem}",
                        "license": "HDA research-only, non-commercial",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        report["per_procedure"][spec.label] = spec.count
        report["total"] += spec.count

    return report


def build_synthetic_holdout(
    faces_dir: Path, out_root: Path, per_procedure: int, seed: int
) -> dict:
    """Build a holdout from a flat directory of face photos (no references).

    Each procedure gets `per_procedure` distinct faces; prompts are assigned
    round-robin from that procedure's prompt list."""
    rng = random.Random(seed)
    faces = sorted(
        p for p in faces_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    needed = per_procedure * len(SPECS)
    if len(faces) < needed:
        raise RuntimeError(
            f"need {needed} faces in {faces_dir} but only found {len(faces)} — "
            f"run fetch_test_faces.py --n {needed}"
        )
    rng.shuffle(faces)

    for sub in ("control", "prompts", "meta"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    report = {"total": 0, "per_procedure": {}, "mode": "synthetic"}
    face_iter = iter(faces)
    next_id = 1
    for spec in SPECS:
        for i in range(per_procedure):
            holdout_id = f"{next_id:05d}"
            next_id += 1
            src = next(face_iter)
            shutil.copyfile(src, out_root / "control" / f"{holdout_id}{src.suffix.lower()}")
            (out_root / "prompts" / f"{holdout_id}.txt").write_text(
                spec.prompts[i % len(spec.prompts)] + "\n", encoding="utf-8"
            )
            (out_root / "meta" / f"{holdout_id}.json").write_text(
                json.dumps(
                    {
                        "procedure": spec.label,
                        "source": str(src),
                        "license": "synthetic face (StyleGAN) — unrestricted",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        report["per_procedure"][spec.label] = per_procedure
        report["total"] += per_procedure
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hda", default=str(DEFAULT_HDA), help="path to HDA database root")
    p.add_argument(
        "--faces-dir",
        default=None,
        help="build a synthetic holdout from this directory of face photos instead of HDA",
    )
    p.add_argument(
        "--per-procedure",
        type=int,
        default=4,
        help="faces per procedure in synthetic mode",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output holdout directory")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.faces_dir:
        report = build_synthetic_holdout(
            Path(args.faces_dir), Path(args.out), args.per_procedure, args.seed
        )
    else:
        report = build_holdout(Path(args.hda), Path(args.out), args.seed)
    print(json.dumps(report, indent=2))
    print(f"wrote holdout to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
