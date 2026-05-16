"""Acceptance gate for workstream 2.

Verifies the eval harness works in both directions before W1 lands:

  - Identity fixture: outputs == sources. The canary MUST fire
    (static_flag = True, static_fraction = 1.0).
  - Edited fixture:  outputs = sources passed through a coarse but
    non-trivial transform (rotation + blur + color jitter). The canary
    MUST stay quiet (static_flag = False) and edit_magnitude should
    clear the 0.05 floor on the majority of entries.

Run from the repo root:
    python ml/scripts/eval_smoke.py [--holdout ml/eval_holdout] [--n 8]

Exits non-zero on assertion failure.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from aura_ml.eval.grid import build_grid, flag_static_checkpoint  # noqa: E402


def _identity_outputs(holdout: Path, out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    control = holdout / "control"
    chosen = sorted(p for p in control.iterdir() if p.suffix.lower() == ".jpg")[:n]
    for p in chosen:
        shutil.copyfile(p, out_dir / p.name)


def _edited_outputs(holdout: Path, out_dir: Path, n: int) -> None:
    """A coarse non-trivial transform — not a real edit, but enough DINO
    displacement to clear the 0.05 floor on most entries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    control = holdout / "control"
    chosen = sorted(p for p in control.iterdir() if p.suffix.lower() == ".jpg")[:n]
    for p in chosen:
        img = Image.open(p).convert("RGB")
        img = img.rotate(7, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
        img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        img = ImageEnhance.Color(img).enhance(0.55)
        img = ImageEnhance.Brightness(img).enhance(1.15)
        img.save(out_dir / p.name, quality=90)


def _build_triples(holdout: Path, outputs: Path) -> list[tuple]:
    triples = []
    for ctrl in sorted(holdout.joinpath("control").iterdir()):
        if ctrl.suffix.lower() != ".jpg":
            continue
        stem = ctrl.stem
        out_path = outputs / f"{stem}.jpg"
        prompt_path = holdout / "prompts" / f"{stem}.txt"
        if not out_path.exists() or not prompt_path.exists():
            continue
        ref_path = holdout / "reference" / f"{stem}.jpg"
        ref = Image.open(ref_path) if ref_path.exists() else None
        triples.append(
            (
                stem,
                Image.open(ctrl),
                Image.open(out_path),
                prompt_path.read_text(encoding="utf-8").strip(),
                ref,
            )
        )
    return triples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--holdout",
        default=str(REPO_ROOT / "ml" / "eval_holdout"),
        help="path to the eval holdout directory",
    )
    p.add_argument(
        "--runs-dir",
        default=str(REPO_ROOT / "ml" / "runs"),
        help="where to drop fixture grids",
    )
    p.add_argument(
        "--n",
        type=int,
        default=8,
        help="number of holdout entries to use in each fixture (kept small for speed)",
    )
    args = p.parse_args()

    holdout = Path(args.holdout)
    runs = Path(args.runs_dir)
    if not (holdout / "control").is_dir():
        print(
            f"[fatal] holdout not found at {holdout}. Run build_eval_holdout.py first.",
            file=sys.stderr,
        )
        return 2

    # Identity fixture
    id_outputs = runs / "smoke_identity" / "outputs"
    _identity_outputs(holdout, id_outputs, args.n)
    id_triples = _build_triples(holdout, id_outputs)
    id_report = build_grid(id_triples, meta={"fixture": "identity"})
    id_report.to_html(runs / "smoke_identity" / "grid.html")
    id_report.to_json(runs / "smoke_identity" / "grid.json")
    id_quarantine, id_reason = flag_static_checkpoint(id_report)
    print(
        f"[identity]  n={len(id_triples)} static_fraction={id_report.static_fraction:.0%} "
        f"flag={'TRIP' if id_quarantine else 'ok'}"
    )

    # Edited fixture
    ed_outputs = runs / "smoke_edited" / "outputs"
    _edited_outputs(holdout, ed_outputs, args.n)
    ed_triples = _build_triples(holdout, ed_outputs)
    ed_report = build_grid(ed_triples, meta={"fixture": "edited"})
    ed_report.to_html(runs / "smoke_edited" / "grid.html")
    ed_report.to_json(runs / "smoke_edited" / "grid.json")
    ed_quarantine, ed_reason = flag_static_checkpoint(ed_report)
    print(
        f"[edited]    n={len(ed_triples)} static_fraction={ed_report.static_fraction:.0%} "
        f"flag={'TRIP' if ed_quarantine else 'ok'}"
    )

    failures = []
    if not id_quarantine:
        failures.append(
            f"identity fixture should trip the canary but didn't: {id_report.static_fraction:.0%} static"
        )
    if ed_quarantine:
        failures.append(
            f"edited fixture should NOT trip the canary but did: {ed_report.static_fraction:.0%} static. "
            f"reason: {ed_reason}"
        )

    if failures:
        print("[FAIL]", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("[OK] both fixtures behaved as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
