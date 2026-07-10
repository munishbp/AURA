"""PyTorch Dataset over (control, target, instruction) triples.

On-disk schema is documented in SCHEMA.md (sibling file):

    <root>/control/<id>.jpg     the "before" image (input to the edit)
    <root>/target/<id>.jpg      the "after" image (desired output)
    <root>/prompts/<id>.txt     one instruction per pair
    <root>/meta/<id>.json       optional provenance sidecar (ignored here)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class PairSample:
    pair_id: str
    control: Image.Image
    target: Image.Image
    instruction: str
    meta: dict[str, Any] | None = None


def _find_image(root: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _resize_center_crop(img: Image.Image, resolution: int) -> Image.Image:
    """Resize the short edge to `resolution`, then center-crop a square."""
    img = img.convert("RGB")
    w, h = img.size
    scale = resolution / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    w, h = img.size
    left = (w - resolution) // 2
    top = (h - resolution) // 2
    return img.crop((left, top, left + resolution, top + resolution))


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor in [-1, 1], CxHxW (VAE input convention)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t * 2.0 - 1.0


class PairDataset(Dataset):
    """Loads (control, target, instruction) triples from disk.

    Args:
        root: directory containing control/, target/, prompts/, optional meta/
        resolution: training resolution. Images are resized + center-cropped.
        transform: optional callable applied AFTER resize/crop. Receives a
            PIL.Image and returns a torch.Tensor. Defaults to [-1,1] float.
        return_paths: include source paths in the returned dict (debug aid).
    """

    def __init__(
        self,
        root: str | Path,
        resolution: int = 768,
        transform=None,
        return_paths: bool = False,
    ) -> None:
        self.root = Path(root)
        self.resolution = resolution
        self.transform = transform or _to_tensor
        self.return_paths = return_paths

        self.control_dir = self.root / "control"
        self.target_dir = self.root / "target"
        self.prompts_dir = self.root / "prompts"

        self._pair_ids = self._discover_pair_ids()
        if not self._pair_ids:
            raise ValueError(f"no complete (control,target,prompt) triples under {self.root}")

    def _discover_pair_ids(self) -> list[str]:
        """Find basenames present in control/ AND target/ AND prompts/.
        Partial entries are warned about, not silently dropped."""
        if not self.control_dir.is_dir():
            raise FileNotFoundError(f"{self.control_dir} does not exist")
        stems = []
        for p in sorted(self.control_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = p.stem
            missing = []
            if _find_image(self.target_dir, stem) is None:
                missing.append("target")
            if not (self.prompts_dir / f"{stem}.txt").exists():
                missing.append("prompt")
            if missing:
                warnings.warn(f"pair '{stem}' missing {'+'.join(missing)} — skipped", stacklevel=2)
                continue
            stems.append(stem)
        return stems

    @property
    def pair_ids(self) -> list[str]:
        return list(self._pair_ids)

    def __len__(self) -> int:
        return len(self._pair_ids)

    def load_pil(self, idx: int) -> PairSample:
        """Load one sample as PIL images (no resize) — for eval/inspection."""
        stem = self._pair_ids[idx]
        return PairSample(
            pair_id=stem,
            control=Image.open(_find_image(self.control_dir, stem)).convert("RGB"),
            target=Image.open(_find_image(self.target_dir, stem)).convert("RGB"),
            instruction=(self.prompts_dir / f"{stem}.txt").read_text(encoding="utf-8").strip(),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        stem = self._pair_ids[idx]
        ctrl_path = _find_image(self.control_dir, stem)
        tgt_path = _find_image(self.target_dir, stem)

        control = _resize_center_crop(Image.open(ctrl_path), self.resolution)
        target = _resize_center_crop(Image.open(tgt_path), self.resolution)
        instruction = (self.prompts_dir / f"{stem}.txt").read_text(encoding="utf-8").strip()

        item: dict[str, Any] = {
            "id": stem,
            "control": self.transform(control),
            "target": self.transform(target),
            "instruction": instruction,
        }
        if self.return_paths:
            item["control_path"] = str(ctrl_path)
            item["target_path"] = str(tgt_path)
        return item


def validate_dataset(
    root: str | Path,
    strict: bool = False,
    check_edit_magnitude: bool = False,
    min_edit_magnitude: float = 0.02,
) -> dict[str, Any]:
    """Sanity-check a dataset directory.

    Structural checks always run. With `check_edit_magnitude=True`, each
    (control, target) pair is also run through the DINO edit-magnitude metric
    — pairs where control ≈ target teach the model to copy the input, which
    is the exact identity-collapse mode that killed the hackathon LoRAs.
    (Requires the eval extra + a GPU; it loads DINOv2.)

    Returns a report dict. Raises ValueError if strict=True and issues exist.
    """
    root = Path(root)
    report: dict[str, Any] = {"root": str(root), "ok": 0, "issues": []}

    for sub in ("control", "target", "prompts"):
        if not (root / sub).is_dir():
            report["issues"].append(f"missing {sub}/ directory")
    if report["issues"]:
        if strict:
            raise ValueError(f"dataset invalid: {report['issues']}")
        return report

    control_stems = {
        p.stem for p in (root / "control").iterdir() if p.suffix.lower() in IMAGE_EXTS
    }
    target_stems = {
        p.stem for p in (root / "target").iterdir() if p.suffix.lower() in IMAGE_EXTS
    }
    prompt_stems = {p.stem for p in (root / "prompts").glob("*.txt")}

    for stem in sorted(control_stems - target_stems):
        report["issues"].append(f"{stem}: control without target")
    for stem in sorted(target_stems - control_stems):
        report["issues"].append(f"{stem}: target without control")
    for stem in sorted((control_stems & target_stems) - prompt_stems):
        report["issues"].append(f"{stem}: pair without prompt")

    for sub in ("control", "target"):
        for p in (root / sub).iterdir():
            if p.is_file() and p.suffix.lower() not in IMAGE_EXTS:
                report["issues"].append(f"non-image file: {sub}/{p.name}")

    complete = sorted(control_stems & target_stems & prompt_stems)
    report["ok"] = len(complete)

    if check_edit_magnitude and complete:
        from aura_ml.eval.metrics import edit_magnitude  # heavy import, on demand

        static_pairs = []
        for stem in complete:
            ctrl = Image.open(_find_image(root / "control", stem))
            tgt = Image.open(_find_image(root / "target", stem))
            em = edit_magnitude(ctrl, tgt)
            if em < min_edit_magnitude:
                static_pairs.append({"id": stem, "edit_magnitude": round(em, 4)})
        if static_pairs:
            report["issues"].append(
                f"{len(static_pairs)} pair(s) below edit-magnitude floor "
                f"{min_edit_magnitude} (control ≈ target — identity-collapse risk)"
            )
            report["static_pairs"] = static_pairs

    if strict and report["issues"]:
        raise ValueError(f"dataset invalid: {report['issues']}")
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="dataset root containing control/ target/ prompts/")
    p.add_argument("--strict", action="store_true")
    p.add_argument(
        "--check-edit-magnitude",
        action="store_true",
        help="also run the DINO static-pair canary over every pair (needs GPU)",
    )
    args = p.parse_args()
    report = validate_dataset(
        args.root, strict=args.strict, check_edit_magnitude=args.check_edit_magnitude
    )
    print(json.dumps(report, indent=2))
