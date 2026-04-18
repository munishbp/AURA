"""PyTorch Dataset over (control, target, instruction) triples.

On-disk schema is documented in SCHEMA.md (sibling file).

Workstream 4 implements `PairDataset.__getitem__` and `validate_dataset`.
Workstream 6 may add VAE-latent caching to `cache_latents()` for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class PairDataset(Dataset):
    """Loads (control, target, instruction) triples from disk.

    Args:
        root: directory containing control/, target/, prompts/, optional meta/
        resolution: training resolution. Images are resized + center-cropped.
        transform: optional callable applied AFTER resize/crop. Receives a
            PIL.Image and returns a torch.Tensor.
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
        self.transform = transform
        self.return_paths = return_paths

        self.control_dir = self.root / "control"
        self.target_dir = self.root / "target"
        self.prompts_dir = self.root / "prompts"

        self._pair_ids = self._discover_pair_ids()

    def _discover_pair_ids(self) -> list[str]:
        """Find basenames present in BOTH control/ and target/ AND prompts/."""
        # TODO(workstream 4):
        #   - List control/*.{jpg,jpeg,png,webp}
        #   - For each, check target/<id>.<ext> exists and prompts/<id>.txt exists
        #   - Skip + warn on partial entries (don't silently drop)
        #   - Return sorted list of stems
        raise NotImplementedError("workstream 4: implement pair discovery")

    def __len__(self) -> int:
        return len(self._pair_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # TODO(workstream 4):
        #   - Load PIL.Image for control + target
        #   - Resize + center-crop to self.resolution
        #   - Apply self.transform if provided
        #   - Read prompts/<id>.txt
        #   - Return dict {"control": ..., "target": ..., "instruction": str, "id": str}
        raise NotImplementedError("workstream 4: implement __getitem__")

    def cache_latents(self, vae, device: str = "cuda") -> None:
        """Pre-encode all control/target images via the VAE and cache to disk.

        Skips a major recurring cost in training. Optional speed-up.
        """
        # TODO(workstream 6): write to <root>/.latent_cache/<id>.pt
        raise NotImplementedError("workstream 6: implement latent caching")


def validate_dataset(root: str | Path, strict: bool = False) -> dict[str, Any]:
    """Sanity-check a dataset directory.

    Returns a report dict with counts and any warnings. Raises ValueError if
    `strict=True` and there are any issues.

    Checks:
        - control/, target/, prompts/ exist
        - For every control/<id>, target/<id> and prompts/<id>.txt exist
        - No non-image files in control/target
        - For each pair, edit-magnitude (1 - DINO cosine) is above a floor
          (the static-image canary applied at dataset-load time — see
          aura_ml.eval.metrics.edit_magnitude). This is the workstream 6
          guard against re-introducing the synthetic-pair collapse mode.
    """
    # TODO(workstream 4): implement basic structural checks
    # TODO(workstream 6): add edit-magnitude per-pair guard
    raise NotImplementedError("workstream 4: implement validate_dataset")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="dataset root containing control/ target/ prompts/")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    report = validate_dataset(args.root, strict=args.strict)
    print(report)
