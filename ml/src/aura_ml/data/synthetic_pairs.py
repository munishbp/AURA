"""Bootstrap a paired training set from unpaired source photos.

Pattern: for each source image, run zero-shot Qwen-Image-Edit with a procedure
prompt to generate a candidate "after" image. Then have Qwen3.5-9B critique
the (source, candidate, instruction) triple. Keep pairs the critic scores well
AND that pass the edit-magnitude floor in `aura_ml.eval.metrics.edit_magnitude`.

Workstream 6. Important caveat: this is the same general pattern that produced
the bad hackathon dataset. Curate aggressively or this WILL re-introduce the
static-image collapse mode at training time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from aura_ml.inference.qwen_edit import QwenImageEditPipeline
from aura_ml.prompt_expander.qwen35 import Procedure, Qwen35PromptExpander


@dataclass
class SyntheticPairConfig:
    procedure: Procedure
    instructions: list[str]  # variants to sample from per source image
    samples_per_source: int = 3  # multiple candidates per source for filtering
    min_edit_magnitude: float = 0.05  # below this, the candidate is "static" — drop
    min_arcface_cosine: float = 0.6  # below this, identity not preserved — drop
    max_lpips: float = 0.4  # above this, the change is too dramatic — drop
    seed: int = 0


def generate_candidate(
    source: Image.Image,
    instruction: str,
    edit_pipeline: QwenImageEditPipeline,
    seed: int,
) -> Image.Image:
    """Single zero-shot edit. Returns a candidate "after" image."""
    # TODO(workstream 6): edit_pipeline.generate(source, instruction, seed=seed)
    raise NotImplementedError("workstream 6: implement candidate generation")


def critique_pair(
    source: Image.Image,
    candidate: Image.Image,
    instruction: str,
    expander: Qwen35PromptExpander,
) -> dict[str, Any]:
    """Use the VLM as a judge. Returns {"score": float in [0,1], "reason": str}.

    The score combines: (a) does the candidate plausibly look like the
    instruction was followed? (b) does it look anatomically realistic? (c) is
    the subject's identity preserved?

    The numeric metrics in `aura_ml.eval.metrics` are computed alongside; this
    function adds the qualitative judgment.
    """
    # TODO(workstream 6): structured prompt to the VLM, parse a numeric score
    raise NotImplementedError("workstream 6: implement VLM critic")


def curate_synthetic_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    config: SyntheticPairConfig,
    edit_pipeline: QwenImageEditPipeline,
    expander: Qwen35PromptExpander,
) -> dict[str, Any]:
    """Walk source_dir, generate candidates, filter, write to output_dir
    in the schema described in SCHEMA.md.

    Returns a report dict: candidates considered, kept, dropped per reason.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    (out / "control").mkdir(parents=True, exist_ok=True)
    (out / "target").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    # TODO(workstream 6):
    #   for each image in src:
    #     for each instruction in config.instructions[: config.samples_per_source]:
    #       candidate = generate_candidate(...)
    #       compute edit_magnitude, arcface_cosine, lpips
    #       if all pass thresholds: critique with VLM
    #       if VLM score > floor: write to control/, target/, prompts/, meta/
    #     log running counts for the final report
    raise NotImplementedError("workstream 6: implement curation loop")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("source_dir")
    p.add_argument("output_dir")
    p.add_argument("--procedure", choices=["rhinoplasty", "facelift", "blepharoplasty"])
    args = p.parse_args()
    cfg = SyntheticPairConfig(procedure=args.procedure, instructions=[])  # TODO: load from YAML
    edit = QwenImageEditPipeline()
    vlm = Qwen35PromptExpander()
    report = curate_synthetic_dataset(args.source_dir, args.output_dir, cfg, edit, vlm)
    print(report)
