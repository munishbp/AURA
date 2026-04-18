"""HTML eval grid renderer + checkpoint-level static-image flagger.

For each (source, instruction, output) triple in the holdout set, render a
row showing both images side-by-side with the four metrics annotated.
Aggregate at the bottom. Save to a single self-contained HTML file.

Per the plan, this is the highest-leverage workstream: without it we can't
tell whether a trained LoRA is helping. The static-image flagger
(`flag_static_checkpoint`) is the direct mitigation for the failure mode that
killed the hackathon LoRAs.

Workstream 2.
"""

from __future__ import annotations

import base64
import io
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from aura_ml.eval.metrics import all_metrics, is_static


@dataclass
class GridEntry:
    pair_id: str
    source: Image.Image
    output: Image.Image
    instruction: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class GridReport:
    entries: list[GridEntry]
    aggregate: dict[str, float]
    static_flag: bool
    static_fraction: float

    def to_html(self, out_path: str | Path) -> None:
        """Write a self-contained HTML file with embedded base64 images."""
        # TODO(workstream 2): render template, embed thumbnails, write file
        raise NotImplementedError("workstream 2: implement HTML render")


def _img_to_data_uri(img: Image.Image, max_side: int = 384) -> str:
    """PIL Image -> 'data:image/jpeg;base64,...' for inline embedding."""
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_grid(
    triples: list[tuple[str, Image.Image, Image.Image, str]],
) -> GridReport:
    """Compute metrics on each triple and aggregate.

    Args:
        triples: list of (pair_id, source_image, output_image, instruction).
                 source comes from the holdout set; output is what the
                 checkpoint produced for that source + instruction.

    Returns:
        A GridReport with per-entry metrics and aggregate stats.
    """
    entries: list[GridEntry] = []
    for pair_id, src, out, instr in triples:
        m = all_metrics(src, out, instr)
        entries.append(GridEntry(pair_id, src, out, instr, m))

    # Aggregate (mean of non-NaN values per metric)
    keys = ["edit_magnitude", "arcface_cosine", "lpips", "clip_score"]
    agg: dict[str, float] = {}
    for k in keys:
        vals = [e.metrics[k] for e in entries if e.metrics.get(k) is not None]
        agg[k] = statistics.fmean(vals) if vals else float("nan")

    # Static-image canary: aggregate over the holdout
    static_count = sum(1 for e in entries if is_static(e.metrics))
    static_fraction = static_count / max(len(entries), 1)
    static_flag = static_fraction > 0.5  # > half the outputs were static = bad checkpoint

    return GridReport(
        entries=entries,
        aggregate=agg,
        static_flag=static_flag,
        static_fraction=static_fraction,
    )


def flag_static_checkpoint(report: GridReport) -> tuple[bool, str]:
    """Returns (should_quarantine, reason). Used by the training loop to drop
    a checkpoint into a 'suspect' bucket so it isn't auto-promoted as best.
    """
    if report.static_flag:
        return (
            True,
            f"static-image collapse: {report.static_fraction:.0%} of outputs "
            f"had edit_magnitude < threshold",
        )
    return (False, "ok")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Render an eval grid from a checkpoint vs a holdout set"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--holdout", required=True, help="dir w/ {control,prompts}")
    p.add_argument("--out", required=True, help="output HTML path")
    args = p.parse_args()
    # TODO(workstream 2):
    #   - Build pipeline w/ given checkpoint
    #   - For each (control, instruction) in holdout, run inference
    #   - Pass to build_grid + GridReport.to_html
    raise NotImplementedError("workstream 2: implement CLI")
