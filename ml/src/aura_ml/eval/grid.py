"""HTML eval grid renderer + checkpoint-level static-image flagger.

For each (source, instruction, output) triple in the holdout set, render a
row showing both images side-by-side with the four metrics annotated.
Aggregate at the bottom. Save to a single self-contained HTML file.

Per the plan, this is the highest-leverage workstream: without it we can't
tell whether a trained LoRA is helping. The static-image flagger
(`flag_static_checkpoint`) is the direct mitigation for the failure mode that
killed the hackathon LoRAs.
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from aura_ml.eval.metrics import all_metrics, is_static

_METRIC_KEYS = ["edit_magnitude", "arcface_cosine", "lpips", "clip_score"]


@dataclass
class GridEntry:
    pair_id: str
    source: Image.Image
    output: Image.Image
    instruction: str
    reference: Image.Image | None = None  # optional ground-truth "after" image
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class GridReport:
    entries: list[GridEntry]
    aggregate: dict[str, float]
    static_flag: bool
    static_fraction: float
    meta: dict[str, str] = field(default_factory=dict)

    def to_html(self, out_path: str | Path, baseline: dict[str, float] | None = None) -> None:
        """Write a self-contained HTML file with embedded base64 images."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html = _render_html(self, baseline)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(html, encoding="utf-8")
        os.replace(tmp, out_path)

    def to_json(self, out_path: str | Path) -> None:
        """Write a JSON sidecar with the same basename as the HTML — diffable
        across runs without re-parsing markup.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "aggregate": self.aggregate,
            "static_flag": self.static_flag,
            "static_fraction": self.static_fraction,
            "entries": [
                {"pair_id": e.pair_id, "instruction": e.instruction, "metrics": e.metrics}
                for e in self.entries
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _img_to_data_uri(img: Image.Image, max_side: int = 384) -> str:
    """PIL Image -> 'data:image/jpeg;base64,...' for inline embedding."""
    img = img.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _is_real(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def build_grid(
    triples: list[tuple],
    meta: dict[str, str] | None = None,
) -> GridReport:
    """Compute metrics on each triple and aggregate.

    Args:
        triples: list of (pair_id, source_image, output_image, instruction)
                 OR (pair_id, source_image, output_image, instruction, reference_image).
        meta:    optional metadata dict surfaced in the HTML header
                 (e.g. {"checkpoint": "...", "mode": "outputs-dir"}).

    Returns:
        A GridReport with per-entry metrics and aggregate stats.
    """
    entries: list[GridEntry] = []
    for tup in triples:
        if len(tup) == 5:
            pair_id, src, out, instr, ref = tup
        else:
            pair_id, src, out, instr = tup
            ref = None
        m = all_metrics(src, out, instr)
        entries.append(GridEntry(pair_id, src, out, instr, reference=ref, metrics=m))

    agg: dict[str, float] = {}
    for k in _METRIC_KEYS:
        vals = [e.metrics[k] for e in entries if _is_real(e.metrics.get(k))]
        agg[k] = statistics.fmean(vals) if vals else float("nan")

    static_count = sum(1 for e in entries if is_static(e.metrics))
    static_fraction = static_count / max(len(entries), 1)
    static_flag = static_fraction > 0.5  # > half static = bad checkpoint

    # Sort flagged entries to the top for visibility
    entries.sort(key=lambda e: (not is_static(e.metrics), e.pair_id))

    return GridReport(
        entries=entries,
        aggregate=agg,
        static_flag=static_flag,
        static_fraction=static_fraction,
        meta=meta or {},
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


# --- HTML rendering ---------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 24px; color: #222; background: #fafafa; }
h1 { margin: 0 0 8px 0; font-size: 22px; }
.meta { color: #666; font-size: 13px; margin-bottom: 16px; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 6px;
         font-weight: 600; font-size: 13px; color: #fff; }
.badge.ok { background: #1e8e3e; }
.badge.bad { background: #c5221f; }
.agg { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
       margin: 16px 0; max-width: 720px; }
.agg .card { background: #fff; border: 1px solid #ddd; border-radius: 8px;
             padding: 10px 14px; }
.agg .k { font-size: 12px; color: #666; text-transform: uppercase;
          letter-spacing: 0.05em; }
.agg .v { font-size: 20px; font-weight: 600; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #eee;
         vertical-align: top; font-size: 13px; }
th { background: #f1f3f4; font-weight: 600; }
tr.flagged { background: #fff5f5; }
.thumb { display: block; max-width: 200px; max-height: 200px; border-radius: 4px; }
.cell-bad { color: #c5221f; font-weight: 600; }
.cell-noface { color: #888; font-style: italic; }
.cell-good { color: #1e8e3e; font-weight: 600; }
.pair-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           color: #555; font-size: 12px; }
.instr { max-width: 320px; font-size: 12px; color: #444; }
"""


def _fmt(v: float) -> str:
    if not _is_real(v):
        return "—"
    return f"{v:.4f}"


def _cell(value: float, kind: str, baseline: float | None = None) -> str:
    if not _is_real(value):
        # Distinguish NaN (no face detected) for arcface from genuinely absent
        if kind == "arcface_cosine":
            return '<td class="cell-noface">no face</td>'
        return "<td>—</td>"
    cls = ""
    if kind == "edit_magnitude" and value < 0.05:
        cls = "cell-bad"
    elif kind == "arcface_cosine" and value < 0.6:
        cls = "cell-bad"
    elif kind == "clip_score" and baseline is not None and _is_real(baseline):
        cls = "cell-good" if value > baseline else "cell-bad"
    attr = f' class="{cls}"' if cls else ""
    return f"<td{attr}>{_fmt(value)}</td>"


def _render_html(report: GridReport, baseline: dict[str, float] | None) -> str:
    badge_cls = "bad" if report.static_flag else "ok"
    badge_txt = (
        f"CANARY TRIPPED — {report.static_fraction:.0%} static"
        if report.static_flag
        else f"OK — {report.static_fraction:.0%} static"
    )
    meta_bits = [f"<b>{k}</b>: {v}" for k, v in report.meta.items()]
    meta_bits.append(f"<b>timestamp</b>: {_dt.datetime.now().isoformat(timespec='seconds')}")
    meta_bits.append(f"<b>n</b>: {len(report.entries)}")
    meta_html = " · ".join(meta_bits)

    agg_cards = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{_fmt(report.aggregate[k])}</div></div>'
        for k in _METRIC_KEYS
    )

    rows: list[str] = []
    base_clip = baseline.get("clip_score") if baseline else None
    for e in report.entries:
        flagged = is_static(e.metrics)
        ref_cell = (
            f'<td><img class="thumb" src="{_img_to_data_uri(e.reference)}" alt="ref"></td>'
            if e.reference is not None
            else "<td></td>"
        )
        row = (
            f'<tr class="{"flagged" if flagged else ""}">'
            f'<td><span class="pair-id">{e.pair_id}</span></td>'
            f'<td><img class="thumb" src="{_img_to_data_uri(e.source)}" alt="src"></td>'
            f'<td><img class="thumb" src="{_img_to_data_uri(e.output)}" alt="out"></td>'
            f"{ref_cell}"
            f'<td class="instr">{_html_escape(e.instruction)}</td>'
            f'{_cell(e.metrics.get("edit_magnitude"), "edit_magnitude")}'
            f'{_cell(e.metrics.get("arcface_cosine"), "arcface_cosine")}'
            f'{_cell(e.metrics.get("lpips"), "lpips")}'
            f'{_cell(e.metrics.get("clip_score"), "clip_score", baseline=base_clip)}'
            "</tr>"
        )
        rows.append(row)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Aura eval grid</title>
<style>{_CSS}</style></head><body>
<h1>Aura eval grid <span class="badge {badge_cls}">{badge_txt}</span></h1>
<div class="meta">{meta_html}</div>
<div class="agg">{agg_cards}</div>
<table>
<thead><tr>
<th>id</th><th>source</th><th>output</th><th>reference</th><th>instruction</th>
<th>edit_mag</th><th>arcface</th><th>lpips</th><th>clip</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</body></html>
"""


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --- CLI --------------------------------------------------------------------


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _find_image(root: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTS:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load_holdout(holdout: Path) -> list[tuple[str, Image.Image, str, Image.Image | None]]:
    """Returns list of (pair_id, control_image, instruction, optional_reference)."""
    control_dir = holdout / "control"
    prompts_dir = holdout / "prompts"
    reference_dir = holdout / "reference"
    if not control_dir.is_dir() or not prompts_dir.is_dir():
        raise FileNotFoundError(
            f"holdout {holdout} missing control/ or prompts/ subdir"
        )
    items = []
    for ctrl_path in sorted(control_dir.iterdir()):
        if ctrl_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        stem = ctrl_path.stem
        prompt_path = prompts_dir / f"{stem}.txt"
        if not prompt_path.exists():
            print(f"[skip] no prompt for {stem}", file=sys.stderr)
            continue
        ref_path = _find_image(reference_dir, stem) if reference_dir.is_dir() else None
        items.append(
            (
                stem,
                Image.open(ctrl_path),
                prompt_path.read_text(encoding="utf-8").strip(),
                Image.open(ref_path) if ref_path else None,
            )
        )
    return items


def _run_outputs_mode(holdout: Path, outputs_dir: Path) -> list[tuple]:
    holdout_items = _load_holdout(holdout)
    triples = []
    missing = 0
    for stem, ctrl, instr, ref in holdout_items:
        out_path = _find_image(outputs_dir, stem)
        if out_path is None:
            print(f"[skip] no output for {stem} in {outputs_dir}", file=sys.stderr)
            missing += 1
            continue
        triples.append((stem, ctrl, Image.open(out_path), instr, ref))
    if missing:
        print(f"[warn] {missing} entries missing outputs", file=sys.stderr)
    return triples


def _run_checkpoint_mode(holdout: Path, checkpoint: Path) -> list[tuple]:
    try:
        from aura_ml.inference.qwen_edit import QwenEditConfig, QwenImageEditPipeline
    except Exception as e:
        print(f"[fatal] cannot import inference pipeline: {e}", file=sys.stderr)
        sys.exit(2)

    pipe = QwenImageEditPipeline(QwenEditConfig())
    try:
        pipe.load()
    except NotImplementedError:
        print(
            "[fatal] workstream 1 inference pipeline is not yet implemented — "
            "use --outputs-dir instead, or wait on W1.",
            file=sys.stderr,
        )
        sys.exit(2)

    # If a LoRA checkpoint dir is given, load it (W7 will fill this in).
    if checkpoint.is_dir():
        try:
            pipe.load_lora(str(checkpoint), name=checkpoint.name)
            pipe.set_active_loras([checkpoint.name], [1.0])
        except NotImplementedError:
            print(
                "[warn] LoRA loading not yet implemented (W7); running base model "
                "without adapter.",
                file=sys.stderr,
            )

    holdout_items = _load_holdout(holdout)
    triples = []
    for stem, ctrl, instr, ref in holdout_items:
        try:
            out = pipe.generate(image=ctrl, prompt=instr)
        except NotImplementedError:
            print(
                "[fatal] generate() not yet implemented — use --outputs-dir.",
                file=sys.stderr,
            )
            sys.exit(2)
        triples.append((stem, ctrl, out, instr, ref))
    return triples


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Render an eval grid from a checkpoint or pre-computed outputs"
    )
    p.add_argument("--holdout", required=True, help="dir w/ {control,prompts,reference}")
    p.add_argument("--out", required=True, help="output HTML path")
    p.add_argument(
        "--baseline-grid",
        default=None,
        help="optional path to a previous grid.json for CLIPScore coloring",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--outputs-dir",
        help="pre-computed outputs (basenames match holdout/control/)",
    )
    src.add_argument(
        "--checkpoint",
        help="LoRA checkpoint dir; requires W1 (qwen_edit.py.generate)",
    )

    args = p.parse_args()
    holdout = Path(args.holdout)
    out_html = Path(args.out)

    if args.outputs_dir:
        mode = "outputs-dir"
        triples = _run_outputs_mode(holdout, Path(args.outputs_dir))
        meta = {"mode": mode, "outputs_dir": args.outputs_dir, "holdout": str(holdout)}
    else:
        mode = "checkpoint"
        triples = _run_checkpoint_mode(holdout, Path(args.checkpoint))
        meta = {"mode": mode, "checkpoint": args.checkpoint, "holdout": str(holdout)}

    if not triples:
        print("[fatal] no triples to score", file=sys.stderr)
        sys.exit(1)

    report = build_grid(triples, meta=meta)

    baseline = None
    if args.baseline_grid:
        baseline_path = Path(args.baseline_grid)
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8")).get(
                "aggregate"
            )

    report.to_html(out_html, baseline=baseline)
    report.to_json(out_html.with_suffix(".json"))

    should_quarantine, reason = flag_static_checkpoint(report)
    print(
        f"wrote {out_html} (n={len(report.entries)}, static_fraction="
        f"{report.static_fraction:.0%}, flag={'TRIP' if should_quarantine else 'ok'})"
    )
    print(f"  aggregate: {report.aggregate}")
    if should_quarantine:
        print(f"  reason: {reason}")


if __name__ == "__main__":
    main()
