"""Gradio demo for Aura — surgical outcome visualization.

Upload a face photo, pick a procedure, type the physician's instruction.
The Qwen3.5-9B expander turns it into a precise edit instruction, the
Qwen-Image-Edit-2511 editor applies it, and the eval metrics score the
result live (including the static-image canary).

Run from ml/:
    uv run python -m app.demo
    uv run python -m app.demo --no-expander      # save ~7 GB VRAM
    uv run python -m app.demo --checkpoints checkpoints/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gradio as gr
from PIL import Image

from aura_ml.inference.pipeline import (
    PROCEDURES,
    AuraInferencePipeline,
    build_default_pipeline,
)
from aura_ml.prompt_expander.qwen35 import OutOfScopeError

try:
    from aura_ml.eval.metrics import all_metrics, is_static

    _HAVE_METRICS = True
except ImportError:  # eval extra not installed — demo still works
    _HAVE_METRICS = False


EXAMPLE_INSTRUCTIONS = {
    "rhinoplasty": "Subtle dorsal hump reduction with refined nasal tip",
    "facelift": "Tighten the lower-face jawline and reduce nasolabial fold",
    "blepharoplasty": "Reduce upper-lid skin redundancy and refine the supratarsal crease",
}

CSS = """
/* ── Aura demo skin ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #0f0f12 !important;
    color: #e8e4dc !important;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    letter-spacing: -0.02em;
}

.aura-header {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    border-bottom: 1px solid #2a2a35;
    margin-bottom: 1.5rem;
}
.aura-header h1 {
    font-size: 2.8rem;
    background: linear-gradient(135deg, #c8b99a 0%, #e8dcc8 60%, #a89878 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.aura-header p {
    color: #888;
    font-size: 0.95rem;
    margin: 0.4rem 0 0;
}

button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg, #c8a96e, #a07840) !important;
    border: none !important;
    color: #0f0f12 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
button.primary-btn:hover { opacity: 0.88; }

.image-panel label { color: #a89878 !important; font-size: 0.82rem !important; }
textarea { background: #1e1e28 !important; border-color: #3a3a48 !important; color: #e8e4dc !important; }
"""

_DISCLAIMER = """
<div style="margin-top:1.5rem; padding:1rem; background:#1a1a22;
            border:1px solid #2a2a35; border-radius:8px;
            font-size:0.82rem; color:#666; text-align:center;">
  ⚠️ For physician consultation and research purposes only.
  Not a medical device. Outcomes are illustrative and non-binding.
</div>
"""


def _fmt_metrics(metrics: dict[str, float]) -> str:
    """Render the four metrics + canary verdict as markdown."""
    def fmt(v: float) -> str:
        return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.3f}"

    static = is_static(metrics)
    canary = (
        '🔴 **CANARY** — output ≈ input (static-image collapse)'
        if static
        else "🟢 real edit detected"
    )
    identity_note = ""
    af = metrics.get("arcface_cosine")
    if af is not None and not math.isnan(af) and af < 0.6:
        identity_note = " · ⚠️ identity drift (ArcFace < 0.6)"
    return (
        f"{canary}{identity_note}\n\n"
        f"| edit magnitude | ArcFace identity | LPIPS | CLIPScore |\n"
        f"|---|---|---|---|\n"
        f"| {fmt(metrics.get('edit_magnitude'))} | {fmt(metrics.get('arcface_cosine'))} "
        f"| {fmt(metrics.get('lpips'))} | {fmt(metrics.get('clip_score'))} |"
    )


def build_ui(pipeline: AuraInferencePipeline) -> gr.Blocks:
    def expand_only(face_image, instruction: str, procedure: str, seed_raw):
        if face_image is None:
            raise gr.Error("Please upload a face photo first.")
        if not instruction.strip():
            raise gr.Error("Please enter an instruction (e.g. 'narrow the nasal tip').")
        seed = int(seed_raw) if seed_raw and int(seed_raw) > 0 else None
        try:
            result = pipeline.expand_prompt(face_image, instruction, procedure, seed=seed)
        except OutOfScopeError as e:
            raise gr.Error(f"Instruction out of scope for {procedure}: {e}") from e
        note = " (rule-based fallback)" if result.used_fallback else ""
        gr.Info(f"Prompt expanded in {result.latency_s:.1f}s{note}")
        return result.prompt

    def run(face_image, instruction: str, procedure: str, prompt_override: str,
            steps: int, seed_raw, use_expander: bool):
        if face_image is None:
            raise gr.Error("Please upload a face photo first.")
        if not instruction.strip() and not prompt_override.strip():
            raise gr.Error("Please enter an instruction (e.g. 'narrow the nasal tip').")

        seed = int(seed_raw) if seed_raw and int(seed_raw) > 0 else None
        face_image = face_image.convert("RGB")

        # A hand-edited prompt in the expanded box wins over re-expansion.
        if prompt_override.strip():
            prompt_used = prompt_override.strip()
        elif use_expander:
            try:
                prompt_used = pipeline.expand_prompt(
                    face_image, instruction, procedure, seed=seed
                ).prompt
            except OutOfScopeError as e:
                raise gr.Error(f"Instruction out of scope for {procedure}: {e}") from e
        else:
            prompt_used = instruction.strip()

        edited, _ = pipeline.generate(
            face_image, prompt_used, procedure,
            num_steps=int(steps), seed=seed, expand=False,
        )

        metrics_md = "*(eval extra not installed — `uv sync --extra eval`)*"
        if _HAVE_METRICS:
            metrics_md = _fmt_metrics(all_metrics(face_image, edited, prompt_used))

        return edited, prompt_used, metrics_md

    def fill_example(procedure: str) -> str:
        return EXAMPLE_INSTRUCTIONS.get(procedure, "")

    theme = gr.themes.Base(primary_hue="amber", neutral_hue="slate")
    with gr.Blocks(title="Aura — Surgical Preview", css=CSS, theme=theme) as demo:
        gr.HTML("""
        <div class="aura-header">
          <h1>Aura</h1>
          <p>Surgical outcome visualization · Qwen-Image-Edit-2511 + Qwen3.5-9B</p>
        </div>
        """)

        with gr.Row():
            # ── Left column: inputs ─────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                face = gr.Image(type="pil", label="Face photo", elem_classes=["image-panel"])
                proc = gr.Dropdown(choices=list(PROCEDURES), value="rhinoplasty", label="Procedure")
                instr = gr.Textbox(
                    lines=3,
                    label="Physician instruction",
                    placeholder="e.g. narrow the nasal tip and reduce the dorsal hump",
                )
                with gr.Row():
                    example_btn = gr.Button("Fill example", size="sm")
                    expand_btn = gr.Button("Expand only", size="sm")
                    clear_btn = gr.Button("Clear", size="sm")

                expanded_prompt = gr.Textbox(
                    label="Expanded prompt (editable — used as-is if filled)",
                    lines=5,
                )

                with gr.Accordion("Advanced options", open=False):
                    steps = gr.Slider(10, 60, value=40, step=1, label="Diffusion steps")
                    seed_input = gr.Number(label="Seed (0 = random)", value=0, precision=0)
                    use_expander = gr.Checkbox(
                        label="Use VLM prompt expander", value=pipeline.expander is not None,
                        interactive=pipeline.expander is not None,
                    )

                generate_btn = gr.Button(
                    "Generate Preview", variant="primary", elem_classes=["primary-btn"]
                )

            # ── Right column: outputs ────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                out_img = gr.Image(
                    type="pil", label="Surgical preview",
                    elem_classes=["image-panel"], interactive=False,
                )
                metrics_box = gr.Markdown(label="Eval metrics")

        gr.HTML(_DISCLAIMER)

        # ── Event wiring ─────────────────────────────────────────────────
        generate_btn.click(
            fn=run,
            inputs=[face, instr, proc, expanded_prompt, steps, seed_input, use_expander],
            outputs=[out_img, expanded_prompt, metrics_box],
        )
        expand_btn.click(
            fn=expand_only,
            inputs=[face, instr, proc, seed_input],
            outputs=[expanded_prompt],
        )
        example_btn.click(fn=fill_example, inputs=[proc], outputs=[instr])
        proc.change(fn=fill_example, inputs=[proc], outputs=[instr])
        # A stale expanded prompt must not override a freshly edited instruction.
        instr.change(fn=lambda: "", outputs=[expanded_prompt])
        clear_btn.click(fn=lambda: ("", None, "", ""), outputs=[instr, out_img, expanded_prompt, metrics_box])

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Aura surgical preview demo")
    p.add_argument(
        "--checkpoints",
        default="checkpoints",
        help="dir containing per-procedure LoRA checkpoint subdirs",
    )
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="public Gradio share link")
    p.add_argument(
        "--no-expander",
        action="store_true",
        help="skip the Qwen3.5-9B expander (~7 GB VRAM saved; rule-based expansion instead)",
    )
    args = p.parse_args()

    pipeline = build_default_pipeline(
        Path(args.checkpoints), use_prompt_expander=not args.no_expander
    )
    demo = build_ui(pipeline)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
