"""Gradio demo for Aura — surgical outcome visualization.

Takes a face photo + procedure selection + instruction and runs
Qwen2-VL as an instruction-following image editor.

Run with:
    pip install gradio transformers torch pillow qwen-vl-utils
    python demo.py

Or from the repo root:
    uv run python -m app.demo --checkpoints checkpoints/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Inline pipeline (no import dependency on the full aura_ml package)
# ---------------------------------------------------------------------------

def _build_pipeline(checkpoints_dir: Path, use_lora: bool = False):
    """Build a QwenImageEditPipeline, optionally loading LoRA checkpoints."""
    # Try importing from the installed package first; fall back to local file.
    try:
        from aura_ml.inference.qwen_edit import QwenEditConfig, QwenImageEditPipeline
    except ImportError:
        # Running standalone — load from the same directory as this file
        import importlib.util, os
        here = Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "qwen_edit", here / "qwen_edit.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        QwenEditConfig = mod.QwenEditConfig
        QwenImageEditPipeline = mod.QwenImageEditPipeline

    cfg = QwenEditConfig(
        auto_device=True,
    )
    pipe = QwenImageEditPipeline(cfg)

    if use_lora and checkpoints_dir.is_dir():
        for proc in ("rhinoplasty", "facelift", "blepharoplasty"):
            ckpt = checkpoints_dir / proc
            if ckpt.is_dir():
                pipe.load_lora(str(ckpt), name=proc)
                print(f"[demo] loaded LoRA: {proc}")

    return pipe


# ---------------------------------------------------------------------------
# Prompt expander (optional — falls back to raw instruction if unavailable)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a surgical-visualization prompt assistant. Given a
face photo and a brief instruction from a physician, produce a single detailed
prompt for a diffusion image-editor.

Rules:
- Stay anatomically realistic. Describe changes in terms of anatomical
  landmarks (dorsal hump, alar base, nasolabial fold, supratarsal crease,
  jowl, etc.).
- Use conservative quantifiers. Prefer "subtle", "moderate", "refined".
  Avoid "dramatic", "much smaller", "completely".
- Preserve identity. Always include language asking the editor to maintain
  the subject's bone structure, skin texture, and core proportions.
- Output ONE paragraph, no preamble, no explanation. Just the prompt.
"""

PROCEDURE_HINTS = {
    "rhinoplasty": (
        "Procedure context: rhinoplasty. Focus on nasal bridge, "
        "dorsal hump, tip projection, alar base, columella."
    ),
    "facelift": (
        "Procedure context: rhytidectomy. Focus on jawline definition, "
        "midface volume, nasolabial fold, marionette lines, jowl."
    ),
    "blepharoplasty": (
        "Procedure context: blepharoplasty. Focus on upper-lid "
        "skin redundancy, supratarsal crease, lower-lid bags, periorbital hollows."
    ),
}


def _expand_prompt_simple(instruction: str, procedure: str) -> str:
    """Rule-based prompt enrichment when the LLM expander isn't available."""
    hint = PROCEDURE_HINTS.get(procedure, "")
    return (
        f"{hint} "
        f"Apply the following change conservatively and realistically: {instruction}. "
        "Preserve the subject's identity, skin texture, and overall facial proportions. "
        "Maintain bilateral symmetry."
    ).strip()


# ---------------------------------------------------------------------------
# UI builders
# ---------------------------------------------------------------------------

PROCEDURES = ["rhinoplasty", "facelift", "blepharoplasty"]

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

/* Header */
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

/* Cards */
.card {
    background: #16161e;
    border: 1px solid #2a2a35;
    border-radius: 12px;
    padding: 1.25rem;
}

/* Buttons */
button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg, #c8a96e, #a07840) !important;
    border: none !important;
    color: #0f0f12 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
button.primary-btn:hover {
    opacity: 0.88;
}

/* Status badge */
.status-ok   { color: #5aad7a; font-weight: 600; }
.status-warn { color: #d4955a; font-weight: 600; }

/* Image panels */
.image-panel label { color: #a89878 !important; font-size: 0.82rem !important; }

/* Instruction box */
textarea { background: #1e1e28 !important; border-color: #3a3a48 !important; color: #e8e4dc !important; }
"""


def build_ui(checkpoints_dir: Path) -> gr.Blocks:
    # Lazy-load the pipeline on first use
    _state: dict = {"pipe": None}

    def get_pipe():
        if _state["pipe"] is None:
            _state["pipe"] = _build_pipeline(checkpoints_dir)
        return _state["pipe"]

    def run(
        face_image,
        instruction: str,
        procedure: str,
        seed_raw,
        use_expander: bool,
    ):
        if face_image is None:
            raise gr.Error("Please upload a face photo first.")
        if not instruction.strip():
            raise gr.Error("Please enter an instruction (e.g. 'narrow the nasal tip').")

        seed = int(seed_raw) if seed_raw and int(seed_raw) > 0 else None

        # Expand prompt
        if use_expander:
            expanded = _expand_prompt_simple(instruction, procedure)
        else:
            expanded = instruction

        pipe = get_pipe()

        # Activate procedure LoRA if loaded
        if procedure in pipe._loaded_loras:
            pipe.set_active_loras([procedure], [0.7])

        edited = pipe.generate(
            image=face_image,
            prompt=expanded,
            seed=seed,
        )
        return edited, expanded

    def fill_example(procedure: str) -> str:
        return EXAMPLE_INSTRUCTIONS.get(procedure, "")

    with gr.Blocks(title="Aura — Surgical Preview") as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="aura-header">
          <h1>Aura</h1>
          <p>Surgical outcome visualization · Powered by Qwen-Image-Edit</p>
        </div>
        """)

        # ── Main layout ─────────────────────────────────────────────────
        with gr.Row():

            # ── Left column: inputs ─────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                face = gr.Image(
                    type="pil",
                    label="Face photo",
                    elem_classes=["image-panel"],
                )

                proc = gr.Dropdown(
                    choices=PROCEDURES,
                    value="rhinoplasty",
                    label="Procedure",
                )

                instr = gr.Textbox(
                    lines=4,
                    label="Physician instruction",
                    placeholder="e.g. narrow the nasal tip and reduce the dorsal hump",
                )

                with gr.Row():
                    example_btn = gr.Button("Fill example", size="sm")
                    clear_btn = gr.Button("Clear", size="sm")

                with gr.Accordion("Advanced options", open=False):
                    seed_input = gr.Number(
                        label="Seed (0 = random)",
                        value=0,
                        precision=0,
                    )
                    use_expander = gr.Checkbox(
                        label="Use prompt expander (anatomical enrichment)",
                        value=True,
                    )

                generate_btn = gr.Button(
                    "Generate Preview",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

            # ── Right column: outputs ────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Output")

                out_img = gr.Image(
                    type="pil",
                    label="Surgical preview",
                    elem_classes=["image-panel"],
                    interactive=False,
                )

                expanded_prompt = gr.Textbox(
                    label="Expanded prompt sent to model",
                    lines=5,
                    interactive=False,
                )

        # ── Disclaimer ───────────────────────────────────────────────────
        gr.HTML("""
        <div style="margin-top:1.5rem; padding:1rem; background:#1a1a22;
                    border:1px solid #2a2a35; border-radius:8px;
                    font-size:0.82rem; color:#666; text-align:center;">
          ⚠️ For physician consultation and research purposes only.
          Not a medical device. Outcomes are illustrative and non-binding.
        </div>
        """)

        # ── Event wiring ─────────────────────────────────────────────────
        generate_btn.click(
            fn=run,
            inputs=[face, instr, proc, seed_input, use_expander],
            outputs=[out_img, expanded_prompt],
        )

        example_btn.click(
            fn=fill_example,
            inputs=[proc],
            outputs=[instr],
        )

        clear_btn.click(
            fn=lambda: ("", None, ""),
            outputs=[instr, out_img, expanded_prompt],
        )

        # Auto-fill example when procedure changes
        proc.change(
            fn=fill_example,
            inputs=[proc],
            outputs=[instr],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        "--no-lora",
        action="store_true",
        help="skip LoRA loading (useful when checkpoints aren't ready yet)",
    )
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoints)
    demo = build_ui(ckpt_dir)
    demo.launch(
        server_port=args.port,
        share=args.share,
        css=CSS,
        theme=gr.themes.Base(primary_hue="amber", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()