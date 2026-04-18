"""Gradio demo. Single Python file. Replaces the entire hackathon
Express + React + iOS stack with one upload-image → click-button page.

Workstream 9. Run with:

    uv run python -m app.demo --checkpoints checkpoints/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from aura_ml.inference.pipeline import Procedure, build_default_pipeline


def build_ui(checkpoints_dir: Path) -> gr.Blocks:
    pipeline = build_default_pipeline(checkpoints_dir)

    PROCEDURES: list[Procedure] = ["rhinoplasty", "facelift", "blepharoplasty"]

    def run(
        face_image,
        instruction: str,
        procedure: str,
        seed: int | None,
    ):
        if face_image is None:
            raise gr.Error("Upload a face photo first.")
        if not instruction.strip():
            raise gr.Error("Add an instruction (e.g. 'narrow the nasal tip').")
        edited, used_prompt = pipeline.generate(
            face_image=face_image,
            user_instruction=instruction,
            procedure=procedure,
            seed=seed if seed and seed > 0 else None,
        )
        return edited, used_prompt

    with gr.Blocks(title="Aura — surgical preview") as demo:
        gr.Markdown(
            "# Aura\n"
            "Upload a face photo, pick a procedure, and describe what you "
            "want changed. The prompt expander rewrites your instruction; "
            "the diffusion model generates the preview."
        )
        with gr.Row():
            with gr.Column():
                face = gr.Image(type="pil", label="Face photo")
                proc = gr.Dropdown(PROCEDURES, value="rhinoplasty", label="Procedure")
                instr = gr.Textbox(
                    lines=3,
                    label="Instruction",
                    placeholder="e.g. narrow the nasal tip and reduce the dorsal hump",
                )
                seed = gr.Number(label="Seed (optional, 0 = random)", value=0, precision=0)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Preview", type="pil")
                out_prompt = gr.Textbox(label="Expanded prompt", lines=4, interactive=False)

        btn.click(run, inputs=[face, instr, proc, seed], outputs=[out_img, out_prompt])

    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoints",
        default="checkpoints",
        help="dir containing per-procedure LoRA checkpoint subdirs",
    )
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="public Gradio share link")
    args = p.parse_args()

    demo = build_ui(Path(args.checkpoints))
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
