"""Full inference pipeline: prompt expander -> diffusion edit -> identity LoRA.

This is what the Gradio demo calls. It's the user-facing composition layer.

Workstream 1: just the diffusion step (no expander, no identity).
Workstream 3: add the prompt expander.
Workstream 7: add identity-preservation LoRA composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from aura_ml.inference.qwen_edit import QwenEditConfig, QwenImageEditPipeline
from aura_ml.prompt_expander.qwen35 import Qwen35PromptExpander

Procedure = Literal["rhinoplasty", "facelift", "blepharoplasty"]


@dataclass
class AuraConfig:
    qwen_edit: QwenEditConfig
    procedure_lora_paths: dict[str, str]  # e.g. {"rhinoplasty": "checkpoints/rhino-best"}
    procedure_lora_scales: dict[str, float]
    identity_lora_path: str | None = None  # optional per-subject LoRA
    identity_lora_scale: float = 0.7
    use_prompt_expander: bool = True


class AuraInferencePipeline:
    """End-to-end pipeline. Construct once per process; reuse across requests."""

    def __init__(self, config: AuraConfig) -> None:
        self.config = config
        self.diffuser = QwenImageEditPipeline(config.qwen_edit)
        self.expander: Qwen35PromptExpander | None = None
        if config.use_prompt_expander:
            # Note: in production, run the expander as a separate process.
            # For the demo we co-locate; both fit in 32 GB at 4-bit.
            self.expander = Qwen35PromptExpander()

        # TODO(workstream 7): preload all procedure LoRAs at startup.
        # for name, path in config.procedure_lora_paths.items():
        #     self.diffuser.load_lora(path, name=name, scale=config.procedure_lora_scales[name])
        # if config.identity_lora_path:
        #     self.diffuser.load_lora(config.identity_lora_path, name="identity",
        #                              scale=config.identity_lora_scale)

    def generate(
        self,
        face_image: Image.Image,
        user_instruction: str,
        procedure: Procedure,
        num_steps: int = 30,
        seed: int | None = None,
    ) -> tuple[Image.Image, str]:
        """Returns (edited_image, expanded_prompt_used)."""
        # 1. Prompt expansion
        if self.expander is not None:
            prompt = self.expander.expand(face_image, user_instruction, procedure)
        else:
            prompt = user_instruction

        # 2. Activate procedure LoRA (+ identity LoRA if configured)
        # TODO(workstream 7):
        #   names = [procedure]
        #   weights = [self.config.procedure_lora_scales[procedure]]
        #   if self.config.identity_lora_path:
        #       names.append("identity")
        #       weights.append(self.config.identity_lora_scale)
        #   self.diffuser.set_active_loras(names, weights)

        # 3. Run diffusion edit
        edited = self.diffuser.generate(
            image=face_image, prompt=prompt, num_steps=num_steps, seed=seed
        )
        return edited, prompt


def build_default_pipeline(checkpoints_dir: str | Path) -> AuraInferencePipeline:
    """Construct an AuraInferencePipeline with sensible defaults for local dev."""
    ckpt = Path(checkpoints_dir)
    cfg = AuraConfig(
        qwen_edit=QwenEditConfig(),
        procedure_lora_paths={
            "rhinoplasty": str(ckpt / "rhinoplasty"),
            "facelift": str(ckpt / "facelift"),
            "blepharoplasty": str(ckpt / "blepharoplasty"),
        },
        procedure_lora_scales={
            "rhinoplasty": 0.7,
            "facelift": 0.7,
            "blepharoplasty": 0.7,
        },
        identity_lora_path=None,
        use_prompt_expander=True,
    )
    return AuraInferencePipeline(cfg)
