"""Full inference pipeline: prompt expander -> diffusion edit -> LoRA composition.

This is what the Gradio demo calls. It's the user-facing composition layer.

    user instruction
        ↓  Qwen3.5-9B prompt expander (4-bit, ~7 GB)
    detailed anatomical prompt
        ↓  Qwen-Image-Edit-2511 (NF4, ~17 GB)
        ↓  optional procedure LoRA + identity LoRA via set_adapters
    edited face image
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from aura_ml.inference.qwen_edit import QwenEditConfig, QwenImageEditPipeline
from aura_ml.prompt_expander.qwen35 import (
    ExpanderResult,
    OutOfScopeError,
    Qwen35PromptExpander,
    expand_template,
)

Procedure = Literal["rhinoplasty", "facelift", "blepharoplasty"]

PROCEDURES: tuple[str, ...] = ("rhinoplasty", "facelift", "blepharoplasty")

# Procedure LoRA at 0.7 is a starting point; tune downward if outputs
# overshoot, upward if the LoRA isn't contributing.
_DEFAULT_LORA_SCALE = 0.7
_DEFAULT_IDENTITY_SCALE = 0.5


@dataclass
class AuraConfig:
    qwen_edit: QwenEditConfig = field(default_factory=QwenEditConfig)
    # procedure name → path to the best LoRA checkpoint for that procedure.
    procedure_lora_paths: dict[str, str] = field(default_factory=dict)
    procedure_lora_scales: dict[str, float] = field(default_factory=dict)
    identity_lora_path: str | None = None  # optional per-subject LoRA
    identity_lora_scale: float = _DEFAULT_IDENTITY_SCALE
    # False = skip Qwen3.5-9B entirely and use the rule-based template
    # (saves ~7 GB VRAM; useful when co-locating with a training run).
    use_prompt_expander: bool = True


class AuraInferencePipeline:
    """End-to-end pipeline. Construct once per process; reuse across requests."""

    def __init__(self, config: AuraConfig | None = None) -> None:
        self.config = config or AuraConfig()
        self.diffuser = QwenImageEditPipeline(self.config.qwen_edit)
        self.expander: Qwen35PromptExpander | None = (
            Qwen35PromptExpander() if self.config.use_prompt_expander else None
        )
        self._loras_registered = False
        # Serializes diffusion runs when several frontends (API worker,
        # Gradio UI) share this pipeline — two concurrent 20B denoise loops
        # would OOM the card.
        self.gpu_lock = threading.Lock()

    # ------------------------------------------------------------------

    def _register_loras(self) -> None:
        """Load configured LoRAs once, after the diffuser itself is loaded.
        Missing checkpoint paths are skipped (not yet trained — not an error)."""
        if self._loras_registered:
            return
        for name, path in self.config.procedure_lora_paths.items():
            if Path(path).exists():
                self.diffuser.load_lora(
                    path,
                    name=name,
                    scale=self.config.procedure_lora_scales.get(name, _DEFAULT_LORA_SCALE),
                )
        if self.config.identity_lora_path and Path(self.config.identity_lora_path).exists():
            self.diffuser.load_lora(
                self.config.identity_lora_path,
                name="identity",
                scale=self.config.identity_lora_scale,
            )
        self._loras_registered = True

    def expand_prompt(
        self,
        face_image: Image.Image,
        user_instruction: str,
        procedure: Procedure,
        seed: int | None = None,
    ) -> ExpanderResult:
        """Prompt expansion only (no diffusion). Raises OutOfScopeError when
        the instruction falls outside the selected procedure."""
        if self.expander is not None:
            result = self.expander.expand_detailed(
                face_image, user_instruction, procedure, seed=seed
            )
            if result.out_of_scope:
                raise OutOfScopeError(result.reason)
            return result
        return ExpanderResult(
            prompt=expand_template(user_instruction, procedure),
            procedure=procedure,
            used_fallback=True,
            reason="expander disabled by config",
        )

    def generate(
        self,
        face_image: Image.Image,
        user_instruction: str,
        procedure: Procedure,
        num_steps: int | None = None,
        seed: int | None = None,
        expand: bool = True,
    ) -> tuple[Image.Image, ExpanderResult]:
        """Run the full pipeline. Returns (edited_image, expander_result).

        The demo displays expander_result.prompt so users can see exactly what
        was sent to the diffusion model.
        """
        # --- Step 1: prompt expansion ------------------------------------
        if expand:
            expansion = self.expand_prompt(face_image, user_instruction, procedure, seed=seed)
        else:
            expansion = ExpanderResult(
                prompt=user_instruction, procedure=procedure, used_fallback=True,
                reason="expansion disabled for this call",
            )

        # --- Step 2: activate LoRAs --------------------------------------
        self.diffuser.load()
        self._register_loras()

        names: list[str] = []
        weights: list[float] = []
        if procedure in self.diffuser._loaded_loras:
            names.append(procedure)
            weights.append(self.config.procedure_lora_scales.get(procedure, _DEFAULT_LORA_SCALE))
        if "identity" in self.diffuser._loaded_loras:
            names.append("identity")
            weights.append(self.config.identity_lora_scale)

        if names:
            self.diffuser.set_active_loras(names, weights)
        else:
            self.diffuser.disable_loras()

        # --- Step 3: diffusion edit --------------------------------------
        edited = self.diffuser.generate(
            image=face_image,
            prompt=expansion.prompt,
            num_steps=num_steps,
            seed=seed,
        )
        return edited, expansion


def build_default_pipeline(
    checkpoints_dir: str | Path = "checkpoints",
    use_prompt_expander: bool = True,
) -> AuraInferencePipeline:
    """Construct an AuraInferencePipeline with sensible defaults for local dev.

    `checkpoints_dir` is scanned for per-procedure subdirectories
    (checkpoints/rhinoplasty/, ...). Missing directories are skipped, so the
    pipeline runs in zero-shot mode before any LoRAs are trained.
    """
    ckpt = Path(checkpoints_dir)
    lora_paths = {p: str(ckpt / p) for p in PROCEDURES if (ckpt / p).exists()}

    return AuraInferencePipeline(
        AuraConfig(
            qwen_edit=QwenEditConfig(),
            procedure_lora_paths=lora_paths,
            use_prompt_expander=use_prompt_expander,
        )
    )
