"""Full inference pipeline: prompt expander -> diffusion edit -> identity LoRA.

This is what the Gradio demo calls. It's the user-facing composition layer.

Workstream 1: just the diffusion step (no expander, no identity).
Workstream 3: add the prompt expander.
Workstream 7: add identity-preservation LoRA composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from aura_ml.inference.qwen_edit import QwenEditConfig, QwenImageEditPipeline

Procedure = Literal["rhinoplasty", "facelift", "blepharoplasty"]

# Default LoRA scales — procedure LoRA at 0.7 is a starting point; tune
# downward if outputs overshoot, upward if the LoRA isn't contributing.
_DEFAULT_LORA_SCALE = 0.7
_DEFAULT_IDENTITY_SCALE = 0.5


@dataclass
class AuraConfig:
    qwen_edit: QwenEditConfig
    # Maps procedure name → path to the best LoRA checkpoint for that procedure.
    # Leave empty until Workstream 6 produces trained checkpoints.
    procedure_lora_paths: dict[str, str] = field(default_factory=dict)
    procedure_lora_scales: dict[str, float] = field(default_factory=dict)
    identity_lora_path: str | None = None   # optional per-subject LoRA (Workstream 7)
    identity_lora_scale: float = _DEFAULT_IDENTITY_SCALE
    use_prompt_expander: bool = True         # disable to skip Qwen3.5-9B entirely


class AuraInferencePipeline:
    """End-to-end pipeline. Construct once per process; reuse across requests.

    Composition order:
        user instruction
            ↓  [Workstream 3: Qwen3.5-9B prompt expander]
        detailed anatomical prompt
            ↓  [Workstream 1: Qwen-Image-Edit-2509]
            ↓  [Workstream 7: procedure LoRA + identity LoRA via set_adapters]
        edited face image

    For now (Workstream 1), the prompt expander is skipped if not loaded,
    and no LoRAs are composed — this gives the zero-shot baseline.
    """

    def __init__(self, config: AuraConfig) -> None:
        self.config = config

        # Always build the diffusion pipeline — this is the core of W1.
        self.diffuser = QwenImageEditPipeline(config.qwen_edit)

        # Prompt expander: lazy-import so that if Qwen3.5-9B isn't downloaded
        # yet (pre-W3), the pipeline still works — it just passes the raw
        # user instruction to the diffusion model.
        self.expander = None
        if config.use_prompt_expander:
            try:
                from aura_ml.prompt_expander.qwen35 import Qwen35PromptExpander
                self.expander = Qwen35PromptExpander()
                # NOTE: expander.load() is lazy — called on first expand() call.
            except NotImplementedError:
                # W3 not yet implemented — fall through to raw instruction.
                print(
                    "[pipeline] Prompt expander not yet implemented (Workstream 3). "
                    "Using raw user instructions."
                )
            except Exception as e:
                print(f"[pipeline] Could not load prompt expander: {e}. "
                      "Using raw user instructions.")

        # Workstream 7: preload procedure LoRAs at startup so each generate()
        # call doesn't pay the load cost. Currently a no-op until W7 is done.
        self._loras_loaded = False
        self._try_load_loras()

    def _try_load_loras(self) -> None:
        """Attempt to load all configured procedure + identity LoRAs.

        Silently skips if:
        - LoRA loading isn't implemented yet (NotImplementedError → W7)
        - A checkpoint path doesn't exist on disk yet (W6 not done)
        """
        if not self.config.procedure_lora_paths:
            return  # No LoRAs configured — zero-shot baseline mode.

        for name, path in self.config.procedure_lora_paths.items():
            if not Path(path).exists():
                # Checkpoint hasn't been trained yet — not an error.
                continue
            scale = self.config.procedure_lora_scales.get(name, _DEFAULT_LORA_SCALE)
            try:
                self.diffuser.load_lora(path, name=name, scale=scale)
            except NotImplementedError:
                # W7 not done yet.
                break
            except Exception as e:
                print(f"[pipeline] Warning: could not load LoRA '{name}': {e}")

        if self.config.identity_lora_path:
            id_path = Path(self.config.identity_lora_path)
            if id_path.exists():
                try:
                    self.diffuser.load_lora(
                        str(id_path),
                        name="identity",
                        scale=self.config.identity_lora_scale,
                    )
                except NotImplementedError:
                    pass  # W7 not done
                except Exception as e:
                    print(f"[pipeline] Warning: could not load identity LoRA: {e}")

    def generate(
        self,
        face_image: Image.Image,
        user_instruction: str,
        procedure: Procedure,
        num_steps: int = 120,
        seed: int | None = None,
    ) -> tuple[Image.Image, str]:
        """Run the full pipeline. Returns (edited_image, prompt_used).

        `prompt_used` is either the raw user instruction (W1, W3 not done)
        or the expander output (W3+). The Gradio demo displays it so users
        can see exactly what was sent to the diffusion model.

        Steps:
        1. Expand the prompt (if expander is loaded and implemented).
        2. Activate the procedure LoRA + optional identity LoRA (if W7 done).
        3. Run diffusion edit and return (image, prompt).
        """
        # --- Step 1: Prompt expansion ------------------------------------
        # W1 baseline: expander is None (use_prompt_expander=False) or not
        # yet implemented. Either way we fall through to the raw instruction.
        prompt = user_instruction
        if self.expander is not None:
            try:
                prompt = self.expander.expand(face_image, user_instruction, procedure)
            except NotImplementedError:
                # W3 not done yet — just use the raw instruction.
                pass
            except Exception as e:
                print(f"[pipeline] Expander error: {e}. Using raw instruction.")

        # --- Step 2: Activate LoRAs (Workstream 7) -----------------------
        # Until W7 is implemented this block is a no-op. Once W7 lands,
        # remove the try/except and let the errors surface.
        try:
            lora_names = []
            lora_weights = []

            if procedure in self.diffuser._loaded_loras:
                lora_names.append(procedure)
                lora_weights.append(
                    self.config.procedure_lora_scales.get(procedure, _DEFAULT_LORA_SCALE)
                )

            if "identity" in self.diffuser._loaded_loras:
                lora_names.append("identity")
                lora_weights.append(self.config.identity_lora_scale)

            if lora_names:
                self.diffuser.set_active_loras(lora_names, lora_weights)

        except NotImplementedError:
            pass  # W7 not done yet — run base model
        except Exception as e:
            print(f"[pipeline] LoRA activation error: {e}. Running base model.")

        # --- Step 3: Diffusion edit --------------------------------------
        edited = self.diffuser.generate(
            image=face_image,
            prompt=prompt,
            num_steps=num_steps,
            seed=seed,
        )

        return edited, prompt


def build_default_pipeline(
    checkpoints_dir: str | Path,
    use_prompt_expander: bool = True,
) -> AuraInferencePipeline:
    """Construct an AuraInferencePipeline with sensible defaults for local dev.

    Called by the Gradio demo (`app/demo.py`). The `checkpoints_dir` is
    scanned for per-procedure subdirectories. Missing directories are silently
    skipped — this lets the demo run in zero-shot mode before any LoRAs are
    trained (Workstream 1 goal).

    Args:
        checkpoints_dir: root directory containing per-procedure checkpoint
            subdirs, e.g.  checkpoints/rhinoplasty/, checkpoints/facelift/, ...
        use_prompt_expander: set False to skip Qwen3.5-9B and pass raw
            user instructions directly to the diffusion model. Useful when
            debugging W1 in isolation before W3 is done.
    """
    ckpt = Path(checkpoints_dir)

    procedures = ["rhinoplasty", "facelift", "blepharoplasty"]
    lora_paths = {}
    lora_scales = {}

    for proc in procedures:
        p = ckpt / proc
        if p.exists():
            lora_paths[proc] = str(p)
            lora_scales[proc] = _DEFAULT_LORA_SCALE

    cfg = AuraConfig(
        qwen_edit=QwenEditConfig(
            # Use the training-compatible variant for now. Switch to 2511
            # once our own LoRAs are trained (it bakes community LoRAs into
            # the base weights and is the recommended inference model).
            model_id="Qwen/Qwen-Image-Edit-2509",
            quantize_4bit=True,
            dtype="bfloat16",
            device="cuda",
            enable_torch_compile=False,  # flip True if doing batch eval
        ),
        procedure_lora_paths=lora_paths,
        procedure_lora_scales=lora_scales,
        identity_lora_path=None,        # set once W7 produces a checkpoint
        use_prompt_expander=use_prompt_expander,
    )

    return AuraInferencePipeline(cfg)