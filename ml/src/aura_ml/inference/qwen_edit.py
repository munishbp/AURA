"""Qwen-Image-Edit pipeline wrapper.

Wraps the diffusers `QwenImageEditPipeline` with:
- NF4 quantization of the base model (fits on a 32 GB 5090)
- Optional FP8 inference path via torchao (post-baseline)
- LoRA load + multi-adapter composition (procedure LoRA + identity LoRA)
- A static-image canary on the first call (warns if output ≈ input — see
  pre-mortem in the repo plan)

Workstream 1 implements `generate()` against Qwen/Qwen-Image-Edit-2509.
Workstream 7 adds LoRA composition via `pipe.set_adapters()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

# Default model IDs. 2509 is the LoRA-training-friendly variant; 2511 bakes
# popular community LoRAs into the base and is preferred for inference once
# we have our own LoRAs trained.
DEFAULT_TRAIN_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_INFER_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"


@dataclass
class LoadedLora:
    name: str
    path: str
    scale: float = 1.0


@dataclass
class QwenEditConfig:
    model_id: str = DEFAULT_TRAIN_MODEL_ID
    dtype: str = "bfloat16"  # bf16 stays consistent with training (pre-mortem #5)
    device: str = "cuda"
    quantize_4bit: bool = True
    enable_cpu_offload: bool = False  # unnecessary on a 5090 at NF4
    enable_torch_compile: bool = False  # opt-in; warmup cost
    loras: list[LoadedLora] = field(default_factory=list)


class QwenImageEditPipeline:
    """Thin wrapper around diffusers' QwenImageEditPipeline.

    Usage:
        pipe = QwenImageEditPipeline(QwenEditConfig())
        pipe.load_lora("checkpoints/rhino-best", name="rhino", scale=0.7)
        pipe.set_active_loras(["rhino"], [0.7])
        out = pipe.generate(face_img, "narrowed nasal tip ...", num_steps=30)
    """

    def __init__(self, config: QwenEditConfig | None = None) -> None:
        self.config = config or QwenEditConfig()
        self._pipe = None  # diffusers pipeline, lazy-loaded
        self._loaded_loras: dict[str, LoadedLora] = {}

    # --- model lifecycle -------------------------------------------------

    def load(self) -> None:
        """Load the base diffusers pipeline. Lazy — call `generate()` to trigger."""
        # TODO(workstream 1):
        #   - Build BitsAndBytesConfig (NF4) when self.config.quantize_4bit
        #   - from diffusers import QwenImageEditPipeline as _Pipe
        #   - self._pipe = _Pipe.from_pretrained(self.config.model_id, ...)
        #   - self._pipe.to(self.config.device)
        #   - if self.config.enable_torch_compile: self._pipe.transformer = torch.compile(...)
        raise NotImplementedError("workstream 1: implement model loading")

    # --- LoRA management -------------------------------------------------

    def load_lora(self, path: str | Path, name: str, scale: float = 1.0) -> None:
        """Register a LoRA adapter under `name`. Doesn't activate it yet."""
        # TODO(workstream 7):
        #   - self._pipe.load_lora_weights(str(path), adapter_name=name)
        #   - self._loaded_loras[name] = LoadedLora(name, str(path), scale)
        #   - validate target_modules match training-time sidecar (pre-mortem #4)
        raise NotImplementedError("workstream 7: implement LoRA loading")

    def set_active_loras(self, names: list[str], weights: list[float]) -> None:
        """Activate a subset of loaded LoRAs at given weights."""
        # TODO(workstream 7): self._pipe.set_adapters(names, adapter_weights=weights)
        raise NotImplementedError("workstream 7: implement adapter switching")

    # --- generation ------------------------------------------------------

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_steps: int = 30,
        guidance_scale: float = 4.0,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Run the editing pipeline. Returns a PIL.Image."""
        if self._pipe is None:
            self.load()
        # TODO(workstream 1): generator = torch.Generator(device).manual_seed(seed)
        # TODO(workstream 1): out = self._pipe(image=image, prompt=prompt, ...).images[0]
        # TODO(workstream 2): wire static-image canary — DINO cosine vs input
        raise NotImplementedError("workstream 1: implement generate")


def _smoke_test() -> None:
    """Local smoke test — load the pipeline and run one inference."""
    pipe = QwenImageEditPipeline()
    img = Image.new("RGB", (768, 768), color=(128, 128, 128))
    out = pipe.generate(img, "make the subject smile", num_steps=8)
    out.save("smoke_test_output.png")
    print("OK: wrote smoke_test_output.png")


if __name__ == "__main__":
    _smoke_test()
