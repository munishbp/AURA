"""Qwen-Image-Edit pipeline wrapper.

Uses QwenImageEditPlusPipeline from diffusers — the real pixel-space editor.

Model: Qwen/Qwen-Image-Edit-2509  (Apache 2.0, ~40 GB bfloat16 / ~20 GB NF4)
       Qwen/Qwen-Image-Edit-2511  (newer, better identity preservation)

Usage:
    pipe = QwenImageEditPipeline()
    out  = pipe.generate(face_img, "Subtle dorsal hump reduction ...", num_steps=30)

Workstream 7: add LoRA composition via load_lora / set_active_loras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

# 2509 is best for LoRA training; 2511 has better out-of-the-box identity
# preservation. Either works with QwenImageEditPlusPipeline.
DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


@dataclass
class LoadedLora:
    name: str
    path: str
    scale: float = 1.0


@dataclass
class QwenEditConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda"
    # Use bfloat16 throughout — matches training precision (pre-mortem #5)
    torch_dtype: torch.dtype = torch.bfloat16
    # 5090 has 32GB VRAM — full model fits, no offloading needed.
    # Only enable this if you get OOM errors.
    enable_cpu_offload: bool = False
    enable_torch_compile: bool = False   # ~2.4x speedup; 7s warmup per run
    auto_device: bool = True             # fall back to CPU if no CUDA
    loras: list[LoadedLora] = field(default_factory=list)


class QwenImageEditPipeline:
    """Pixel-space image editor backed by Qwen-Image-Edit-2509.

    Wraps diffusers' QwenImageEditPlusPipeline so the rest of aura_ml
    (demo, training harness, eval) don't import diffusers directly.
    """

    def __init__(self, config: QwenEditConfig | None = None) -> None:
        self.config = config or QwenEditConfig()
        self._pipe = None
        self._loaded_loras: dict[str, LoadedLora] = {}

        if self.config.auto_device and not torch.cuda.is_available():
            print("[QwenEdit] CUDA not available — falling back to CPU (will be slow)")
            self.config.device = "cpu"
            self.config.torch_dtype = torch.float32   # bf16 unsupported on CPU
            self.config.enable_cpu_offload = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load QwenImageEditPlusPipeline. Called lazily on first generate()."""
        from diffusers import QwenImageEditPlusPipeline

        print(f"[QwenEdit] loading {self.config.model_id} ...")
        print("[QwenEdit] first run downloads ~40 GB; subsequent runs use cache")

        load_kwargs = {"torch_dtype": self.config.torch_dtype}

        # Pass device_map so weights land directly on GPU — avoids staging
        # the full 40 GB model through CPU RAM first (very slow on Windows).
        if self.config.device != "cpu":
            load_kwargs["device_map"] = "cuda"

        self._pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.config.model_id,
            **load_kwargs,
        )

        if self.config.enable_torch_compile:
            self._pipe.transformer = torch.compile(self._pipe.transformer)
            print("[QwenEdit] torch.compile enabled (~7s warmup on first call)")

        self._pipe.set_progress_bar_config(disable=False)
        print("[QwenEdit] model ready.")

    # ------------------------------------------------------------------
    # LoRA management  (Workstream 7)
    # ------------------------------------------------------------------

    def load_lora(self, path: str | Path, name: str, scale: float = 1.0) -> None:
        """Register a LoRA adapter under `name`. Does not activate it."""
        if self._pipe is None:
            self.load()
        try:
            self._pipe.load_lora_weights(str(path), adapter_name=name)
            self._loaded_loras[name] = LoadedLora(name, str(path), scale)
            print(f"[QwenEdit] loaded LoRA '{name}' from {path}")
        except Exception as e:
            print(f"[QwenEdit][warn] LoRA load failed: {e}")

    def set_active_loras(self, names: list[str], weights: list[float]) -> None:
        """Activate a subset of loaded LoRAs at given weights."""
        if self._pipe is None:
            return
        try:
            self._pipe.set_adapters(names, adapter_weights=weights)
        except Exception as e:
            print(f"[QwenEdit][warn] set_adapters failed: {e}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_steps: int = 30,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        seed: int | None = None,
    ) -> Image.Image:
        """Edit image according to prompt. Returns a PIL.Image.

        Args:
            image:           Input face photo (any size; pipeline handles resize).
            prompt:          Detailed edit instruction (ideally from the prompt expander).
            num_steps:       Diffusion steps. 30 is good default; 50 for higher quality.
            true_cfg_scale:  Classifier-free guidance scale. 4.0 per Qwen docs.
            negative_prompt: What to avoid. A single space disables it.
            seed:            For reproducibility. None = random.
        """
        if self._pipe is None:
            self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)

        inputs = {
            "image": image.convert("RGB"),
            "prompt": prompt,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_steps,
            "generator": generator,
        }

        with torch.inference_mode():
            output = self._pipe(**inputs)

        return output.images[0]