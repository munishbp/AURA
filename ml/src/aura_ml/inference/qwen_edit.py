"""Qwen-Image-Edit pipeline wrapper.

Uses QwenImageEditPlusPipeline from diffusers — the real pixel-space editor.

Model: Qwen/Qwen-Image-Edit-2511 (Apache 2.0). Newest of the open Qwen edit
line (Dec 2025): less image drift and much better character/identity
consistency than 2509 — identity preservation is the metric this project is
graded on, so 2511 is the default for both inference and LoRA training
(training and serving on the same base avoids the adapter/base mismatch from
the hackathon pre-mortem).

VRAM budget on a 32 GB RTX 5090 (the target box):
    transformer  NF4   ~11 GB      (20B MMDiT; 40 GB at bf16 — does NOT fit raw)
    text_encoder NF4    ~5 GB      (Qwen2.5-VL-7B)
    VAE          bf16   <1 GB
    activations         ~2-4 GB
    ------------------------------
    ~18-20 GB, leaving room for the 4-bit Qwen3.5-9B prompt expander (~7 GB).

Usage:
    pipe = QwenImageEditPipeline()
    out  = pipe.generate(face_img, "Subtle dorsal hump reduction ...")
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

# Inference recipe from the 2511 model card.
DEFAULT_NUM_STEPS = 40
DEFAULT_TRUE_CFG = 4.0
DEFAULT_NEGATIVE = " "


@dataclass
class LoadedLora:
    name: str
    path: str
    scale: float = 1.0


@dataclass
class QwenEditConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    # "nf4" quantizes transformer + text encoder with bitsandbytes (fits the
    # 5090); "none" loads bf16 (needs ~45 GB VRAM — H100 class only).
    quantize: Literal["nf4", "none"] = "nf4"
    # Model-level CPU offload trades speed for VRAM headroom. Not needed on a
    # 32 GB card with NF4; enable if co-locating with a training job.
    enable_cpu_offload: bool = False
    enable_torch_compile: bool = False
    auto_device: bool = True  # fall back to CPU if no CUDA (slow; debug only)
    loras: list[LoadedLora] = field(default_factory=list)


class QwenImageEditPipeline:
    """Pixel-space image editor backed by Qwen-Image-Edit-2511.

    Wraps diffusers' QwenImageEditPlusPipeline so the rest of aura_ml
    (demo, training harness, eval) don't import diffusers directly.
    """

    def __init__(self, config: QwenEditConfig | None = None) -> None:
        self.config = config or QwenEditConfig()
        self._pipe = None
        self._loaded_loras: dict[str, LoadedLora] = {}
        self._active_loras: list[str] = []

        if self.config.auto_device and not torch.cuda.is_available():
            print("[QwenEdit] CUDA not available — falling back to CPU (will be slow)")
            self.config.device = "cpu"
            self.config.torch_dtype = torch.float32  # bf16 unsupported on CPU
            self.config.quantize = "none"  # bitsandbytes needs CUDA
            self.config.enable_cpu_offload = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load QwenImageEditPlusPipeline. Called lazily on first generate()."""
        if self._pipe is not None:
            return
        from diffusers import QwenImageEditPlusPipeline

        print(f"[QwenEdit] loading {self.config.model_id} (quantize={self.config.quantize}) ...")

        load_kwargs: dict = {"torch_dtype": self.config.torch_dtype}

        if self.config.quantize == "nf4":
            # The transformer is a diffusers model, the text encoder is a
            # transformers model (Qwen2.5-VL-7B) — each needs its own library's
            # BitsAndBytesConfig in the quant mapping.
            from diffusers import BitsAndBytesConfig as DiffusersBnb
            from diffusers.quantizers import PipelineQuantizationConfig
            from transformers import BitsAndBytesConfig as TransformersBnb

            load_kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": DiffusersBnb(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=self.config.torch_dtype,
                    ),
                    "text_encoder": TransformersBnb(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=self.config.torch_dtype,
                    ),
                }
            )

        self._pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.config.model_id, **load_kwargs
        )

        if self.config.enable_cpu_offload:
            self._pipe.enable_model_cpu_offload()
        elif self.config.device != "cpu":
            self._pipe.to(self.config.device)

        if self.config.enable_torch_compile:
            self._pipe.transformer = torch.compile(self._pipe.transformer)
            print("[QwenEdit] torch.compile enabled (slow first call, faster after)")

        self._pipe.set_progress_bar_config(disable=False)
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 2**30
            print(f"[QwenEdit] model ready ({used:.1f} GiB allocated).")
        else:
            print("[QwenEdit] model ready (CPU).")

    def unload(self) -> None:
        """Free the pipeline and reclaim VRAM."""
        self._pipe = None
        self._loaded_loras.clear()
        self._active_loras = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def load_lora(self, path: str | Path, name: str, scale: float = 1.0) -> None:
        """Register a LoRA adapter under `name`. Does not activate it.

        If a `training_meta.json` sidecar exists next to the weights (written
        by aura_ml.training.train), assert the adapter was trained on the same
        base model — the silent base/adapter mismatch was pre-mortem cause #4.
        """
        if self._pipe is None:
            self.load()

        path = Path(path)
        sidecar = path / "training_meta.json" if path.is_dir() else path.parent / "training_meta.json"
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            trained_on = meta.get("base_model_id")
            if trained_on and trained_on != self.config.model_id:
                raise ValueError(
                    f"LoRA '{name}' was trained on {trained_on} but this pipeline "
                    f"is running {self.config.model_id}. Refusing to load silently."
                )

        self._pipe.load_lora_weights(str(path), adapter_name=name)
        self._loaded_loras[name] = LoadedLora(name, str(path), scale)
        print(f"[QwenEdit] loaded LoRA '{name}' from {path}")

    def set_active_loras(self, names: list[str], weights: list[float]) -> None:
        """Activate a subset of loaded LoRAs at given weights."""
        if self._pipe is None:
            raise RuntimeError("pipeline not loaded — call load() first")
        unknown = [n for n in names if n not in self._loaded_loras]
        if unknown:
            raise KeyError(f"LoRA(s) not loaded: {unknown}")
        self._pipe.set_adapters(names, adapter_weights=weights)
        self._active_loras = list(names)

    def disable_loras(self) -> None:
        """Run the base model with no adapter contribution."""
        if self._pipe is not None and self._loaded_loras:
            self._pipe.disable_lora()
        self._active_loras = []

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Image.Image | list[Image.Image],
        prompt: str,
        num_steps: int = DEFAULT_NUM_STEPS,
        true_cfg_scale: float = DEFAULT_TRUE_CFG,
        negative_prompt: str = DEFAULT_NEGATIVE,
        seed: int | None = None,
    ) -> Image.Image:
        """Edit image according to prompt. Returns a PIL.Image.

        Args:
            image:           Input face photo, or a list of reference images
                             (2511 supports multi-image conditioning).
            prompt:          Detailed edit instruction (ideally from the prompt expander).
            num_steps:       Diffusion steps. 40 per the 2511 model card.
            true_cfg_scale:  Classifier-free guidance scale. 4.0 per Qwen docs.
            negative_prompt: What to avoid. A single space disables it.
            seed:            For reproducibility. None = random.
        """
        if self._pipe is None:
            self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)

        images = image if isinstance(image, list) else [image]
        images = [im.convert("RGB") for im in images]

        # 2511 is not guidance-distilled: true CFG via negative prompt, no
        # guidance_scale (the pipeline would just warn and ignore it).
        inputs = {
            "image": images,
            "prompt": prompt,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_steps,
            "generator": generator,
        }

        with torch.inference_mode():
            output = self._pipe(**inputs)

        return output.images[0]
