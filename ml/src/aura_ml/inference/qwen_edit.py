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

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

# Default model IDs. 2509 is the LoRA-training-friendly variant; 2511 bakes
# popular community LoRAs into the base and is preferred for inference once
# we have our own LoRAs trained.
DEFAULT_TRAIN_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_INFER_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

# Static-image canary threshold — edit_magnitude below this is flagged.
_STATIC_THRESHOLD = 0.05


@dataclass
class LoadedLora:
    name: str
    path: str
    scale: float = 1.0


@dataclass
class QwenEditConfig:
    model_id: str = DEFAULT_TRAIN_MODEL_ID
    dtype: str = "bfloat16"          # bf16 throughout — matches training (pre-mortem #5)
    device: str = "cuda"
    quantize_4bit: bool = True        # NF4 via bitsandbytes — ~10-12 GB on 5090
    enable_cpu_offload: bool = False  # unnecessary on a 5090 at NF4
    enable_torch_compile: bool = False  # opt-in; warmup cost ~2-3 min
    loras: list[LoadedLora] = field(default_factory=list)


class QwenImageEditPipeline:
    """Thin wrapper around diffusers' QwenImageEditPipeline.

    Usage:
        pipe = QwenImageEditPipeline(QwenEditConfig())
        out = pipe.generate(face_img, "narrowed nasal tip ...", num_steps=30)

        # With LoRAs (Workstream 7):
        pipe.load_lora("checkpoints/rhino-best", name="rhino", scale=0.7)
        pipe.set_active_loras(["rhino"], [0.7])
        out = pipe.generate(face_img, "narrowed nasal tip ...", num_steps=30)
    """

    def __init__(self, config: QwenEditConfig | None = None) -> None:
        self.config = config or QwenEditConfig()
        self._pipe = None                         # diffusers pipeline, lazy-loaded
        self._loaded_loras: dict[str, LoadedLora] = {}

    # --- model lifecycle -------------------------------------------------

    def load(self) -> None:
        """Load the base diffusers pipeline.

        Called automatically on the first `generate()` call, but you can call
        it eagerly at startup to pay the cost once.

        What happens here:
        1. If quantize_4bit is set, build a BitsAndBytesConfig requesting NF4
           quantisation. NF4 is the most storage-efficient 4-bit format; it
           stores each weight in 4 bits using a normal-distribution codebook.
           Combined with bf16 compute dtype, this gives ~10-12 GB model footprint
           on the 5090 — well within the 32 GB budget.
        2. Load the diffusers QwenImageEditPipeline from HuggingFace. The
           `quantization_config` kwarg is passed through to `from_pretrained`
           which applies it automatically to the transformer weights.
        3. Optionally enable sequential CPU offload (moves layers to CPU when
           not in use — useful on smaller GPUs but not needed on the 5090).
        4. Optionally torch.compile the transformer for faster inference.
           Adds ~2-3 min warmup on first call; subsequent calls are faster.
        """
        from diffusers import QwenImageEditPipeline as _DiffusersPipe
        from diffusers import PipelineQuantizationConfig

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        # --- NF4 quantisation config -------------------------------------
        # This uses the diffusers PipelineQuantizationConfig wrapper rather
        # than the older transformers BitsAndBytesConfig object.
        quant_config = None
        if self.config.quantize_4bit:
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                },
            )

        print(f"[qwen_edit] Loading {self.config.model_id} "
              f"({'NF4' if self.config.quantize_4bit else torch_dtype}) ...")

        # `device_map="auto"` lets accelerate shard across GPU + CPU if needed.
        # On a single 5090 it will keep everything on GPU.
        device_map = None
        if self.config.quantize_4bit:
            if self.config.device == "cuda":
                device_map = "cuda"
            elif self.config.device == "cpu":
                device_map = "cpu"
            else:
                device_map = "cuda"

        self._pipe = _DiffusersPipe.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            device_map=device_map,
        )

        # If not quantised (or device_map wasn't used), move to the target device.
        if not self.config.quantize_4bit or device_map is None:
            self._pipe = self._pipe.to(self.config.device)

        # Sequential CPU offload: moves layers to CPU when idle, back to GPU
        # when needed. Saves VRAM at the cost of slower throughput. Skip on 5090.
        if self.config.enable_cpu_offload:
            self._pipe.enable_sequential_cpu_offload()

        # torch.compile wraps the transformer's forward pass with TorchInductor.
        # Gives ~20-40% speedup after the JIT warmup. Only worthwhile for many
        # sequential inferences (e.g. batch eval), not one-off demo calls.
        if self.config.enable_torch_compile:
            print("[qwen_edit] torch.compile enabled — first call will be slow (~2-3 min)")
            self._pipe.transformer = torch.compile(
                self._pipe.transformer,
                mode="reduce-overhead",  # good balance of compile time vs speedup
                fullgraph=False,         # safer; fullgraph=True can fail on dynamic shapes
            )

        print("[qwen_edit] Model loaded.")

    # --- LoRA management -------------------------------------------------

    def load_lora(self, path: str | Path, name: str, scale: float = 1.0) -> None:
        """Register a LoRA adapter under `name`. Doesn't activate it yet.

        This stores the adapter in the pipeline's internal adapter registry via
        `load_lora_weights(..., adapter_name=name)`. The adapter stays dormant
        until you call `set_active_loras()`.

        Also validates that the LoRA's `training_meta.json` sidecar (written
        by train.py at save time) has target_modules consistent with the
        currently-loaded base model — mitigation for pre-mortem issue #4.

        Workstream 7 fills this in fully.
        """
        # TODO(workstream 7):
        #   path = Path(path)
        #   sidecar = path / "training_meta.json"
        #   if sidecar.exists():
        #       import json
        #       meta = json.loads(sidecar.read_text())
        #       # Assert target_modules match what we expect
        #       expected = {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"}
        #       if set(meta.get("target_modules", [])) != expected:
        #           raise ValueError(
        #               f"LoRA target_modules mismatch: {meta.get('target_modules')} "
        #               f"vs expected {expected}"
        #           )
        #   self._pipe.load_lora_weights(str(path), adapter_name=name)
        #   self._loaded_loras[name] = LoadedLora(name=name, path=str(path), scale=scale)
        raise NotImplementedError("workstream 7: implement LoRA loading")

    def set_active_loras(self, names: list[str], weights: list[float]) -> None:
        """Activate a subset of loaded LoRAs at given weights.

        This calls `pipe.set_adapters(names, adapter_weights=weights)` which
        is the diffusers multi-adapter composition API. You can compose a
        procedure LoRA + an identity LoRA at different scales, e.g.:
            set_active_loras(["rhino", "identity"], [0.7, 0.5])

        Workstream 7 fills this in fully.
        """
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
        """Run the editing pipeline. Returns a PIL.Image.

        Steps:
        1. Lazy-load the model if not already loaded.
        2. Build a seeded `torch.Generator` for reproducibility. When seed is
           None, generation is non-deterministic (different every call).
        3. Call the diffusers pipeline. The QwenImageEditPipeline expects:
           - `image`: the source PIL image
           - `prompt`: the instruction text
           - `num_inference_steps`: number of denoising steps
           - `guidance_scale`: classifier-free guidance scale. 4.0 is a
             reasonable default for instruction-following edits; lower = more
             creative / less instruction-bound.
           - `generator`: the seeded Generator for reproducibility
        4. Run the static-image canary: compute DINO edit_magnitude between
           source and output. Warn loudly if < 0.05 — this is the failure mode
           that killed the hackathon LoRAs. This is a WARNING not an error;
           the caller (eval harness) decides what to do with it.
        """
        if self._pipe is None:
            self.load()

        # Build a seeded generator. Keep it on CPU — diffusers will move it
        # to the right device internally. None seed = random each call.
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Run inference.
        # The diffusers QwenImageEditPipeline call signature:
        #   pipeline(
        #       image=...,           # PIL.Image source
        #       prompt=...,          # instruction string
        #       num_inference_steps=...,
        #       guidance_scale=...,
        #       negative_prompt=..., # optional
        #       generator=...,
        #   ).images[0]
        result = self._pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
        )
        output: Image.Image = result.images[0]

        # --- Static-image canary -----------------------------------------
        # Compute edit_magnitude = 1 - cosine(DINO(source), DINO(output)).
        # We import lazily here to avoid loading DINO until it's actually needed.
        # The canary fires a warning rather than raising so that:
        #   (a) the eval harness can aggregate across the holdout and decide
        #       whether to quarantine the checkpoint
        #   (b) single inference calls in the demo don't crash on a borderline
        #       output
        try:
            from aura_ml.eval.metrics import edit_magnitude as _edit_magnitude
            em = _edit_magnitude(image, output)
            if em < _STATIC_THRESHOLD:
                warnings.warn(
                    f"[static-image canary] edit_magnitude={em:.4f} < "
                    f"{_STATIC_THRESHOLD} — output may be a copy of the input. "
                    "Check your LoRA weights, target_modules, and dataset divergence.",
                    stacklevel=2,
                )
        except Exception as canary_err:
            # Don't let a canary failure crash inference — just warn.
            warnings.warn(
                f"[static-image canary] could not compute edit_magnitude: {canary_err}",
                stacklevel=2,
            )

        return output


def _smoke_test() -> None:
    """Local smoke test — load the pipeline and run one inference.

    Run from the ml/ directory:
        uv run python -m aura_ml.inference.qwen_edit

    Writes smoke_test_output.png to the current directory. Intended to verify:
    - Model loads without OOM
    - generate() returns a PIL image
    - Output file is readable / not corrupted

    Note: uses a grey placeholder image, not a real face photo.
    Expected edit_magnitude on a grey image: will likely trigger the canary
    (grey → grey), which is fine for a smoke test.
    """
    pipe = QwenImageEditPipeline()
    img = Image.new("RGB", (768, 768), color=(128, 128, 128))
    out = pipe.generate(img, "make the subject smile", num_steps=8)
    out.save("smoke_test_output.png")
    print("OK: wrote smoke_test_output.png")


if __name__ == "__main__":
    _smoke_test()