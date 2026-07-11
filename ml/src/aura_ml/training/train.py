"""LoRA training loop for Qwen-Image-Edit-2511 on a 32 GB RTX 5090.

Design (QLoRA + flow matching, mirroring QwenImageEditPlusPipeline exactly):

  Phase 1 — cache. Load the VAE + text encoder (NF4), walk the dataset once,
  and write per-pair tensors to <dataset>/.cond_cache/: packed control/target
  latents, prompt embeds (instruction + control image through Qwen2.5-VL),
  negative embeds, and img_shapes. Then both encoders are freed — training
  never pays their VRAM.

  Phase 2 — train. Load only the transformer (NF4, ~11 GB), attach a LoRA
  (attention + MLP on both streams), and optimize with AdamW-8bit on the
  rectified-flow objective:  x_t = (1-σ)·x0 + σ·ε,  target v = ε − x0,
  conditioned by concatenating control latents along the sequence dim —
  exactly what the inference pipeline does at every denoise step.

  Eval — every `eval_every_epochs`, run the real K-step CFG denoise loop over
  the holdout's cached conditioning, decode with the VAE, and score with the
  eval harness. A checkpoint that trips the static-image canary is written to
  a .QUARANTINE file instead of being promoted to `best/`.

Pre-mortem mitigations baked in:
- #4 (target modules): the actual target_modules list is written to a sidecar
  JSON next to each checkpoint; the inference loader asserts base-model match.
- #5 (precision): bf16 compute end-to-end, same as inference.
- #1 (identity collapse): the toy task uses pairs with visible source/target
  divergence, so collapse fails loudly on the eval harness from epoch 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

# Sequence-packing geometry from the QwenImageEditPlus pipeline.
CONDITION_IMAGE_AREA = 384 * 384  # text-encoder conditioning image
NEGATIVE_PROMPT = " "


# --- Config ----------------------------------------------------------------


@dataclass
class TrainConfig:
    # Data
    dataset_root: str
    resolution: int = 768  # 512 if VRAM tight; 1024 if H100
    # Model
    base_model_id: str = "Qwen/Qwen-Image-Edit-2511"
    quantize_4bit: bool = True  # NF4 transformer (required on 32 GB)
    # LoRA — attention + MLP on both image and text streams
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: [
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "img_mlp.net.0.proj", "img_mlp.net.2",
            "txt_mlp.net.0.proj", "txt_mlp.net.2",
        ]
    )
    # Optimizer
    learning_rate: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"
    # Flow-matching timestep sampling: "logit_normal" (SD3/FLUX-style) or "uniform"
    timestep_sampling: str = "logit_normal"
    # Schedule
    epochs: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    # Eval / save
    output_dir: str = "outputs/run"
    save_every_epochs: int = 5
    eval_every_epochs: int = 5
    eval_holdout_dir: str = ""  # dir with {control,prompts}; "" disables eval
    eval_num_steps: int = 20
    eval_true_cfg_scale: float = 4.0
    eval_max_samples: int = 8
    # Repro
    seed: int = 0


def load_config(path: str | Path) -> TrainConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


# --- Conditioning cache (phase 1) -------------------------------------------


def _cache_key(cfg: TrainConfig) -> str:
    payload = f"{cfg.base_model_id}|{cfg.resolution}|v1"
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _resize_area(img: Image.Image, area: int, multiple: int = 32) -> Image.Image:
    """Resize preserving aspect ratio to ~area pixels, dims divisible by `multiple`
    (mirrors calculate_dimensions in the diffusers pipeline)."""
    w, h = img.size
    ratio = w / h
    width = round(math.sqrt(area * ratio) / multiple) * multiple
    height = round(math.sqrt(area / ratio) / multiple) * multiple
    return img.resize((max(width, multiple), max(height, multiple)), Image.LANCZOS)


class ConditioningCacher:
    """Loads VAE + text encoder once, encodes everything, then frees them."""

    def __init__(self, cfg: TrainConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self._vae = None
        self._text_encoder = None
        self._processor = None
        self._tokenizer = None
        # Prompt template constants from QwenImageEditPlusPipeline.
        self.prompt_template = (
            "<|im_start|>system\nDescribe the key features of the input image "
            "(color, shape, size, texture, objects, background), then explain "
            "how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while "
            "maintaining consistency with the original input where appropriate."
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.template_drop_idx = 64

    def load_encoders(self) -> None:
        from diffusers import AutoencoderKLQwenImage
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        print("[cache] loading VAE + text encoder ...")
        self._vae = AutoencoderKLQwenImage.from_pretrained(
            self.cfg.base_model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        te_kwargs: dict = {"dtype": torch.bfloat16}
        if self.cfg.quantize_4bit:
            te_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            te_kwargs["device_map"] = self.device
        self._text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.base_model_id, subfolder="text_encoder", **te_kwargs
        ).eval()
        if not self.cfg.quantize_4bit:
            self._text_encoder.to(self.device)
        self._processor = AutoProcessor.from_pretrained(
            self.cfg.base_model_id, subfolder="processor"
        )

    def free_encoders(self) -> None:
        import gc

        self._vae = None
        self._text_encoder = None
        self._processor = None
        gc.collect()
        torch.cuda.empty_cache()

    # -- encoders ------------------------------------------------------

    @torch.no_grad()
    def encode_image_vae(self, img: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        """PIL -> packed latents [1, seq, C*4]; also returns latent (h/2, w/2) grid."""
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1  # [1,3,H,W]
        t = t.unsqueeze(2).to(self.device, torch.bfloat16)  # [1,3,1,H,W] video-style
        z = self._vae.encode(t).latent_dist.mode()  # argmax sampling
        mean = torch.tensor(self._vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
        std = torch.tensor(self._vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
        z = (z - mean) / std  # [1, C, 1, h, w]
        z = z[:, :, 0]  # [1, C, h, w]
        _, c, h, w = z.shape
        packed = z.view(1, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
        packed = packed.reshape(1, (h // 2) * (w // 2), c * 4)
        return packed.cpu(), (h // 2, w // 2)

    @torch.no_grad()
    def decode_packed_latents(self, packed: torch.Tensor, grid: tuple[int, int]) -> Image.Image:
        """Inverse of encode_image_vae: packed [1, seq, C*4] -> PIL."""
        gh, gw = grid
        c4 = packed.shape[-1]
        c = c4 // 4
        z = packed.view(1, gh, gw, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
        z = z.reshape(1, c, 1, gh * 2, gw * 2).to(self.device, torch.bfloat16)
        mean = torch.tensor(self._vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
        std = torch.tensor(self._vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
        z = z * std + mean
        img = self._vae.decode(z, return_dict=False)[0][:, :, 0]  # [1,3,H,W]
        img = (img.float() / 2 + 0.5).clamp(0, 1)
        arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(arr)

    @torch.no_grad()
    def encode_prompt(
        self, instruction: str, condition_image: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(instruction, control image) -> (prompt_embeds [1,L,D], mask [1,L]).
        Mirrors QwenImageEditPlusPipeline._get_qwen_prompt_embeds."""
        img = _resize_area(condition_image.convert("RGB"), CONDITION_IMAGE_AREA)
        txt = self.prompt_template.format(
            "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" + instruction
        )
        model_inputs = self._processor(
            text=[txt], images=[img], padding=True, return_tensors="pt"
        ).to(self.device)
        out = self._text_encoder(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs.get("pixel_values"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1][0]  # [L, D] (batch of 1)
        valid = model_inputs["attention_mask"][0].bool()
        hidden = hidden[valid][self.template_drop_idx:]
        embeds = hidden.unsqueeze(0).to(torch.bfloat16)
        mask = torch.ones(1, embeds.shape[1], dtype=torch.long, device=embeds.device)
        return embeds.cpu(), mask.cpu()

    # -- dataset walk ----------------------------------------------------

    def build_cache(self, dataset_root: Path, holdout_dir: Path | None) -> Path:
        """Encode all training pairs (+ holdout controls) to .cond_cache/."""
        from aura_ml.data.pair_loader import PairDataset

        cache_dir = dataset_root / ".cond_cache" / _cache_key(self.cfg)
        cache_dir.mkdir(parents=True, exist_ok=True)
        marker = cache_dir / "COMPLETE"
        holdout_cache = cache_dir / "holdout"

        need_train = not marker.exists()
        need_holdout = holdout_dir is not None and not (holdout_cache / "COMPLETE").exists()
        if not need_train and not need_holdout:
            print(f"[cache] reusing {cache_dir}")
            return cache_dir

        self.load_encoders()

        if need_train:
            ds = PairDataset(dataset_root, resolution=self.cfg.resolution)
            for i in tqdm(range(len(ds)), desc="caching train pairs"):
                sample = ds.load_pil(i)
                res = self.cfg.resolution
                ctrl = _center_crop_resize(sample.control, res)
                tgt = _center_crop_resize(sample.target, res)
                ctrl_lat, ctrl_grid = self.encode_image_vae(ctrl)
                tgt_lat, tgt_grid = self.encode_image_vae(tgt)
                embeds, mask = self.encode_prompt(sample.instruction, sample.control)
                neg_embeds, neg_mask = self.encode_prompt(NEGATIVE_PROMPT, sample.control)
                torch.save(
                    {
                        "id": sample.pair_id,
                        "control_latents": ctrl_lat,
                        "control_grid": ctrl_grid,
                        "target_latents": tgt_lat,
                        "target_grid": tgt_grid,
                        "prompt_embeds": embeds,
                        "prompt_mask": mask,
                        "neg_embeds": neg_embeds,
                        "neg_mask": neg_mask,
                        "instruction": sample.instruction,
                    },
                    cache_dir / f"{sample.pair_id}.pt",
                )
            marker.write_text("ok")

        if need_holdout:
            holdout_cache.mkdir(parents=True, exist_ok=True)
            entries = _load_holdout_entries(holdout_dir)[: self.cfg.eval_max_samples]
            for stem, ctrl_img, instruction in tqdm(entries, desc="caching holdout"):
                res = self.cfg.resolution
                ctrl = _center_crop_resize(ctrl_img, res)
                ctrl_lat, ctrl_grid = self.encode_image_vae(ctrl)
                embeds, mask = self.encode_prompt(instruction, ctrl_img)
                neg_embeds, neg_mask = self.encode_prompt(NEGATIVE_PROMPT, ctrl_img)
                torch.save(
                    {
                        "id": stem,
                        "control_latents": ctrl_lat,
                        "control_grid": ctrl_grid,
                        "prompt_embeds": embeds,
                        "prompt_mask": mask,
                        "neg_embeds": neg_embeds,
                        "neg_mask": neg_mask,
                        "instruction": instruction,
                        "control_size": ctrl.size,
                    },
                    holdout_cache / f"{stem}.pt",
                )
            (holdout_cache / "COMPLETE").write_text("ok")

        # Keep the VAE for eval-time decode (it's ~150 MB); free the 7B encoder.
        import gc

        self._text_encoder = None
        self._processor = None
        gc.collect()
        torch.cuda.empty_cache()
        return cache_dir


def _center_crop_resize(img: Image.Image, resolution: int) -> Image.Image:
    from aura_ml.data.pair_loader import _resize_center_crop

    return _resize_center_crop(img, resolution)


def _load_holdout_entries(holdout: Path) -> list[tuple[str, Image.Image, str]]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    entries = []
    control_dir = holdout / "control"
    prompts_dir = holdout / "prompts"
    for p in sorted(control_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        prompt_path = prompts_dir / f"{p.stem}.txt"
        if not prompt_path.exists():
            continue
        entries.append(
            (p.stem, Image.open(p).convert("RGB"), prompt_path.read_text().strip())
        )
    return entries


# --- Model setup (phase 2) ---------------------------------------------------


def load_transformer(cfg: TrainConfig, device: str = "cuda"):
    from diffusers import QwenImageTransformer2DModel

    print(f"[train] loading transformer from {cfg.base_model_id} "
          f"(4bit={cfg.quantize_4bit}) ...")
    kwargs: dict = {"torch_dtype": torch.bfloat16}
    if cfg.quantize_4bit:
        from diffusers import BitsAndBytesConfig

        # Same selective-NF4 recipe as inference (qwen_edit.py): first/last
        # blocks + in/out projections stay bf16 so quantization grain doesn't
        # compound over denoise steps — and the LoRA trains against the same
        # base it will serve on.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[
                "transformer_blocks.0.",
                "transformer_blocks.59.",
                "img_in",
                "txt_in",
                "proj_out",
            ],
        )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        cfg.base_model_id, subfolder="transformer", **kwargs
    )
    if not cfg.quantize_4bit:
        transformer.to(device)
    transformer.requires_grad_(False)
    return transformer


def attach_lora(transformer, cfg: TrainConfig) -> list[torch.nn.Parameter]:
    """Wrap the transformer with a LoRA adapter. Returns trainable params."""
    from peft import LoraConfig

    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_cfg)
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    # LoRA params train in fp32 for optimizer stability (QLoRA convention).
    for p in trainable:
        p.data = p.data.to(torch.float32)
    n = sum(p.numel() for p in trainable)
    print(f"[train] LoRA attached: {n/1e6:.1f}M trainable params "
          f"(r={cfg.rank}, α={cfg.alpha}, {len(cfg.target_modules)} module patterns)")
    return trainable


def write_target_modules_sidecar(out_dir: Path, cfg: TrainConfig) -> None:
    """Write target_modules + key hyperparams next to the checkpoint so the
    inference loader can assert base-model equality (pre-mortem #4)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "base_model_id": cfg.base_model_id,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "dropout": cfg.dropout,
        "target_modules": cfg.target_modules,
        "training_dtype": cfg.mixed_precision,
        "training_resolution": cfg.resolution,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(sidecar, indent=2))


def save_lora(transformer, out_dir: Path, cfg: TrainConfig) -> None:
    from diffusers import QwenImageEditPlusPipeline
    from peft.utils import get_peft_model_state_dict

    out_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(transformer)
    # Cast back to bf16 for a compact, inference-ready checkpoint.
    state = {k: v.to(torch.bfloat16) for k, v in state.items()}
    QwenImageEditPlusPipeline.save_lora_weights(
        out_dir, transformer_lora_layers=state, safe_serialization=True
    )
    write_target_modules_sidecar(out_dir, cfg)
    print(f"[train] saved LoRA -> {out_dir}")


# --- Flow-matching pieces ----------------------------------------------------


def sample_sigmas(batch: int, mode: str, device) -> torch.Tensor:
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(batch, device=device))
    return torch.rand(batch, device=device)  # uniform


def make_img_shapes(target_grid: tuple[int, int], control_grid: tuple[int, int]) -> list:
    return [[(1, *target_grid), (1, *control_grid)]]


def training_step(transformer, item: dict, cfg: TrainConfig, device) -> torch.Tensor:
    x0 = item["target_latents"].to(device, torch.bfloat16)      # [1, seq_t, C4]
    ctrl = item["control_latents"].to(device, torch.bfloat16)   # [1, seq_c, C4]
    embeds = item["prompt_embeds"].to(device, torch.bfloat16)   # [1, L, D]
    mask = item["prompt_mask"].to(device)

    sigma = sample_sigmas(1, cfg.timestep_sampling, device).to(torch.bfloat16)
    noise = torch.randn_like(x0)
    x_t = (1.0 - sigma) * x0 + sigma * noise
    target_v = noise - x0

    model_input = torch.cat([x_t, ctrl], dim=1)
    img_shapes = make_img_shapes(item["target_grid"], item["control_grid"])

    pred = transformer(
        hidden_states=model_input,
        timestep=sigma,  # pipeline passes t/1000 == sigma
        guidance=None,
        encoder_hidden_states_mask=mask,
        encoder_hidden_states=embeds,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
    pred = pred[:, : x0.shape[1]]

    return F.mse_loss(pred.float(), target_v.float())


# --- Eval hook ----------------------------------------------------------------


@torch.no_grad()
def denoise_holdout_sample(
    transformer, item: dict, cfg: TrainConfig, device
) -> torch.Tensor:
    """Run the real K-step CFG denoise loop over one cached holdout entry.
    Returns packed output latents. Mirrors the inference pipeline (incl.
    dynamic-shift sigma schedule and norm-preserving CFG)."""
    ctrl = item["control_latents"].to(device, torch.bfloat16)
    embeds = item["prompt_embeds"].to(device, torch.bfloat16)
    mask = item["prompt_mask"].to(device)
    neg_embeds = item["neg_embeds"].to(device, torch.bfloat16)
    neg_mask = item["neg_mask"].to(device)
    grid = item["control_grid"]
    img_shapes = make_img_shapes(grid, grid)

    seq_len = ctrl.shape[1]
    latents = torch.randn(
        (1, seq_len, ctrl.shape[2]),
        device=device, dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(cfg.seed),
    )

    # Dynamic-shift schedule (calculate_shift in the pipeline).
    base_len, max_len, base_shift, max_shift = 256, 4096, 0.5, 1.15
    m = (max_shift - base_shift) / (max_len - base_len)
    mu = seq_len * m + (base_shift - base_len * m)
    sigmas = np.linspace(1.0, 1.0 / cfg.eval_num_steps, cfg.eval_num_steps)
    shifted = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))  # time shift
    sigmas_t = torch.from_numpy(np.append(shifted, 0.0)).to(device, torch.float32)

    for i in range(cfg.eval_num_steps):
        sigma = sigmas_t[i]
        t = sigma.expand(1).to(torch.bfloat16)
        model_input = torch.cat([latents, ctrl], dim=1)

        pred = transformer(
            hidden_states=model_input, timestep=t, guidance=None,
            encoder_hidden_states_mask=mask, encoder_hidden_states=embeds,
            img_shapes=img_shapes, return_dict=False,
        )[0][:, :seq_len]

        if cfg.eval_true_cfg_scale > 1.0:
            neg_pred = transformer(
                hidden_states=model_input, timestep=t, guidance=None,
                encoder_hidden_states_mask=neg_mask, encoder_hidden_states=neg_embeds,
                img_shapes=img_shapes, return_dict=False,
            )[0][:, :seq_len]
            comb = neg_pred + cfg.eval_true_cfg_scale * (pred - neg_pred)
            cond_norm = torch.norm(pred, dim=-1, keepdim=True)
            comb_norm = torch.norm(comb, dim=-1, keepdim=True)
            pred = comb * (cond_norm / comb_norm)

        dt = sigmas_t[i + 1] - sigmas_t[i]
        latents = (latents.float() + pred.float() * dt).to(torch.bfloat16)

    return latents


def run_eval_grid(
    transformer, cacher: ConditioningCacher, holdout_cache: Path,
    cfg: TrainConfig, epoch: int,
) -> dict:
    """Denoise the cached holdout, decode, score with the eval harness, write
    the HTML grid, and quarantine on canary trip. Returns aggregate metrics."""
    from aura_ml.eval.grid import build_grid, flag_static_checkpoint

    entries = sorted(holdout_cache.glob("*.pt"))[: cfg.eval_max_samples]
    if not entries:
        return {}

    device = "cuda"
    triples = []
    eval_dir = Path(cfg.output_dir) / "eval" / f"epoch_{epoch:04d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    transformer.eval()
    for pt_path in tqdm(entries, desc=f"eval@{epoch}"):
        item = torch.load(pt_path, weights_only=False)
        out_latents = denoise_holdout_sample(transformer, item, cfg, device)
        out_img = cacher.decode_packed_latents(out_latents, item["control_grid"])
        out_img.save(eval_dir / f"{item['id']}.jpg", quality=92)
        # Reconstruct the control image from cached latents for a fair
        # comparison (same VAE round-trip as the output).
        ctrl_img = cacher.decode_packed_latents(
            item["control_latents"], item["control_grid"]
        )
        triples.append((item["id"], ctrl_img, out_img, item["instruction"]))
    transformer.train()

    report = build_grid(triples, meta={"epoch": str(epoch), "run": cfg.output_dir})
    report.to_html(eval_dir / "grid.html")
    report.to_json(eval_dir / "grid.json")

    quarantine, reason = flag_static_checkpoint(report)
    if quarantine:
        (Path(cfg.output_dir) / f"epoch_{epoch:04d}.QUARANTINE").write_text(reason)
        print(f"[eval] !! CANARY TRIPPED at epoch {epoch}: {reason}")
    print(f"[eval] epoch {epoch}: {report.aggregate} "
          f"(static={report.static_fraction:.0%})")
    return report.aggregate


# --- Main loop -----------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    device = "cuda"
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset_root = Path(cfg.dataset_root)
    holdout_dir = Path(cfg.eval_holdout_dir) if cfg.eval_holdout_dir else None
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: conditioning cache (loads + frees the encoders).
    cacher = ConditioningCacher(cfg, device)
    cache_dir = cacher.build_cache(dataset_root, holdout_dir)
    if cacher._vae is None:  # cache was fully reused — still need VAE for eval
        from diffusers import AutoencoderKLQwenImage

        cacher._vae = AutoencoderKLQwenImage.from_pretrained(
            cfg.base_model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ).to(device).eval()
    holdout_cache = cache_dir / "holdout"

    train_items = sorted(p for p in cache_dir.glob("*.pt"))
    if not train_items:
        raise RuntimeError(f"no cached training pairs in {cache_dir}")
    print(f"[train] {len(train_items)} cached pairs")

    # Phase 2: transformer + LoRA.
    transformer = load_transformer(cfg, device)
    trainable = attach_lora(transformer, cfg)
    transformer.train()

    if cfg.optimizer == "adamw_8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

    steps_per_epoch = math.ceil(len(train_items) / cfg.gradient_accumulation_steps)
    total_steps = max(steps_per_epoch * cfg.epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.lr_min
    )

    log_path = out_dir / "train_log.jsonl"
    best_metric = -1.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(train_items)))
        random.shuffle(order)
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(order, desc=f"epoch {epoch}/{cfg.epochs}")
        for i, idx in enumerate(pbar):
            item = torch.load(train_items[idx], weights_only=False)
            loss = training_step(transformer, item, cfg, device)
            (loss / cfg.gradient_accumulation_steps).backward()
            epoch_losses.append(loss.item())

            if (i + 1) % cfg.gradient_accumulation_steps == 0 or (i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix(loss=f"{epoch_losses[-1]:.4f}")

        mean_loss = float(np.mean(epoch_losses))
        record = {
            "epoch": epoch,
            "loss": round(mean_loss, 5),
            "lr": scheduler.get_last_lr()[0],
            "step": global_step,
        }

        # Eval + canary quarantine
        if holdout_dir and holdout_cache.exists() and epoch % cfg.eval_every_epochs == 0:
            aggregate = run_eval_grid(transformer, cacher, holdout_cache, cfg, epoch)
            record["eval"] = {k: round(v, 4) for k, v in aggregate.items()
                              if not math.isnan(v)}
            quarantined = (out_dir / f"epoch_{epoch:04d}.QUARANTINE").exists()
            score = aggregate.get("clip_score", float("nan"))
            if not quarantined and not math.isnan(score) and score > best_metric:
                best_metric = score
                save_lora(transformer, out_dir / "best", cfg)
                record["promoted_best"] = True

        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs:
            save_lora(transformer, out_dir / f"epoch_{epoch:04d}", cfg)

        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[train] epoch {epoch}: loss={mean_loss:.4f} lr={record['lr']:.2e}")

    print(f"[train] done. checkpoints in {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f)

    train(cfg)


if __name__ == "__main__":
    main()
