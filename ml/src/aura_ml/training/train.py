"""Minimal LoRA training loop for Qwen-Image-Edit-2509.

~150–200 lines once filled in. The point is pedagogical clarity, not
performance — for real runs, prefer the ai-toolkit configs under `configs/`
once parity has been demonstrated against this script in the eval harness.

Workstream 4 implements the loop. The hyperparameters here mirror the plan's
recommendations, lowered for Qwen-Image-Edit (which prefers a lower LR than
FLUX-Kontext recipes).

Pre-mortem mitigations baked in:
- #4 (target modules): logs the actual `target_modules` list to a sidecar
  JSON next to each checkpoint so the inference loader can assert equality.
- #5 (precision): training in bf16 throughout. Inference must match.
- #1 (identity collapse): the toy task in workstream 4 uses pairs with
  visible source/target divergence, so identity collapse fails loudly on
  the eval harness from epoch 1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


# --- Config ----------------------------------------------------------------


@dataclass
class TrainConfig:
    # Data
    dataset_root: str
    resolution: int = 768  # 512 if VRAM tight; 1024 if Vast.ai H100
    # Model
    base_model_id: str = "Qwen/Qwen-Image-Edit-2509"
    quantize_4bit: bool = True
    # LoRA
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ]
    )
    # Optimizer
    learning_rate: float = 1e-4  # lower than FLUX-Kontext recipes
    lr_min: float = 1e-6
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"
    # Schedule
    epochs: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    # Eval / save
    output_dir: str = "outputs/run"
    save_every_epochs: int = 5
    eval_every_epochs: int = 5
    eval_holdout_dir: str = ""  # path to {control,prompts} for grid eval
    # Repro
    seed: int = 0


def load_config(path: str | Path) -> TrainConfig:
    """Load a YAML config into TrainConfig. YAML keys must match dataclass
    field names exactly.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


# --- Setup -----------------------------------------------------------------


def build_pipeline(cfg: TrainConfig):
    """Load the base diffusers pipeline with optional NF4 quantization."""
    # TODO(workstream 4):
    #   - BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    #                        bnb_4bit_compute_dtype=torch.bfloat16)
    #   - QwenImageEditPipeline.from_pretrained(cfg.base_model_id, ...)
    #   - pipe.text_encoder.requires_grad_(False)
    #   - pipe.vae.requires_grad_(False)
    #   - pipe.transformer.requires_grad_(False)  # base frozen, LoRA wraps it
    raise NotImplementedError("workstream 4: build_pipeline")


def attach_lora(pipe, cfg: TrainConfig):
    """Wrap the transformer with a LoRA adapter via PEFT."""
    # TODO(workstream 4):
    #   from peft import LoraConfig, get_peft_model
    #   lora_cfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha,
    #                          lora_dropout=cfg.dropout,
    #                          target_modules=cfg.target_modules,
    #                          init_lora_weights="gaussian")
    #   pipe.transformer = get_peft_model(pipe.transformer, lora_cfg)
    raise NotImplementedError("workstream 4: attach_lora")


def write_target_modules_sidecar(out_dir: Path, cfg: TrainConfig) -> None:
    """Write target_modules + key training hyperparams next to the checkpoint
    so the inference loader can assert equality (pre-mortem mitigation #4).
    """
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
    with open(out_dir / "training_meta.json", "w") as f:
        json.dump(sidecar, f, indent=2)


# --- Eval hook -------------------------------------------------------------


def run_eval_grid(pipe, cfg: TrainConfig, epoch: int) -> dict:
    """Run the eval harness on the holdout set and write an HTML grid.

    Returns the aggregate metrics dict so the training loop can log it.
    Auto-quarantines the checkpoint if the static-image canary trips.
    """
    # TODO(workstream 4):
    #   from aura_ml.eval.grid import build_grid, flag_static_checkpoint
    #   triples = []
    #   for each (control, instruction) in cfg.eval_holdout_dir:
    #       output = pipe(image=control, prompt=instruction, num_steps=20).images[0]
    #       triples.append((id, control, output, instruction))
    #   report = build_grid(triples)
    #   report.to_html(f"{cfg.output_dir}/eval/epoch_{epoch:04d}.html")
    #   should_quarantine, reason = flag_static_checkpoint(report)
    #   if should_quarantine:
    #       (Path(cfg.output_dir) / f"epoch_{epoch:04d}.QUARANTINE").write_text(reason)
    #   return report.aggregate
    raise NotImplementedError("workstream 4: run_eval_grid")


# --- Main loop -------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    """The training loop. Steps:

    1. Build pipeline, attach LoRA, freeze base.
    2. Load PairDataset, optional VAE-latent caching.
    3. Build optimizer (AdamW-8bit) + cosine LR schedule.
    4. For each epoch:
        a. For each batch: forward (predict noise), MSE loss, backward,
           gradient accumulation, optimizer.step, scheduler.step.
        b. Every save_every_epochs: save LoRA weights + sidecar.
        c. Every eval_every_epochs: run_eval_grid (auto-quarantine on canary).
    """
    # TODO(workstream 4): full loop
    raise NotImplementedError("workstream 4: implement train")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config")
    p.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f)

    train(cfg)


if __name__ == "__main__":
    main()
