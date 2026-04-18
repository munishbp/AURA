"""aura_ml — plastic-surgery outcome visualization pipeline.

Top-level package. The two main entry points are:

- `aura_ml.inference.pipeline.AuraInferencePipeline` — production-style pipeline
  that composes the prompt expander, the diffusion model, and identity LoRAs.
- `aura_ml.training.train` — minimal LoRA training script (workstream 4).

For a high-level Gradio demo, see `app/demo.py` at the repo root.
"""

__version__ = "0.0.0"
