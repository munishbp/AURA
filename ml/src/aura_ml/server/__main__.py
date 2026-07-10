"""Entrypoint: uv run python -m aura_ml.server [--host 0.0.0.0] [--ui] ..."""

from __future__ import annotations

import argparse
import os

# GPU VRAM belongs to the editor + expander; run metric models on CPU unless
# the operator explicitly says otherwise. Must be set before metrics import.
os.environ.setdefault("AURA_METRICS_DEVICE", "cpu")

import uvicorn

from aura_ml.server.api import ServerConfig, create_app


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aura REST API server (OpenAPI docs at /docs)"
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; 0.0.0.0 to serve your LAN (phone uploads)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--checkpoints", default="checkpoints",
                   help="dir with per-procedure LoRA checkpoint subdirs")
    p.add_argument("--no-expander", action="store_true",
                   help="skip the Qwen3.5-9B expander (~7 GB VRAM saved)")
    p.add_argument("--preload", action="store_true",
                   help="load models at startup instead of on first request")
    p.add_argument("--ui", action="store_true",
                   help="also mount the Gradio demo at /ui (same pipeline)")
    args = p.parse_args()

    app = create_app(
        ServerConfig(
            checkpoints_dir=args.checkpoints,
            use_prompt_expander=not args.no_expander,
            preload=args.preload,
            mount_ui=args.ui,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
