"""FastAPI server exposing the Aura pipeline as a REST API.

Run:  uv run python -m aura_ml.server --help
Docs: http://<host>:<port>/docs (OpenAPI/Swagger, generated)
"""

import os

# In the server, GPU VRAM belongs to the editor + expander; run metric models
# (and onnxruntime's greedy CUDA arena) on CPU unless the operator overrides.
# Must be set before aura_ml.eval.metrics resolves its device on first use.
os.environ.setdefault("AURA_METRICS_DEVICE", "cpu")

from aura_ml.server.api import ServerConfig, create_app  # noqa: E402

__all__ = ["ServerConfig", "create_app"]
