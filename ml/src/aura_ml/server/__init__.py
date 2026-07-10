"""FastAPI server exposing the Aura pipeline as a REST API.

Run:  uv run python -m aura_ml.server --help
Docs: http://<host>:<port>/docs (OpenAPI/Swagger, generated)
"""

from aura_ml.server.api import ServerConfig, create_app

__all__ = ["ServerConfig", "create_app"]
