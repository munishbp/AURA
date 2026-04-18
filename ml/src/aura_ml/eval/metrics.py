"""Quantitative eval metrics.

Four metrics are required by the plan:

1. ArcFace cosine          — identity preservation between source and output
2. **Edit magnitude**      — `1 - cosine(DINO(source), DINO(output))`. The
                              static-image canary. Near zero means the model
                              copied the input — exactly the failure mode that
                              killed the hackathon LoRAs.
3. LPIPS                   — perceptual distance, source vs output
4. CLIPScore               — instruction-text vs output image alignment

All metric models are loaded lazily and cached as module-level singletons.

Workstream 2 implements all four.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

# --- Edit magnitude (DINO) ---------------------------------------------------


@lru_cache(maxsize=1)
def _load_dino():
    """DINOv2 ViT-B/14. Used for edit-magnitude — does the output differ from
    the input in semantically meaningful ways?
    """
    # TODO(workstream 2):
    #   from transformers import AutoModel, AutoImageProcessor
    #   processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    #   model = AutoModel.from_pretrained("facebook/dinov2-base").eval().cuda()
    #   return processor, model
    raise NotImplementedError("workstream 2: load DINOv2")


def edit_magnitude(source: Image.Image, output: Image.Image) -> float:
    """Returns 1 - cosine(DINO(source), DINO(output)).

    Near-zero = model copied the input (BAD — see pre-mortem).
    Large    = model made a substantive change (could be good or bad — pair
               with arcface_cosine and clip_score to characterize).

    Recommended floor for "real edit happened": 0.05.
    """
    # TODO(workstream 2): encode both, L2-normalize, return 1 - cos sim
    raise NotImplementedError("workstream 2: implement edit_magnitude")


# --- Identity preservation (ArcFace) ----------------------------------------


@lru_cache(maxsize=1)
def _load_arcface():
    """InsightFace's buffalo_l (ArcFace + RetinaFace) — the standard for
    face-identity cosine sims in 2025/26.
    """
    # TODO(workstream 2):
    #   import insightface
    #   app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    #   app.prepare(ctx_id=0, det_size=(640, 640))
    #   return app
    raise NotImplementedError("workstream 2: load ArcFace")


def arcface_cosine(image_a: Image.Image, image_b: Image.Image) -> float:
    """Returns cosine similarity between the largest-face embeddings in each
    image. Returns NaN if either image has no detected face.

    For a procedure LoRA we expect this to stay ≥ 0.6 — the post-edit face
    should clearly still be the same person.
    """
    # TODO(workstream 2): embed both, return cosine. NaN on no-face.
    raise NotImplementedError("workstream 2: implement arcface_cosine")


# --- Perceptual (LPIPS) -----------------------------------------------------


@lru_cache(maxsize=1)
def _load_lpips():
    """LPIPS-AlexNet. Cheap, well-calibrated."""
    # TODO(workstream 2):
    #   import lpips
    #   return lpips.LPIPS(net="alex").eval().cuda()
    raise NotImplementedError("workstream 2: load LPIPS")


def lpips_score(image_a: Image.Image, image_b: Image.Image) -> float:
    """LPIPS distance in [0, ~1]. Higher = more perceptually different."""
    # TODO(workstream 2): normalize to [-1, 1] tensors, run net, return scalar
    raise NotImplementedError("workstream 2: implement lpips_score")


# --- Edit fidelity (CLIPScore) ----------------------------------------------


@lru_cache(maxsize=1)
def _load_clip():
    """OpenCLIP ViT-L/14. Used to score (instruction text, output image)
    alignment.
    """
    # TODO(workstream 2):
    #   import open_clip
    #   model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    #   tokenizer = open_clip.get_tokenizer("ViT-L-14")
    #   return model.cuda().eval(), preprocess, tokenizer
    raise NotImplementedError("workstream 2: load CLIP")


def clip_score(image: Image.Image, text: str) -> float:
    """Cosine similarity between CLIP image and text embeddings. Higher =
    output matches the instruction.
    """
    # TODO(workstream 2): encode, normalize, dot
    raise NotImplementedError("workstream 2: implement clip_score")


# --- Aggregate ---------------------------------------------------------------


def all_metrics(
    source: Image.Image,
    output: Image.Image,
    instruction: str,
) -> dict[str, float]:
    """Compute all four metrics on one (source, output, instruction) triple.

    Returns a dict — None values for metrics that errored (e.g., no face
    detected). Stable key order so it can be used directly as a CSV row.
    """
    return {
        "edit_magnitude": edit_magnitude(source, output),
        "arcface_cosine": arcface_cosine(source, output),
        "lpips": lpips_score(source, output),
        "clip_score": clip_score(output, instruction),
    }


def is_static(metrics: dict[str, float], threshold: float = 0.05) -> bool:
    """The canary. True = checkpoint produced a static-image collapse on this
    sample; flag the checkpoint. Apply across the holdout set and aggregate.
    """
    em = metrics.get("edit_magnitude")
    return em is not None and em < threshold
