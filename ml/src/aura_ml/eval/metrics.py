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
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


# --- Edit magnitude (DINO) ---------------------------------------------------


@lru_cache(maxsize=1)
def _load_dino():
    """DINOv2 ViT-B/14. Used for edit-magnitude — does the output differ from
    the input in semantically meaningful ways?
    """
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(_DEVICE)
    return processor, model


@torch.inference_mode()
def _dino_embed(img: Image.Image) -> torch.Tensor:
    processor, model = _load_dino()
    inputs = processor(images=_to_rgb(img), return_tensors="pt").to(_DEVICE)
    out = model(**inputs).last_hidden_state[:, 0]  # CLS token
    return F.normalize(out, dim=-1).squeeze(0)


def edit_magnitude(source: Image.Image, output: Image.Image) -> float:
    """Returns 1 - cosine(DINO(source), DINO(output)).

    Near-zero = model copied the input (BAD — see pre-mortem).
    Large    = model made a substantive change (could be good or bad — pair
               with arcface_cosine and clip_score to characterize).

    Recommended floor for "real edit happened": 0.05.
    """
    a = _dino_embed(source)
    b = _dino_embed(output)
    return float(1.0 - torch.dot(a, b).item())


# --- Identity preservation (ArcFace) ----------------------------------------


@lru_cache(maxsize=1)
def _load_arcface():
    """InsightFace's buffalo_l (ArcFace + RetinaFace). Falls back to CPU if
    onnxruntime-gpu isn't picking up CUDA (common on Windows).
    """
    import insightface

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _largest_face_embedding(img: Image.Image) -> np.ndarray | None:
    app = _load_arcface()
    rgb = _to_rgb(img)
    arr = np.array(rgb)[:, :, ::-1]  # RGB → BGR for insightface
    faces = app.get(arr)
    if not faces:
        # SCRFD misses faces that fill the whole frame (tight portrait crops —
        # exactly what this pipeline processes). Retry with a neutral border;
        # the embedding is landmark-aligned, so padding doesn't distort it.
        from PIL import ImageOps

        pad = max(rgb.width, rgb.height) // 3
        padded = ImageOps.expand(rgb, border=pad, fill=(127, 127, 127))
        faces = app.get(np.array(padded)[:, :, ::-1])
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    emb = faces[0].normed_embedding
    return np.asarray(emb, dtype=np.float32)


def arcface_cosine(image_a: Image.Image, image_b: Image.Image) -> float:
    """Returns cosine similarity between the largest-face embeddings in each
    image. Returns NaN if either image has no detected face.

    For a procedure LoRA we expect this to stay ≥ 0.6 — the post-edit face
    should clearly still be the same person.
    """
    ea = _largest_face_embedding(image_a)
    eb = _largest_face_embedding(image_b)
    if ea is None or eb is None:
        return float("nan")
    return float(np.dot(ea, eb))


# --- Perceptual (LPIPS) -----------------------------------------------------


@lru_cache(maxsize=1)
def _load_lpips():
    """LPIPS-AlexNet. Cheap, well-calibrated."""
    import lpips

    return lpips.LPIPS(net="alex").eval().to(_DEVICE)


def _lpips_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(_to_rgb(img).resize((256, 256), Image.LANCZOS), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW in [0,1]
    return (t * 2.0 - 1.0).to(_DEVICE)


@torch.inference_mode()
def lpips_score(image_a: Image.Image, image_b: Image.Image) -> float:
    """LPIPS distance in [0, ~1]. Higher = more perceptually different."""
    net = _load_lpips()
    return float(net(_lpips_tensor(image_a), _lpips_tensor(image_b)).item())


# --- Edit fidelity (CLIPScore) ----------------------------------------------


@lru_cache(maxsize=1)
def _load_clip():
    """OpenCLIP ViT-L/14. Used to score (instruction text, output image)
    alignment.
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model.to(_DEVICE).eval(), preprocess, tokenizer


@torch.inference_mode()
def clip_score(image: Image.Image, text: str) -> float:
    """Cosine similarity between CLIP image and text embeddings. Higher =
    output matches the instruction.
    """
    model, preprocess, tokenizer = _load_clip()
    img_t = preprocess(_to_rgb(image)).unsqueeze(0).to(_DEVICE)
    tok = tokenizer([text]).to(_DEVICE)
    img_emb = F.normalize(model.encode_image(img_t), dim=-1)
    txt_emb = F.normalize(model.encode_text(tok), dim=-1)
    return float((img_emb @ txt_emb.T).item())


# --- Aggregate ---------------------------------------------------------------


def all_metrics(
    source: Image.Image,
    output: Image.Image,
    instruction: str,
) -> dict[str, float]:
    """Compute all four metrics on one (source, output, instruction) triple.

    Returns a dict — NaN values for metrics that errored (e.g., no face
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
    if em is None or (isinstance(em, float) and math.isnan(em)):
        return False
    return em < threshold
