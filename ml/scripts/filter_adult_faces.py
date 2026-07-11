"""Filter a test-face directory to clear adult faces only.

StyleGAN face generators emit children and heavily artifacted samples; both
are unusable for a surgical-visualization pipeline (we do not process minors'
faces, even synthetic ones). Uses insightface buffalo_l (detection + age).

Deletes files in place that fail: no detectable face, estimated age < min-age,
or detection confidence below threshold.

Run from ml/:
    uv run python scripts/filter_adult_faces.py data/raw/faces --min-age 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("faces_dir")
    p.add_argument("--min-age", type=int, default=25)
    p.add_argument("--min-det-score", type=float, default=0.6)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import insightface

    app = insightface.app.FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces_dir = Path(args.faces_dir)
    kept, dropped = 0, 0
    for path in sorted(p for p in faces_dir.iterdir() if p.suffix.lower() in EXTS):
        img = Image.open(path).convert("RGB")
        # Pad: SCRFD misses faces that fill the whole frame (tight crops).
        from PIL import ImageOps

        pad = max(img.size) // 3
        img = ImageOps.expand(img, border=pad, fill=(127, 127, 127))
        arr = np.array(img)[:, :, ::-1]  # BGR
        detections = app.get(arr)
        reason = None
        if not detections:
            reason = "no face detected"
        else:
            face = max(detections, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if face.det_score < args.min_det_score:
                reason = f"low detection confidence ({face.det_score:.2f})"
            elif face.age is not None and face.age < args.min_age:
                reason = f"estimated age {face.age} < {args.min_age}"

        if reason:
            dropped += 1
            print(f"DROP {path.name}: {reason}")
            if not args.dry_run:
                path.unlink()
        else:
            kept += 1
            print(f"keep {path.name} (age≈{face.age})")

    print(f"\n{kept} kept, {dropped} dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
