"""Fetch N synthetic face photos for pipeline testing.

Downloads StyleGAN-generated faces from thispersondoesnotexist.com — these
are AI-generated people (nobody's real face), so they're safe to use for
smoke tests, the toy training task, and the synthetic eval holdout without
privacy or licensing concerns.

The real eval holdout should be built from the HDA Facial Plastic Surgery
Database (research-only license) with build_eval_holdout.py once that
database is on this machine — synthetic faces validate the *pipeline*, not
surgical realism.

Run from the repo root:
    uv run python scripts/fetch_test_faces.py --n 16 --out data/raw/faces
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

URL = "https://thispersondoesnotexist.com/random-person.jpeg"


def fetch_one(retries: int = 3) -> Image.Image | None:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (aura-ml test-face fetcher)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(2)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out", default="data/raw/faces")
    p.add_argument("--size", type=int, default=1024, help="long-edge resize (0 = keep original)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    existing = len(list(out.glob("face_*.jpg")))
    fetched = 0
    idx = existing
    while fetched < args.n:
        img = fetch_one()
        if img is None:
            print("[fatal] repeated fetch failures — check network", file=sys.stderr)
            return 1
        if args.size:
            img.thumbnail((args.size, args.size), Image.LANCZOS)
        idx += 1
        path = out / f"face_{idx:04d}.jpg"
        img.save(path, quality=95)
        fetched += 1
        print(f"[{fetched}/{args.n}] {path}")
        time.sleep(1.2)  # be polite; the site serves a new face per request

    print(f"done — {fetched} new faces in {out} ({existing} were already there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
