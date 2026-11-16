"""Download the pretrained DeepFashion2 detector and verify its checksum.

We use a third-party model rather than training our own:
  * Repo:    https://huggingface.co/Bingsu/adetailer  (Apache-2.0)
  * File:    deepfashion2_yolov8s-seg.pt  (YOLOv8s instance segmentation, 13 DF2 classes)
  * SHA-256: pinned below. The download fails if the file ever changes upstream.

Security note: .pt files are Python pickles, and loading one can execute code.
Hugging Face's scanner marks this file "unsafe" because the pickle references
builtins.getattr. We inspected it statically (pickletools, without unpickling).
getattr is used only as getattr(ultralytics...head.Detect, "forward"), the standard
Ultralytics Segment-head pattern. Every other import is a torch.nn layer, a torch
tensor-rebuild helper, or an Ultralytics module. See docs/DETECTION.md.

Usage:
    python -m src.download_weights
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request

from src.config import MODELS

URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt"
SHA256 = "c319c15e86443ea61f70b40620909d737c47ff2a39381536503ac5c54772f8ac"
DEST = MODELS / "deepfashion2_yolov8s-seg.pt"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights() -> str:
    """Return the local path to verified weights, downloading them if needed."""
    if DEST.exists() and sha256(DEST) == SHA256:
        return str(DEST)
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256(tmp)
    if got != SHA256:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for detector weights: {got} != {SHA256}")
    tmp.replace(DEST)
    return str(DEST)


if __name__ == "__main__":
    print(ensure_weights())
