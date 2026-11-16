"""Shared configuration: paths, seeds, and the category taxonomy.

Everything that more than one script needs lives here so the pipeline,
evaluation scripts, notebooks and the demo app all agree on the same values.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
DF2_RAW = RAW / "df2"                 # DeepFashion2 zips as downloaded
DF2 = DATA / "df2"                    # extracted images + annos
YOLO_DS = DATA / "df2_yolo"           # YOLO-format dataset (images/ + labels/)
SPLITS = ROOT / "splits"              # committed split lists (small text files)
CATALOG = DATA / "catalog"            # catalog.csv + images/
ARTIFACTS = ROOT / "artifacts"        # embeddings, FAISS indexes
MODELS = ROOT / "models"              # trained detector weights
RESULTS = ROOT / "results"

# Unofficial (unencrypted) Hugging Face re-upload of DeepFashion2.
DF2_HF_BASE = "https://huggingface.co/datasets/sahirp/deepfashion2/resolve/main"

SEED = 42

# ---------------------------------------------------------------------------
# DeepFashion2 taxonomy (13 classes; DF2 category_id is 1-based, YOLO is 0-based)
# ---------------------------------------------------------------------------
DF2_CLASSES = [
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeve_dress",
    "long_sleeve_dress",
    "vest_dress",
    "sling_dress",
]

# The product catalog has no sleeve-length labels, so retrieval filters on a
# coarser "group". Each DF2 class maps to exactly one group.
DF2_TO_GROUP = {
    "short_sleeve_top": "top",
    "long_sleeve_top": "top",
    "vest": "top",
    "sling": "top",
    "short_sleeve_outwear": "outerwear",
    "long_sleeve_outwear": "outerwear",
    "shorts": "shorts",
    "trousers": "trousers",
    "skirt": "skirt",
    "short_sleeve_dress": "dress",
    "long_sleeve_dress": "dress",
    "vest_dress": "dress",
    "sling_dress": "dress",
}
GROUPS = ["top", "outerwear", "shorts", "trousers", "skirt", "dress"]


def set_seed(seed: int = SEED) -> None:
    """Fix every RNG we use so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is optional for the data-only scripts
        pass


def best_device() -> str:
    """Pick CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
