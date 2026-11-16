"""Embed every catalog product once and cache the vectors for the app.

Writes artifacts/catalog_<model>.npy (rows aligned with data/catalog/catalog.csv).
The FAISS index itself is rebuilt from these vectors at load time (exact search
over 17k vectors builds in milliseconds).

Usage:
    python -m src.build_index --models fashionclip clip color_hist
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import ARTIFACTS, CATALOG, set_seed
from src.embed import get_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["fashionclip"])
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    set_seed()
    cat = pd.read_csv(CATALOG / "catalog.csv")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        out = ARTIFACTS / f"catalog_{name}.npy"
        if out.exists() and len(np.load(out, mmap_mode="r")) == len(cat):
            print(f"[{name}] cached: {out.name}")
            continue
        emb = get_embedder(name)
        chunks, t0 = [], time.time()
        for i in tqdm(range(0, len(cat), args.batch), desc=f"catalog {name}"):
            imgs = [Image.open(CATALOG / p).convert("RGB") for p in cat.image_path[i : i + args.batch]]
            chunks.append(emb.embed(imgs))
        vecs = np.concatenate(chunks)
        np.save(out, vecs)
        print(f"[{name}] {len(vecs):,} products embedded in {time.time() - t0:.0f}s -> {out.name}")


if __name__ == "__main__":
    main()
