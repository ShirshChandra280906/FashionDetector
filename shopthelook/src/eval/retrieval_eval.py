"""Consumer-to-shop retrieval evaluation on DeepFashion2 validation.

Protocol (follows the DeepFashion2 consumer-to-shop task, using ground-truth boxes):
  * query   = every garment in a consumer ("user") photo with style > 0
  * gallery = every garment in a shop photo
  * a gallery item is a correct match when it has the same (pair_id, style)
  * metric  = Recall@k: share of queries with at least one correct match in the top k

We crop with ground-truth boxes so this measures the *embedding* in isolation.
Detector errors are measured separately in the detection evaluation.

Two ranking modes:
  * none  - rank the whole gallery
  * group - rank only gallery items in the query's category group (what the product does)

Usage:
    python -m src.eval.retrieval_eval --models color_hist clip fashionclip
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import ARTIFACTS, DF2, DF2_TO_GROUP, RESULTS, set_seed
from src.embed import get_embedder
from src.index import GroupedIndex
from src.retrieve import first_positive_rank, recall_at_k

TOP_K = 50  # we store the first-positive rank up to 50; anything deeper counts as a miss


def load_crops(image_stem: str, boxes: list[tuple]) -> list[Image.Image]:
    with Image.open(DF2 / "validation" / "image" / f"{image_stem}.jpg") as im:
        im = im.convert("RGB")
        W, H = im.size
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            crops.append(im.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))))
    return crops


def embed_items(items: pd.DataFrame, model: str, chunk_images: int = 256) -> np.ndarray:
    """Embed every row of `items` (cached to artifacts/df2val_<model>.npy)."""
    cache = ARTIFACTS / f"df2val_{model}.npy"
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(items):
            print(f"[{model}] using cached embeddings {cache.name}")
            return emb
    embedder = get_embedder(model)
    groups = list(items.groupby("image", sort=False))  # keep crops of one image together
    out = np.zeros((len(items), embedder.dim), np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        for s in tqdm(range(0, len(groups), chunk_images), desc=f"embed {model}"):
            part = groups[s : s + chunk_images]
            crops_per_img = pool.map(lambda g: load_crops(g[0], list(zip(g[1].x1, g[1].y1, g[1].x2, g[1].y2))), part)
            crops, rows = [], []
            for (_, g), cs in zip(part, crops_per_img):
                crops.extend(cs)
                rows.extend(g.index.tolist())
            out[rows] = embedder.embed(crops)
    print(f"[{model}] embedded {len(items):,} crops in {time.time() - t0:.0f}s")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def evaluate(items: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Return one row per query with its first-positive rank in each mode."""
    q = items[(items.source == "user") & (items["style"] > 0)]
    g = items[items.source == "shop"]
    g_emb, q_emb = emb[g.index], emb[q.index]
    g_key = list(zip(g.pair_id, g["style"]))
    positives: dict[tuple, set] = {}
    for i, key in enumerate(g_key):
        positives.setdefault(key, set()).add(i)

    gindex = GroupedIndex(g_emb, g.category.map(DF2_TO_GROUP).tolist())
    q_keys = list(zip(q.pair_id, q["style"]))
    q_groups = q.category.map(DF2_TO_GROUP).to_numpy()

    ranks = {"none": np.zeros(len(q), int), "group": np.zeros(len(q), int)}
    _, ids = gindex.search(q_emb, TOP_K)
    for i in range(len(q)):
        ranks["none"][i] = first_positive_rank(ids[i], positives[q_keys[i]])
    for grp in np.unique(q_groups):
        rows = np.flatnonzero(q_groups == grp)
        _, ids = gindex.search(q_emb[rows], TOP_K, group=grp)
        for r, row_ids in zip(rows, ids):
            ranks["group"][r] = first_positive_rank(row_ids, positives[q_keys[r]])

    out = q[["image", "item", "category", "occlusion", "zoom_in", "viewpoint", "scale"]].copy()
    out["group"] = q_groups
    out["rank_none"], out["rank_group"] = ranks["none"], ranks["group"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["color_hist", "clip", "fashionclip"])
    args = ap.parse_args()
    set_seed()

    items = pd.read_csv(DF2 / "val_items.csv", dtype={"image": str})
    out_dir = RESULTS / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "df2_c2s_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    for model in args.models:
        emb = embed_items(items, model)
        per_q = evaluate(items, emb)
        per_q.to_csv(out_dir / f"df2_c2s_ranks_{model}.csv", index=False)
        for mode in ("none", "group"):
            row = {"model": model, "filter": mode, "n_queries": len(per_q),
                   "n_gallery": int((items.source == "shop").sum()),
                   **recall_at_k(per_q[f"rank_{mode}"], ks=(1, 5, 10, 20))}
            print(json.dumps(row))
            if not summary.empty:
                summary = summary[~((summary.model == model) & (summary["filter"] == mode))]
            summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
