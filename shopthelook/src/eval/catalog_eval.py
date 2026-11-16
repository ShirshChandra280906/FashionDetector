"""Catalog-side retrieval metrics (no exact ground truth exists for the catalog).

1. Wrong-category rate without filtering: DF2 consumer garments (12,377 GT crops)
   are searched against the whole catalog. We measure the share of top-k
   recommendations whose group differs from the garment's true group. This
   quantifies the "wrong-category match" risk and what the category filter removes.
2. Catalog leave-one-out (category-match accuracy@k): every catalog product is used as a
   query against the rest of its group. We measure the share of top-k neighbours with the
   same articleType and the same baseColour.

Usage:
    python -m src.eval.catalog_eval --models color_hist clip fashionclip
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import ARTIFACTS, CATALOG, DF2, DF2_TO_GROUP, RESULTS
from src.index import FlatIndex, GroupedIndex

KS = (1, 5, 10)


def wrong_category_rate(model: str, items: pd.DataFrame, cat: pd.DataFrame) -> dict:
    q = items[(items.source == "user") & (items["style"] > 0)]
    q_emb = np.load(ARTIFACTS / f"df2val_{model}.npy")[q.index]
    c_emb = np.load(ARTIFACTS / f"catalog_{model}.npy")
    _, ids = FlatIndex(c_emb).search(q_emb, max(KS))
    c_groups = cat["group"].to_numpy()[ids]  # (n_queries, K)
    q_groups = q.category.map(DF2_TO_GROUP).to_numpy()[:, None]
    wrong = c_groups != q_groups
    row = {"model": model, "n_queries": len(q)}
    for k in KS:
        row[f"wrong_cat@{k}"] = float(wrong[:, :k].mean())
    # Break down by the garment's true group (tops dominate the catalog)
    per_group = pd.DataFrame({"group": q_groups[:, 0], "wrong@5": wrong[:, :5].mean(1)}).groupby("group")["wrong@5"].mean()
    return row, per_group.rename(model)


def catalog_loo(model: str, cat: pd.DataFrame) -> dict:
    emb = np.load(ARTIFACTS / f"catalog_{model}.npy")
    gi = GroupedIndex(emb, cat["group"].tolist())
    art, col = cat["articleType"].to_numpy(), cat["baseColour"].fillna("NA").to_numpy()
    hits_art = {k: [] for k in KS}
    hits_col = {k: [] for k in KS}
    for g, rows in gi.ids.items():
        _, nn = gi.search(emb[rows], max(KS) + 1, group=g)
        nn = nn[:, 1:]  # drop the query itself (it is its own nearest neighbour)
        for k in KS:
            hits_art[k].append((art[nn[:, :k]] == art[rows][:, None]).mean(1))
            hits_col[k].append((col[nn[:, :k]] == col[rows][:, None]).mean(1))
    row = {"model": model, "n_products": len(cat)}
    for k in KS:
        row[f"same_articleType@{k}"] = float(np.concatenate(hits_art[k]).mean())
        row[f"same_colour@{k}"] = float(np.concatenate(hits_col[k]).mean())
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["color_hist", "clip", "fashionclip"])
    args = ap.parse_args()
    items = pd.read_csv(DF2 / "val_items.csv", dtype={"image": str})
    cat = pd.read_csv(CATALOG / "catalog.csv")
    out = RESULTS / "retrieval"
    out.mkdir(parents=True, exist_ok=True)

    wrong_rows, per_group, loo_rows = [], [], []
    for m in args.models:
        r, pg = wrong_category_rate(m, items, cat)
        wrong_rows.append(r)
        per_group.append(pg)
        loo_rows.append(catalog_loo(m, cat))
    pd.DataFrame(wrong_rows).to_csv(out / "catalog_wrong_category_unfiltered.csv", index=False)
    pd.concat(per_group, axis=1).to_csv(out / "catalog_wrong_category_by_group.csv", index_label="group")
    pd.DataFrame(loo_rows).to_csv(out / "catalog_leave_one_out.csv", index=False)
    print(pd.DataFrame(wrong_rows).to_string(index=False), "\n")
    print(pd.concat(per_group, axis=1).to_string(), "\n")
    print(pd.DataFrame(loo_rows).to_string(index=False))


if __name__ == "__main__":
    main()
