"""Phase-1 data overview: catalog coverage and DF2 class balance.

Writes results/data/{catalog_groups,df2_class_counts}.csv and data_overview.png.
    python -m src.eval.data_overview
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CATALOG, DF2, DF2_CLASSES, GROUPS, RESULTS, SPLITS
from src.viz import SERIES, hbar, style


def df2_counts(split_dir, list_file) -> pd.Series:
    counts = dict.fromkeys(DF2_CLASSES, 0)
    for s in list_file.read_text().split():
        a = json.loads((split_dir / "annos" / f"{s}.json").read_text())
        for k, v in a.items():
            if k.startswith("item"):
                counts[DF2_CLASSES[v["category_id"] - 1]] += 1
    return pd.Series(counts)


def main() -> None:
    out = RESULTS / "data"
    out.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(CATALOG / "catalog.csv")
    groups = cat["group"].value_counts().reindex(GROUPS).fillna(0).astype(int)
    groups.rename("products").to_frame().assign(share=lambda d: d.products / d.products.sum()).to_csv(
        out / "catalog_groups.csv", index_label="group")

    df2 = pd.DataFrame({
        "train_subset": df2_counts(DF2 / "train", SPLITS / "df2_train_subset.txt"),
        "det_val": df2_counts(DF2 / "validation", SPLITS / "df2_det_val.txt"),
        "det_test": df2_counts(DF2 / "validation", SPLITS / "df2_det_test.txt"),
    })
    df2.to_csv(out / "df2_class_counts.csv", index_label="category")
    print(groups.to_string(), "\n")
    print(df2.to_string())

    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    g = groups.sort_values(ascending=False)
    hbar(a, g.index.tolist(), g.values.tolist(), color=SERIES[0])
    a.set_title(f"Catalog products per retrieval group (n={groups.sum():,})")
    a.set_xlabel("products")
    t = df2["train_subset"].sort_values(ascending=False)
    hbar(b, t.index.tolist(), t.values.tolist(), color=SERIES[0])
    b.set_title(f"DF2 train subset: instances per class ({len((SPLITS / 'df2_train_subset.txt').read_text().split()):,} imgs)")
    b.set_xlabel("instances")
    fig.tight_layout()
    fig.savefig(out / "data_overview.png")
    print(f"\nSaved {out / 'data_overview.png'}")


if __name__ == "__main__":
    main()
