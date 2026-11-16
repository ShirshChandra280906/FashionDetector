"""Catalog retrieval: crop embedding + category -> top-k similar products."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ARTIFACTS, CATALOG
from src.index import GroupedIndex


@dataclass
class Match:
    product_id: int
    name: str
    article_type: str
    colour: str
    group: str
    score: float  # cosine similarity in [-1, 1]
    image_path: str  # absolute path to the product image


class CatalogRetriever:
    """Holds the catalog metadata and its grouped FAISS index."""

    def __init__(self, catalog: pd.DataFrame, emb: np.ndarray, image_root=CATALOG):
        assert len(catalog) == len(emb), "catalog rows and embeddings must align"
        self.catalog = catalog.reset_index(drop=True)
        self.index = GroupedIndex(emb, self.catalog["group"].tolist())
        self.image_root = image_root

    @classmethod
    def load(cls, embedder_name: str) -> "CatalogRetriever":
        """Load catalog.csv and the cached catalog embeddings for one embedder."""
        catalog = pd.read_csv(CATALOG / "catalog.csv")
        emb = np.load(ARTIFACTS / f"catalog_{embedder_name}.npy")
        return cls(catalog, emb)

    def search(self, query_emb: np.ndarray, group: str | None, k: int = 5) -> list[Match]:
        """Top-k products for ONE query vector, restricted to `group` if given."""
        scores, rows = self.index.search(query_emb.reshape(1, -1), k, group=group)
        out = []
        for s, r in zip(scores[0], rows[0]):
            if r < 0:
                continue
            row = self.catalog.iloc[int(r)]
            out.append(
                Match(
                    product_id=int(row["id"]),
                    name=str(row["productDisplayName"]),
                    article_type=str(row["articleType"]),
                    colour=str(row["baseColour"]),
                    group=str(row["group"]),
                    score=float(s),
                    image_path=str(self.image_root / row["image_path"]),
                )
            )
        return out


def first_positive_rank(ranked_ids: np.ndarray, positives: set) -> int:
    """1-based rank of the first relevant item in a ranked list, or 0 if none is found.
    Recall@k for a query is then simply (0 < rank <= k)."""
    for i, x in enumerate(ranked_ids):
        if x in positives:
            return i + 1
    return 0


def recall_at_k(ranks: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    """Fraction of queries whose first positive is within the top k."""
    ranks = np.asarray(ranks)
    return {f"R@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in ks}
