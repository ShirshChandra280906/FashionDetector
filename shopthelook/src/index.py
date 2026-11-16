"""FAISS indexes over L2-normalised embeddings.

We use exact inner-product search (IndexFlatIP). On normalised vectors inner
product equals cosine similarity. At catalog sizes in the tens of thousands,
exact search takes about a millisecond, so an approximate index (IVF/HNSW) is not needed yet.

GroupedIndex keeps one sub-index per category group so we can "filter by
detected category, then rank by cosine", as the product spec asks.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np



class FlatIndex:
    """Exact cosine-similarity search over a fixed set of vectors."""

    def __init__(self, emb: np.ndarray):
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, row_ids), each of shape (n_queries, k). Missing slots have id -1."""
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float32)
        return self.index.search(q, min(k, len(self)) if len(self) else 0)


class GroupedIndex:
    """One FlatIndex per group; search returns ids into the *original* array."""

    def __init__(self, emb: np.ndarray, groups: list[str] | np.ndarray):
        groups = np.asarray(groups)
        self.dim = emb.shape[1]
        self.ids: dict[str, np.ndarray] = {}
        self.sub: dict[str, FlatIndex] = {}
        for g in sorted(set(groups.tolist())):
            rows = np.flatnonzero(groups == g)
            self.ids[g] = rows
            self.sub[g] = FlatIndex(emb[rows])
        self.all = FlatIndex(emb)  # unfiltered index, for the "no category filter" comparison

    def search(self, q: np.ndarray, k: int, group: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Search within `group` (or across everything if group is None)."""
        if group is None:
            return self.all.search(q, k)
        if group not in self.sub:  # category not stocked in this catalog
            n = np.atleast_2d(q).shape[0]
            return np.zeros((n, 0), np.float32), np.zeros((n, 0), np.int64)
        scores, local = self.sub[group].search(q, k)
        return scores, self.ids[group][local]

    # -- persistence --------------------------------------------------------
    @staticmethod
    def save_embeddings(path: Path, emb: np.ndarray, groups: list[str]) -> None:
        """We persist embeddings + groups (small) and rebuild the index on load:
        a flat index is quick to rebuild, and this avoids FAISS file version issues."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), emb.astype(np.float32))
        path.with_suffix(".groups.json").write_text(json.dumps(list(groups)))

    @classmethod
    def load(cls, path: Path) -> "GroupedIndex":
        emb = np.load(path.with_suffix(".npy"))
        groups = json.loads(path.with_suffix(".groups.json").read_text())
        return cls(emb, groups)
