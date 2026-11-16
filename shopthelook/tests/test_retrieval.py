"""Tests for the index and retrieval helpers (no datasets or model downloads needed)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import torch  # noqa: F401  (load torch alongside faiss, as the app does)
from src.embed import ColorHistogramEmbedder, l2_normalize, pad_to_square
from src.index import FlatIndex, GroupedIndex
from src.retrieve import CatalogRetriever, first_positive_rank, recall_at_k


def _random_unit(n, d, seed=0):
    return l2_normalize(np.random.default_rng(seed).normal(size=(n, d)))


def test_flat_index_matches_numpy_exact_search():
    """FAISS top-k must equal brute-force cosine ranking (also a smoke test that
    faiss works next to torch after scripts/fix_macos_openmp.py)."""
    gallery, queries = _random_unit(5000, 64, 1), _random_unit(200, 64, 2)
    scores, ids = FlatIndex(gallery).search(queries, 10)
    sims = queries @ gallery.T
    expected = np.argsort(-sims, axis=1)[:, :10]
    np.testing.assert_array_equal(ids, expected)
    np.testing.assert_allclose(scores, np.take_along_axis(sims, expected, 1), atol=1e-5)


def test_grouped_index_only_returns_requested_group_and_global_ids():
    emb = _random_unit(300, 16, 3)
    groups = np.array(["top", "dress", "skirt"] * 100)
    gi = GroupedIndex(emb, groups)
    q = _random_unit(5, 16, 4)
    scores, ids = gi.search(q, 7, group="dress")
    assert ids.shape == (5, 7)
    assert set(groups[ids.ravel()]) == {"dress"}
    # ids refer to the ORIGINAL rows, so recomputing the score must agree
    np.testing.assert_allclose(scores, np.einsum("qd,qkd->qk", q, emb[ids]), atol=1e-5)
    # scores are sorted best-first
    assert np.all(np.diff(scores, axis=1) <= 1e-6)


def test_grouped_index_unknown_group_returns_empty():
    gi = GroupedIndex(_random_unit(10, 8), ["top"] * 10)
    scores, ids = gi.search(_random_unit(2, 8), 5, group="outerwear")
    assert scores.shape == (2, 0) and ids.shape == (2, 0)


def test_first_positive_rank_and_recall():
    assert first_positive_rank(np.array([5, 3, 9]), {9}) == 3
    assert first_positive_rank(np.array([5, 3, 9]), {1}) == 0  # not found
    r = recall_at_k(np.array([1, 3, 0, 12]), ks=(1, 5, 10))
    assert r == {"R@1": 0.25, "R@5": 0.5, "R@10": 0.5}


def test_catalog_retriever_filters_and_ranks(tmp_path):
    emb = _random_unit(6, 8, 5)
    cat = pd.DataFrame({
        "id": range(6), "group": ["top", "top", "dress", "dress", "top", "skirt"],
        "articleType": "x", "baseColour": "Blue", "productDisplayName": [f"p{i}" for i in range(6)],
        "image_path": [f"images/{i}.jpg" for i in range(6)],
    })
    r = CatalogRetriever(cat, emb, image_root=tmp_path)
    # querying with product 0's own vector must return product 0 first, score ~1
    res = r.search(emb[0], group="top", k=5)
    assert [m.group for m in res] == ["top"] * 3  # only 3 tops exist
    assert res[0].product_id == 0 and res[0].score == pytest.approx(1.0, abs=1e-5)


def test_color_histogram_prefers_same_colour():
    red, red2, blue = (Image.new("RGB", (40, 60), c) for c in [(220, 20, 20), (200, 30, 30), (20, 20, 220)])
    e = ColorHistogramEmbedder().embed([red, red2, blue])
    assert e[0] @ e[1] > e[0] @ e[2]
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)


def test_pad_to_square_keeps_whole_image():
    sq = pad_to_square(Image.new("RGB", (30, 90), (0, 0, 0)))
    assert sq.size == (90, 90)
    assert sq.getpixel((0, 45)) == (255, 255, 255)  # padded margin
    assert sq.getpixel((45, 45)) == (0, 0, 0)  # original content centred
