"""Pipeline tests using fake detector/embedder (no model downloads)."""
import numpy as np
import pandas as pd
from PIL import Image

import src  # noqa: F401
from src.detect import Detection
from src.embed import l2_normalize
from src.pipeline import ShopTheLookPipeline, draw_detections
from src.retrieve import CatalogRetriever


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def __call__(self, image):
        return self.dets


class MeanColourEmbedder:
    """Embeds a crop as its mean RGB colour (deterministic, enough to test wiring)."""
    dim = 3

    def embed(self, images):
        return l2_normalize(np.stack([np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0) + 1e-3 for im in images]))


def _catalog():
    # products: red top, blue top, blue trousers, red trousers
    colours = np.array([[255, 0, 0], [0, 0, 255], [0, 0, 255], [255, 0, 0]], np.float32) + 1e-3
    cat = pd.DataFrame({"id": [1, 2, 3, 4], "group": ["top", "top", "trousers", "trousers"],
                        "articleType": ["Tshirts", "Tshirts", "Jeans", "Trousers"],
                        "baseColour": ["Red", "Blue", "Blue", "Red"],
                        "productDisplayName": ["red tee", "blue tee", "blue jeans", "red trousers"],
                        "image_path": [f"images/{i}.jpg" for i in range(1, 5)]})
    return CatalogRetriever(cat, l2_normalize(colours))


def _outfit():
    # top half red (shirt), bottom half blue (trousers)
    img = Image.new("RGB", (100, 200), (255, 0, 0))
    img.paste((0, 0, 255), (0, 100, 100, 200))
    return img


def test_pipeline_matches_each_garment_within_its_category():
    dets = [Detection("short_sleeve_top", "top", 0.9, (0, 0, 100, 100)),
            Detection("trousers", "trousers", 0.8, (0, 100, 100, 200))]
    pipe = ShopTheLookPipeline(FakeDetector(dets), MeanColourEmbedder(), _catalog(), crop_pad=0)
    res = pipe.run(_outfit(), k=2)
    assert len(res.items) == 2
    top, bottom = res.items
    assert [m.product_id for m in top.matches] == [1, 2]  # red tee ranked first, only tops
    assert [m.product_id for m in bottom.matches] == [3, 4]  # blue jeans first, only trousers
    assert top.matches[0].score > top.matches[1].score
    assert set(res.timings_ms) == {"detect", "embed", "search", "total"}


def test_pipeline_without_category_filter_can_cross_categories():
    dets = [Detection("trousers", "trousers", 0.8, (0, 100, 100, 200))]  # blue region
    pipe = ShopTheLookPipeline(FakeDetector(dets), MeanColourEmbedder(), _catalog(), crop_pad=0)
    groups = {m.group for m in pipe.run(_outfit(), k=4, filter_by_category=False).items[0].matches}
    assert groups == {"top", "trousers"}  # unfiltered search returns a blue *top* too


def test_pipeline_handles_no_detections():
    pipe = ShopTheLookPipeline(FakeDetector([]), MeanColourEmbedder(), _catalog())
    res = pipe.run(_outfit())
    assert res.items == [] and res.timings_ms["total"] >= 0


def test_detection_crop_padding_is_clamped_to_image():
    d = Detection("skirt", "skirt", 0.5, (0, 0, 50, 50))
    assert d.crop(Image.new("RGB", (60, 60)), pad=0.5).size == (60, 60)


def test_draw_detections_returns_same_size_copy():
    img = _outfit()
    out = draw_detections(img, [Detection("trousers", "trousers", 0.8, (0, 100, 100, 200))])
    assert out.size == img.size and out is not img
