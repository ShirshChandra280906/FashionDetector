"""End-to-end Shop the Look: photo -> garments -> similar catalog products.

    pipe = ShopTheLookPipeline.load("fashionclip")
    result = pipe.run(Image.open("outfit.jpg"), k=5)
    for item in result.items:
        print(item.detection.category, [m.name for m in item.matches])

Components are injected, so tests can pass lightweight fakes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from src.detect import Detection
from src.retrieve import Match
from src.viz import SERIES


@dataclass
class ItemResult:
    detection: Detection
    matches: list[Match]


@dataclass
class LookResult:
    items: list[ItemResult]
    timings_ms: dict[str, float] = field(default_factory=dict)


class ShopTheLookPipeline:
    def __init__(self, detector, embedder, retriever, crop_pad: float = 0.02):
        self.detector, self.embedder, self.retriever = detector, embedder, retriever
        self.crop_pad = crop_pad

    @classmethod
    def load(cls, embedder_name: str = "fashionclip", **detector_kwargs) -> "ShopTheLookPipeline":
        from src.detect import GarmentDetector
        from src.embed import get_embedder
        from src.retrieve import CatalogRetriever

        return cls(GarmentDetector(**detector_kwargs), get_embedder(embedder_name), CatalogRetriever.load(embedder_name))

    def run(self, image: Image.Image, k: int = 5, filter_by_category: bool = True) -> LookResult:
        image = image.convert("RGB")
        t0 = time.perf_counter()
        dets = self.detector(image)
        t1 = time.perf_counter()
        crops = [d.crop(image, pad=self.crop_pad) for d in dets]
        embs = self.embedder.embed(crops) if crops else []
        t2 = time.perf_counter()
        items = [ItemResult(d, self.retriever.search(e, d.group if filter_by_category else None, k))
                 for d, e in zip(dets, embs)]
        t3 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1000  # noqa: E731
        return LookResult(items, {"detect": ms(t0, t1), "embed": ms(t1, t2), "search": ms(t2, t3), "total": ms(t0, t3)})


def draw_detections(image: Image.Image, dets: list[Detection]) -> Image.Image:
    """Return a copy of `image` with numbered, labelled boxes (numbers match the result cards)."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    lw = max(2, round(min(out.size) / 200))
    font = ImageFont.load_default(size=max(12, round(min(out.size) / 32)))
    for i, d in enumerate(dets):
        color = SERIES[i % len(SERIES)]
        draw.rectangle(d.box, outline=color, width=lw)
        label = f" {i + 1}. {d.category.replace('_', ' ')} {d.confidence:.2f} "
        th = font.size + 6
        x1, y1 = d.box[0], max(0, d.box[1] - th - lw)
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle((tb[0], tb[1] - 3, tb[2], tb[3] + 3), fill=color)
        draw.text((x1, y1), label, fill="white", font=font)
    return out
