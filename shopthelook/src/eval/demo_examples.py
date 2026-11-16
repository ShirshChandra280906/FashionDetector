"""Render real pipeline outputs on held-out photos for the README.

Picks seeded consumer photos from det_test that have >= 2 detected garments.
Saves results/demo/example_<n>.png (annotated photo + top-5 products per garment)
and copies the source photos to app/examples/ for the demo's example gallery.

    python -m src.eval.demo_examples --n 3
"""
from __future__ import annotations

import argparse
import random
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import DF2, RESULTS, ROOT, SEED, SPLITS
from src.pipeline import ShopTheLookPipeline, draw_detections
from src.viz import SERIES, TEXT_2, style


def render(img, result, path, title):
    style()
    items = result.items
    k = max(len(it.matches) for it in items)
    fig = plt.figure(figsize=(3.2 + 1.9 * k, max(3.6, 2.15 * len(items))))
    gs = fig.add_gridspec(len(items), k + 1, width_ratios=[2.6] + [1] * k, wspace=0.08, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(draw_detections(img, [it.detection for it in items]))
    ax.axis("off")
    for r, it in enumerate(items):
        for c, m in enumerate(it.matches):
            a = fig.add_subplot(gs[r, c + 1])
            th = Image.open(m.image_path).convert("RGB")
            th.thumbnail((240, 240))
            a.imshow(th)
            a.set_xticks([]), a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True)
                s.set_color(SERIES[r % len(SERIES)] if c == 0 else "#e4e3df")
            a.set_xlabel(f"{m.article_type}\n{m.colour} · sim {m.score:.3f}", fontsize=7.5, color=TEXT_2)
            if c == 0:
                a.set_title(f"{r + 1}. {it.detection.category.replace('_', ' ')} → top-5 in '{it.detection.group}'",
                            fontsize=9, loc="left", color="#0b0b0b", fontweight="bold")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default="fashionclip")
    args = ap.parse_args()
    items = pd.read_csv(DF2 / "val_items.csv", dtype={"image": str})
    test = set((SPLITS / "df2_det_test.txt").read_text().split())
    multi = items[(items.source == "user") & items.image.isin(test)].groupby("image").size()
    cands = sorted(multi[multi >= 2].index)
    random.Random(SEED + 1).shuffle(cands)

    pipe = ShopTheLookPipeline.load(args.model)
    out = RESULTS / "demo"
    out.mkdir(parents=True, exist_ok=True)
    ex_dir = ROOT / "app" / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for stem in cands:
        src = DF2 / "validation" / "image" / f"{stem}.jpg"
        img = Image.open(src).convert("RGB")
        res = pipe.run(img, k=5)
        if len(res.items) < 2:
            continue
        done += 1
        render(img, res, out / f"example_{done}.png",
               f"Held-out DeepFashion2 photo {stem}: {len(res.items)} garments, {res.timings_ms['total']:.0f} ms ({args.model})")
        shutil.copy(src, ex_dir / f"df2_{stem}.jpg")
        print(f"example_{done}: {stem} -> {[(it.detection.category, round(it.detection.confidence, 2)) for it in res.items]}")
        if done == args.n:
            break


if __name__ == "__main__":
    main()
