"""End-to-end latency of the pipeline on this machine.

Times detect -> crop -> embed -> category-filtered search for N consumer photos
from det_test, after a warm-up. Image decoding from disk is excluded, which matches
a server that already holds the uploaded bytes in memory.

Usage:
    python -m src.eval.latency --model fashionclip --n 200
"""
from __future__ import annotations

import argparse
import json
import platform
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import DF2, RESULTS, SEED, SPLITS, set_seed
from src.pipeline import ShopTheLookPipeline
from src.viz import SERIES, TEXT_2, style


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fashionclip")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    set_seed()

    import torch

    items = pd.read_csv(DF2 / "val_items.csv", dtype={"image": str})
    test = set(SPLITS.joinpath("df2_det_test.txt").read_text().split())
    user_imgs = sorted(set(items[(items.source == "user") & items.image.isin(test)].image))
    stems = random.Random(SEED).sample(user_imgs, args.n + args.warmup)
    images = [Image.open(DF2 / "validation" / "image" / f"{s}.jpg").convert("RGB") for s in stems]

    pipe = ShopTheLookPipeline.load(args.model)
    for im in images[: args.warmup]:
        pipe.run(im)
    rows = []
    for s, im in zip(stems[args.warmup :], images[args.warmup :]):
        r = pipe.run(im, k=5)
        rows.append({"image": s, "n_items": len(r.items), "width": im.width, "height": im.height, **r.timings_ms})
    df = pd.DataFrame(rows)
    out = RESULTS / "latency"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"latency_{args.model}_per_image.csv", index=False)

    summary = {
        "model": args.model, "n_images": len(df), "warmup": args.warmup,
        "mean_items_per_image": float(df.n_items.mean()),
        **{f"{stage}_ms": {"median": float(df[stage].median()), "p95": float(np.percentile(df[stage], 95)),
                           "mean": float(df[stage].mean())} for stage in ["detect", "embed", "search", "total"]},
        "env": {"machine": platform.machine(), "processor": platform.processor(), "os": platform.platform(),
                "device": pipe.detector.device, "torch": torch.__version__},
    }
    (out / f"latency_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(df.total, bins=30, color=SERIES[0], edgecolor="#fcfcfb", linewidth=1)
    for q, lab in [(summary["total_ms"]["median"], "median"), (summary["total_ms"]["p95"], "p95")]:
        ax.axvline(q, color=TEXT_2, linestyle="--", linewidth=1)
        ax.text(q, ax.get_ylim()[1] * 0.95, f" {lab} {q:.0f} ms", color=TEXT_2, fontsize=9, va="top")
    ax.set_xlabel("end-to-end time per photo (ms)")
    ax.set_ylabel("photos")
    ax.set_title(f"Pipeline latency, {len(df)} photos ({summary['env']['device']}, {args.model})")
    fig.tight_layout()
    fig.savefig(out / f"latency_{args.model}.png")


if __name__ == "__main__":
    main()
