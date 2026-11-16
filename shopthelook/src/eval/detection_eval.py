"""Evaluate a trained detector on the held-out det_test split.

Writes to results/detection/:
  * <tag>_metrics.json   - mAP@50, mAP@50-95, precision, recall, speed, environment
  * <tag>_per_class.csv  - the same metrics per category + number of test instances
  * <tag>_per_class.png  - per-class mAP chart

Usage:
    python -m src.eval.detection_eval --weights models/detector_best.pt
"""
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DF2_CLASSES, RESULTS, YOLO_DS
from src.viz import NEUTRAL, SERIES, style


def split_dirs(data_yaml: str, split: str) -> tuple[Path, Path]:
    """(images_dir, labels_dir) for a split defined in an Ultralytics dataset YAML."""
    import yaml

    cfg = yaml.safe_load(Path(data_yaml).read_text())
    img_dir = Path(cfg["path"]) / cfg[split]
    return img_dir, Path(str(img_dir).replace("/images/", "/labels/"))


def instance_counts(labels_dir: Path) -> Counter:
    c = Counter()
    for f in labels_dir.glob("*.txt"):
        for line in f.read_text().split("\n"):
            if line.strip():
                c[int(line.split()[0])] += 1
    return c


def plot_per_class(df: pd.DataFrame, path: Path, title: str) -> None:
    style()
    d = df.sort_values("mAP50", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["mAP50"], height=0.38, color=SERIES[0], label="mAP@50", edgecolor="#fcfcfb", linewidth=1)
    ax.barh(y - 0.2, d["mAP50_95"], height=0.38, color=SERIES[1], label="mAP@50-95", edgecolor="#fcfcfb", linewidth=1)
    ax.set_yticks(y, [f"{c}  (n={n:,})" for c, n in zip(d["category"], d["test_instances"])])
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, d["mAP50"]):
        ax.text(v + 0.01, yi + 0.2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("average precision on det_test")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(YOLO_DS / "df2.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="det_test")
    ap.add_argument("--title", default="Per-class AP on held-out det_test")
    ap.add_argument("--out", default=str(RESULTS / "detection"))
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    # conf=0.001 / iou=0.7 are the standard mAP evaluation settings (not deployment thresholds).
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, conf=0.001,
                  iou=0.7, device=args.device, plots=True, project=str(out / "val_runs"), name=args.tag,
                  exist_ok=True)

    img_dir, lbl_dir = split_dirs(args.data, args.split)
    counts = instance_counts(lbl_dir)
    rows = []
    for i, cls in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append(dict(category=DF2_CLASSES[int(cls)], test_instances=counts[int(cls)],
                         precision=p, recall=r, mAP50=ap50, mAP50_95=ap))
    per_class = pd.DataFrame(rows)
    per_class.to_csv(out / f"{args.tag}_per_class.csv", index=False)

    device = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    metrics = {
        "weights": str(args.weights),
        "split": args.split,
        "images": len(list(img_dir.glob("*.jpg"))),
        "task": model.task,
        "instances": int(sum(counts.values())),
        "imgsz": args.imgsz,
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        **({"mask_mAP50": float(m.seg.map50), "mask_mAP50_95": float(m.seg.map)} if hasattr(m, "seg") else {}),
        "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},
        "env": {"device": device, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "python": platform.python_version()},
    }
    (out / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_per_class(per_class, out / f"{args.tag}_per_class.png",
                   f"{args.title} ({metrics['images']:,} imgs), box "
                   f"mAP@50={metrics['mAP50']:.3f}")
    print(json.dumps(metrics, indent=2))
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()
