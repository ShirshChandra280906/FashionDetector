"""Fine-tune a YOLO detector on the DeepFashion2 subset.

Designed to survive Colab disconnects: if <project>/<name>/weights/last.pt
exists, training resumes from it instead of starting over.

Usage (Colab, T4):
    python -m src.train_detector --project /content/drive/MyDrive/shopthelook/runs
Local smoke test (a few minutes on a laptop):
    python -m src.train_detector --epochs 1 --fraction 0.02 --name smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ROOT, SEED, YOLO_DS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained starting weights")
    ap.add_argument("--data", default=str(YOLO_DS / "df2.yaml"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=10, help="early-stop after N epochs without val improvement")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train images (smoke tests)")
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers")
    ap.add_argument("--device", default=None, help="0 for first GPU, 'mps', or 'cpu' (auto if omitted)")
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--name", default="yolo11n_df2_15k")
    args = ap.parse_args()

    from ultralytics import YOLO

    last = Path(args.project) / args.name / "weights" / "last.pt"
    if last.exists():
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        fraction=args.fraction,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=SEED,
        deterministic=True,
        plots=True,
        # Defaults are kept for augmentation. Horizontal flips are fine for clothing,
        # and none of our classes depend on left/right.
    )


if __name__ == "__main__":
    main()
