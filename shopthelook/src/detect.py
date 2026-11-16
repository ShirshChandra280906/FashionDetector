"""Garment detection with the pretrained DeepFashion2 YOLOv8s-seg model.

Returns a list of Detection objects. Each carries the DF2 class, its retrieval
group, a confidence score, the box, and (optionally) the instance mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.config import DF2_CLASSES, DF2_TO_GROUP


@dataclass
class Detection:
    category: str  # one of DF2_CLASSES
    group: str  # retrieval group (top, trousers, ...)
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    mask: np.ndarray | None = field(default=None, repr=False)  # HxW bool, optional

    def crop(self, image: Image.Image, pad: float = 0.0) -> Image.Image:
        """Crop the box from `image`, optionally padded by a fraction of its size."""
        x1, y1, x2, y2 = self.box
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        W, H = image.size
        return image.crop((max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(W, int(x2 + dx)), min(H, int(y2 + dy))))


class GarmentDetector:
    """Thin wrapper around an Ultralytics model with deployment thresholds."""

    def __init__(self, weights: str | None = None, conf: float = 0.35, iou: float = 0.5,
                 max_items: int = 6, device: str | None = None):
        from ultralytics import YOLO

        from src.config import best_device
        from src.download_weights import ensure_weights

        self.model = YOLO(weights or ensure_weights())
        self.conf, self.iou, self.max_items = conf, iou, max_items
        self.device = device or best_device()

    def __call__(self, image: Image.Image) -> list[Detection]:
        # agnostic_nms: one garment -> one box. Per-class NMS let the same denim garment come
        # back as both "shorts" and "skirt" (see results/demo/failure_duplicate_boxes_*.png).
        res = self.model.predict(image.convert("RGB"), conf=self.conf, iou=self.iou, device=self.device,
                                 agnostic_nms=True, verbose=False, retina_masks=True)[0]
        dets = []
        boxes = res.boxes
        masks = res.masks.data.cpu().numpy().astype(bool) if res.masks is not None else None
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            name = DF2_CLASSES[cls]  # same 13-class order as the weights (names differ slightly)
            x1, y1, x2, y2 = (int(round(v)) for v in boxes.xyxy[i].tolist())
            dets.append(Detection(name, DF2_TO_GROUP[name], float(boxes.conf[i]), (x1, y1, x2, y2),
                                  masks[i] if masks is not None else None))
        dets.sort(key=lambda d: -d.confidence)
        return dets[: self.max_items]
