"""Box metrics of the seg model against the OFFICIAL DeepFashion2 bounding boxes.

The main eval (detection_eval.py) uses Ultralytics' segmentation validator, which
derives each ground-truth box from the polygon extent. DF2's official boxes
differ (median IoU 0.943 with the polygon extent on det_test), so here we run the
*detection* validator on the box-label dataset. The seg head's mask coefficients
are dropped before NMS.

    python -m src.eval.detection_eval_official_boxes
"""
import json

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from src.config import RESULTS, YOLO_DS


class BoxOnlyValidator(DetectionValidator):
    def postprocess(self, preds):
        # Seg output: ((B, 4+nc+32, N), protos), extras. Keep the decoded boxes + class scores only.
        while isinstance(preds, (list, tuple)):
            preds = preds[0]
        return super().postprocess(preds[:, : 4 + self.nc])


def main(weights="models/deepfashion2_yolov8s-seg.pt", device=None):
    m = YOLO(weights)
    v = BoxOnlyValidator(args=dict(model=weights, data=str(YOLO_DS / "df2.yaml"), split="test", imgsz=640, batch=16,
                                   conf=0.001, iou=0.7, device=device, plots=False,
                                   project=str(RESULTS / "detection" / "val_runs"), name="det_test_official_boxes",
                                   exist_ok=True))
    v(model=m.model)
    mt = v.metrics
    out = {"note": "box metrics vs OFFICIAL DF2 bounding boxes (det_test, 6,005 images)",
           "mAP50": float(mt.box.map50), "mAP50_95": float(mt.box.map),
           "precision": float(mt.box.mp), "recall": float(mt.box.mr)}
    (RESULTS / "detection" / "det_test_official_boxes_metrics.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
