# Detection: model choice, safety check, evaluation

## Decision: use pretrained DeepFashion2 weights instead of training our own

The original plan was to fine-tune YOLO11n on a 15k-image DeepFashion2 subset (the notebook
`notebooks/01_train_detector_colab.ipynb` and `src/train_detector.py` implement this and were
smoke-tested). To move faster, we used a publicly available detector already trained on
DeepFashion2 and **evaluated it ourselves** on a held-out split. We did **not** train or
fine-tune the detector, and the README and resume bullets say so.

| | |
|---|---|
| Model | `deepfashion2_yolov8s-seg.pt`: YOLOv8s instance segmentation, 13 DF2 classes, 11.8M parameters |
| Source | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) on Hugging Face |
| SHA-256 | `c319c15e86443ea61f70b40620909d737c47ff2a39381536503ac5c54772f8ac` (pinned in `src/download_weights.py`) |
| Repo license | Apache-2.0 |
| Checkpoint metadata | Ultralytics 8.3.35, trained 100 epochs at 640 px, batch 32. The checkpoint itself carries Ultralytics' **AGPL-3.0** notice |
| Author's reported numbers | box mAP@50 0.849, mAP@50-95 0.763 (split not stated) |
| Licensing takeaway | Fine for an open, non-commercial portfolio. A commercial product would need a detector trained on owned data and an Ultralytics enterprise license (or a permissively licensed architecture) |

## Safety check of the weights file

`.pt` files are Python pickles, and loading one can execute code. Hugging Face's scanner labels
this file **"unsafe"**. Before loading it we inspected the pickle **statically** (Python
`pickletools`, which parses opcodes without unpickling):

* Imports referenced: `torch.nn` layers (Conv2d, BatchNorm2d, SiLU, …), torch tensor-rebuild helpers,
  `collections.OrderedDict`, Ultralytics model classes (`SegmentationModel`, `C2f`, `Detect`, `Segment`, …),
  and two builtins: `set` and `getattr`.
* The only `getattr` call is `getattr(ultralytics.nn.modules.head.Detect, "forward")`, the standard
  way Ultralytics' `Segment` head reuses the detection forward pass. That builtin is what triggers
  the "unsafe" label.
* No `os`, `subprocess`, `eval`, `exec`, network or file access.

Conclusion: false positive. The project owner approved loading it. `src/download_weights.py`
refuses any file whose SHA-256 differs from the one we inspected.

## Evaluation

**Split:** `det_test`, 6,005 DF2-validation photos (9,744 garments) from 400 randomly chosen
`pair_id`s (`splits/df2_det_test.txt`). Settings: 640 px, conf 0.001, IoU 0.7 (standard mAP settings).
Run on an Apple M4 (MPS).

**Two ground-truth box definitions.** Ultralytics' segmentation validator derives each
ground-truth box from the polygon's extent. DF2 also provides official bounding boxes, and the two
differ (on det_test: mean IoU 0.924, median 0.943, and only 43% of pairs have IoU ≥ 0.95). We report both:

| ground truth | box mAP@50 | box mAP@50-95 | script |
|---|---:|---:|---|
| official DF2 bounding boxes (headline) | 0.842 | 0.636 | `src/eval/detection_eval_official_boxes.py` |
| polygon-extent boxes (+ mask metrics, per-class) | 0.843 | 0.744 | `src/eval/detection_eval.py` |

Exact values: `results/detection/det_test_official_boxes_metrics.json` and `det_test_metrics.json`.

mAP@50 is almost identical under both. mAP@50-95 is lower against the official boxes because
the strict IoU thresholds punish the systematic offset between the two box definitions.

**Train/test contamination check.** The model card doesn't say which DF2 splits were used for
training. Our test photos come from DF2-validation. As evidence, we evaluated the same model on 2,000
images from DF2-*train* (`splits/df2_train_eval_sample.txt`). It scores clearly higher there (box mAP@50
0.941 vs 0.843 on det_test, polygon-extent boxes). That gap is what you'd expect if training used
DF2-train and validation was held out. It is **evidence, not proof**: if validation images had
been in training, the two scores would be close.

**Unexplained:** mask mAP is *lower* on the train sample (0.754) than on det_test (0.820), even though box
mAP is higher. We suspect our conversion of multi-part DF2 polygons into single YOLO polygons,
but haven't confirmed it. The pipeline only uses boxes, so this doesn't affect the product numbers.

**Weak classes:** `short_sleeve_outwear` (44 test boxes, AP@50 ≈ 0.50) and `sling` (35 test boxes,
AP@50 ≈ 0.68) are rare in DF2. With so few test boxes, their per-class AP is noisy.
