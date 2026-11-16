#!/usr/bin/env bash
# One-time setup for the demo: env, weights, catalog, catalog embeddings.
# Downloads ~0.9 GB (catalog) + 24 MB (detector) + ~600 MB per CLIP model.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python scripts/fix_macos_openmp.py          # macOS: share one OpenMP runtime (no-op elsewhere)
.venv/bin/python -m src.download_weights              # pretrained DF2 detector, SHA-256 verified
.venv/bin/python -m src.data.prepare_catalog          # 17,330-product catalog
.venv/bin/python -m src.build_index --models fashionclip
echo "Setup done. Start the demo with: bash run_demo.sh"
