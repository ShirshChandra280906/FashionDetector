#!/usr/bin/env bash
# Start the ShopTheLook demo at http://127.0.0.1:7860 (runs setup first if needed).
set -euo pipefail
cd "$(dirname "$0")"
[ -f artifacts/catalog_${SHOPTHELOOK_EMBEDDER:-fashionclip}.npy ] || bash setup.sh
exec .venv/bin/python app/app.py
