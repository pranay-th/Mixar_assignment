#!/usr/bin/env bash
# Cross-platform runner (Linux + macOS). Prefer this if unsure.

set -euo pipefail

PYTHON_BIN=python3

if ! command -v $PYTHON_BIN >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Install via Homebrew: brew install python"
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "[INFO] Creating virtual environment..."
  $PYTHON_BIN -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "[INFO] Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p results

echo "[INFO] Running pipeline..."
$PYTHON_BIN src/pipeline.py --input_dir meshes --out_dir results --clusters 4 --base_bins 1024 --k_density 16 --alpha 1.0

echo "[INFO] Done. Check results/ and results/run.log"
deactivate
