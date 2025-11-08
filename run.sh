#!/usr/bin/env bash
# Mixar Assignment - Linux Runner (no venv)

set -euo pipefail

PYTHON_BIN=python3

if ! command -v $PYTHON_BIN >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Please install Python 3."
  exit 1
fi

echo "[INFO] Installing required packages globally/user site..."
pip install --user --upgrade pip
pip install --user -r requirements.txt

mkdir -p results

echo "[INFO] Running Mixar pipeline..."
$PYTHON_BIN src/pipeline.py \
  --input_dir meshes \
  --out_dir results \
  --clusters 4 \
  --base_bins 1024 \
  --k_density 16 \
  --alpha 1.0

echo "[INFO] Done. Check results/ and results/run.log"
