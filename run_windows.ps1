# Mixar Assignment - Windows Runner (no venv)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_windows.ps1

Write-Host "[INFO] Mixar pipeline (Windows, no venv)" -ForegroundColor Cyan

# --- Detect Python ---
$py = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Host "[ERROR] Python not found. Install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
$pyexe = $py.Path

# --- Install dependencies globally/user site ---
Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip | Out-Null
pip install -r requirements.txt | Out-Null

# --- Ensure results folder exists ---
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

# --- Run pipeline ---
Write-Host "[INFO] Running Mixar pipeline..." -ForegroundColor Cyan
& $pyexe src\pipeline.py --input_dir meshes --out_dir results --clusters 4 --base_bins 1024 --k_density 16 --alpha 1.0

Write-Host "[INFO] Done. Outputs and logs saved in results\run.log" -ForegroundColor Green
