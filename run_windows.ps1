# run_windows.ps1 - PowerShell runner (Windows 10/11)
param()

Write-Host "[INFO] Mixar pipeline (Windows)" -ForegroundColor Cyan

# find python
$py = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $py) {
    $py = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $py) {
    Write-Host "[ERROR] Python not found. Install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

$pyexe = $py.Path

if (-not (Test-Path -Path ".\venv")) {
    Write-Host "[INFO] Creating virtual environment..." -ForegroundColor Yellow
    & $pyexe -m venv venv
}

# Activate venv
$activate = Join-Path -Path "venv" -ChildPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "[ERROR] Activation script not found: $activate" -ForegroundColor Red
    exit 1
}
. $activate

Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip | Out-Null
pip install -r requirements.txt | Out-Null

if (-not (Test-Path -Path "results")) { New-Item -ItemType Directory -Path "results" | Out-Null }

Write-Host "[INFO] Running pipeline..." -ForegroundColor Cyan
& $pyexe src\pipeline.py --input_dir meshes --out_dir results --clusters 4 --base_bins 1024 --k_density 16 --alpha 1.0

Write-Host "[INFO] Done. See results\run.log" -ForegroundColor Green
deactivate
