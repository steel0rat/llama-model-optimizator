#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Installing project with build extras..."
python -m pip install --upgrade pip
python -m pip install -e ".[gui,build]"

Write-Host "==> Building Windows executable..."
python -m PyInstaller --noconfirm --clean pyinstaller/moe-optimizator.spec

$Exe = Join-Path $Root "dist" "moe-optimizator.exe"
if (-not (Test-Path $Exe)) {
    throw "Build failed: $Exe not found"
}

Write-Host "==> Done: $Exe"
