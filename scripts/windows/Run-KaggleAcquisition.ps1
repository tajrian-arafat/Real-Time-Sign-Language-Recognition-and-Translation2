# Download + verify Kaggle asl-signs into SIGN_LANGUAGE_DATA_ROOT (default D:\PROJECTS\sign language).
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $env:SIGN_LANGUAGE_DATA_ROOT) {
    $env:SIGN_LANGUAGE_DATA_ROOT = 'D:\PROJECTS\sign language'
}
& (Join-Path $PSScriptRoot 'Ensure-DataRoot.ps1')
Set-Location $RepoRoot
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $py) { throw 'Python not found on PATH' }
& $py data/scripts/download_kaggle_islr.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Unzip if zip present and parquets missing
$raw = Join-Path $env:SIGN_LANGUAGE_DATA_ROOT 'raw\kaggle_asl_signs'
$zip = Join-Path $raw 'asl-signs.zip'
$parquetCount = (Get-ChildItem -Path $raw -Recurse -Filter '*.parquet' -ErrorAction SilentlyContinue | Measure-Object).Count
if ((Test-Path $zip) -and $parquetCount -lt 90000) {
    Write-Host "Extracting $zip (this takes a while)..."
    Expand-Archive -Path $zip -DestinationPath $raw -Force
}
& $py scripts/run_data_acquisition.py
exit $LASTEXITCODE
