# Creates canonical Windows data root and subdirs (no downloads).
$ErrorActionPreference = 'Stop'
$Root = if ($env:SIGN_LANGUAGE_DATA_ROOT) { $env:SIGN_LANGUAGE_DATA_ROOT } else { 'D:\PROJECTS\sign language' }
$dirs = @(
    $Root,
    (Join-Path $Root 'raw'),
    (Join-Path $Root 'raw\kaggle_asl_signs'),
    (Join-Path $Root 'processed'),
    (Join-Path $Root 'processed\kaggle_asl_signs'),
    (Join-Path $Root 'models\served')
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Write-Host "Data root ready: $Root"
$env:SIGN_LANGUAGE_DATA_ROOT = $Root
