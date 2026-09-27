param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir
)
$ErrorActionPreference = 'Stop'
$Root = if ($env:SIGN_LANGUAGE_DATA_ROOT) { $env:SIGN_LANGUAGE_DATA_ROOT } else { 'D:\PROJECTS\sign language' }
$destServed = Join-Path $Root 'models\served'
$destBaseline = Join-Path $Root 'models\kaggle_baseline'
New-Item -ItemType Directory -Force -Path $destServed, $destBaseline | Out-Null
$files = @(
    @{ Src = 'model.onnx'; Dst = Join-Path $destServed 'model.onnx' },
    @{ Src = 'label_map.json'; Dst = Join-Path $destServed 'label_map.json' },
    @{ Src = 'best.pt'; Dst = Join-Path $destBaseline 'best.pt' },
    @{ Src = 'bangla_dictionary.json'; Dst = Join-Path $Root 'models\bangla_dictionary.json' }
)
foreach ($f in $files) {
    $srcPath = Join-Path $SourceDir $f.Src
    if (Test-Path $srcPath) {
        Copy-Item -Force $srcPath $f.Dst
        Write-Host "Copied $($f.Src) -> $($f.Dst)"
    }
}
