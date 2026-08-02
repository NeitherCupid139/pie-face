param(
    [string]$DistPath = "dist-win",
    [string]$WorkPath = "build-win"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}
$spec = Join-Path $root "pie_face_windows.spec"
$dist = Join-Path $root $DistPath
$work = Join-Path $root $WorkPath

& $python -m PyInstaller --noconfirm --clean `
    --distpath $dist `
    --workpath $work `
    $spec

$exe = Join-Path $dist "PieFace.exe"
if (-not (Test-Path $exe)) {
    throw "Build completed without producing $exe"
}

Write-Host "Built $((Resolve-Path $exe).Path)"
