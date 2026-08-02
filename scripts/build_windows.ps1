param(
    [string]$DistPath = "dist-win",
    [string]$WorkPath = "build-win"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python -m PyInstaller --noconfirm --clean `
    --distpath $DistPath `
    --workpath $WorkPath `
    (Join-Path $PSScriptRoot "pie_face_windows.spec")

$exe = Join-Path $DistPath "PieFace.exe"
if (-not (Test-Path $exe)) {
    throw "Build completed without producing $exe"
}

Write-Host "Built $((Resolve-Path $exe).Path)"
