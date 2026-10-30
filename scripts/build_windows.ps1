<#
.SYNOPSIS
  Build LiveCaptions.exe on Windows 10/11.

.DESCRIPTION
  Creates a virtual environment, installs the runtime dependencies plus
  PyInstaller, and produces dist\LiveCaptions.exe (single file).

.PARAMETER OneDir
  Produce a folder build (dist\LiveCaptions\) instead of a single exe. Starts
  faster because nothing has to be unpacked to %TEMP% on launch.

.PARAMETER WithArgos
  Bundle Argos Translate (offline translation into ~40 languages). Pulls in
  PyTorch, so the result is several hundred MB larger.

.PARAMETER Console
  Keep a console window attached (useful to see logs / use --no-gui).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
#>
param(
    [switch]$OneDir,
    [switch]$WithArgos,
    [switch]$Console
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "Python 3.9+ is required: https://www.python.org/downloads/windows/ (tick 'Add to PATH')" }

if (-not (Test-Path ".venv-build")) {
    Write-Host ">> creating build venv"
    python -m venv .venv-build
}
$py = Join-Path $root ".venv-build\Scripts\python.exe"

Write-Host ">> installing dependencies"
& $py -m pip install --upgrade pip wheel | Out-Null
& $py -m pip install -r requirements.txt pyinstaller pillow
if ($WithArgos) {
    & $py -m pip install --extra-index-url https://download.pytorch.org/whl/cpu argostranslate
}

Write-Host ">> generating icon"
& $py scripts\make_icon.py

Write-Host ">> running tests"
& $py -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

$specArgs = @()
if ($OneDir)    { $specArgs += "--onedir" }
if ($WithArgos) { $specArgs += "--with-argos" }
if ($Console)   { $specArgs += "--console" }

Write-Host ">> building with PyInstaller"
if (Test-Path build) { Remove-Item -Recurse -Force build }
& $py -m PyInstaller --noconfirm --clean live_captions.spec -- @specArgs
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

if ($OneDir) {
    Write-Host "`nDone: dist\LiveCaptions\LiveCaptions.exe"
} else {
    $exe = Get-Item dist\LiveCaptions.exe
    Write-Host ("`nDone: {0}  ({1:N0} MB)" -f $exe.FullName, ($exe.Length / 1MB))
}
