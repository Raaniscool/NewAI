# PowerShell Build Script for ScratchLM Windows Executable
# Run this script in PowerShell to package ScratchLM into a portable ScratchLM.exe

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Building ScratchLM Windows Executable (ScratchLM.exe)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Ensure PyInstaller is installed
Write-Host "`n1. Verifying PyInstaller installation..." -ForegroundColor Yellow
python -m pip install pyinstaller --quiet

# Clean previous build artifacts
Write-Host "`n2. Cleaning previous build directories..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

# Execute PyInstaller build using ScratchLM.spec
Write-Host "`n3. Running PyInstaller compiler..." -ForegroundColor Yellow
python -m PyInstaller --clean ScratchLM.spec

# Verify build output
if (Test-Path "dist\ScratchLM.exe") {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host " SUCCESS! Executable created successfully:" -ForegroundColor Green
    Write-Host " Location: dist\ScratchLM.exe" -ForegroundColor Green
    Write-Host " Workflow: Double-click dist\ScratchLM.exe to launch GUI!" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "`n==========================================================" -ForegroundColor Red
    Write-Host " Build failed or dist\ScratchLM.exe was not found." -ForegroundColor Red
    Write-Host " Please inspect PyInstaller logs above for details." -ForegroundColor Red
    Write-Host "==========================================================" -ForegroundColor Red
}
