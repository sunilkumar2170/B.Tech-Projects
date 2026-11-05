Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Video Deepfake Detection - Launching Fullstack Platform" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $root "run_backend.ps1") -WorkingDirectory $root
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $root "run_frontend.ps1") -WorkingDirectory $root

Write-Host "Backend API: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Frontend UI: http://localhost:5173" -ForegroundColor Green
