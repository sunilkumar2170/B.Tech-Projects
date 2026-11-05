Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Video Deepfake Detection - Starting React + Vite Frontend" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location -Path (Join-Path $scriptDir "frontend")
npm run dev
