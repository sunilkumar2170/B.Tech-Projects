Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Video Deepfake Detection - Starting FastAPI Backend Server" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
$env:TORCHDYNAMO_DISABLE="1"
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location -Path (Join-Path $scriptDir "backend")
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
