# PowerShell script to start MLflow UI
# Usage: .\start_mlflow_ui.ps1

Write-Host "Starting MLflow UI..." -ForegroundColor Green
Write-Host "MLruns directory: .\models\mlruns" -ForegroundColor Cyan
Write-Host ""
Write-Host "Open in browser: http://localhost:5000" -ForegroundColor Yellow
Write-Host "Press CTRL+C to stop" -ForegroundColor Yellow
Write-Host ""

python -m mlflow ui --backend-store-uri ./models/mlruns --host 127.0.0.1 --port 5000
