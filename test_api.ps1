# PowerShell script för att testa API:et

Write-Host "Testing Avarn ML API" -ForegroundColor Green
Write-Host "====================" -ForegroundColor Green

# 1. Health Check
Write-Host "`n1. Health Check:" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# 2. Single Prediction
Write-Host "`n2. Making Prediction:" -ForegroundColor Yellow
$body = @{
    antal_sektioner = 8
    antal_detektorer = 25
    antal_larmdon = 15
    dörrhållarmagneter = 5
    ventilation = 1
    stad = "Stockholm"
    kvartalsvis = 0
    månadsvis = 1
    årsvis = 0
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Prediction successful!" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    if ($_.ErrorDetails.Message) {
        Write-Host $_.ErrorDetails.Message -ForegroundColor Red
    }
}

Write-Host "`nDone! Open http://localhost:8000/docs for Swagger UI" -ForegroundColor Cyan
