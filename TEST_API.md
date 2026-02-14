# Testa API:et

## Metod 1: PowerShell (Rekommenderat för Windows)

Kör PowerShell-scriptet:
```powershell
.\test_api.ps1
```

Eller kör kommandon manuellt:

### Health Check
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

### Single Prediction
```powershell
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

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

## Metod 2: Webbläsare (Enklast för GET)

Öppna i webbläsare:
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000/

I Swagger UI kan du testa alla endpoints direkt!

## Metod 3: curl (Om installerad)

Om du har curl installerad (Windows 10+ har det inbyggt):

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"antal_sektioner\": 8, \"antal_detektorer\": 25, \"antal_larmdon\": 15, \"dörrhållarmagneter\": 5, \"ventilation\": 1, \"stad\": \"Stockholm\", \"kvartalsvis\": 0, \"månadsvis\": 1, \"årsvis\": 0}"
```

## Metod 4: Python Script

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
data = {
    "antal_sektioner": 8,
    "antal_detektorer": 25,
    "antal_larmdon": 15,
    "dörrhållarmagneter": 5,
    "ventilation": 1,
    "stad": "Stockholm",
    "kvartalsvis": 0,
    "månadsvis": 1,
    "årsvis": 0
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Rekommendation

**Använd Swagger UI** - Det är enklast!
1. Starta API:et: `python run_api.py`
2. Öppna: http://localhost:8000/docs
3. Klicka på "Try it out" på valfritt endpoint
4. Fyll i data och klicka "Execute"
