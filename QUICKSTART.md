# 🚀 Quick Start Guide

## 1. Installera Dependencies

```bash
pip install -r requirements.txt
```

## 2. Träna Modell (med MLflow)

```bash
# Med MLflow tracking
cd app/models
python train_with_mlflow.py --data ../../data/training_data.csv

# Eller med gamla scriptet (utan MLflow)
cd scripts
python train_model.py --data ../data/training_data.csv
```

## 3. Starta API

```bash
# Enklast
python run_api.py

# Eller
python main.py

# Eller direkt med uvicorn
uvicorn app.api.main:app --reload
```

API:et körs på: http://localhost:8000

## 4. Testa API

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "antal_sektioner": 8,
    "antal_detektorer": 25,
    "antal_larmdon": 15,
    "dörrhållarmagneter": 5,
    "ventilation": 1,
    "stad": "Stockholm",
    "kvartalsvis": 0,
    "månadsvis": 1,
    "årsvis": 0
  }'
```

### API Documentation
Öppna i webbläsare: http://localhost:8000/docs

## 5. Docker (Production)

```bash
# Bygg image
docker build -t avarn-ml-api .

# Kör container
docker run -p 8000:8000 avarn-ml-api

# Eller med docker-compose
docker-compose up
```

## 📊 MLflow UI

För att se experiment tracking:

```bash
mlflow ui --backend-store-uri ./models/mlruns
```

Öppna: http://localhost:5000

## 📁 Viktiga Filer

- `app/api/main.py` - FastAPI application
- `app/models/model_loader.py` - Model loading
- `app/features/feature_pipeline.py` - Feature engineering
- `app/monitoring/logger.py` - Prediction logging
- `Dockerfile` - Docker configuration
- `requirements.txt` - Dependencies

## 🔍 Monitoring

Prediction logs sparas i:
- `logs/predictions.csv`
- `logs/predictions_YYYY-MM-DD.jsonl`

## ✅ Checklista

- [ ] Dependencies installerade
- [ ] Modell tränad
- [ ] API startad
- [ ] Health check fungerar
- [ ] Test prediction gjord
- [ ] Docker testad (valfritt)

---

**Systemet är nu redo för produktion!** 🎉
