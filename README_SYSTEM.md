# Avarn ML System - Production Architecture

Komplett ML-system för prediktering av brandlarmstestningspriser. Detta är ett **produktionsklart system**, inte bara en modell.

## 🏗️ Systemarkitektur

```
Data → Feature Engineering → Model → API → Monitoring
```

### Komponenter

1. **Feature Pipeline** (`app/features/`)
   - Versionerad feature engineering
   - Automatisk encoding och scaling
   - Schema-validering

2. **Model Training** (`app/models/`)
   - MLflow integration för versionering
   - Automatisk experiment tracking
   - Model registry

3. **API Layer** (`app/api/`)
   - FastAPI för inference
   - Pydantic schemas för validering
   - OpenAPI dokumentation
   - Batch predictions

4. **Monitoring** (`app/monitoring/`)
   - Prediction logging
   - Performance tracking
   - Error monitoring

5. **Docker** 
   - Containeriserad deployment
   - Health checks
   - Production-ready

## 🚀 Quick Start

### 1. Installera Dependencies

```bash
pip install -r requirements.txt
```

### 2. Träna Modell med MLflow

```bash
cd app/models
python train_with_mlflow.py --data ../../data/training_data.csv
```

### 3. Starta API

```bash
# Lokalt
python main.py

# Eller med uvicorn direkt
uvicorn app.api.main:app --reload
```

### 4. Testa API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
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

### 5. Docker Deployment

```bash
# Bygg image
docker build -t avarn-ml-api .

# Kör container
docker run -p 8000:8000 avarn-ml-api

# Eller med docker-compose
docker-compose up
```

## 📊 API Endpoints

### `GET /`
Root endpoint med systeminfo.

### `GET /health`
Health check för monitoring.

### `POST /predict`
Single prediction.

**Request:**
```json
{
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
```

**Response:**
```json
{
  "predicted_price": 45230.50,
  "confidence_interval_lower": 40707.45,
  "confidence_interval_upper": 49753.55,
  "model_version": "gradient_boosting",
  "feature_pipeline_version": "v1.0",
  "prediction_id": "pred_abc123"
}
```

### `POST /predict/batch`
Batch predictions (max 100 items).

## 🔍 Monitoring

### Prediction Logs

Alla predictions loggas automatiskt till:
- `logs/predictions.csv` - CSV format för analys
- `logs/predictions_YYYY-MM-DD.jsonl` - JSONL per dag

### MLflow Tracking

MLflow UI för experiment tracking:
```bash
mlflow ui --backend-store-uri ./models/mlruns
```

Öppna: http://localhost:5000

## 🏭 Production Deployment

### Render.com

1. Connect GitHub repo
2. Set build command: `docker build -t avarn-ml-api .`
3. Set start command: `docker run -p $PORT:8000 avarn-ml-api`
4. Add environment variables

### Environment Variables

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=./models/mlruns
```

## 📁 Projektstruktur

```
Avarn/
├── app/
│   ├── api/              # FastAPI application
│   │   ├── main.py      # API endpoints
│   │   └── schemas.py   # Pydantic schemas
│   ├── features/        # Feature engineering
│   │   └── feature_pipeline.py
│   ├── models/          # Model management
│   │   ├── model_loader.py
│   │   └── train_with_mlflow.py
│   └── monitoring/      # Logging & monitoring
│       └── logger.py
├── models/              # Saved models
├── data/                # Training data
├── logs/                # Prediction logs
├── scripts/             # Utility scripts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🔧 Development

### Code Quality

```bash
# Format code
black app/

# Lint
ruff check app/
```

### Testing

```bash
# Test API locally
pytest tests/  # (om du lägger till tests)
```

## 📈 MLOps Features

✅ Model versionering (MLflow)
✅ Feature pipeline versionering
✅ Experiment tracking
✅ Prediction logging
✅ Health checks
✅ Docker containerization
✅ API documentation (OpenAPI)
✅ Schema validation (Pydantic)

## 🎯 Nästa Steg

1. **PostgreSQL Integration**
   - Lagra predictions i databas
   - Feature store
   - Historical data

2. **Django Frontend**
   - Admin panel
   - Prediction interface
   - Analytics dashboard

3. **Advanced Monitoring**
   - Drift detection
   - Performance metrics
   - Alerting

4. **CI/CD**
   - GitHub Actions
   - Automated testing
   - Deployment pipeline

## 📝 Notes

- Modellen laddas automatiskt vid startup
- Alla predictions loggas för monitoring
- MLflow tracking för experiment management
- Docker för enkel deployment
- Production-ready error handling

---

**Detta är ett komplett ML-system, inte bara en modell.** 🚀
