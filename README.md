# Avarn Fire Alarm Testing Price Prediction System

Production-ready ML system for predicting fire alarm testing prices. Built with FastAPI, scikit-learn, MLflow, and Streamlit.

## 🎯 Overview

This is a complete ML system that includes:
- **ML Pipeline**: Feature engineering with OneHotEncoder, model training with MLflow
- **REST API**: FastAPI-based inference API with conformal prediction intervals
- **Web UI**: Streamlit interface for interactive predictions
- **Monitoring**: Prediction logging and performance tracking
- **MLOps**: Model versioning, experiment tracking, and deployment-ready architecture

## 🏗️ Architecture

```
Avarn/
├── app/                    # Production application code
│   ├── api/                # FastAPI REST API
│   │   ├── main.py        # API endpoints (/predict, /health)
│   │   └── schemas.py     # Pydantic request/response models
│   ├── features/          # Feature engineering pipeline
│   │   └── feature_pipeline.py  # OneHotEncoder + ColumnTransformer
│   ├── models/            # Model management
│   │   ├── model_loader.py      # Production model loader
│   │   └── train_with_mlflow.py # Training with MLflow tracking
│   ├── monitoring/        # Prediction logging
│   │   └── logger.py      # JSONL prediction logs
│   └── ui/                # Streamlit web interface
│       └── streamlit_app.py
├── models/                # Saved models and artifacts
│   ├── best_model.pkl     # Production model
│   ├── preprocessor.pkl   # Feature pipeline
│   └── mlruns/            # MLflow experiment tracking
├── data/                  # Training data
│   └── training_data.csv
├── tests/                 # Pytest test suite
│   ├── test_schemas.py    # Schema validation tests
│   ├── test_api_health.py # Health endpoint tests
│   └── test_api_predict.py # Prediction endpoint tests
└── scripts/               # Utility scripts (data generation, etc.)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
python app/models/train_with_mlflow.py
```

This will:
- Load training data from `data/training_data.csv`
- Train multiple models (Gradient Boosting, Random Forest, Ridge)
- Track experiments in MLflow
- Save best model to `models/best_model.pkl`
- Calculate conformal prediction intervals from test set residuals

### 3. Start API Server

```bash
python run_api.py
```

API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 4. Start Web UI (Optional)

```bash
python run_streamlit.py
```

Streamlit UI will be available at `http://localhost:8501`

## 📡 API Usage

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

**Response:**
```json
{
  "predicted_price": 45230.50,
  "confidence_interval_lower": 40000.00,
  "confidence_interval_upper": 50000.00,
  "model_version": "gradient_boosting",
  "feature_pipeline_version": "v2.0",
  "prediction_id": "pred_1234567890"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
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
    ]
  }'
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Tests cover:
- Schema validation (Pydantic models)
- API endpoints (/health, /predict)
- Error handling and edge cases

## 🔧 Key Features

### Feature Engineering
- **OneHotEncoder** for categorical features (not LabelEncoder!)
- **ColumnTransformer** for proper feature preprocessing
- Handles encoding issues with Swedish characters

### Model Training
- Multiple algorithms: Gradient Boosting, Random Forest, Ridge
- **MLflow** for experiment tracking and model versioning
- Automatic model selection based on test RMSE
- Residual-based conformal prediction intervals

### Production API
- **FastAPI** with automatic OpenAPI documentation
- Pydantic schema validation
- Conformal prediction intervals (calibrated on test set)
- Prediction logging to JSONL files
- Health check endpoint

### Confidence Intervals
- **Conformal prediction** based on test set residuals
- 90% and 95% confidence intervals
- Calibrated on holdout set (not hardcoded ±10%)

### Model Loading
- **Local artifact mode**: Always loads from `models/best_model.pkl`
- No MLflow dependency in production
- Robust for containers/deployments

## 📊 Model Performance

Current best model (Gradient Boosting):
- **Test RMSE**: ~3167 SEK
- **Test R²**: 0.986
- **90% Confidence Margin**: ~6718 SEK (calibrated on test set)

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -t avarn-api .

# Run container
docker run -p 8000:8000 avarn-api
```

### Render.com

See `render.yaml` for deployment configuration. The system is configured for:
- FastAPI service on port 8000
- Streamlit UI on port 8501 (optional)

## 📁 Project Structure Details

### `app/api/` - REST API
- `main.py`: FastAPI application with `/predict`, `/predict/batch`, `/health` endpoints
- `schemas.py`: Pydantic models for request/response validation

### `app/features/` - Feature Engineering
- `feature_pipeline.py`: Production feature pipeline with OneHotEncoder
- Handles numeric scaling and categorical encoding
- Versioned pipeline for reproducibility

### `app/models/` - Model Management
- `model_loader.py`: Loads models from local files (production-ready)
- `train_with_mlflow.py`: Training script with MLflow integration

### `app/monitoring/` - Observability
- `logger.py`: Logs all predictions to JSONL files
- Enables tracking of prediction performance over time

### `app/ui/` - Web Interface
- `streamlit_app.py`: Interactive Streamlit UI for predictions
- Connects to FastAPI backend

## 🔬 MLflow

View experiment tracking:

```bash
python start_mlflow_ui.py
```

Then open `http://localhost:5000` to see:
- Experiment runs and metrics
- Model artifacts
- Feature pipeline versions

## 📝 Data Format

Training data should be CSV with columns:
- `antal_sektioner`: Number of fire alarm sections (1-50)
- `antal_detektorer`: Number of detectors (1-200)
- `antal_larmdon`: Number of alarm devices (1-100)
- `dörrhållarmagneter`: Number of door holder magnets (0-50)
- `ventilation`: Ventilation system (0=no, 1=yes)
- `stad`: City (Stockholm, Göteborg, Malmö, etc.)
- `kvartalsvis`: Quarterly testing (0=no, 1=yes)
- `månadsvis`: Monthly testing (0=no, 1=yes)
- `årsvis`: Yearly testing (0=no, 1=yes)
- `price`: Target variable (price in SEK)

## 🛠️ Development

### Generate Training Data

```bash
python scripts/generate_dummy_data.py
```

### Run Tests

```bash
pytest tests/ -v
```

### Code Quality

The project follows best practices:
- Type hints with Pydantic
- Comprehensive test coverage
- Production-ready error handling
- Logging and monitoring

## 📚 Additional Documentation

- `QUICKSTART.md`: Step-by-step getting started guide
- `TEST_API.md`: API testing examples
- `streamlit_deploy.md`: Streamlit deployment guide

## 🎯 Production Checklist

- ✅ OneHotEncoder for categorical features
- ✅ Conformal prediction intervals (not hardcoded)
- ✅ Local artifact mode (no MLflow in production)
- ✅ Comprehensive test suite
- ✅ Docker support
- ✅ Monitoring and logging
- ✅ API documentation (OpenAPI)

## 📄 License

[Your License Here]

## 👤 Author

[Your Name/Organization]
