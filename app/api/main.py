"""
FastAPI application for ML inference API.
Production-ready API with validation, monitoring, and error handling.
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.config import Config
from app.models.model_loader import ModelLoader
from app.monitoring.logger import PredictionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Avarn Fire Alarm Testing Price Prediction API",
    description="ML API for predicting fire alarm testing prices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS if "*" not in Config.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loader
model_loader = ModelLoader()
prediction_logger = PredictionLogger()


@app.on_event("startup")
async def startup_event():
    """Load model and feature pipeline on startup."""
    try:
        model_loader.load_latest_model()
        logger.info("Model and feature pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Avarn Fire Alarm Testing Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "unhealthy",
        version="1.0.0",
        model_loaded=model_loader.is_loaded(),
        feature_pipeline_loaded=model_loader.is_loaded(),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict fire alarm testing price for a single request.
    """
    try:
        # Validate frequency (exactly one must be selected)
        frequency_sum = request.kvartalsvis + request.månadsvis + request.årsvis
        if frequency_sum != 1:
            raise HTTPException(
                status_code=400,
                detail="Exactly one frequency must be selected (kvartalsvis, månadsvis, or årsvis)",
            )

        # Generate prediction ID
        prediction_id = f"pred_{uuid.uuid4().hex[:10]}"

        # Make prediction with conformal prediction interval (calibrated on test set)
        prediction, lower, upper = model_loader.predict_with_interval(
            request.dict(), confidence=0.90
        )

        response = PredictionResponse(
            predicted_price=round(prediction, 2),
            confidence_interval_lower=round(lower, 2),
            confidence_interval_upper=round(upper, 2),
            model_version=model_loader.get_model_version(),
            feature_pipeline_version=model_loader.get_pipeline_version(),
            prediction_id=prediction_id,
        )

        # Log prediction
        prediction_logger.log_prediction(
            prediction_id=prediction_id,
            request=request.dict(),
            response=response.dict(),
            timestamp=datetime.now(),
        )

        logger.info(f"Prediction made: {prediction_id} -> {prediction:.2f} SEK")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict prices for multiple requests in batch.
    """
    try:
        predictions = []

        for item in request.items:
            # Validate frequency
            frequency_sum = item.kvartalsvis + item.månadsvis + item.årsvis
            if frequency_sum != 1:
                continue  # Skip invalid items

            prediction_id = f"pred_{uuid.uuid4().hex[:10]}"
            # Use conformal prediction for confidence intervals
            prediction, lower, upper = model_loader.predict_with_interval(
                item.dict(), confidence=0.90
            )

            predictions.append(
                PredictionResponse(
                    predicted_price=round(prediction, 2),
                    confidence_interval_lower=round(lower, 2),
                    confidence_interval_upper=round(upper, 2),
                    model_version=model_loader.get_model_version(),
                    feature_pipeline_version=model_loader.get_pipeline_version(),
                    prediction_id=prediction_id,
                )
            )

            # Log each prediction
            prediction_logger.log_prediction(
                prediction_id=prediction_id,
                request=item.dict(),
                response=predictions[-1].dict(),
                timestamp=datetime.now(),
            )

        logger.info(f"Batch prediction: {len(predictions)} predictions made")

        return BatchPredictionResponse(
            predictions=predictions,
            total_items=len(predictions),
            model_version=model_loader.get_model_version(),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
