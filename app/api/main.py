"""
FastAPI application for ML inference API.
Production-ready API with validation, monitoring, and error handling.
"""

import logging
import uuid
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
        # Frequency validation is now handled in schema (root_validator)
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
    Returns both successful predictions and errors for failed items.
    """
    from app.api.schemas import BatchPredictionError

    predictions = []
    errors = []

    for index, item in enumerate(request.items):
        try:
            # Frequency validation is handled in schema (root_validator)
            # If validation fails, Pydantic will raise ValidationError
            prediction_id = f"pred_{uuid.uuid4().hex[:10]}"
            # Use conformal prediction for confidence intervals
            prediction, lower, upper = model_loader.predict_with_interval(
                item.dict(), confidence=0.90
            )

            response = PredictionResponse(
                predicted_price=round(prediction, 2),
                confidence_interval_lower=round(lower, 2),
                confidence_interval_upper=round(upper, 2),
                model_version=model_loader.get_model_version(),
                feature_pipeline_version=model_loader.get_pipeline_version(),
                prediction_id=prediction_id,
            )

            predictions.append(response)

            # Log each prediction
            prediction_logger.log_prediction(
                prediction_id=prediction_id,
                request=item.dict(),
                response=response.dict(),
                timestamp=datetime.now(),
            )

        except Exception as e:
            # Capture error for this item
            error_detail = str(e)
            if hasattr(e, "detail"):
                error_detail = e.detail
            errors.append(BatchPredictionError(index=index, detail=error_detail))
            logger.warning(f"Batch item {index} failed: {error_detail}")

    logger.info(
        f"Batch prediction: {len(predictions)} successful, {len(errors)} failed"
    )

    return BatchPredictionResponse(
        predictions=predictions,
        errors=errors,
        total_items=len(request.items),
        successful=len(predictions),
        failed=len(errors),
        model_version=model_loader.get_model_version(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
