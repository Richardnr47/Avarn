"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class StadEnum(str, Enum):
    """Supported cities."""
    STOCKHOLM = "Stockholm"
    GOTEBORG = "Göteborg"
    MALMO = "Malmö"
    UPPSALA = "Uppsala"
    LINKOPING = "Linköping"
    OREBRO = "Örebro"
    VASTERAS = "Västerås"
    HELSINGBORG = "Helsingborg"


class PredictionRequest(BaseModel):
    """Request schema for price prediction."""
    
    antal_sektioner: int = Field(..., ge=1, le=50, description="Number of fire alarm sections")
    antal_detektorer: int = Field(..., ge=1, le=200, description="Number of detectors")
    antal_larmdon: int = Field(..., ge=1, le=100, description="Number of alarm devices")
    dörrhållarmagneter: int = Field(..., ge=0, le=50, description="Number of door holder magnets")
    ventilation: int = Field(..., ge=0, le=1, description="Ventilation system (0=no, 1=yes)")
    stad: StadEnum = Field(..., description="City")
    kvartalsvis: int = Field(..., ge=0, le=1, description="Quarterly testing (0=no, 1=yes)")
    månadsvis: int = Field(..., ge=0, le=1, description="Monthly testing (0=no, 1=yes)")
    årsvis: int = Field(..., ge=0, le=1, description="Yearly testing (0=no, 1=yes)")
    
    @validator('kvartalsvis', 'månadsvis', 'årsvis')
    def validate_frequency(cls, v, values):
        """Ensure exactly one frequency is selected."""
        # This will be checked in the endpoint
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response schema for price prediction."""
    
    predicted_price: float = Field(..., description="Predicted price in SEK")
    confidence_interval_lower: Optional[float] = Field(None, description="Lower bound of confidence interval")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper bound of confidence interval")
    model_version: str = Field(..., description="Model version used for prediction")
    feature_pipeline_version: str = Field(..., description="Feature pipeline version used")
    prediction_id: str = Field(..., description="Unique prediction ID for tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 45230.50,
                "confidence_interval_lower": 42200.00,
                "confidence_interval_upper": 48260.00,
                "model_version": "v1.0",
                "feature_pipeline_version": "v1.0",
                "prediction_id": "pred_1234567890"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    items: List[PredictionRequest] = Field(..., min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse]
    total_items: int
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [],
                "total_items": 1,
                "model_version": "v1.0"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    version: str
    model_loaded: bool
    feature_pipeline_loaded: bool
    timestamp: str
