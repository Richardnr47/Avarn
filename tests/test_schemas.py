"""
Tests for Pydantic schema validation.
"""

import pytest
from pydantic import ValidationError
from app.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    StadEnum,
    BatchPredictionRequest,
    HealthResponse
)


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""
    
    def test_valid_request(self):
        """Test valid prediction request."""
        request = PredictionRequest(
            antal_sektioner=8,
            antal_detektorer=25,
            antal_larmdon=15,
            dörrhållarmagneter=5,
            ventilation=1,
            stad=StadEnum.STOCKHOLM,
            kvartalsvis=0,
            månadsvis=1,
            årsvis=0
        )
        assert request.antal_sektioner == 8
        assert request.stad == StadEnum.STOCKHOLM
    
    def test_invalid_sektioner_too_low(self):
        """Test validation error for antal_sektioner < 1."""
        with pytest.raises(ValidationError):
            PredictionRequest(
                antal_sektioner=0,  # Invalid: must be >= 1
                antal_detektorer=25,
                antal_larmdon=15,
                dörrhållarmagneter=5,
                ventilation=1,
                stad=StadEnum.STOCKHOLM,
                kvartalsvis=0,
                månadsvis=1,
                årsvis=0
            )
    
    def test_invalid_sektioner_too_high(self):
        """Test validation error for antal_sektioner > 50."""
        with pytest.raises(ValidationError):
            PredictionRequest(
                antal_sektioner=51,  # Invalid: must be <= 50
                antal_detektorer=25,
                antal_larmdon=15,
                dörrhållarmagneter=5,
                ventilation=1,
                stad=StadEnum.STOCKHOLM,
                kvartalsvis=0,
                månadsvis=1,
                årsvis=0
            )
    
    def test_invalid_stad(self):
        """Test validation error for invalid city."""
        with pytest.raises(ValidationError):
            PredictionRequest(
                antal_sektioner=8,
                antal_detektorer=25,
                antal_larmdon=15,
                dörrhållarmagneter=5,
                ventilation=1,
                stad="InvalidCity",  # Invalid: not in StadEnum
                kvartalsvis=0,
                månadsvis=1,
                årsvis=0
            )
    
    def test_valid_stad_enum(self):
        """Test all valid city enums."""
        for stad in StadEnum:
            request = PredictionRequest(
                antal_sektioner=8,
                antal_detektorer=25,
                antal_larmdon=15,
                dörrhållarmagneter=5,
                ventilation=1,
                stad=stad,
                kvartalsvis=0,
                månadsvis=1,
                årsvis=0
            )
            assert request.stad == stad


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""
    
    def test_valid_response(self):
        """Test valid prediction response."""
        response = PredictionResponse(
            predicted_price=45230.50,
            confidence_interval_lower=42200.00,
            confidence_interval_upper=48260.00,
            model_version="v1.0",
            feature_pipeline_version="v2.0",
            prediction_id="pred_1234567890"
        )
        assert response.predicted_price == 45230.50
        assert response.confidence_interval_lower == 42200.00
        assert response.confidence_interval_upper == 48260.00


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""
    
    def test_valid_batch_request(self):
        """Test valid batch prediction request."""
        request = BatchPredictionRequest(
            items=[
                PredictionRequest(
                    antal_sektioner=8,
                    antal_detektorer=25,
                    antal_larmdon=15,
                    dörrhållarmagneter=5,
                    ventilation=1,
                    stad=StadEnum.STOCKHOLM,
                    kvartalsvis=0,
                    månadsvis=1,
                    årsvis=0
                )
            ]
        )
        assert len(request.items) == 1
    
    def test_empty_batch_request(self):
        """Test validation error for empty batch."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(items=[])


class TestHealthResponse:
    """Tests for HealthResponse schema."""
    
    def test_valid_health_response(self):
        """Test valid health response."""
        from datetime import datetime
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            model_loaded=True,
            feature_pipeline_loaded=True,
            timestamp=datetime.now().isoformat()
        )
        assert response.status == "healthy"
        assert response.model_loaded is True
