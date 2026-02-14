"""
Tests for /predict endpoint.
"""

import pytest
from fastapi import status
from app.api.schemas import StadEnum


def test_predict_endpoint_success(client, sample_prediction_request):
    """Test successful prediction request."""
    response = client.post("/predict", json=sample_prediction_request)
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    
    # Check response structure
    assert "predicted_price" in data
    assert "confidence_interval_lower" in data
    assert "confidence_interval_upper" in data
    assert "model_version" in data
    assert "feature_pipeline_version" in data
    assert "prediction_id" in data
    
    # Check data types
    assert isinstance(data["predicted_price"], (int, float))
    assert isinstance(data["confidence_interval_lower"], (int, float))
    assert isinstance(data["confidence_interval_upper"], (int, float))
    assert isinstance(data["model_version"], str)
    assert isinstance(data["feature_pipeline_version"], str)
    assert isinstance(data["prediction_id"], str)
    
    # Check that prediction_id starts with "pred_"
    assert data["prediction_id"].startswith("pred_")
    
    # Check that confidence interval is valid
    assert data["confidence_interval_lower"] <= data["predicted_price"]
    assert data["confidence_interval_upper"] >= data["predicted_price"]


def test_predict_endpoint_invalid_frequency(client, sample_prediction_request):
    """Test prediction request with invalid frequency (none selected)."""
    invalid_request = sample_prediction_request.copy()
    invalid_request["kvartalsvis"] = 0
    invalid_request["månadsvis"] = 0
    invalid_request["årsvis"] = 0  # No frequency selected
    
    response = client.post("/predict", json=invalid_request)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "frequency" in response.json()["detail"].lower()


def test_predict_endpoint_multiple_frequencies(client, sample_prediction_request):
    """Test prediction request with multiple frequencies selected."""
    invalid_request = sample_prediction_request.copy()
    invalid_request["kvartalsvis"] = 1
    invalid_request["månadsvis"] = 1  # Both selected
    invalid_request["årsvis"] = 0
    
    response = client.post("/predict", json=invalid_request)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "frequency" in response.json()["detail"].lower()


def test_predict_endpoint_invalid_stad(client, sample_prediction_request):
    """Test prediction request with invalid city."""
    invalid_request = sample_prediction_request.copy()
    invalid_request["stad"] = "InvalidCity"
    
    response = client.post("/predict", json=invalid_request)
    
    # Should fail at schema validation level
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_endpoint_different_cities(client):
    """Test prediction with different valid cities."""
    for stad in StadEnum:
        request = {
            "antal_sektioner": 8,
            "antal_detektorer": 25,
            "antal_larmdon": 15,
            "dörrhållarmagneter": 5,
            "ventilation": 1,
            "stad": stad.value,
            "kvartalsvis": 0,
            "månadsvis": 1,
            "årsvis": 0
        }
        
        response = client.post("/predict", json=request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predicted_price" in data
        assert data["predicted_price"] > 0  # Price should be positive


def test_predict_endpoint_missing_field(client, sample_prediction_request):
    """Test prediction request with missing required field."""
    invalid_request = sample_prediction_request.copy()
    del invalid_request["antal_sektioner"]  # Remove required field
    
    response = client.post("/predict", json=invalid_request)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
