"""
Tests for /health endpoint.
"""

import pytest
from fastapi import status


def test_health_endpoint(client):
    """Test that /health endpoint returns 200 and correct structure."""
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data
    assert "feature_pipeline_loaded" in data
    assert "timestamp" in data

    # Check that status is either "healthy" or "unhealthy"
    assert data["status"] in ["healthy", "unhealthy"]

    # Check that version is a string
    assert isinstance(data["version"], str)

    # Check that model_loaded and feature_pipeline_loaded are booleans
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["feature_pipeline_loaded"], bool)


def test_health_endpoint_structure(client):
    """Test that /health endpoint returns correct data types."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    # Validate types
    assert isinstance(data["status"], str)
    assert isinstance(data["version"], str)
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["feature_pipeline_loaded"], bool)
    assert isinstance(data["timestamp"], str)
