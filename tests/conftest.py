"""
Pytest configuration and fixtures for testing.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_model_loader():
    """Mock model loader for testing."""
    mock_loader = MagicMock()
    mock_loader.is_loaded.return_value = True
    mock_loader.get_model_version.return_value = "test_v1.0"
    mock_loader.get_pipeline_version.return_value = "v2.0"
    mock_loader.predict.return_value = 45000.0
    mock_loader.predict_with_interval.return_value = (45000.0, 40000.0, 50000.0)
    mock_loader.residual_90_percentile = 5000.0
    mock_loader.residual_95_percentile = 7500.0
    return mock_loader


@pytest.fixture
def client(mock_model_loader):
    """Create a test client for the FastAPI app."""
    # Import app and mock the model loader before startup
    from app.api import main
    
    # Mock the model loader
    main.model_loader = mock_model_loader
    
    # Clear startup events to avoid loading model
    main.app.router.on_startup = []
    
    return TestClient(main.app)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {
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
