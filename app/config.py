"""
Configuration management for the application.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Application configuration."""

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "./models/mlruns")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fire_alarm_price_prediction")

    # Paths
    # config.py is at app/config.py, so parent.parent is project root (not parent.parent.parent)
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # Handle MODELS_DIR: if absolute path, use as-is; if relative, join with PROJECT_ROOT
    models_dir_env = os.getenv("MODELS_DIR", "models")
    MODELS_DIR: Path = Path(models_dir_env) if Path(models_dir_env).is_absolute() else PROJECT_ROOT / models_dir_env
    
    data_dir_env = os.getenv("DATA_DIR", "data")
    DATA_DIR: Path = Path(data_dir_env) if Path(data_dir_env).is_absolute() else PROJECT_ROOT / data_dir_env
    
    log_dir_env = os.getenv("LOG_DIR", "logs")
    LOG_DIR: Path = Path(log_dir_env) if Path(log_dir_env).is_absolute() else PROJECT_ROOT / log_dir_env

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # Streamlit
    STREAMLIT_API_URL: str = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    @classmethod
    def get_mlflow_uri(cls) -> str:
        """Get MLflow tracking URI with proper file:// prefix for Windows."""
        uri = cls.MLFLOW_TRACKING_URI
        if not uri.startswith(("file://", "http://", "https://")):
            # Convert to absolute path and format for Windows
            abs_path = Path(uri).resolve()
            uri = f"file:///{str(abs_path).replace(chr(92), '/')}"
        return uri
