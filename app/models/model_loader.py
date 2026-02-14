"""
Model loader for production inference.
Handles model loading, versioning, and prediction.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.config import Config

# MLflow imports removed - only used for training, not serving

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Production model loader.
    
    Always loads from local files (models/best_model.pkl + preprocessor.pkl).
    MLflow is only used for training and experiment tracking locally, not for serving.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory containing saved models (defaults to Config.MODELS_DIR)
        """
        if models_dir is None:
            models_dir = Config.MODELS_DIR
        
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_pipeline = None
        self.model_version = None
        self.pipeline_version = None
        
        # Residual statistics for conformal prediction
        self.residual_90_percentile = None  # For 90% confidence intervals
        self.residual_95_percentile = None  # For 95% confidence intervals
        
        # MLflow is only used for training, not for serving
        # Production always uses local files (best_model.pkl)
    
    def load_latest_model(self) -> None:
        """
        Load model and feature pipeline from local files.
        
        For production: Always use local files (models/best_model.pkl + preprocessor.pkl)
        MLflow is only used for training and comparison locally, not for serving.
        """
        try:
            # Always use local files for production serving
            # This is more robust and works reliably in containers/deployments
            self._load_local_model()
            logger.info("Loaded model from local files")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    # MLflow loading removed - production always uses local files
    # MLflow is only used for training and experiment tracking locally
    
    def _load_local_model(self) -> None:
        """Load model from local pickle files."""
        model_path = self.models_dir / "best_model.pkl"
        pipeline_path = self.models_dir / "preprocessor.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.model_version = model_data.get('model_name', 'unknown')
        
        # Load residual statistics for conformal prediction
        self.residual_90_percentile = model_data.get('residual_90_percentile')
        self.residual_95_percentile = model_data.get('residual_95_percentile')
        
        if self.residual_90_percentile is None:
            logger.warning("No residual statistics found. Using fallback confidence intervals.")
        
        # Load feature pipeline
        # Try new pipeline format first
        try:
            from app.features.feature_pipeline import FeaturePipeline
            self.feature_pipeline = FeaturePipeline.load(str(pipeline_path))
            self.pipeline_version = self.feature_pipeline.version
            
            # Fix encoding issues with Swedish characters in feature columns
            # This can happen when pickle saves/loads with wrong encoding
            if self.feature_pipeline.feature_columns:
                # Normalize column names to handle encoding issues
                import unicodedata
                normalized_columns = []
                for col in self.feature_pipeline.feature_columns:
                    # Try to fix common encoding issues
                    col_fixed = col.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
                    if col_fixed != col:
                        normalized_columns.append(col_fixed)
                    else:
                        normalized_columns.append(col)
                
                # If we have a mismatch, try to map columns
                if any('' in col for col in self.feature_pipeline.feature_columns):
                    # Create mapping for common Swedish characters
                    char_map = {
                        'drrhllarmagneter': 'dörrhållarmagneter',
                        'mnadsvis': 'månadsvis',
                        'rsvis': 'årsvis'
                    }
                    # Update feature_columns with correct encoding
                    fixed_columns = []
                    for col in self.feature_pipeline.feature_columns:
                        fixed_col = char_map.get(col, col)
                        fixed_columns.append(fixed_col)
                    self.feature_pipeline.feature_columns = fixed_columns
                    
        except Exception as e:
            logger.warning(f"Failed to load new pipeline format: {e}")
            # Fallback to old preprocessor format
            try:
                from scripts.preprocess import DataPreprocessor
                self.feature_pipeline = DataPreprocessor()
                self.feature_pipeline.load_preprocessor(str(pipeline_path))
                self.pipeline_version = "v1.0"
            except Exception as e2:
                logger.error(f"Failed to load old preprocessor format: {e2}")
                raise
    
    def predict(self, request_dict: Dict[str, Any]) -> float:
        """
        Make prediction on request data.
        
        Args:
            request_dict: Dictionary with feature values
            
        Returns:
            Predicted price
        """
        if self.model is None or self.feature_pipeline is None:
            raise RuntimeError("Model not loaded. Call load_latest_model() first.")
        
        # Convert request to DataFrame
        df = pd.DataFrame([request_dict])
        
        # Transform features
        # Handle both new and old pipeline formats
        if hasattr(self.feature_pipeline, 'transform'):
            # New FeaturePipeline format
            X, _ = self.feature_pipeline.transform(df, fit=False)
        else:
            # Old DataPreprocessor format
            X = self.feature_pipeline.prepare_features(df, fit=False)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Ensure non-negative
        prediction = max(0, float(prediction))
        
        return prediction
    
    def predict_with_interval(self, request_dict: Dict[str, Any], confidence: float = 0.90) -> tuple:
        """
        Make prediction with confidence interval using conformal prediction.
        
        Args:
            request_dict: Dictionary with feature values
            confidence: Confidence level (0.90 for 90%, 0.95 for 95%)
            
        Returns:
            Tuple of (prediction, lower_bound, upper_bound)
        """
        prediction = self.predict(request_dict)
        
        # Use conformal prediction based on residual percentiles
        if confidence == 0.90 and self.residual_90_percentile is not None:
            margin = self.residual_90_percentile
        elif confidence == 0.95 and self.residual_95_percentile is not None:
            margin = self.residual_95_percentile
        else:
            # Fallback: use 90% percentile if available, otherwise 10% of prediction
            if self.residual_90_percentile is not None:
                margin = self.residual_90_percentile
            else:
                logger.warning("No residual statistics available. Using fallback 10% margin.")
                margin = prediction * 0.1
        
        lower = max(0, prediction - margin)
        upper = prediction + margin
        
        return prediction, lower, upper
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.feature_pipeline is not None
    
    def get_model_version(self) -> str:
        """Get current model version."""
        return self.model_version or "unknown"
    
    def get_pipeline_version(self) -> str:
        """Get current feature pipeline version."""
        return self.pipeline_version or "unknown"
