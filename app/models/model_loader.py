"""
Model loader for production inference.
Handles model loading, versioning, and prediction.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Production model loader with MLflow integration.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory containing saved models
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"
        
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_pipeline = None
        self.model_version = None
        self.pipeline_version = None
        
        # Try to load MLflow model if available
        self.mlflow_model = None
        self.mlflow_uri = None
    
    def load_latest_model(self) -> None:
        """Load the latest model and feature pipeline."""
        try:
            # Try MLflow first
            if self._load_mlflow_model():
                logger.info("Loaded model from MLflow")
                return
            
            # Fallback to local files
            self._load_local_model()
            logger.info("Loaded model from local files")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_mlflow_model(self) -> bool:
        """Try to load model from MLflow."""
        try:
            # Check if MLflow is configured
            mlflow_uri = Path(self.models_dir) / "mlruns"
            if not mlflow_uri.exists():
                return False
            
            mlflow.set_tracking_uri(str(mlflow_uri))
            
            # Get latest model from registry
            # This is simplified - in production, use proper model registry
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Get latest run
            experiments = client.search_experiments()
            if not experiments:
                return False
            
            latest_run = None
            for exp in experiments:
                runs = client.search_runs(exp.experiment_id, order_by=["start_time desc"], max_results=1)
                if runs:
                    latest_run = runs[0]
                    break
            
            if latest_run:
                model_uri = f"runs:/{latest_run.info.run_id}/model"
                self.mlflow_model = mlflow.sklearn.load_model(model_uri)
                self.model = self.mlflow_model
                self.model_version = latest_run.info.run_id
                self.pipeline_version = "v1.0"  # Extract from run if available
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"MLflow load failed: {e}")
            return False
    
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
        
        # Load feature pipeline
        # Try new pipeline format first
        try:
            from app.features.feature_pipeline import FeaturePipeline
            self.feature_pipeline = FeaturePipeline.load(str(pipeline_path))
            self.pipeline_version = self.feature_pipeline.version
        except:
            # Fallback to old preprocessor format
            from scripts.preprocess import DataPreprocessor
            self.feature_pipeline = DataPreprocessor()
            self.feature_pipeline.load_preprocessor(str(pipeline_path))
            self.pipeline_version = "v1.0"
    
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
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.feature_pipeline is not None
    
    def get_model_version(self) -> str:
        """Get current model version."""
        return self.model_version or "unknown"
    
    def get_pipeline_version(self) -> str:
        """Get current feature pipeline version."""
        return self.pipeline_version or "unknown"
