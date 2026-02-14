"""
Model training with MLflow integration.
Tracks experiments, versions models, and stores artifacts.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import mlflow
import mlflow.sklearn
from pathlib import Path
import sys
from datetime import datetime

# Suppress MLflow deprecation warnings (they're internal to MLflow)
warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.features.feature_pipeline import FeaturePipeline
from scripts.preprocess import DataPreprocessor


def train_with_mlflow(
    data_path: str,
    experiment_name: str = "fire_alarm_price_prediction",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train models with MLflow tracking.
    
    Args:
        data_path: Path to training data CSV
        experiment_name: MLflow experiment name
        test_size: Proportion of test set
        random_state: Random seed
    """
    # Set up MLflow
    # Set tracking URI explicitly with file:// prefix for Windows
    models_dir = Path(__file__).parent.parent.parent / "models"
    mlruns_path = models_dir / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to file:// URI format for Windows
    mlruns_uri = f"file:///{str(mlruns_path.absolute()).replace(chr(92), '/')}"
    mlflow.set_tracking_uri(mlruns_uri)
    
    # Set experiment (creates if doesn't exist)
    mlflow.set_experiment(experiment_name)
    
    # Load and prepare data
    print("Loading data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(df)
    X, y = preprocessor.prepare_features(df, fit=True)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Models to train
    models = {
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        'ridge': Ridge(alpha=1.0)
    }
    
    best_model = None
    best_score = float('inf')
    best_name = None
    
    # Train each model with MLflow tracking
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log model (using default pickle format - works reliably)
            # Note: skops format requires trusted_types configuration, pickle is simpler
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature pipeline
            models_dir = Path(__file__).parent.parent.parent / "models"
            pipeline_path = models_dir / "preprocessor.pkl"
            if pipeline_path.exists():
                mlflow.log_artifact(str(pipeline_path), "feature_pipeline")
            
            # Log data info
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Test R²: {test_r2:.4f}")
            
            # Track best model
            if test_rmse < best_score:
                best_score = test_rmse
                best_model = model
                best_name = model_name
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_name} (Test RMSE: {best_score:.2f})")
    print(f"{'='*60}")
    
    # Save best model locally
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    import pickle
    model_data = {
        'model': best_model,
        'model_name': best_name,
        'timestamp': datetime.now().isoformat()
    }
    
    best_model_path = models_dir / "best_model.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Best model saved to {best_model_path}")
    
    return best_model, best_name


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models with MLflow')
    parser.add_argument('--data', type=str, default='../data/training_data.csv',
                        help='Path to training data')
    parser.add_argument('--experiment', type=str, default='fire_alarm_price_prediction',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    train_with_mlflow(
        data_path=args.data,
        experiment_name=args.experiment
    )
