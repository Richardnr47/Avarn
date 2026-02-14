"""
Model training script for fire alarm testing price prediction.
Supports multiple regression algorithms and model evaluation.
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from preprocess import DataPreprocessor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR


class ModelTrainer:
    """Handles model training and evaluation for price prediction."""

    def __init__(self):
        """Initialize the model trainer."""
        self.models = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            "svr": SVR(kernel="rbf", C=100, gamma="scale"),
        }
        self.trained_models = {}
        self.preprocessor = DataPreprocessor()
        self.best_model = None
        self.best_model_name = None

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all available models and evaluate their performance.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with model performance metrics
        """
        results = {}
        best_score = float("inf")

        print("\n" + "=" * 60)
        print("Training Models")
        print("=" * 60)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            results[name] = {
                "train_rmse": float(train_rmse),
                "test_rmse": float(test_rmse),
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
            }

            print(f"  Train RMSE: {train_rmse:.2f}")
            print(f"  Test RMSE:  {test_rmse:.2f}")
            print(f"  Train MAE:  {train_mae:.2f}")
            print(f"  Test MAE:   {test_mae:.2f}")
            print(f"  Train R²:   {train_r2:.4f}")
            print(f"  Test R²:    {test_r2:.4f}")

            # Track best model (lowest test RMSE)
            if test_rmse < best_score:
                best_score = test_rmse
                self.best_model = model
                self.best_model_name = name

        print("\n" + "=" * 60)
        print(f"Best Model: {self.best_model_name} (Test RMSE: {best_score:.2f})")
        print("=" * 60)

        return results

    def save_model(self, model_name, file_path):
        """
        Save a trained model to disk.

        Args:
            model_name: Name of the model to save
            file_path: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")

        model_data = {
            "model": self.trained_models[model_name],
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {file_path}")

    def save_best_model(self, file_path):
        """Save the best performing model to disk."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")

        model_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "timestamp": datetime.now().isoformat(),
        }

        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Best model ({self.best_model_name}) saved to {file_path}")

    def save_results(self, results, file_path):
        """Save training results to JSON file."""
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {file_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train fire alarm testing price prediction models"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../data/training_data.csv",
        help="Path to training data CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../models",
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Initialize trainer
    trainer = ModelTrainer()
    preprocessor = trainer.preprocessor

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = preprocessor.load_data(args.data)
    df = preprocessor.clean_data(df)
    X, y = preprocessor.prepare_features(df, fit=True)

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"\nDataset Info:")
    print(f"  Total samples: {len(X)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X.shape[1]}")

    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)

    # Save preprocessor
    preprocessor_path = os.path.join(args.output, "preprocessor.pkl")
    preprocessor.save_preprocessor(preprocessor_path)

    # Save best model
    best_model_path = os.path.join(args.output, "best_model.pkl")
    trainer.save_best_model(best_model_path)

    # Save all models
    for model_name in trainer.trained_models.keys():
        model_path = os.path.join(args.output, f"{model_name}.pkl")
        trainer.save_model(model_name, model_path)

    # Save results
    results_path = os.path.join(args.output, "training_results.json")
    trainer.save_results(results, results_path)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
