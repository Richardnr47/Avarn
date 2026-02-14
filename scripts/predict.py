"""
Prediction script for fire alarm testing price prediction.
Loads trained models and makes predictions on new data.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from preprocess import DataPreprocessor


class PricePredictor:
    """Handles price predictions using trained models."""

    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the predictor with a trained model and preprocessor.

        Args:
            model_path: Path to the saved model file
            preprocessor_path: Path to the saved preprocessor file
        """
        # Load model
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.model_name = model_data.get("model_name", "unknown")
        print(f"Loaded model: {self.model_name}")

        # Load preprocessor
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_preprocessor(preprocessor_path)
        print("Loaded preprocessor")

    def predict(self, data):
        """
        Make price predictions on new data.

        Args:
            data: DataFrame or path to CSV file with features

        Returns:
            Array of predicted prices
        """
        # Load data if path is provided
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        # Preprocess data (without target column)
        X = self.preprocessor.prepare_features(df, fit=False)

        # Make predictions
        predictions = self.model.predict(X)

        return predictions

    def predict_with_details(self, data):
        """
        Make predictions and return detailed information.

        Args:
            data: DataFrame or path to CSV file with features

        Returns:
            DataFrame with original data and predictions
        """
        # Load data if path is provided
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        # Make predictions
        predictions = self.predict(df)

        # Create result DataFrame
        result_df = df.copy()
        result_df["predicted_price"] = predictions

        return result_df


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Predict fire alarm testing prices")
    parser.add_argument(
        "--model",
        type=str,
        default="../models/best_model.pkl",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="../models/preprocessor.pkl",
        help="Path to preprocessor file",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file with features"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save predictions (CSV file)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["simple", "detailed"],
        default="detailed",
        help="Output format: simple (just predictions) or detailed (with input features)",
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.preprocessor):
        raise FileNotFoundError(f"Preprocessor file not found: {args.preprocessor}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Initialize predictor
    predictor = PricePredictor(args.model, args.preprocessor)

    # Make predictions
    print(f"\nMaking predictions on {args.input}...")

    if args.format == "detailed":
        results = predictor.predict_with_details(args.input)
        print(f"\nPredictions completed. Shape: {results.shape}")
        print("\nFirst few predictions:")
        print(results.head())
    else:
        predictions = predictor.predict(args.input)
        results = pd.DataFrame({"predicted_price": predictions})
        print(f"\nPredictions completed. {len(predictions)} predictions made")
        print("\nFirst few predictions:")
        print(results.head())

    # Save results if output path is provided
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")
    else:
        print("\nPredictions displayed above. Use --output to save to file.")


if __name__ == "__main__":
    main()
