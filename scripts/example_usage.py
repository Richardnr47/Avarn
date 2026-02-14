"""
Example usage script demonstrating how to use the fire alarm testing
price prediction system programmatically.
"""

import os
import sys

import pandas as pd

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import PricePredictor
from preprocess import DataPreprocessor
from train_model import ModelTrainer


def example_training():
    """Example of training models programmatically."""
    print("=" * 60)
    print("Example: Training Models")
    print("=" * 60)

    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()

    # Load and preprocess data
    data_path = "../data/training_data.csv"
    df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(df)
    X, y = preprocessor.prepare_features(df, fit=True)

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)

    # Save preprocessor and best model
    preprocessor.save_preprocessor("../models/preprocessor.pkl")
    trainer.save_best_model("../models/best_model.pkl")

    return trainer, preprocessor, results


def example_prediction():
    """Example of making predictions programmatically."""
    print("\n" + "=" * 60)
    print("Example: Making Predictions")
    print("=" * 60)

    # Initialize predictor
    model_path = "../models/best_model.pkl"
    preprocessor_path = "../models/preprocessor.pkl"

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("Models not found. Please train models first.")
        return

    predictor = PricePredictor(model_path, preprocessor_path)

    # Create sample data for prediction
    sample_data = pd.DataFrame(
        {
            "building_size": [600, 900, 1200],
            "num_alarms": [12, 20, 25],
            "location": ["Stockholm", "Göteborg", "Malmö"],
            "service_type": ["Standard", "Premium", "Premium"],
            "building_age": [5, 3, 1],
        }
    )

    print("\nSample input data:")
    print(sample_data)

    # Make predictions
    predictions = predictor.predict(sample_data)

    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred:.2f} SEK")

    # Get detailed predictions
    detailed_results = predictor.predict_with_details(sample_data)
    print("\nDetailed results:")
    print(detailed_results)


def example_single_prediction():
    """Example of making a single prediction."""
    print("\n" + "=" * 60)
    print("Example: Single Prediction")
    print("=" * 60)

    model_path = "../models/best_model.pkl"
    preprocessor_path = "../models/preprocessor.pkl"

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("Models not found. Please train models first.")
        return

    predictor = PricePredictor(model_path, preprocessor_path)

    # Single prediction
    single_data = pd.DataFrame(
        {
            "building_size": [750],
            "num_alarms": [15],
            "location": ["Stockholm"],
            "service_type": ["Standard"],
            "building_age": [8],
        }
    )

    prediction = predictor.predict(single_data)[0]
    print(f"\nPredicted price: {prediction:.2f} SEK")
    print(f"Input: {single_data.iloc[0].to_dict()}")


if __name__ == "__main__":
    print("Fire Alarm Testing Price Prediction - Example Usage\n")

    # Check if models exist
    model_exists = os.path.exists("../models/best_model.pkl")

    if not model_exists:
        print("Models not found. Training models first...\n")
        example_training()

    # Run prediction examples
    example_prediction()
    example_single_prediction()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
