"""
Data preprocessing module for fire alarm testing price prediction.
Handles data loading, cleaning, feature engineering, and preparation.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    """Handles data preprocessing for fire alarm testing price prediction."""

    def __init__(self, config=None):
        """
        Initialize the preprocessor.

        Args:
            config: Dictionary with configuration parameters
        """
        self.preprocessor = None  # ColumnTransformer
        self.config = config or {}
        self.feature_columns = None
        self.numeric_features = None
        self.categorical_features = None
        self.target_column = "price"

    def load_data(self, file_path):
        """
        Load data from CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
        return df

    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and outliers.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_count:
            print(f"Removed {initial_count - len(df)} duplicate records")

        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"Filled missing values in {col} with median")

        # Handle missing values in categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(
                    df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown",
                    inplace=True,
                )
                print(f"Filled missing values in {col} with mode")

        return df

    def _create_preprocessor(self, df, fit=True):
        """
        Create ColumnTransformer with OneHotEncoder for categorical features.

        Args:
            df: Input DataFrame
            fit: Whether to fit transformers

        Returns:
            ColumnTransformer
        """
        # Identify numeric and categorical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # Exclude target column
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)

        transformers = []

        # Numeric features: StandardScaler
        if numeric_cols:
            transformers.append(("num", StandardScaler(), numeric_cols))

        # Categorical features: OneHotEncoder (not LabelEncoder!)
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                    categorical_cols,
                )
            )

        if transformers:
            preprocessor = ColumnTransformer(
                transformers=transformers, remainder="drop"
            )
            if fit:
                preprocessor.fit(df)
            return preprocessor, numeric_cols, categorical_cols

        return None, numeric_cols, categorical_cols

    def prepare_features(self, df, fit=True):
        """
        Prepare features for model training/prediction using OneHotEncoder.

        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for prediction)

        Returns:
            Tuple of (features, target) or just features if target doesn't exist
        """
        df = df.copy()

        # Separate features and target
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            X = df
            y = None

        # Store feature columns for later use
        if fit:
            self.feature_columns = X.columns.tolist()

        # Create or use existing preprocessor
        if fit or self.preprocessor is None:
            self.preprocessor, self.numeric_features, self.categorical_features = (
                self._create_preprocessor(X, fit=fit)
            )

        if self.preprocessor is None:
            raise ValueError("Could not create preprocessor")

        # Transform features
        if fit:
            X_transformed = self.preprocessor.fit_transform(X)
        else:
            X_transformed = self.preprocessor.transform(X)

        # Get feature names after transformation
        feature_names = []

        # Numeric features (same names)
        if self.numeric_features:
            feature_names.extend(self.numeric_features)

        # Categorical features (OneHot creates multiple columns)
        if self.categorical_features and hasattr(self.preprocessor, "transformers_"):
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == "cat" and hasattr(transformer, "get_feature_names_out"):
                    cat_names = transformer.get_feature_names_out(
                        self.categorical_features
                    )
                    feature_names.extend(cat_names)

        # If we couldn't get feature names, use default
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        if y is not None:
            return X_df, y
        else:
            return X_df

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.

        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_preprocessor(self, file_path):
        """Save the preprocessor (ColumnTransformer) to disk."""
        preprocessor_data = {
            "preprocessor": self.preprocessor,
            "feature_columns": self.feature_columns,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_column": self.target_column,
        }

        with open(file_path, "wb") as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor saved to {file_path}")

    def load_preprocessor(self, file_path):
        """Load the preprocessor from disk."""
        with open(file_path, "rb") as f:
            preprocessor_data = pickle.load(f)

        # Handle backward compatibility
        if "preprocessor" in preprocessor_data:
            # New format with ColumnTransformer
            self.preprocessor = preprocessor_data["preprocessor"]
            self.numeric_features = preprocessor_data.get("numeric_features", [])
            self.categorical_features = preprocessor_data.get(
                "categorical_features", []
            )
        else:
            # Old format - would need migration
            raise ValueError(
                "Old preprocessor format detected. Please retrain with new pipeline."
            )

        self.feature_columns = preprocessor_data["feature_columns"]
        self.target_column = preprocessor_data.get("target_column", "price")
        print(f"Preprocessor loaded from {file_path}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()

    # Load and process data
    df = preprocessor.load_data("../data/training_data.csv")
    df = preprocessor.clean_data(df)
    X, y = preprocessor.prepare_features(df, fit=True)

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {X_train.shape[1]}")
