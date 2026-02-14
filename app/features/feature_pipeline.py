"""
Feature engineering pipeline with versioning.
Handles feature transformation, validation, and storage.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeaturePipeline:
    """
    Production-ready feature pipeline with versioning.
    """

    def __init__(self, version: str = "v1.0"):
        """
        Initialize feature pipeline.

        Args:
            version: Pipeline version for tracking
        """
        self.version = version
        self.preprocessor = None  # ColumnTransformer
        self.feature_columns: Optional[List[str]] = None
        self.numeric_features: Optional[List[str]] = None
        self.categorical_features: Optional[List[str]] = None
        self.target_column = "price"
        self.metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "feature_columns": None,
            "numeric_features": None,
            "categorical_features": None,
        }

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """
        Fit the feature pipeline on training data.

        Args:
            df: Training DataFrame with features and target

        Returns:
            Self for chaining
        """
        df = df.copy()

        # Store feature columns (exclude target)
        if self.target_column in df.columns:
            feature_cols = [col for col in df.columns if col != self.target_column]
        else:
            feature_cols = df.columns.tolist()

        self.feature_columns = sorted(feature_cols)
        self.metadata["feature_columns"] = self.feature_columns

        # Identify numeric and categorical features
        numeric_cols = df[self.feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = (
            df[self.feature_columns].select_dtypes(include=["object"]).columns.tolist()
        )

        self.numeric_features = numeric_cols
        self.categorical_features = categorical_cols
        self.metadata["numeric_features"] = numeric_cols
        self.metadata["categorical_features"] = categorical_cols

        # Create ColumnTransformer with proper preprocessing
        transformers = []

        # Numeric features: StandardScaler
        if numeric_cols:
            transformers.append(("num", StandardScaler(), numeric_cols))

        # Categorical features: OneHotEncoder (not LabelEncoder!)
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    categorical_cols,
                )
            )

        # Create preprocessor
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",  # Drop any columns not explicitly handled
            )

            # Fit the preprocessor
            X = df[self.feature_columns]
            self.preprocessor.fit(X)

        return self

    def transform(
        self, df: pd.DataFrame, fit: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform features for model input.

        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training)

        Returns:
            Tuple of (features, target) or (features, None) if no target
        """
        df = df.copy()

        if self.preprocessor is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # Separate target if present
        target = None
        if self.target_column in df.columns:
            target = df[self.target_column]
            df = df.drop(columns=[self.target_column])

        # Ensure all required features are present
        # Handle encoding issues by creating a mapping
        column_mapping = {}
        missing = []

        for expected_col in self.feature_columns:
            if expected_col in df.columns:
                # Direct match
                column_mapping[expected_col] = expected_col
            else:
                # Try to find matching column (handle encoding issues)
                found = False
                for df_col in df.columns:
                    # Normalize both for comparison
                    norm_expected = (
                        expected_col.lower().replace("ö", "o").replace("ä", "a").replace("å", "a")
                    )
                    norm_df = df_col.lower().replace("ö", "o").replace("ä", "a").replace("å", "a")
                    if norm_expected == norm_df:
                        column_mapping[expected_col] = df_col
                        found = True
                        break

                if not found:
                    missing.append(expected_col)

        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Rename columns to match expected names
        if column_mapping and any(k != v for k, v in column_mapping.items()):
            df = df.rename(columns={v: k for k, v in column_mapping.items()})

        # Select and order features
        X = df[self.feature_columns]

        # Transform using ColumnTransformer
        if fit:
            X_transformed = self.preprocessor.fit_transform(X)
        else:
            X_transformed = self.preprocessor.transform(X)

        # Get feature names after transformation
        # OneHotEncoder creates multiple columns per categorical feature
        feature_names = []

        # Numeric features (same names)
        if self.numeric_features:
            feature_names.extend(self.numeric_features)

        # Categorical features (OneHot creates multiple columns)
        if self.categorical_features and hasattr(self.preprocessor, "transformers_"):
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == "cat" and hasattr(transformer, "get_feature_names_out"):
                    # Get OneHot encoded feature names
                    cat_names = transformer.get_feature_names_out(self.categorical_features)
                    feature_names.extend(cat_names)

        # If we couldn't get feature names, use default
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        # Convert to DataFrame
        df_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        return df_transformed, target

    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        pipeline_data = {
            "version": self.version,
            "preprocessor": self.preprocessor,
            "feature_columns": self.feature_columns,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_column": self.target_column,
            "metadata": self.metadata,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(pipeline_data, f)

        # Also save metadata as JSON for readability
        metadata_path = path.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Load pipeline from disk."""
        with open(path, "rb") as f:
            pipeline_data = pickle.load(f)

        pipeline = cls(version=pipeline_data.get("version", "v1.0"))
        pipeline.preprocessor = pipeline_data.get("preprocessor")

        # Handle backward compatibility with old format
        if "preprocessor" not in pipeline_data:
            # Old format with scaler and label_encoders - would need migration
            raise ValueError("Old pipeline format detected. Please retrain with new pipeline.")

        # Fix encoding issues with Swedish characters
        # When pickle saves/loads, encoding can get corrupted
        feature_columns = pipeline_data["feature_columns"]

        # Known correct column names in expected order
        correct_columns = [
            "antal_sektioner",
            "antal_detektorer",
            "antal_larmdon",
            "dörrhållarmagneter",
            "ventilation",
            "stad",
            "kvartalsvis",
            "månadsvis",
            "årsvis",
        ]

        # Create mapping: normalize both corrupted and correct names for matching
        fixed_columns = []
        for i, col in enumerate(feature_columns):
            # Normalize the corrupted column name (remove special chars)
            col_normalized = "".join(c for c in col.lower() if c.isalnum())

            # Try to match against correct columns
            matched = False
            for correct_col in correct_columns:
                correct_normalized = "".join(c for c in correct_col.lower() if c.isalnum())
                if col_normalized == correct_normalized:
                    fixed_columns.append(correct_col)
                    matched = True
                    break

            if not matched:
                # If we can't match, try by position (assuming order is preserved)
                if i < len(correct_columns):
                    fixed_columns.append(correct_columns[i])
                else:
                    # Keep original as fallback
                    fixed_columns.append(col)

        pipeline.feature_columns = fixed_columns
        pipeline.numeric_features = pipeline_data.get("numeric_features", [])
        pipeline.categorical_features = pipeline_data.get("categorical_features", [])
        pipeline.target_column = pipeline_data.get("target_column", "price")
        pipeline.metadata = pipeline_data.get("metadata", {})

        return pipeline


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    data_path = Path(__file__).parent.parent.parent / "data" / "training_data.csv"
    df = pd.read_csv(data_path)

    # Basic cleaning
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    pipeline = FeaturePipeline(version="v2.0")  # v2.0 = OneHotEncoder
    X, y = pipeline.fit(df).transform(df, fit=True)

    print(f"Pipeline version: {pipeline.version}")
    print(f"Original features: {len(pipeline.feature_columns)}")
    print(f"  - Numeric: {len(pipeline.numeric_features)}")
    print(f"  - Categorical: {len(pipeline.categorical_features)}")
    print(f"Transformed shape: {X.shape}")
    print(f"  (OneHotEncoder expands categorical features)")
