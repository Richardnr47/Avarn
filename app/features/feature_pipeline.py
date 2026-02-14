"""
Feature engineering pipeline with versioning.
Handles feature transformation, validation, and storage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json


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
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: Optional[List[str]] = None
        self.target_column = 'price'
        self.metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'feature_columns': None
        }
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
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
        self.metadata['feature_columns'] = self.feature_columns
        
        # Encode categorical features
        categorical_cols = df[self.feature_columns].select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Scale numeric features
        numeric_cols = df[self.feature_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_numeric = df[numeric_cols]
            self.scaler.fit(df_numeric)
        
        return self
    
    def transform(self, df: pd.DataFrame, fit: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform features for model input.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training)
            
        Returns:
            Tuple of (features, target) or (features, None) if no target
        """
        df = df.copy()
        
        if self.feature_columns is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Separate target if present
        target = None
        if self.target_column in df.columns:
            target = df[self.target_column]
            df = df.drop(columns=[self.target_column])
        
        # Ensure all required features are present
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Select and order features
        df = df[self.feature_columns]
        
        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in self.label_encoders:
                # Handle unseen categories
                unique_values = set(df[col].astype(str).unique())
                known_values = set(self.label_encoders[col].classes_)
                unknown_values = unique_values - known_values
                
                if unknown_values:
                    # Replace unknown with most common known value
                    default_value = self.label_encoders[col].classes_[0]
                    df[col] = df[col].astype(str).replace(list(unknown_values), default_value)
                
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
            elif fit:
                # Fit new encoder
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Scale numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if fit:
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # Convert to DataFrame with proper column names
        df_transformed = pd.DataFrame(df, columns=self.feature_columns)
        
        return df_transformed, target
    
    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        pipeline_data = {
            'version': self.version,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'metadata': self.metadata
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        # Also save metadata as JSON for readability
        metadata_path = path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'FeaturePipeline':
        """Load pipeline from disk."""
        with open(path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        pipeline = cls(version=pipeline_data.get('version', 'v1.0'))
        pipeline.scaler = pipeline_data['scaler']
        pipeline.label_encoders = pipeline_data['label_encoders']
        pipeline.feature_columns = pipeline_data['feature_columns']
        pipeline.target_column = pipeline_data.get('target_column', 'price')
        pipeline.metadata = pipeline_data.get('metadata', {})
        
        return pipeline


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "data" / "training_data.csv"
    df = pd.read_csv(data_path)
    
    pipeline = FeaturePipeline(version="v1.0")
    X, y = pipeline.transform(df, fit=True)
    
    print(f"Pipeline version: {pipeline.version}")
    print(f"Features: {len(pipeline.feature_columns)}")
    print(f"Transformed shape: {X.shape}")
