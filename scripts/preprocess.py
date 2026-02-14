"""
Data preprocessing module for fire alarm testing price prediction.
Handles data loading, cleaning, feature engineering, and preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os


class DataPreprocessor:
    """Handles data preprocessing for fire alarm testing price prediction."""
    
    def __init__(self, config=None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.config = config or {}
        self.feature_columns = None
        self.target_column = 'price'
        
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
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
                print(f"Filled missing values in {col} with mode")
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Exclude target column if it exists
        if self.target_column in categorical_cols:
            categorical_cols = categorical_cols.drop(self.target_column)
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unknown_values = unique_values - known_values
                    
                    if unknown_values:
                        print(f"Warning: Unknown categories in {col}: {unknown_values}")
                        # Replace unknown with most common known value
                        df[col] = df[col].astype(str).replace(list(unknown_values), 
                                                               self.label_encoders[col].classes_[0])
                    
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    print(f"Warning: No encoder found for {col}, skipping encoding")
        
        return df
    
    def prepare_features(self, df, fit=True):
        """
        Prepare features for model training/prediction.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scaler (True for training, False for prediction)
            
        Returns:
            Tuple of (features, target) or just features if target doesn't exist
        """
        df = df.copy()
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Separate features and target
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Store feature columns for later use
            if fit:
                self.feature_columns = X.columns.tolist()
            
            # Scale features
            if fit:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            return pd.DataFrame(X_scaled, columns=self.feature_columns), y
        else:
            # Prediction mode - no target column
            if self.feature_columns is None:
                raise ValueError("Feature columns not defined. Train the model first.")
            
            # Ensure columns match training data
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing features: {missing_cols}")
            
            X = df[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=self.feature_columns)
    
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
        """Save the preprocessor (scaler and encoders) to disk."""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path):
        """Load the preprocessor from disk."""
        with open(file_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_columns = preprocessor_data['feature_columns']
        self.target_column = preprocessor_data.get('target_column', 'price')
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
