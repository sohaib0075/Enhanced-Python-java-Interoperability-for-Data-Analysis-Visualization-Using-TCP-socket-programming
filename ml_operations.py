"""
Machine Learning Operations Module
Provides ML model training, prediction, and evaluation capabilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
import pickle
import os

# Global model storage
trained_models = {}
MODEL_DIR = "saved_models"

def ensure_model_dir():
    """Create model directory if it doesn't exist"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def train_regression_model(df: pd.DataFrame, target_col: int, feature_cols: list = None, 
                           model_type: str = 'linear', test_size: float = 0.2) -> dict:
    """
    Train a regression model
    
    Args:
        df: DataFrame with data
        target_col: Index of target column
        feature_cols: List of feature column indices (None = use all except target)
        model_type: 'linear' or 'random_forest'
        test_size: Proportion of data for testing
    
    Returns:
        Dictionary with model info and metrics
    """
    try:
        if target_col >= len(df.columns):
            raise ValueError(f"Target column {target_col} out of range")
        
        # Prepare features and target
        if feature_cols is None:
            feature_cols = [i for i in range(len(df.columns)) if i != target_col]
        
        X = df.iloc[:, feature_cols].select_dtypes(include='number')
        y = df.iloc[:, target_col]
        
        if X.empty:
            raise ValueError("No numeric features found")
        
        # Remove rows with NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            raise ValueError("Not enough data for training (need at least 10 samples)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Store model
        model_name = f"regression_{model_type}_{len(trained_models)}"
        trained_models[model_name] = {
            'model': model,
            'type': 'regression',
            'model_type': model_type,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'feature_names': X.columns.tolist()
        }
        
        result = {
            'model_name': model_name,
            'model_type': f'{model_type}_regression',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_count': len(feature_cols)
        }
        
        logging.info(f"Regression model trained: {model_name}, R²={test_r2:.4f}")
        return result
        
    except Exception as e:
        logging.error(f"Error training regression model: {e}", exc_info=True)
        raise

def train_classification_model(df: pd.DataFrame, target_col: int, feature_cols: list = None,
                               model_type: str = 'logistic', test_size: float = 0.2) -> dict:
    """
    Train a classification model
    
    Args:
        df: DataFrame with data
        target_col: Index of target column
        feature_cols: List of feature column indices
        model_type: 'logistic' or 'random_forest'
        test_size: Proportion of data for testing
    
    Returns:
        Dictionary with model info and metrics
    """
    try:
        if target_col >= len(df.columns):
            raise ValueError(f"Target column {target_col} out of range")
        
        if feature_cols is None:
            feature_cols = [i for i in range(len(df.columns)) if i != target_col]
        
        X = df.iloc[:, feature_cols].select_dtypes(include='number')
        y = df.iloc[:, target_col]
        
        if X.empty:
            raise ValueError("No numeric features found")
        
        # Remove rows with NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            raise ValueError("Not enough data for training")
        
        # Check if classification (discrete target)
        unique_values = y.nunique()
        if unique_values > 20:
            raise ValueError(f"Target has {unique_values} unique values. Consider regression instead.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Store model
        model_name = f"classification_{model_type}_{len(trained_models)}"
        trained_models[model_name] = {
            'model': model,
            'type': 'classification',
            'model_type': model_type,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'feature_names': X.columns.tolist()
        }
        
        result = {
            'model_name': model_name,
            'model_type': f'{model_type}_classification',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classes': unique_values,
            'feature_count': len(feature_cols)
        }
        
        logging.info(f"Classification model trained: {model_name}, Accuracy={test_acc:.4f}")
        return result
        
    except Exception as e:
        logging.error(f"Error training classification model: {e}", exc_info=True)
        raise

def predict(model_name: str, features: list) -> float:
    """
    Make prediction using trained model
    
    Args:
        model_name: Name of trained model
        features: List of feature values
    
    Returns:
        Prediction value
    """
    if model_name not in trained_models:
        raise ValueError(f"Model '{model_name}' not found")
    
    model_info = trained_models[model_name]
    model = model_info['model']
    
    # Convert to numpy array
    features_array = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features_array)[0]
    return float(prediction)

def evaluate_model(model_name: str, df: pd.DataFrame) -> dict:
    """
    Evaluate model on new data
    
    Args:
        model_name: Name of trained model
        df: DataFrame with test data
    
    Returns:
        Evaluation metrics
    """
    if model_name not in trained_models:
        raise ValueError(f"Model '{model_name}' not found")
    
    model_info = trained_models[model_name]
    model = model_info['model']
    feature_cols = model_info['feature_cols']
    target_col = model_info['target_col']
    
    X = df.iloc[:, feature_cols].select_dtypes(include='number')
    y = df.iloc[:, target_col]
    
    # Remove NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    y_pred = model.predict(X)
    
    if model_info['type'] == 'regression':
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return {
            'mse': mse,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'samples': len(X)
        }
    else:
        acc = accuracy_score(y, y_pred)
        return {
            'accuracy': acc,
            'samples': len(X)
        }

def feature_importance(model_name: str) -> dict:
    """
    Get feature importance for tree-based models
    
    Args:
        model_name: Name of trained model
    
    Returns:
        Dictionary with feature importances
    """
    if model_name not in trained_models:
        raise ValueError(f"Model '{model_name}' not found")
    
    model_info = trained_models[model_name]
    model = model_info['model']
    
    if model_info['model_type'] in ['random_forest', 'random_forest_classification']:
        importances = model.feature_importances_
        feature_names = model_info['feature_names']
        return dict(zip(feature_names, importances.tolist()))
    else:
        # For linear models, use coefficients
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_names = model_info['feature_names']
            return dict(zip(feature_names, abs(coefs).tolist()))
        return {}

def list_models() -> list:
    """List all trained models"""
    return [
        {
            'name': name,
            'type': info['type'],
            'model_type': info['model_type'],
            'target_col': info['target_col']
        }
        for name, info in trained_models.items()
    ]

def save_model(model_name: str, filename: str = None):
    """Save model to file"""
    if model_name not in trained_models:
        raise ValueError(f"Model '{model_name}' not found")
    
    ensure_model_dir()
    if filename is None:
        filename = f"{model_name}.pkl"
    
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(trained_models[model_name], f)
    
    logging.info(f"Model saved: {filepath}")
    return filepath

def load_model(filepath: str):
    """Load model from file"""
    with open(filepath, 'rb') as f:
        model_info = pickle.load(f)
    
    model_name = os.path.basename(filepath).replace('.pkl', '')
    trained_models[model_name] = model_info
    logging.info(f"Model loaded: {filepath}")
    return model_name

# Initialize model directory
ensure_model_dir()





