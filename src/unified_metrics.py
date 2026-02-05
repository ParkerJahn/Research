#!/usr/bin/env python3
"""
Unified Metrics Calculation Module
===================================

This module provides a single source of truth for all forecast evaluation metrics
used across the research project. This ensures consistency in RMSE, MAE, and other
metrics reported in figures, tables, and analysis scripts.

All scripts should import and use these functions instead of calculating metrics
independently to avoid inconsistencies.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional
import warnings


def validate_inputs(actual: np.ndarray, predicted: np.ndarray, 
                   min_observations: int = 10) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Validate and clean input arrays for metric calculation.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    predicted : np.ndarray
        Predicted/forecast values
    min_observations : int
        Minimum number of valid observations required
        
    Returns
    -------
    actual_clean : np.ndarray
        Cleaned actual values
    predicted_clean : np.ndarray
        Cleaned predicted values
    is_valid : bool
        Whether the data meets minimum requirements
    """
    # Convert to numpy arrays if needed
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # Check shape compatibility
    if actual.shape != predicted.shape:
        warnings.warn(f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}")
        return actual, predicted, False
    
    # Remove NaN and Inf values
    valid_mask = ~(np.isnan(actual) | np.isnan(predicted) | 
                   np.isinf(actual) | np.isinf(predicted))
    
    actual_clean = actual[valid_mask]
    predicted_clean = predicted[valid_mask]
    
    # Check minimum observations
    is_valid = len(actual_clean) >= min_observations
    
    if not is_valid:
        warnings.warn(f"Insufficient valid observations: {len(actual_clean)} < {min_observations}")
    
    return actual_clean, predicted_clean, is_valid


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray, 
                   validate: bool = True) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    This is the AUTHORITATIVE RMSE calculation for the entire project.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    predicted : np.ndarray
        Predicted/forecast values
    validate : bool
        Whether to validate inputs (default: True)
        
    Returns
    -------
    rmse : float
        Root Mean Squared Error, or np.nan if validation fails
    """
    if validate:
        actual, predicted, is_valid = validate_inputs(actual, predicted)
        if not is_valid:
            return np.nan
    
    return np.sqrt(mean_squared_error(actual, predicted))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray,
                 validate: bool = True) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    predicted : np.ndarray
        Predicted/forecast values
    validate : bool
        Whether to validate inputs (default: True)
        
    Returns
    -------
    mae : float
        Mean Absolute Error, or np.nan if validation fails
    """
    if validate:
        actual, predicted, is_valid = validate_inputs(actual, predicted)
        if not is_valid:
            return np.nan
    
    return mean_absolute_error(actual, predicted)


def calculate_qlike(actual: np.ndarray, predicted: np.ndarray,
                   validate: bool = True, epsilon: float = 1e-10) -> float:
    """
    Calculate QLIKE (Quasi-Likelihood) loss for volatility forecasting.
    
    QLIKE = mean(actual/predicted - log(actual/predicted) - 1)
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values (must be positive)
    predicted : np.ndarray
        Predicted/forecast values (must be positive)
    validate : bool
        Whether to validate inputs (default: True)
    epsilon : float
        Small constant to prevent division by zero
        
    Returns
    -------
    qlike : float
        QLIKE loss, or np.nan if validation fails
    """
    if validate:
        actual, predicted, is_valid = validate_inputs(actual, predicted)
        if not is_valid:
            return np.nan
    
    # Ensure positive values for volatility
    actual = np.maximum(actual, epsilon)
    predicted = np.maximum(predicted, epsilon)
    
    ratio = actual / predicted
    qlike = np.mean(ratio - np.log(ratio) - 1)
    
    return qlike


def calculate_mse(actual: np.ndarray, predicted: np.ndarray,
                 validate: bool = True) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    predicted : np.ndarray
        Predicted/forecast values
    validate : bool
        Whether to validate inputs (default: True)
        
    Returns
    -------
    mse : float
        Mean Squared Error, or np.nan if validation fails
    """
    if validate:
        actual, predicted, is_valid = validate_inputs(actual, predicted)
        if not is_valid:
            return np.nan
    
    return mean_squared_error(actual, predicted)


def calculate_directional_accuracy(actual: np.ndarray, predicted: np.ndarray,
                                   validate: bool = True) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed changes
    predicted : np.ndarray
        Predicted changes
    validate : bool
        Whether to validate inputs (default: True)
        
    Returns
    -------
    accuracy : float
        Directional accuracy (0-1), or np.nan if validation fails
    """
    if validate:
        actual, predicted, is_valid = validate_inputs(actual, predicted)
        if not is_valid:
            return np.nan
    
    correct = np.sign(actual) == np.sign(predicted)
    return np.mean(correct)


def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray,
                         include_qlike: bool = True,
                         include_directional: bool = False) -> Dict[str, float]:
    """
    Calculate all standard forecast evaluation metrics.
    
    This is the UNIFIED function that should be used throughout the project
    to ensure consistency across all outputs.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    predicted : np.ndarray
        Predicted/forecast values
    include_qlike : bool
        Whether to include QLIKE (for volatility forecasting)
    include_directional : bool
        Whether to include directional accuracy
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'RMSE': Root Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'MSE': Mean Squared Error
        - 'QLIKE': QLIKE loss (if include_qlike=True)
        - 'Directional_Accuracy': Directional accuracy (if include_directional=True)
        - 'N': Number of valid observations
    """
    # Validate once for all metrics
    actual_clean, predicted_clean, is_valid = validate_inputs(actual, predicted)
    
    if not is_valid:
        metrics = {
            'RMSE': np.nan,
            'MAE': np.nan,
            'MSE': np.nan,
            'N': len(actual_clean)
        }
        if include_qlike:
            metrics['QLIKE'] = np.nan
        if include_directional:
            metrics['Directional_Accuracy'] = np.nan
        return metrics
    
    # Calculate all metrics without re-validating
    metrics = {
        'RMSE': calculate_rmse(actual_clean, predicted_clean, validate=False),
        'MAE': calculate_mae(actual_clean, predicted_clean, validate=False),
        'MSE': calculate_mse(actual_clean, predicted_clean, validate=False),
        'N': len(actual_clean)
    }
    
    if include_qlike:
        metrics['QLIKE'] = calculate_qlike(actual_clean, predicted_clean, validate=False)
    
    if include_directional:
        metrics['Directional_Accuracy'] = calculate_directional_accuracy(
            actual_clean, predicted_clean, validate=False
        )
    
    return metrics


def calculate_improvement(baseline_metric: float, model_metric: float,
                         as_percentage: bool = True) -> float:
    """
    Calculate improvement of model over baseline.
    
    Parameters
    ----------
    baseline_metric : float
        Baseline model metric (e.g., RMSE)
    model_metric : float
        Model metric to compare
    as_percentage : bool
        Return as percentage (default: True)
        
    Returns
    -------
    improvement : float
        Improvement metric. Positive means model is better than baseline.
        If as_percentage=True, returns percentage improvement.
    """
    if np.isnan(baseline_metric) or np.isnan(model_metric):
        return np.nan
    
    if baseline_metric == 0:
        warnings.warn("Baseline metric is zero, cannot calculate improvement")
        return np.nan
    
    improvement = (baseline_metric - model_metric) / baseline_metric
    
    if as_percentage:
        improvement *= 100
    
    return improvement


def format_metrics_for_display(metrics: Dict[str, float], 
                               precision: int = 3) -> Dict[str, str]:
    """
    Format metrics dictionary for display in tables/figures.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric values
    precision : int
        Number of decimal places
        
    Returns
    -------
    formatted : dict
        Dictionary with formatted string values
    """
    formatted = {}
    for key, value in metrics.items():
        if key == 'N':
            formatted[key] = str(int(value)) if not np.isnan(value) else 'N/A'
        elif np.isnan(value):
            formatted[key] = 'N/A'
        else:
            formatted[key] = f"{value:.{precision}f}"
    
    return formatted
