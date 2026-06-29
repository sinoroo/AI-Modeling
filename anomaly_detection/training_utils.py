"""
Utility functions for training optimization and debugging.
"""

import numpy as np
import time
from typing import Tuple, Dict, Any


def validate_training_data(X: np.ndarray, y: np.ndarray = None, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and clean training data.
    
    Args:
        X: Feature matrix
        y: Optional labels
        verbose: Print validation messages
        
    Returns:
        Cleaned X and y (if provided)
    """
    if verbose:
        print(f"[Validation] Starting data validation...")
        print(f"  Input shape: X={X.shape}, y={y.shape if y is not None else 'None'}")
    
    # Check for NaN values
    nan_mask = ~np.isnan(X).any(axis=1) if X.ndim == 2 else ~np.isnan(X).any(axis=(1, 2))
    nan_removed = (~nan_mask).sum()
    
    if nan_removed > 0:
        if verbose:
            print(f"  Removing {nan_removed} rows with NaN values")
        X = X[nan_mask]
        if y is not None:
            y = y[nan_mask]
    
    # Check for Inf values
    inf_mask = np.isfinite(X).all(axis=1) if X.ndim == 2 else np.isfinite(X).all(axis=(1, 2))
    inf_removed = (~inf_mask).sum()
    
    if inf_removed > 0:
        if verbose:
            print(f"  Removing {inf_removed} rows with Inf values")
        X = X[inf_mask]
        if y is not None:
            y = y[inf_mask]
    
    # Check data statistics
    if verbose:
        if X.ndim >= 2:
            flat_X = X.reshape(X.shape[0], -1) if X.ndim > 2 else X
            print(f"  Data statistics:")
            print(f"    Min: {flat_X.min():.4f}, Max: {flat_X.max():.4f}")
            print(f"    Mean: {flat_X.mean():.4f}, Std: {flat_X.std():.4f}")
            
            # Check for zero variance
            variances = np.var(flat_X, axis=0)
            zero_var_cols = np.sum(variances == 0)
            if zero_var_cols > 0:
                print(f"  WARNING: {zero_var_cols} features have zero variance!")
        
        print(f"  Final shape: X={X.shape}, y={y.shape if y is not None else 'None'}")
    
    return X, y if y is not None else X


def timed_training(model_class, method_name: str, *args, verbose: bool = True, **kwargs):
    """
    Train a model with timing information.
    
    Args:
        model_class: Class or function for training
        method_name: Name of the model
        *args: Positional arguments for training
        verbose: Print timing info
        **kwargs: Keyword arguments for training
        
    Returns:
        Trained model, elapsed time
    """
    if verbose:
        print(f"[Timing] Starting '{method_name}' training...")
    
    start_time = time.time()
    
    try:
        result = model_class(*args, **kwargs)
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"[Timing] '{method_name}' completed in {elapsed:.2f}s ({elapsed/60:.2f}m)")
        
        return result, elapsed
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Timing] '{method_name}' failed after {elapsed:.2f}s")
        raise e


def get_data_summary(X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
    """
    Get summary statistics of training data.
    
    Args:
        X: Feature matrix
        y: Optional labels
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "X_shape": X.shape,
        "y_shape": y.shape if y is not None else None,
        "n_samples": X.shape[0],
        "n_features": X.shape[1] if X.ndim == 2 else np.prod(X.shape[1:]),
    }
    
    if X.ndim > 2:
        flat_X = X.reshape(X.shape[0], -1)
    else:
        flat_X = X
    
    summary.update({
        "min_value": float(flat_X.min()),
        "max_value": float(flat_X.max()),
        "mean_value": float(flat_X.mean()),
        "std_value": float(flat_X.std()),
        "nan_count": int(np.isnan(flat_X).sum()),
        "inf_count": int(np.isinf(flat_X).sum()),
    })
    
    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        summary["class_distribution"] = dict(zip(map(int, unique), map(int, counts)))
    
    return summary


def print_data_summary(summary: Dict[str, Any]):
    """Pretty print data summary."""
    print("\n[Data Summary]")
    print(f"  Shape: X={summary['X_shape']}, y={summary['y_shape']}")
    print(f"  Samples: {summary['n_samples']}, Features: {summary['n_features']}")
    print(f"  Range: [{summary['min_value']:.4f}, {summary['max_value']:.4f}]")
    print(f"  Mean: {summary['mean_value']:.4f}, Std: {summary['std_value']:.4f}")
    print(f"  NaN: {summary['nan_count']}, Inf: {summary['inf_count']}")
    
    if summary.get('class_distribution'):
        print(f"  Classes: {summary['class_distribution']}")
