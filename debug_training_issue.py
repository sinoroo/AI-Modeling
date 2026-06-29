"""
Debug script to analyze training data and identify One-Class SVM hang issue.
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anomaly_detection import config, data_loader, preprocessing

def analyze_data_sizes():
    """Analyze data file sizes and structure."""
    print("\n" + "="*80)
    print("1. DATA FILES ANALYSIS")
    print("="*80)
    
    train_dir = config.TRAIN_DATA_DIR
    print(f"Train data directory: {train_dir}\n")
    
    total_rows = 0
    file_info = []
    
    for filename in sorted(os.listdir(train_dir)):
        if filename.endswith(".csv"):
            filepath = os.path.join(train_dir, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Total lines - 9 metadata lines = data lines
            data_lines = len(lines) - 9
            total_rows += data_lines
            
            file_info.append({
                'File': filename,
                'Total Lines': len(lines),
                'Data Lines': data_lines
            })
            
            print(f"  {filename:30s}: {len(lines):6d} lines ({data_lines:5d} data rows)")
    
    print(f"\nTotal data rows: {total_rows}")
    return total_rows

def analyze_preprocessing_output():
    """Load and analyze preprocessed data."""
    print("\n" + "="*80)
    print("2. PREPROCESSING OUTPUT ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    loader_train = data_loader.DataLoader()
    train_df = loader_train.load_data(config.TRAIN_DATA_DIR)
    
    loader_val = data_loader.DataLoader()
    val_df = loader_val.load_data(config.VAL_DATA_DIR)
    
    loader_test = data_loader.DataLoader()
    test_df = loader_test.load_data(config.TEST_DATA_DIR)
    
    # Analyze data
    analysis = loader_train.analyze_columns()
    feature_cols = analysis["feature_cols"]
    label_col = analysis["label_col"]
    
    print(f"Feature columns: {feature_cols}")
    print(f"Label column: {label_col}")
    print(f"\nData shapes:")
    print(f"  Train: {train_df.shape}")
    print(f"  Val:   {val_df.shape}")
    print(f"  Test:  {test_df.shape}")
    
    # Check for NaN values
    print(f"\nMissing values:")
    print(f"  Train: {train_df[feature_cols].isna().sum().sum()}")
    print(f"  Val:   {val_df[feature_cols].isna().sum().sum()}")
    print(f"  Test:  {test_df[feature_cols].isna().sum().sum()}")
    
    return train_df, val_df, test_df, feature_cols, label_col

def analyze_processed_arrays(train_df, val_df, test_df, feature_cols, label_col):
    """Preprocess and analyze resulting arrays."""
    print("\n" + "="*80)
    print("3. PROCESSED ARRAYS ANALYSIS")
    print("="*80)
    
    print("\nPreprocessing data...")
    preprocessed = preprocessing.preprocess_data(
        train_df, val_df, test_df, feature_cols, label_col
    )
    
    X_train = preprocessed["X_train"]
    y_train = preprocessed["y_train"]
    X_val = preprocessed["X_val"]
    y_val = preprocessed["y_val"]
    X_test = preprocessed["X_test"]
    y_test = preprocessed["y_test"]
    
    print(f"\nArray shapes:")
    print(f"  X_train: {X_train.shape} ({X_train.size} elements)")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape} ({X_val.size} elements)")
    print(f"  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape} ({X_test.size} elements)")
    print(f"  y_test:  {y_test.shape}")
    
    # Flatten for classical ML
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"\nFlattened shapes (for Classical ML):")
    print(f"  X_train_flat: {X_train_flat.shape} ({X_train_flat.size} elements)")
    print(f"  X_val_flat:   {X_val_flat.shape} ({X_val_flat.size} elements)")
    print(f"  X_test_flat:  {X_test_flat.shape} ({X_test_flat.size} elements)")
    
    # Data statistics
    print(f"\nData statistics (training features):")
    print(f"  Min: {X_train_flat.min():.6f}")
    print(f"  Max: {X_train_flat.max():.6f}")
    print(f"  Mean: {X_train_flat.mean():.6f}")
    print(f"  Std: {X_train_flat.std():.6f}")
    print(f"  Median: {np.median(X_train_flat):.6f}")
    
    # Check for NaN, Inf values
    print(f"\nData integrity:")
    print(f"  NaN count: {np.isnan(X_train_flat).sum()}")
    print(f"  Inf count: {np.isinf(X_train_flat).sum()}")
    
    # Class distribution
    print(f"\nClass distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({c/len(y_train)*100:.1f}%)")
    
    return X_train_flat, X_val_flat, X_test_flat, y_train, y_val, y_test

def estimate_oneclass_svm_time(n_samples, n_features):
    """Estimate One-Class SVM training time."""
    print("\n" + "="*80)
    print("4. ONE-CLASS SVM COMPLEXITY ANALYSIS")
    print("="*80)
    
    # One-Class SVM complexity is O(n^2.5) to O(n^3) for RBF kernel
    # with typical complexity around O(n^2.7)
    
    print(f"\nInput dimensions:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    
    print(f"\nOne-Class SVM Configuration:")
    print(f"  Kernel: rbf")
    print(f"  Gamma: auto (1/{n_features} = {1/n_features:.6f})")
    print(f"  Nu: 0.05")
    
    # Rough complexity estimate
    complexity_exponent = 2.5  # Conservative estimate
    
    # Baseline: 1000 samples takes ~1 second
    baseline_samples = 1000
    baseline_time = 1.0
    
    estimated_time = baseline_time * ((n_samples / baseline_samples) ** complexity_exponent)
    
    print(f"\nEstimated training time (rough):")
    print(f"  Complexity: O(n^{complexity_exponent})")
    print(f"  Estimated time: ~{estimated_time:.1f} seconds (~{estimated_time/60:.1f} minutes)")
    
    # Warning if too large
    if estimated_time > 60:
        print(f"\n⚠️  WARNING: One-Class SVM may take >1 minute!")
        print(f"   Consider using smaller gamma or different anomaly detection method")
    
    return estimated_time

def main():
    """Run all analysis."""
    print("\n" + "█"*80)
    print("AI-Modeling FFT | TRAINING DEBUG ANALYSIS")
    print("█"*80)
    
    # Step 1: Analyze files
    total_rows = analyze_data_sizes()
    
    # Step 2: Load and analyze data
    train_df, val_df, test_df, feature_cols, label_col = analyze_preprocessing_output()
    
    # Step 3: Preprocess and analyze arrays
    X_train_flat, X_val_flat, X_test_flat, y_train, y_val, y_test = analyze_processed_arrays(
        train_df, val_df, test_df, feature_cols, label_col
    )
    
    # Step 4: Estimate One-Class SVM time
    estimate_oneclass_svm_time(X_train_flat.shape[0], X_train_flat.shape[1])
    
    print("\n" + "█"*80)
    print("Analysis complete!")
    print("█"*80 + "\n")

if __name__ == "__main__":
    main()
