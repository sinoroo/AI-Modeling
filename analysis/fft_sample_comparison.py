"""
FFT Sample Size Comparison Script

Compares model performance across different FFT sample sizes (window sizes).
Tests various window sizes to determine optimal sample size for FFT feature extraction.

Usage:
    python analysis/fft_sample_comparison.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import pickle
import torch

# Add package to path (analysis 폴더의 부모 디렉토리)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection import (
    config,
    data_loader,
    preprocessing,
    model_training,
    evaluation,
)

# Result storage (analysis 폴더 하위)
COMPARISON_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "fft_comparison_results")
Path(COMPARISON_RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def train_and_evaluate_model(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             model_type: str = "RandomForest") -> Dict:
    """
    Train a single model and evaluate on test set.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        model_type: Type of model to train
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n  [Model Training] Training {model_type}...")
    
    try:
        # Flatten for classical ML models
        X_test_flat = X_test.reshape(X_test.shape[0], -1) if X_test.ndim == 3 else X_test
        
        if model_type == "RandomForest":
            model = model_training.ClassicalModelTrainer.train_random_forest(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "GradientBoosting":
            model = model_training.ClassicalModelTrainer.train_gradient_boosting(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "AdaBoost":
            model = model_training.ClassicalModelTrainer.train_adaboost(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "ExtraTrees":
            model = model_training.ClassicalModelTrainer.train_extra_trees(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "DecisionTree":
            model = model_training.ClassicalModelTrainer.train_decision_tree(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "KNearestNeighbors":
            model = model_training.ClassicalModelTrainer.train_knn(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "GaussianNB":
            model = model_training.ClassicalModelTrainer.train_gaussian_nb(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "LogisticRegression":
            model = model_training.ClassicalModelTrainer.train_logistic_regression(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "SVM":
            model = model_training.ClassicalModelTrainer.train_svm(X_train, y_train)
            y_pred = model.predict(X_test_flat)
            
        elif model_type == "IsolationForest":
            model = model_training.ClassicalModelTrainer.train_isolation_forest(X_train)
            y_pred = model.predict(X_test_flat)
            y_pred = np.where(y_pred == -1, 1, 0)
            
        elif model_type == "OneClassSVM":
            model = model_training.ClassicalModelTrainer.train_one_class_svm(X_train)
            y_pred = model.predict(X_test_flat)
            y_pred = np.where(y_pred == -1, 1, 0)
            
        elif model_type == "LocalOutlierFactor":
            model = model_training.ClassicalModelTrainer.train_local_outlier_factor(X_train)
            y_pred = model.predict(X_test_flat)
            y_pred = np.where(y_pred == -1, 1, 0)
            
        elif model_type == "EllipticEnvelope":
            model = model_training.ClassicalModelTrainer.train_elliptic_envelope(X_train)
            y_pred = model.predict(X_test_flat)
            y_pred = np.where(y_pred == -1, 1, 0)
            
        elif model_type == "Autoencoder":
            model = model_training.DeepLearningTrainer.train_autoencoder(
                X_train, X_val, params=config.DL_MODELS["Autoencoder"]
            )
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            X_test_tensor = torch.from_numpy(X_test_flat).float()
            model.eval()
            with torch.no_grad():
                X_recon = model(X_test_tensor).numpy()
            reconstruction_error = np.mean((X_test_flat - X_recon) ** 2, axis=1)
            threshold = np.percentile(reconstruction_error, config.DL_MODELS["Autoencoder"].get("threshold_percentile", 95))
            y_pred = (reconstruction_error > threshold).astype(int)
            
        elif model_type == "LSTM":
            model = model_training.DeepLearningTrainer.train_lstm(
                X_train, X_val, params=config.DL_MODELS["LSTM"]
            )
            X_test_tensor = torch.from_numpy(X_test).float()
            model.eval()
            with torch.no_grad():
                X_pred = model(X_test_tensor).numpy()
            prediction_error = np.mean((X_test[:, -1, :] - X_pred) ** 2, axis=1)
            threshold = np.percentile(prediction_error, config.DL_MODELS["LSTM"].get("threshold_percentile", 95))
            y_pred = (prediction_error > threshold).astype(int)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        metrics = evaluation.ModelEvaluator.compute_metrics(y_test, y_pred)
        return metrics
        
    except Exception as e:
        print(f"    Error training {model_type}: {e}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e)
        }


def run_fft_sample_comparison(
    train_path: str = None,
    val_path: str = None,
    test_path: str = None,
    sample_sizes: List[int] = None,
    models_to_test: List[str] = None
) -> Dict:
    """
    Run comparison across different FFT sample sizes.
    
    Args:
        train_path: Training data directory
        val_path: Validation data directory
        test_path: Test data directory
        sample_sizes: List of sample sizes to test
        models_to_test: List of model types to test
    
    Returns:
        Comparison results dictionary
    """
    print("\n" + "="*80)
    print("FFT SAMPLE SIZE COMPARISON")
    print("="*80)
    
    train_path = train_path or config.TRAIN_DATA_DIR
    val_path = val_path or config.VAL_DATA_DIR
    test_path = test_path or config.TEST_DATA_DIR
    sample_sizes = sample_sizes or config.FFT_SAMPLE_SIZES
    models_to_test = models_to_test or ["RandomForest", "Autoencoder"]
    
    if not all([train_path, val_path, test_path]):
        raise ValueError("Data paths not configured")
    
    print("\n[Step 1] Loading data...")
    loader_train = data_loader.DataLoader()
    train_df = loader_train.load_data(train_path)
    
    loader_val = data_loader.DataLoader()
    val_df = loader_val.load_data(val_path)
    
    loader_test = data_loader.DataLoader()
    test_df = loader_test.load_data(test_path)
    
    print("[Step 2] Analyzing data...")
    analysis = loader_train.analyze_columns()
    feature_cols = analysis["feature_cols"]
    label_col = analysis["label_col"]
    
    print(f"  Feature columns: {feature_cols}")
    print(f"  Label column: {label_col}")
    print(f"  Testing sample sizes: {sample_sizes}")
    print(f"  Testing models: {models_to_test}")
    
    results = {
        "sample_sizes": sample_sizes,
        "models": models_to_test,
        "results_by_sample_size": {}
    }
    
    for sample_size in sample_sizes:
        print(f"\n{'='*80}")
        print(f"Testing Sample Size: {sample_size}")
        print(f"{'='*80}")
        
        print(f"\n[Preprocessing with window size={sample_size}]")
        preprocessed = preprocessing.preprocess_data(
            train_df, val_df, test_df, feature_cols, label_col,
            window_size=sample_size
        )
        
        X_train = preprocessed["X_train"]
        y_train = preprocessed["y_train"]
        X_val = preprocessed["X_val"]
        y_val = preprocessed["y_val"]
        X_test = preprocessed["X_test"]
        y_test = preprocessed["y_test"]
        
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        
        sample_size_results = {}
        for model_type in models_to_test:
            metrics = train_and_evaluate_model(
                X_train, y_train, X_val, y_val, X_test, y_test,
                model_type=model_type
            )
            sample_size_results[model_type] = metrics
            print(f"    {model_type}:")
            print(f"      Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"      Precision: {metrics.get('precision', 0):.4f}")
            print(f"      Recall:    {metrics.get('recall', 0):.4f}")
            print(f"      F1 Score:  {metrics.get('f1', 0):.4f}")
        
        results["results_by_sample_size"][sample_size] = sample_size_results
    
    return results


def save_comparison_results(results: Dict, output_file: str = None):
    """Save comparison results to JSON file."""
    if output_file is None:
        output_file = os.path.join(COMPARISON_RESULTS_DIR, "fft_sample_comparison.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved] Results saved to {output_file}")
    except Exception as e:
        print(f"[Error] Failed to save results: {e}")


def print_comparison_summary(results: Dict):
    """Print comparison summary as a formatted table."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    sample_sizes = results.get("sample_sizes", [])
    models = results.get("models", [])
    results_by_size = results.get("results_by_sample_size", {})
    
    for model_type in models:
        print(f"\n{model_type}:")
        print("-" * 80)
        print(f"{'Sample Size':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
        print("-" * 80)
        
        for sample_size in sample_sizes:
            if sample_size in results_by_size:
                model_results = results_by_size[sample_size].get(model_type, {})
                acc = model_results.get("accuracy", 0)
                prec = model_results.get("precision", 0)
                rec = model_results.get("recall", 0)
                f1 = model_results.get("f1", 0)
                
                print(f"{sample_size:<15} {acc:<15.4f} {prec:<15.4f} {rec:<15.4f} {f1:<15.4f}")
        
        print()


def main():
    """Main execution."""
    try:
        results = run_fft_sample_comparison()
        print_comparison_summary(results)
        save_comparison_results(results)
        print("\n✓ FFT Sample Size Comparison Complete!")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
