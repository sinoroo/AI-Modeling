"""
Comprehensive Anomaly Detection Model Comparison Script

Compares all anomaly detection models:
- IsolationForest
- OneClassSVM
- LocalOutlierFactor
- EllipticEnvelope
- RobustCovariance
- MinCovDet
- KMeansAnomaly
- PCAAnomaly
- DBSCAN
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

# Add package to path (analysis 폴더의 부모 디렉토리)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection import config
from anomaly_detection import data_loader
from anomaly_detection import preprocessing
from anomaly_detection import model_training
from anomaly_detection import evaluation


# ============================================================================
# ANOMALY DETECTION MODELS
# ============================================================================

ANOMALY_DETECTION_MODELS = [
    "IsolationForest",
    "OneClassSVM",
    "LocalOutlierFactor",
    "EllipticEnvelope",
    "RobustCovariance",
    "MinCovDet",
    "KMeansAnomaly",
    "PCAAnomaly",
    "DBSCAN",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_anomaly_predictions(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from anomaly detection model.
    
    Returns:
    - predictions: -1 for anomaly, 1 for normal
    - scores: anomaly scores (higher = more anomalous)
    """
    X_flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
    
    model_name = model.__class__.__name__
    
    if hasattr(model, 'predict'):
        predictions = model.predict(X_flat)
    else:
        predictions = np.ones(X_flat.shape[0])
    
    scores = np.zeros(X_flat.shape[0])
    
    if hasattr(model, 'score_samples'):
        scores = -model.score_samples(X_flat)
    elif hasattr(model, 'decision_function'):
        scores = -model.decision_function(X_flat)
    elif model_name == 'KMeans':
        scores = np.min(model.transform(X_flat), axis=1)
    elif model_name == 'PCA':
        X_transformed = model.transform(X_flat)
        X_reconstructed = model.inverse_transform(X_transformed)
        scores = np.mean((X_flat - X_reconstructed) ** 2, axis=1)
    elif model_name == 'DBSCAN':
        predictions = model.labels_
        scores = np.where(predictions == -1, 1.0, 0.0)
    
    return predictions, scores


def convert_predictions_to_binary(predictions: np.ndarray) -> np.ndarray:
    """Convert anomaly predictions to binary labels (0=normal, 1=anomaly)."""
    return np.where(predictions == -1, 1, 0)


def calculate_anomaly_detection_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray
) -> Dict:
    """
    Calculate metrics for anomaly detection.
    
    y_true: Binary labels (1=anomaly, 0=normal)
    predictions: Model predictions (-1=anomaly, 1=normal)
    scores: Anomaly scores
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, roc_curve, auc
    )
    
    y_pred = convert_predictions_to_binary(predictions)
    
    if len(np.unique(y_pred)) == 1:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        try:
            roc_auc = roc_auc_score(y_true, scores_norm)
        except:
            roc_auc = 0.5
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
    except Exception as e:
        print(f"[WARNING] Error calculating metrics: {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_and_evaluate_anomaly_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """Train and evaluate a single anomaly detection model."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    try:
        method_map = {
            "IsolationForest": "train_isolation_forest",
            "OneClassSVM": "train_one_class_svm",
            "LocalOutlierFactor": "train_local_outlier_factor",
            "EllipticEnvelope": "train_elliptic_envelope",
            "RobustCovariance": "train_robust_covariance",
            "MinCovDet": "train_min_cov_det",
            "KMeansAnomaly": "train_kmeans_anomaly",
            "PCAAnomaly": "train_pca_anomaly",
            "DBSCAN": "train_dbscan",
        }
        
        method_name = method_map.get(model_name)
        if method_name is None:
            print(f"[ERROR] Model {model_name} not found in mapping")
            return None
        
        model = getattr(model_training.ClassicalModelTrainer, method_name)(X_train)
        
        y_train_pred, y_train_scores = get_anomaly_predictions(model, X_train)
        y_val_pred, y_val_scores = get_anomaly_predictions(model, X_val)
        y_test_pred, y_test_scores = get_anomaly_predictions(model, X_test)
        
        y_train_binary = np.zeros(len(y_train))
        
        train_metrics = calculate_anomaly_detection_metrics(y_train_binary, y_train_pred, y_train_scores)
        val_metrics = calculate_anomaly_detection_metrics(y_val, y_val_pred, y_val_scores)
        test_metrics = calculate_anomaly_detection_metrics(y_test, y_test_pred, y_test_scores)
        
        print(f"\n[{model_name}] Evaluation Results:")
        print(f"  Validation F1: {val_metrics['f1']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        
        return {
            "model_name": model_name,
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "y_test_pred": y_test_pred,
            "y_test_scores": y_test_scores,
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to train {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def run_anomaly_detection_comparison(
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    models_to_test: List[str] = None,
    window_size: int = None,
    output_dir: str = "anomaly_detection_results"
) -> Dict:
    """
    Run comprehensive anomaly detection model comparison.
    
    注意: output_dir는 analysis 폴더 하위에 생성됨
    """
    
    if models_to_test is None:
        models_to_test = ANOMALY_DETECTION_MODELS
    
    if window_size is None:
        window_size = config.WINDOW_SIZE
    
    # analysis 폴더 하위에 결과 디렉토리 생성
    analysis_dir = os.path.dirname(__file__)
    output_dir = os.path.join(analysis_dir, output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data if not provided
    if X_train is None:
        print("[DataLoader] Loading data...")
        try:
            loader_train = data_loader.DataLoader(config.TRAIN_DATA_DIR)
            loader_val = data_loader.DataLoader(config.VAL_DATA_DIR)
            loader_test = data_loader.DataLoader(config.TEST_DATA_DIR)
            
            X_train = loader_train.load_data(config.TRAIN_DATA_DIR).values
            X_val = loader_val.load_data(config.VAL_DATA_DIR).values
            X_test = loader_test.load_data(config.TEST_DATA_DIR).values
            
            y_train = np.ones(X_train.shape[0])
            y_val = np.ones(X_val.shape[0])
            y_test = np.ones(X_test.shape[0])
            
            print(f"[DataLoader] Loaded data shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
        except Exception as e:
            print(f"[WARNING] Failed to load data from directories: {e}")
            print("[DataLoader] Using dummy data for demonstration...")
            n_train, n_val, n_test = 100, 30, 30
            X_train = np.random.randn(n_train, 64)
            X_val = np.random.randn(n_val, 64)
            X_test = np.random.randn(n_test, 64)
            y_train = np.zeros(n_train)
            y_val = np.zeros(n_val)
            y_test = np.concatenate([np.zeros(n_test//2), np.ones(n_test//2)])
    
    # Preprocess data
    if X_train.ndim == 2:
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        X_train = X_train.reshape(n_samples, -1, 1) if X_train.shape[1] <= window_size else X_train.reshape(n_samples, window_size, -1)
    if X_val.ndim == 2:
        n_samples = X_val.shape[0]
        X_val = X_val.reshape(n_samples, -1, 1) if X_val.shape[1] <= window_size else X_val.reshape(n_samples, window_size, -1)
    if X_test.ndim == 2:
        n_samples = X_test.shape[0]
        X_test = X_test.reshape(n_samples, -1, 1) if X_test.shape[1] <= window_size else X_test.reshape(n_samples, window_size, -1)
    
    X_train_prep = X_train
    X_val_prep = X_val
    X_test_prep = X_test
    
    # Train and evaluate all models
    results = {}
    for model_name in models_to_test:
        result = train_and_evaluate_anomaly_model(
            model_name,
            X_train_prep,
            y_train,
            X_val_prep,
            y_val,
            X_test_prep,
            y_test
        )
        
        if result is not None:
            results[model_name] = result
    
    # Create comparison report
    print(f"\n{'='*60}")
    print("ANOMALY DETECTION MODEL COMPARISON REPORT")
    print(f"{'='*60}\n")
    
    # Create metrics dataframe
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.append({
            "Model": model_name,
            "Val_F1": result["val_metrics"]["f1"],
            "Val_Precision": result["val_metrics"]["precision"],
            "Val_Recall": result["val_metrics"]["recall"],
            "Val_ROC_AUC": result["val_metrics"]["roc_auc"],
            "Test_F1": result["test_metrics"]["f1"],
            "Test_Precision": result["test_metrics"]["precision"],
            "Test_Recall": result["test_metrics"]["recall"],
            "Test_ROC_AUC": result["test_metrics"]["roc_auc"],
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics = df_metrics.sort_values("Test_F1", ascending=False)
    
    print("Test Set Performance Ranking:")
    print(df_metrics.to_string(index=False))
    print()
    
    # Save results
    results_json = {
        "comparison_date": pd.Timestamp.now().isoformat(),
        "window_size": window_size,
        "models_tested": list(results.keys()),
        "metrics_summary": df_metrics.to_dict(orient="records"),
        "results": {}
    }
    
    for model_name, result in results.items():
        results_json["results"][model_name] = {
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
        }
    
    # Save JSON report
    json_path = os.path.join(output_dir, "anomaly_detection_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[SAVED] Results saved to {json_path}")
    
    # Save CSV report
    csv_path = os.path.join(output_dir, "anomaly_detection_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"[SAVED] Metrics CSV saved to {csv_path}")
    
    # Create visualization
    create_anomaly_detection_visualizations(df_metrics, output_dir)
    
    return results_json


def create_anomaly_detection_visualizations(df_metrics: pd.DataFrame, output_dir: str):
    """Create comparison visualizations."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Anomaly Detection Model Comparison - Test Set Performance", 
                     fontsize=16, fontweight='bold')
        
        ax = axes[0, 0]
        df_sorted = df_metrics.sort_values("Test_F1", ascending=True)
        ax.barh(df_sorted["Model"], df_sorted["Test_F1"], color="skyblue")
        ax.set_xlabel("F1 Score")
        ax.set_title("F1 Score Comparison")
        ax.set_xlim(0, 1)
        
        ax = axes[0, 1]
        df_sorted = df_metrics.sort_values("Test_Precision", ascending=True)
        ax.barh(df_sorted["Model"], df_sorted["Test_Precision"], color="lightcoral")
        ax.set_xlabel("Precision")
        ax.set_title("Precision Comparison")
        ax.set_xlim(0, 1)
        
        ax = axes[1, 0]
        df_sorted = df_metrics.sort_values("Test_Recall", ascending=True)
        ax.barh(df_sorted["Model"], df_sorted["Test_Recall"], color="lightgreen")
        ax.set_xlabel("Recall")
        ax.set_title("Recall Comparison")
        ax.set_xlim(0, 1)
        
        ax = axes[1, 1]
        df_sorted = df_metrics.sort_values("Test_ROC_AUC", ascending=True)
        ax.barh(df_sorted["Model"], df_sorted["Test_ROC_AUC"], color="lightyellow")
        ax.set_xlabel("ROC-AUC")
        ax.set_title("ROC-AUC Comparison")
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, "anomaly_detection_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Visualization saved to {fig_path}")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics_cols = ["Test_F1", "Test_Precision", "Test_Recall", "Test_ROC_AUC"]
        heatmap_data = df_metrics.set_index("Model")[metrics_cols]
        
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn", 
                    vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Score"})
        ax.set_title("Anomaly Detection Models - Performance Matrix", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Models")
        
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, "anomaly_detection_heatmap.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Heatmap saved to {fig_path}")
        plt.close()
        
    except Exception as e:
        print(f"[WARNING] Failed to create visualizations: {e}")


# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

def example_1_basic_anomaly_detection():
    """Example 1: Basic anomaly detection with 3 core models."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Anomaly Detection (3 Core Models)")
    print("="*70)
    
    results = run_anomaly_detection_comparison(
        models_to_test=["IsolationForest", "OneClassSVM", "LocalOutlierFactor"],
        output_dir="anomaly_detection_results_example1"
    )
    return results


def example_5_quick_test():
    """Example 5: Quick test with minimal models (recommended for first run)."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Quick Test (Recommended for First Run)")
    print("="*70)
    
    results = run_anomaly_detection_comparison(
        models_to_test=["IsolationForest", "KMeansAnomaly", "PCAAnomaly"],
        output_dir="anomaly_detection_results_quick"
    )
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\nAnomaly Detection Model Comparison Script")
    print("=" * 70)
    print("Results will be saved in: analysis/anomaly_detection_results_quick/")
    print("=" * 70)
    
    # Run Example 5 (quick test) by default
    result = example_5_quick_test()
