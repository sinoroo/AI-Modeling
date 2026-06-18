"""
Model evaluation module for anomaly detection.

Computes:
- Classification metrics (Accuracy, Precision, Recall, F1-score)
- ROC-AUC
- Confusion Matrix
- Model comparison and ranking
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from typing import Dict, Tuple, Any
import json
import os

from . import config


class ModelEvaluator:
    """Evaluate trained models."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # ROC-AUC (for binary classification only)
        if y_proba is not None and len(np.unique(y_true)) <= 2:
            try:
                # Handle different y_proba shapes
                if y_proba.ndim == 2:
                    if y_proba.shape[1] >= 2:
                        y_proba_auc = y_proba[:, 1]  # Binary: use positive class prob
                    elif y_proba.shape[1] == 1:
                        y_proba_auc = y_proba[:, 0]  # Single column: use as-is
                    else:
                        y_proba_auc = None
                else:
                    y_proba_auc = y_proba  # Already 1D
                
                if y_proba_auc is not None:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba_auc)
                else:
                    metrics["roc_auc"] = None
            except:
                metrics["roc_auc"] = None

        return metrics

    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray,
                                y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def print_metrics(model_name: str, metrics: Dict[str, float]):
        """Print metrics in formatted manner."""
        print(f"\n  {model_name}")
        print("  " + "-" * 40)
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"    {metric_name}: {value:.4f}")

    @staticmethod
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             model_name: str,
                             output_dir: str = "results") -> str:
        """Plot and save confusion matrix."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(output_dir, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        filepath = os.path.join(output_dir, f"cm_{model_name}.png")
        plt.savefig(filepath, dpi=100)
        plt.close()
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray,
                      y_proba: np.ndarray,
                      model_name: str,
                      output_dir: str = "results") -> str:
        """Plot and save ROC curve."""
        import matplotlib.pyplot as plt
        
        if len(np.unique(y_true)) > 2:
            return None  # Multi-class ROC not plotted here
        
        if y_proba is None:
            return None

        os.makedirs(output_dir, exist_ok=True)

        # Handle different y_proba shapes
        if y_proba.ndim == 2:
            if y_proba.shape[1] >= 2:
                y_proba = y_proba[:, 1]  # Binary: use positive class prob
            elif y_proba.shape[1] == 1:
                y_proba = y_proba[:, 0]  # Single column: use as-is
            else:
                return None  # No valid probability column
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve: {model_name}")
            plt.legend(loc="lower right")
            plt.tight_layout()

            filepath = os.path.join(output_dir, f"roc_{model_name}.png")
            plt.savefig(filepath, dpi=100)
            plt.close()

            return filepath
        except:
            # ROC curve generation failed (e.g., if only 1 class in predictions)
            return None


def evaluate_classical_model(model: Any,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            model_name: str) -> Dict[str, Any]:
    """
    Evaluate classical ML model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Model name

    Returns:
        Evaluation results dictionary
    """
    print(f"\n[Evaluation] Evaluating {model_name}...")

    # Flatten if needed
    if X_test.ndim > 2:
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_test_flat = X_test

    # Predictions
    y_pred = model.predict(X_test_flat)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_flat)

    # Compute metrics
    metrics = ModelEvaluator.compute_metrics(y_test, y_pred, y_proba)
    ModelEvaluator.print_metrics(model_name, metrics)

    # Confusion matrix
    cm = ModelEvaluator.compute_confusion_matrix(y_test, y_pred)

    # Plots
    ModelEvaluator.plot_confusion_matrix(y_test, y_pred, model_name)
    if y_proba is not None:
        ModelEvaluator.plot_roc_curve(y_test, y_proba, model_name)

    return {
        "model_name": model_name,
        "metrics": metrics,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


def evaluate_deep_learning_model(model: Any,
                                X_test: np.ndarray,
                                y_test: np.ndarray = None,
                                model_name: str = "DL_Model",
                                threshold: float = None,
                                device: str = "cpu") -> Dict[str, Any]:
    """
    Evaluate deep learning model.

    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test labels (optional, for unsupervised we use threshold)
        model_name: Model name
        threshold: Anomaly threshold (for reconstruction error)
        device: Device to use

    Returns:
        Evaluation results dictionary
    """
    print(f"\n[Evaluation] Evaluating {model_name}...")

    import torch

    model.eval()
    model = model.to(device)

    # Handle LSTM (needs 3D: batch, seq_len, features) vs Autoencoder (needs 2D)
    if "LSTM" in model_name:
        # LSTM expects 3D input
        X_tensor = torch.from_numpy(X_test).float().to(device)
    else:
        # Autoencoder expects 2D input (flattened time-series)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_tensor = torch.from_numpy(X_test_flat).float().to(device)

    # Get model outputs (reconstruction or predictions)
    with torch.no_grad():
        outputs = model(X_tensor)

    # For anomaly detection, compute reconstruction error
    if "LSTM" in model_name:
        # LSTM output is (batch, features), but input was (batch, seq_len, features)
        # Use LSTM output which represents the last time step's encoding
        # Compute error across the entire sequence window
        X_mean = torch.mean(X_tensor, dim=1)  # (batch, features)
        reconstruction_error = torch.mean((X_mean - outputs) ** 2, dim=1).cpu().numpy()
    else:
        # Autoencoder: X_tensor is already 2D, outputs is 2D
        reconstruction_error = torch.mean((X_tensor - outputs) ** 2, dim=1).cpu().numpy()

    # Anomaly labels
    if threshold is None:
        threshold = np.percentile(reconstruction_error, 95)

    y_pred = (reconstruction_error > threshold).astype(int)

    results = {
        "model_name": model_name,
        "reconstruction_error": reconstruction_error,
        "threshold": threshold,
        "predictions": y_pred,
    }

    if y_test is not None:
        metrics = ModelEvaluator.compute_metrics(y_test, y_pred)
        ModelEvaluator.print_metrics(model_name, metrics)
        results["metrics"] = metrics
        
        cm = ModelEvaluator.compute_confusion_matrix(y_test, y_pred)
        results["confusion_matrix"] = cm

        ModelEvaluator.plot_confusion_matrix(y_test, y_pred, model_name)

    return results


def compare_models(evaluation_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare models and rank them.

    Args:
        evaluation_results: Dictionary of model evaluation results

    Returns:
        DataFrame with model rankings
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON & RANKING")
    print("=" * 80)

    comparison_data = []

    for model_name, result in evaluation_results.items():
        row = {"Model": model_name}
        
        if "metrics" in result:
            row.update(result["metrics"])
        
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # Score models (F1 is primary metric)
    if "f1" in df_comparison.columns:
        df_comparison = df_comparison.sort_values("f1", ascending=False)
    elif "accuracy" in df_comparison.columns:
        df_comparison = df_comparison.sort_values("accuracy", ascending=False)

    print("\n" + df_comparison.to_string(index=False))

    # Select best model
    if "f1" in df_comparison.columns:
        best_model = df_comparison.iloc[0]["Model"]
        best_score = df_comparison.iloc[0]["f1"]
        print(f"\n[Best Model] {best_model} (F1: {best_score:.4f})")
    elif "accuracy" in df_comparison.columns:
        best_model = df_comparison.iloc[0]["Model"]
        best_score = df_comparison.iloc[0]["accuracy"]
        print(f"\n[Best Model] {best_model} (Accuracy: {best_score:.4f})")

    print("=" * 80)

    return df_comparison


def generate_evaluation_report(evaluation_results: Dict[str, Dict],
                              output_path: str = "evaluation_report.json",
                              output_dir: str = "results"):
    """
    Generate comprehensive evaluation reports.
    
    Creates:
    1. Main evaluation_report.json with overall summary
    2. Individual model_<name>_evaluation.json for each model

    Args:
        evaluation_results: Evaluation results from all models
        output_path: Path to save main report
        output_dir: Directory to save individual model reports
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[Report] Generating evaluation reports...")

    # Main report with all models
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_models": len(evaluation_results),
        "models": {}
    }

    for model_name, result in evaluation_results.items():
        model_report = {
            "metrics": result.get("metrics", {}),
            "confusion_matrix": result.get("confusion_matrix", "").tolist() if isinstance(result.get("confusion_matrix"), np.ndarray) else None,
        }
        report["models"][model_name] = model_report

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Main report saved to {output_path}")

    # Individual model reports
    for model_name, result in evaluation_results.items():
        model_report_path = os.path.join(output_dir, f"model_{model_name.lower()}_evaluation.json")
        
        model_detail = {
            "model_name": model_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": result.get("metrics", {}),
            "confusion_matrix": result.get("confusion_matrix", "").tolist() if isinstance(result.get("confusion_matrix"), np.ndarray) else None,
            "predictions_count": len(result.get("predictions", [])),
            "files": {
                "confusion_matrix_plot": f"cm_{model_name}.png",
                "roc_curve_plot": f"roc_{model_name}.png"
            }
        }
        
        # Add reconstruction error info for DL models
        if "reconstruction_error" in result:
            reconstruction_error = result["reconstruction_error"]
            model_detail["reconstruction_error"] = {
                "threshold": float(result.get("threshold", 0)),
                "min": float(np.min(reconstruction_error)),
                "max": float(np.max(reconstruction_error)),
                "mean": float(np.mean(reconstruction_error)),
                "std": float(np.std(reconstruction_error)),
                "percentile_95": float(np.percentile(reconstruction_error, 95))
            }
        
        # Add probabilities info if available
        if result.get("probabilities") is not None:
            y_proba = result["probabilities"]
            if y_proba.ndim == 2:
                model_detail["probabilities"] = {
                    "shape": list(y_proba.shape),
                    "min": float(np.min(y_proba)),
                    "max": float(np.max(y_proba)),
                    "mean": float(np.mean(y_proba))
                }
        
        with open(model_report_path, 'w', encoding='utf-8') as f:
            json.dump(model_detail, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Model report saved to {model_report_path}")

    print(f"[Report] All reports generated successfully!")


if __name__ == "__main__":
    print("Evaluation module loaded.")
