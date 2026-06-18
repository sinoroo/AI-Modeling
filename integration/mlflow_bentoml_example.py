"""
Example script demonstrating MLflow and BentoML integration.

This script shows how to:
1. Load models from MLflow
2. Register models in BentoML
3. Create a service and build it
4. Serve predictions via REST API
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path

# Add parent package to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection import config
from integration import (
    MLflowTracker,
    FeatureStore,
    AnomalyDetectionService,
)
import bentoml


def load_models_and_artifacts(model_dir: str = "models") -> dict:
    """
    Load trained models and artifacts.

    Args:
        model_dir: Directory containing saved models

    Returns:
        Dictionary with loaded models
    """
    models = {}
    
    print("[Setup] Loading models from disk...")
    
    # Load classical ML models
    for model_name in ["RandomForest", "IsolationForest", "OneClassSVM"]:
        model_path = os.path.join(model_dir, f"{model_name.lower()}_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
            print(f"  ✓ Loaded {model_name}")

    # Load deep learning models
    if os.path.exists(os.path.join(model_dir, "autoencoder_model.pt")):
        from anomaly_detection.model_training import Autoencoder
        # Would need to know input_dim - load from feature_store schema
        # For now, skip
        print(f"  • Autoencoder model found (requires feature schema for loading)")

    if os.path.exists(os.path.join(model_dir, "lstm_model.pt")):
        print(f"  • LSTM model found (requires feature schema for loading)")

    return models


def setup_bentoml_service(models: dict, 
                         preprocessor_path: str = None,
                         feature_schema_path: str = None) -> AnomalyDetectionService:
    """
    Setup BentoML service with models.

    Args:
        models: Dictionary of loaded models
        preprocessor_path: Path to preprocessor
        feature_schema_path: Path to feature schema

    Returns:
        Configured service instance
    """
    print("\n[Setup] Initializing BentoML service...")
    
    service = AnomalyDetectionService()
    
    # Load models into service
    for model_name, model in models.items():
        # Save model temporarily for service to load
        temp_path = f"/tmp/{model_name}_temp.pkl"
        with open(temp_path, 'wb') as f:
            pickle.dump(model, f)
        
        service.load_model(model_name, temp_path)
        os.remove(temp_path)
    
    # Load preprocessor if available
    if preprocessor_path and os.path.exists(preprocessor_path):
        service.load_preprocessor(preprocessor_path)
        print(f"  ✓ Preprocessor loaded")

    # Load feature store if available
    if feature_schema_path:
        fs = FeatureStore()
        # Load schema
        import json
        if os.path.exists(feature_schema_path):
            with open(feature_schema_path, 'r') as f:
                fs.schema = json.load(f)
            service.load_feature_store(fs)
            print(f"  ✓ Feature store loaded with schema")

    service.set_ready(True)
    print(f"  ✓ Service ready with {len(models)} models")
    
    return service


def register_model_with_mlflow(model_name: str, 
                              model_uri: str,
                              tags: dict = None) -> str:
    """
    Register a model with MLflow Model Registry.

    Args:
        model_name: Name for the registered model
        model_uri: URI of the model in MLflow
        tags: Optional tags

    Returns:
        Registered model version
    """
    print(f"\n[MLflow] Registering model: {model_name}")
    
    tracker = MLflowTracker(
        experiment_name="anomaly_detection",
        tracking_uri="sqlite:///integration/mlflow.db"
    )
    
    try:
        version = tracker.register_model(model_uri, model_name)
        print(f"  ✓ Registered model version: {version}")
        return version
    except Exception as e:
        print(f"  ✗ Registration failed: {e}")
        return None


def compare_best_models(experiment_name: str = "anomaly_detection",
                       metric_name: str = "f1",
                       top_n: int = 5) -> None:
    """
    Compare best models from MLflow.

    Args:
        experiment_name: MLflow experiment name
        metric_name: Metric to optimize
        top_n: Number of top runs to display
    """
    print(f"\n[MLflow] Comparing top {top_n} models by {metric_name}...")
    
    tracker = MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri="sqlite:///integration/mlflow.db"
    )
    
    comparison = tracker.compare_runs(metric_name, top_n)
    
    if comparison is not None and len(comparison) > 0:
        print(f"\n{'Rank':<6} {'Model':<20} {'Metric':<15} {'Run ID':<40}")
        print("-" * 81)
        
        for idx, row in comparison.iterrows():
            metric_value = row.get(f"metrics.{metric_name}", "N/A")
            run_id = row.get("run_id", "N/A")
            model_name = row.get("tags.model_type", "N/A")
            
            print(f"{idx+1:<6} {str(model_name):<20} {str(metric_value):<15} {str(run_id):<40}")
    else:
        print("  No runs found")


def build_bentoml_service(service_name: str = "anomaly_detection_service",
                         build_dir: str = "./bento") -> str:
    """
    Build BentoML service for deployment.

    Args:
        service_name: Name of the service
        build_dir: Directory to build the service

    Returns:
        Path to built service
    """
    print(f"\n[BentoML] Building service: {service_name}")
    
    # Note: This is a simplified example
    # In production, you would use:
    # bentoml build --bentofile bentofile.yaml
    
    print(f"  ℹ To build the service, run:")
    print(f"    bentoml build -f bentofile.yaml")
    print(f"  ℹ To serve the model locally, run:")
    print(f"    bentoml serve anomaly_detection_service:latest")
    
    return build_dir


def test_inference_api(service: AnomalyDetectionService,
                       test_data: np.ndarray,
                       model_name: str = "RandomForest") -> dict:
    """
    Test inference API with sample data.

    Args:
        service: BentoML service instance
        test_data: Test data array
        model_name: Model to use for inference

    Returns:
        Prediction result
    """
    print(f"\n[Test] Running inference with {model_name}...")
    
    result = service._predict_single_impl(test_data, model_name)
    
    print(f"  Prediction: {'ANOMALY' if result['prediction'] == 1 else 'NORMAL'}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MLflow and BentoML Integration Example"
    )
    parser.add_argument("--setup", action="store_true", 
                       help="Setup and test BentoML service")
    parser.add_argument("--compare", action="store_true",
                       help="Compare models from MLflow")
    parser.add_argument("--build", action="store_true",
                       help="Build BentoML service")
    parser.add_argument("--test", action="store_true",
                       help="Test inference with sample data")
    parser.add_argument("--all", action="store_true",
                       help="Run all steps")

    args = parser.parse_args()

    if not (args.setup or args.compare or args.build or args.test or args.all):
        args.all = True

    # Step 1: Compare MLflow models
    if args.compare or args.all:
        compare_best_models(metric_name="f1", top_n=5)

    # Step 2: Setup BentoML service
    if args.setup or args.all:
        models = load_models_and_artifacts()
        service = setup_bentoml_service(
            models,
            preprocessor_path="models/preprocessor.pkl",
            feature_schema_path="integration/feature_store/feature_schema.json"
        )

        # Test inference
        if args.test or args.all:
            # Create sample test data
            test_data = np.random.normal(loc=50, scale=10, size=(100, 5))
            test_inference_api(service, test_data, "RandomForest")

    # Step 3: Build BentoML service
    if args.build or args.all:
        build_bentoml_service()

    print("\n✓ MLflow and BentoML integration example complete!")


if __name__ == "__main__":
    main()
