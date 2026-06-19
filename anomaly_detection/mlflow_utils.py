"""
MLflow integration utilities for model tracking and management.

Manages:
- Experiment tracking
- Model logging and versioning
- Metrics and parameters logging
- Model registration
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import os
from typing import Dict, Any, Optional
import json
import numpy as np


class MLflowTracker:
    """Manage MLflow experiment tracking."""

    def __init__(self, experiment_name: str = "anomaly_detection", 
                 tracking_uri: str = None):
        """
        Initialize MLflow Tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI (None for local SQLite)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI (SQLite by default instead of file store)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use SQLite instead of file store (MLflow 2.0+)
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        self.active_run = None

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Name of the run
            tags: Optional tags for the run

        Returns:
            Run ID
        """
        self.active_run = mlflow.start_run(run_name=run_name)
        
        if tags:
            mlflow.set_tags(tags)
        
        print(f"[MLflow] Started run: {run_name} (ID: {self.active_run.info.run_id})")
        return self.active_run.info.run_id

    def end_run(self):
        """End current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            print(f"[MLflow] Ended run: {self.active_run.info.run_id}")
            self.active_run = None

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        print(f"[MLflow] Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value, step=step)
        print(f"[MLflow] Logged {len(metrics)} metrics")

    def log_dict(self, data: Dict, name: str):
        """
        Log dictionary as JSON artifact.

        Args:
            data: Dictionary to log
            name: Artifact name (without extension)
        """
        # Convert non-JSON serializable objects
        def convert_to_serializable(obj):
            """Convert numpy arrays and other objects to JSON-serializable format."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert data to JSON-serializable format
        serializable_data = convert_to_serializable(data)
        
        artifact_path = f"{name}.json"
        with open(artifact_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        mlflow.log_artifact(artifact_path)
        os.remove(artifact_path)
        print(f"[MLflow] Logged artifact: {name}")

    def log_pytorch_model(self, model, model_name: str, 
                         artifact_path: str = "pytorch_models"):
        """
        Log PyTorch model.

        Args:
            model: PyTorch model
            model_name: Name of the model
            artifact_path: Artifact path
        """
        # Use pickle serialization format to avoid requiring input_example
        mlflow.pytorch.log_model(model, artifact_path, serialization_format="pickle")
        print(f"[MLflow] Logged PyTorch model: {model_name}")

    def log_sklearn_model(self, model, model_name: str,
                         artifact_path: str = "sklearn_models"):
        """
        Log scikit-learn model.

        Args:
            model: sklearn model
            model_name: Name of the model
            artifact_path: Artifact path
        """
        mlflow.sklearn.log_model(model, artifact_path)
        print(f"[MLflow] Logged sklearn model: {model_name}")

    def register_model(self, model_uri: str, model_name: str) -> str:
        """
        Register model in MLflow Model Registry.

        Args:
            model_uri: URI of the model (e.g., runs:/run_id/path)
            model_name: Name for registered model

        Returns:
            Version of registered model
        """
        result = mlflow.register_model(model_uri, model_name)
        print(f"[MLflow] Registered model: {model_name} (version: {result.version})")
        return result.version

    def log_model_with_schema(self, model, model_type: str, model_name: str,
                             input_example = None, signature = None):
        """
        Log model with schema information.

        Args:
            model: Model object
            model_type: Type of model ('pytorch', 'sklearn')
            model_name: Name of the model
            input_example: Example input for schema inference
            signature: Optional model signature
        """
        if model_type == "pytorch":
            mlflow.pytorch.log_model(model, "model", 
                                    input_example=input_example,
                                    signature=signature)
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(model, "model",
                                    input_example=input_example,
                                    signature=signature)
        print(f"[MLflow] Logged model with schema: {model_name}")

    def log_feature_schema(self, schema: Dict[str, Any], name: str = "feature_schema"):
        """
        Log feature schema.

        Args:
            schema: Feature schema dictionary
            name: Artifact name
        """
        self.log_dict(schema, name)

    def get_best_run(self, metric_name: str, order_by: str = "DESC"):
        """
        Get best run by metric.

        Args:
            metric_name: Name of the metric to optimize
            order_by: "ASC" or "DESC"

        Returns:
            Best run object
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {order_by}"],
            max_results=1
        )
        
        if len(runs) > 0:
            return runs.iloc[0]
        return None

    def compare_runs(self, metric_name: str, top_n: int = 5):
        """
        Compare top N runs.

        Args:
            metric_name: Metric to compare
            top_n: Number of top runs to return

        Returns:
            DataFrame of top runs
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=top_n
        )
        
        return runs
