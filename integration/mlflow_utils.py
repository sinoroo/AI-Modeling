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
        
        # Create or get experiment
        try:
            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                # Create new experiment if it doesn't exist
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"[MLflow] Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                print(f"[MLflow] Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
            
            # Set active experiment
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"[MLflow] Error initializing experiment '{experiment_name}': {e}")
            # Still set experiment even if there's an error
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
        try:
            # End any existing run first
            if self.active_run:
                mlflow.end_run()
                print(f"[MLflow] Ended previous run: {self.active_run.info.run_id}")
            
            # Start new run with explicit experiment
            self.active_run = mlflow.start_run(run_name=run_name)
            
            if tags:
                mlflow.set_tags(tags)
            
            print(f"[MLflow] Started run: {run_name}")
            print(f"         Experiment: {self.experiment_name}")
            print(f"         Run ID: {self.active_run.info.run_id}")
            return self.active_run.info.run_id
        except Exception as e:
            print(f"[MLflow] Error starting run '{run_name}': {e}")
            return None

    def end_run(self):
        """End current MLflow run."""
        if self.active_run:
            try:
                run_id = self.active_run.info.run_id
                mlflow.end_run()
                print(f"[MLflow] Ended run: {run_id}")
                self.active_run = None
            except Exception as e:
                print(f"[MLflow] Error ending run: {e}")
        else:
            print(f"[MLflow] No active run to end")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        if not params:
            print(f"[MLflow] No parameters to log")
            return
        
        try:
            mlflow.log_params(params)
            print(f"[MLflow] Logged {len(params)} parameters")
        except Exception as e:
            print(f"[MLflow] Error logging parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        logged_count = 0
        skipped_count = 0
        
        for key, value in metrics.items():
            if value is not None:
                try:
                    mlflow.log_metric(key, value, step=step)
                    logged_count += 1
                except Exception as e:
                    print(f"  [Warning] Failed to log metric '{key}': {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
        
        if logged_count > 0 or skipped_count > 0:
            msg = f"[MLflow] Logged {logged_count} metrics"
            if skipped_count > 0:
                msg += f" ({skipped_count} skipped due to None/error)"
            print(msg)
        else:
            print(f"[MLflow] No metrics to log (received {len(metrics)} items, all None or empty)")

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
                         artifact_path: str = "pytorch_models",
                         input_example = None):
        """
        Log PyTorch model.

        Args:
            model: PyTorch model
            model_name: Name of the model
            artifact_path: Artifact path (model name in MLflow)
            input_example: Example input for pt2 format (torch.Tensor or numpy array)
                          If provided, uses safe pt2 serialization format
        """
        try:
            # Use pt2 format if input_example provided, otherwise use default pickle format
            if input_example is not None:
                mlflow.pytorch.log_model(
                    model, 
                    name=artifact_path,
                    input_example=input_example,
                    serialization_format='pt2'
                )
                print(f"[MLflow] Logged PyTorch model: {model_name} (pt2 format)")
            else:
                # Use default pickle format (may show warning but works)
                mlflow.pytorch.log_model(
                    model, 
                    name=artifact_path
                )
                print(f"[MLflow] Logged PyTorch model: {model_name} (pickle format)")
        except Exception as e:
            print(f"[MLflow] Error logging PyTorch model '{model_name}': {e}")

    def log_sklearn_model(self, model, model_name: str,
                         artifact_path: str = "sklearn_models"):
        """
        Log scikit-learn model.

        Args:
            model: sklearn model
            model_name: Name of the model
            artifact_path: Artifact path (model name in MLflow)
        """
        try:
            # Use name parameter instead of deprecated artifact_path
            mlflow.sklearn.log_model(model, name=artifact_path)
            print(f"[MLflow] Logged sklearn model: {model_name}")
        except Exception as e:
            print(f"[MLflow] Error logging sklearn model '{model_name}': {e}")

    def register_model(self, model_uri: str, model_name: str) -> str:
        """
        Register model in MLflow Model Registry.

        Args:
            model_uri: URI of the model (e.g., runs:/run_id/path)
            model_name: Name for registered model

        Returns:
            Version of registered model
        """
        try:
            result = mlflow.register_model(model_uri, model_name)
            print(f"[MLflow] Registered model: {model_name} (version: {result.version})")
            return result.version
        except Exception as e:
            print(f"[MLflow] Error registering model '{model_name}': {e}")
            return None

    def log_model_with_schema(self, model, model_type: str, model_name: str,
                             input_example = None, signature = None):
        """
        Log model with schema information.

        Args:
            model: Model object
            model_type: Type of model ('pytorch', 'sklearn')
            model_name: Name of the model
            input_example: Example input for schema inference (required for pt2)
            signature: Optional model signature
        """
        try:
            if model_type == "pytorch":
                # Use pt2 format if input_example provided
                if input_example is not None:
                    mlflow.pytorch.log_model(
                        model, 
                        name="model", 
                        input_example=input_example,
                        signature=signature,
                        serialization_format='pt2'
                    )
                    print(f"[MLflow] Logged model with schema: {model_name} (pt2 format)")
                else:
                    mlflow.pytorch.log_model(
                        model, 
                        name="model",
                        signature=signature
                    )
                    print(f"[MLflow] Logged model with schema: {model_name} (pickle format)")
            elif model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model, 
                    name="model",
                    input_example=input_example,
                    signature=signature
                )
                print(f"[MLflow] Logged model with schema: {model_name}")
        except Exception as e:
            print(f"[MLflow] Error logging model with schema '{model_name}': {e}")

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
