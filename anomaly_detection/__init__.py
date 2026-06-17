"""
Anomaly Detection Pipeline Package

Modules:
- config: Configuration management
- data_loader: Data loading and EDA
- preprocessing: Data preprocessing and feature engineering
- model_training: Model training (classical ML + DL)
- evaluation: Model evaluation and comparison
- inference_serial: Real-time serial inference
- feature_store: Feature schema and metadata management
- mlflow_utils: MLflow integration for model tracking
- bentoml_service: BentoML service for model serving
"""

__version__ = "1.0.0"
__author__ = "AI/ML Engineering Team"

from . import config
from . import data_loader
from . import preprocessing
from . import model_training
from . import evaluation
from . import inference_serial
from . import feature_store
from . import mlflow_utils
from . import bentoml_service

__all__ = [
    "config",
    "data_loader",
    "preprocessing",
    "model_training",
    "evaluation",
    "inference_serial",
    "feature_store",
    "mlflow_utils",
    "bentoml_service",
]
