"""
Anomaly Detection Pipeline Package

Modules:
- config: Configuration management
- data_loader: Data loading and EDA
- preprocessing: Data preprocessing and feature engineering
- model_training: Model training (classical ML + DL)
- evaluation: Model evaluation and comparison
- inference_serial: Real-time serial inference
"""

__version__ = "1.0.0"
__author__ = "AI/ML Engineering Team"

from . import config
from . import data_loader
from . import preprocessing
from . import model_training
from . import evaluation
from . import inference_serial

__all__ = [
    "config",
    "data_loader",
    "preprocessing",
    "model_training",
    "evaluation",
    "inference_serial",
]
