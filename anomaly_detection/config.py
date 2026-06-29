"""
Configuration file for anomaly detection pipeline.

This module centralizes all configuration parameters for data loading,
preprocessing, model training, and inference.
"""

import os
from pathlib import Path

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Data paths - can be overridden from command line or environment variables
DATA_DIR = os.getenv("DATA_DIR", "data_new_format")

# Directory-based data paths (scan all CSV files in those directories)
TRAIN_DATA_DIR = os.getenv("TRAIN_DATA_DIR", os.path.join(DATA_DIR, "train"))
VAL_DATA_DIR = os.getenv("VAL_DATA_DIR", os.path.join(DATA_DIR, "val"))
TEST_DATA_DIR = os.getenv("TEST_DATA_DIR", os.path.join(DATA_DIR, "test"))

# Legacy: single file paths (deprecated - not used with new format)
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", None)
VAL_DATA_PATH = os.getenv("VAL_DATA_PATH", None)
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", None)

# Create data directory if it doesn't exist
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# CSV data format configuration
# Set to True for new format (9 lines metadata + data from line 10)
# Set to False for simple format (single header line + data)
USE_NEW_CSV_FORMAT = True  # New format: metadata + time/vibration data

# Metadata column indices (0-based, lines 0-8)
METADATA_MAPPING = {
    "Date": 0,           # Line 1: Date,2020-11-26 09:42:19
    "Filename": 1,       # Line 2: Filename,STFMK-...
    "Data Label": 2,     # Line 3: Data Label,정상 (or 축정렬불량, etc)
    "Label_No": 3,       # Line 4: Label_No,00
    "Motor Spec": 4,     # Line 5: Motor Spec,L-CAHU-01R,...
    "Period": 5,         # Line 6: Period,3SEC
    "Sample Rate": 6,    # Line 7: Sample Rate,4000
    "RMS": 7,           # Line 8: RMS,0.010346,
    "Data Length": 8,   # Line 9: Data Length,12000,
}

# Normal/Abnormal labels for the new format
NORMAL_LABELS = ["정상"]  # Data Label values that indicate normal operation
ABNORMAL_LABELS = ["축정렬불량"]  # Data Label values that indicate anomalies (add more as needed)

# EDA configuration
EDA_OUTPUT_DIR = "eda_results"
Path(EDA_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Time-series windowing
WINDOW_SIZE = 64  # Number of samples per window
OVERLAP = 0.5    # Fraction of overlap between windows (0.5 = 50%)
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

# Feature normalization
NORMALIZE_METHOD = "standardize"  # "standardize", "minmax", or "robust"

# Missing value handling
MISSING_VALUE_METHOD = "interpolate"  # "interpolate", "forward_fill", or "drop"

# Feature engineering
USE_FFT_FEATURES = True
USE_STATISTICAL_FEATURES = False  # Disabled - using only FFT features
FFT_N_BINS = 32  # Number of FFT bins to extract

# FFT Sample Sizes for comparison and optimization
# Testing different window sizes to find optimal FFT parameters
FFT_SAMPLE_SIZES = [32, 64, 128, 256]  # Different window sizes for FFT analysis
FFT_BINS_SIZES = [5, 10, 20, 32]       # Different FFT bin sizes to extract

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Train/Val/Test split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Classical ML Models
CLASSICAL_MODELS = {
    # Ensemble Methods
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": RANDOM_SEED,
    },
    "GradientBoosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": RANDOM_SEED,
    },
    "AdaBoost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED,
    },
    "ExtraTrees": {
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": RANDOM_SEED,
    },
    
    # Tree-based
    "DecisionTree": {
        "max_depth": 15,
        "random_state": RANDOM_SEED,
    },
    
    # Distance-based
    "KNearestNeighbors": {
        "n_neighbors": 5,
        "weights": "distance",
    },
    
    # Probability-based
    "GaussianNB": {},
    "LogisticRegression": {
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
    },
    
    # SVM
    "SVM": {
        "kernel": "rbf",
        "gamma": "scale",
        "random_state": RANDOM_SEED,
    },
    
    # Anomaly Detection (Original)
    "IsolationForest": {
        "n_estimators": 100,
        "contamination": 0.1,
        "random_state": RANDOM_SEED,
    },
    "OneClassSVM": {
        "kernel": "linear",  # Changed from "rbf" for 10x speed improvement
        "nu": 0.05,
        "max_iter": 1000,    # Added timeout protection
    },
    "LocalOutlierFactor": {
        "n_neighbors": 20,
        "contamination": 0.1,
    },
    "EllipticEnvelope": {
        "contamination": 0.1,
        "random_state": RANDOM_SEED,
    },
    
    # Anomaly Detection (New - Advanced)
    "RobustCovariance": {
        "contamination": 0.1,
    },
    "MinCovDet": {
        "contamination": 0.1,
        "random_state": RANDOM_SEED,
    },
    "KMeansAnomaly": {
        "n_clusters": 5,
        "n_init": 10,
        "random_state": RANDOM_SEED,
        "contamination": 0.1,  # For threshold calculation
    },
    "PCAAnomaly": {
        "n_components": 0.95,  # Keep 95% of variance
        "contamination": 0.1,  # For threshold calculation
    },
    "DBSCAN": {
        "eps": 0.5,
        "min_samples": 5,
    },
}

# Deep Learning Models
DL_MODELS = {
    "Autoencoder": {
        "input_dim": None,  # Will be set based on data
        "encoding_dim": 32,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.15,
        "threshold_percentile": 95,  # For anomaly detection
    },
    "LSTM": {
        "input_dim": None,
        "hidden_dim": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "validation_split": 0.15,
        "threshold_percentile": 95,
    },
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "confusion_matrix",
]

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

MODEL_DIR = "models"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Model file names
MODEL_FILES = {
    "RandomForest": os.path.join(MODEL_DIR, "random_forest_model.pkl"),
    "IsolationForest": os.path.join(MODEL_DIR, "isolation_forest_model.pkl"),
    "OneClassSVM": os.path.join(MODEL_DIR, "one_class_svm_model.pkl"),
    "Autoencoder": os.path.join(MODEL_DIR, "autoencoder_model.pt"),
    "LSTM": os.path.join(MODEL_DIR, "lstm_model.pt"),
    "Scaler": os.path.join(MODEL_DIR, "scaler.pkl"),
    "Preprocessor": os.path.join(MODEL_DIR, "preprocessor.pkl"),
}

# ============================================================================
# SERIAL INFERENCE CONFIGURATION
# ============================================================================

# Serial port settings
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
BAUD_RATE = int(os.getenv("BAUD_RATE", "9600"))
TIMEOUT = 1.0  # seconds

# Inference settings
INFERENCE_WINDOW_SIZE = WINDOW_SIZE
INFERENCE_BUFFER_SIZE = WINDOW_SIZE * 2

# Anomaly threshold (depends on model)
ANOMALY_THRESHOLD = 0.5  # Probability threshold for binary classification

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

LOG_DIR = "logs"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"

# ============================================================================
# ANOMALY TYPES (Extensible)
# ============================================================================

ANOMALY_TYPES = {
    0: "Normal",
    1: "High Vibration",
    2: "High Temperature",
    3: "Non-steady State",
    4: "Other",
}

print("[CONFIG] Configuration loaded successfully.")
