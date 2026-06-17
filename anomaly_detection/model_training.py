"""
Model training module for anomaly detection.

Implements:
- Classical ML models: RandomForest, IsolationForest, OneClassSVM
- Deep Learning models: Autoencoder, LSTM
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from . import config
from . import evaluation
from . import mlflow_utils


# ============================================================================
# CLASSICAL ML MODELS
# ============================================================================

class ClassicalModelTrainer:
    """Trainer for classical ML models."""

    @staticmethod
    def train_random_forest(X_train: np.ndarray, 
                           y_train: np.ndarray,
                           params: Dict = None) -> RandomForestClassifier:
        """Train Random Forest classifier."""
        if params is None:
            params = config.CLASSICAL_MODELS["RandomForest"]

        print("[ClassicalML] Training Random Forest...")
        
        # Flatten 3D array to 2D for classical ML
        if X_train.ndim == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        model = RandomForestClassifier(**params)
        model.fit(X_train_flat, y_train)
        print("[ClassicalML] Random Forest training complete")

        return model

    @staticmethod
    def train_isolation_forest(X_train: np.ndarray,
                              params: Dict = None) -> IsolationForest:
        """Train Isolation Forest (unsupervised anomaly detection)."""
        if params is None:
            params = config.CLASSICAL_MODELS["IsolationForest"]

        print("[ClassicalML] Training Isolation Forest...")
        
        # Flatten 3D array to 2D for classical ML
        if X_train.ndim == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        model = IsolationForest(**params)
        model.fit(X_train_flat)
        print("[ClassicalML] Isolation Forest training complete")

        return model

    @staticmethod
    def train_one_class_svm(X_train: np.ndarray,
                           params: Dict = None) -> OneClassSVM:
        """Train One-Class SVM (unsupervised anomaly detection)."""
        if params is None:
            params = config.CLASSICAL_MODELS["OneClassSVM"]

        print("[ClassicalML] Training One-Class SVM...")
        
        # Flatten windows for classical ML
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        model = OneClassSVM(**params)
        model.fit(X_train_flat)
        print("[ClassicalML] One-Class SVM training complete")

        return model

    @staticmethod
    def save_model(model: Any, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"[ClassicalML] Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[ClassicalML] Model loaded from {filepath}")
        return model


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection."""

    def __init__(self, input_dim: int, encoding_dim: int = 32):
        """
        Initialize Autoencoder.

        Args:
            input_dim: Input feature dimension
            encoding_dim: Dimensions of bottleneck layer
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTM(nn.Module):
    """LSTM-based anomaly detection model."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        """
        Initialize LSTM.

        Args:
            input_dim: Number of features
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
        """
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """Forward pass (x: shape [batch, seq_len, n_features])."""
        lstm_out, _ = self.lstm(x)
        # Use last output
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


class DeepLearningTrainer:
    """Trainer for deep learning models."""

    @staticmethod
    def prepare_data(X: np.ndarray, y: np.ndarray = None,
                    batch_size: int = 32) -> DataLoader:
        """Prepare data for PyTorch."""
        # Flatten windows: (n_windows, window_size, n_features) -> (n_windows, window_size*n_features)
        X_flat = X.reshape(X.shape[0], -1)
        
        X_tensor = torch.from_numpy(X_flat).float()
        
        if y is not None:
            y_tensor = torch.from_numpy(y).long()
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @staticmethod
    def train_autoencoder(X_train: np.ndarray,
                         X_val: np.ndarray,
                         params: Dict = None,
                         device: str = "cpu") -> Autoencoder:
        """Train Autoencoder."""
        if params is None:
            params = config.DL_MODELS["Autoencoder"]

        print("[DL] Training Autoencoder...")
        
        # Flatten data
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        input_dim = X_train_flat.shape[1]
        encoding_dim = params.get("encoding_dim", 32)
        
        model = Autoencoder(input_dim, encoding_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params.get("learning_rate", 0.001))
        criterion = nn.MSELoss()
        
        # Prepare data
        train_loader = DeepLearningTrainer.prepare_data(
            X_train, batch_size=params.get("batch_size", 32)
        )
        
        epochs = params.get("epochs", 50)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for X_batch, in train_loader:  # Unpack tuple from TensorDataset
                X_batch = X_batch.to(device)
                
                # Forward
                X_recon = model(X_batch)
                loss = criterion(X_recon, X_batch)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
        
        print("[DL] Autoencoder training complete")
        return model

    @staticmethod
    def train_lstm(X_train: np.ndarray,
                   X_val: np.ndarray,
                   params: Dict = None,
                   device: str = "cpu") -> LSTM:
        """Train LSTM."""
        if params is None:
            params = config.DL_MODELS["LSTM"]

        print("[DL] Training LSTM...")
        
        n_features = X_train.shape[2]  # (n_windows, window_size, n_features)
        hidden_dim = params.get("hidden_dim", 64)
        num_layers = params.get("num_layers", 2)
        
        model = LSTM(n_features, hidden_dim, num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params.get("learning_rate", 0.001))
        criterion = nn.MSELoss()
        
        # Prepare data
        train_loader = DeepLearningTrainer.prepare_lstm_data(
            X_train, batch_size=params.get("batch_size", 32)
        )
        
        epochs = params.get("epochs", 100)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for X_batch, in train_loader:  # Unpack tuple from TensorDataset
                X_batch = X_batch.to(device)
                
                # Forward
                X_pred = model(X_batch)
                # Compare prediction with last timestep
                loss = criterion(X_pred, X_batch[:, -1, :])
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
        
        print("[DL] LSTM training complete")
        return model

    @staticmethod
    def prepare_lstm_data(X: np.ndarray,
                         batch_size: int = 32) -> DataLoader:
        """Prepare data for LSTM."""
        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @staticmethod
    def save_model(model: nn.Module, filepath: str):
        """Save PyTorch model."""
        torch.save(model.state_dict(), filepath)
        print(f"[DL] Model saved to {filepath}")

    @staticmethod
    def load_model(model_class, filepath: str, **kwargs):
        """Load PyTorch model."""
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(filepath))
        print(f"[DL] Model loaded from {filepath}")
        return model


# ============================================================================
# UNIFIED TRAINING FUNCTION
# ============================================================================

def train_all_models(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     model_names: list = None,
                     mlflow_tracker: mlflow_utils.MLflowTracker = None) -> Dict[str, Any]:
    """
    Train all models and return results.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_names: List of models to train
        mlflow_tracker: MLflow tracker instance

    Returns:
        Dictionary with trained models and evaluations
    """
    if model_names is None:
        model_names = ["RandomForest", "IsolationForest", "OneClassSVM", 
                      "Autoencoder", "LSTM"]

    models = {}
    results = {}

    # Classical ML Models
    if "RandomForest" in model_names:
        if mlflow_tracker:
            # Log with model-specific prefix to avoid parameter conflicts
            params = {f"RandomForest_{k}": v for k, v in config.CLASSICAL_MODELS.get("RandomForest", {}).items()}
            mlflow_tracker.log_params(params)
        
        model = ClassicalModelTrainer.train_random_forest(X_train, y_train)
        models["RandomForest"] = model
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_pred = model.predict(X_train_flat)
        acc = accuracy_score(y_train, y_pred)
        results["RandomForest"] = {
            "train_acc": acc,
        }
        
        if mlflow_tracker:
            mlflow_tracker.log_metrics({"train_accuracy": acc})
            mlflow_tracker.log_sklearn_model(model, "RandomForest")

    if "IsolationForest" in model_names:
        if mlflow_tracker:
            params = {f"IsolationForest_{k}": v for k, v in config.CLASSICAL_MODELS.get("IsolationForest", {}).items()}
            mlflow_tracker.log_params(params)
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        model = ClassicalModelTrainer.train_isolation_forest(X_train_flat)
        models["IsolationForest"] = model
        
        y_pred = model.predict(X_train_flat)
        y_pred = np.where(y_pred == -1, 1, 0)  # Convert to binary
        acc = accuracy_score(np.where(y_train == 0, 0, 1), y_pred)
        results["IsolationForest"] = {
            "train_acc": acc,
        }
        
        if mlflow_tracker:
            mlflow_tracker.log_metrics({"IsolationForest_accuracy": acc})
            mlflow_tracker.log_sklearn_model(model, "IsolationForest")

    if "OneClassSVM" in model_names:
        if mlflow_tracker:
            params = {f"OneClassSVM_{k}": v for k, v in config.CLASSICAL_MODELS.get("OneClassSVM", {}).items()}
            mlflow_tracker.log_params(params)
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        model = ClassicalModelTrainer.train_one_class_svm(X_train_flat)
        models["OneClassSVM"] = model
        
        y_pred = model.predict(X_train_flat)
        y_pred = np.where(y_pred == -1, 1, 0)
        acc = accuracy_score(np.where(y_train == 0, 0, 1), y_pred)
        results["OneClassSVM"] = {
            "train_acc": acc,
        }
        
        if mlflow_tracker:
            mlflow_tracker.log_metrics({"OneClassSVM_accuracy": acc})
            mlflow_tracker.log_sklearn_model(model, "OneClassSVM")

    # Deep Learning Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Training] Using device: {device}")

    if "Autoencoder" in model_names:
        if mlflow_tracker:
            params = {f"Autoencoder_{k}": v for k, v in config.DL_MODELS.get("Autoencoder", {}).items()}
            mlflow_tracker.log_params(params)
        
        model = DeepLearningTrainer.train_autoencoder(X_train, X_val, device=device)
        models["Autoencoder"] = model
        results["Autoencoder"] = {"status": "trained"}
        
        if mlflow_tracker:
            mlflow_tracker.log_pytorch_model(model, "Autoencoder")

    if "LSTM" in model_names:
        if mlflow_tracker:
            params = {f"LSTM_{k}": v for k, v in config.DL_MODELS.get("LSTM", {}).items()}
            mlflow_tracker.log_params(params)
        
        model = DeepLearningTrainer.train_lstm(X_train, X_val, device=device)
        models["LSTM"] = model
        results["LSTM"] = {"status": "trained"}
        
        if mlflow_tracker:
            mlflow_tracker.log_pytorch_model(model, "LSTM")

    return models, results


if __name__ == "__main__":
    print("Model training module loaded.")
