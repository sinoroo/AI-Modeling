"""
BentoML service for anomaly detection model serving.

Exposes:
- Anomaly detection API
- Batch inference
- Model information endpoint
"""

import bentoml
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Any, Optional
import torch


@bentoml.service
class AnomalyDetectionService:
    """Anomaly detection service class."""
    
    def __init__(self):
        """Initialize service."""
        self.models = {}
        self.preprocessor = None
        self.feature_store = None
        self.is_ready = False

    def load_model(self, model_name: str, model_path: str):
        """
        Load a model.

        Args:
            model_name: Name of the model
            model_path: Path to model file
        """
        try:
            if model_name in ["autoencoder", "lstm"]:
                # PyTorch models
                model = torch.load(model_path, map_location="cpu")
                model.eval()
            else:
                # sklearn models
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            self.models[model_name] = model
            print(f"[BentoML] Loaded model: {model_name}")
            return True
        except Exception as e:
            print(f"[BentoML] Error loading model {model_name}: {e}")
            return False

    def load_preprocessor(self, preprocessor_path: str):
        """Load preprocessor."""
        try:
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            print(f"[BentoML] Loaded preprocessor")
            return True
        except Exception as e:
            print(f"[BentoML] Error loading preprocessor: {e}")
            return False

    def load_feature_store(self, feature_store):
        """Load feature store."""
        self.feature_store = feature_store
        print(f"[BentoML] Loaded feature store")

    def set_ready(self, ready: bool = True):
        """Set service ready status."""
        self.is_ready = ready
        print(f"[BentoML] Service ready: {ready}")

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess input data."""
        if self.preprocessor:
            return self.preprocessor.normalize(data, fit=False)
        return data

    def _predict_single_impl(self, data: np.ndarray, model_name: str = "autoencoder") -> Dict[str, Any]:
        """
        Predict for single sample.

        Args:
            data: Input array (2D or 3D)
            model_name: Model to use

        Returns:
            Prediction dictionary
        """
        if not self.is_ready:
            raise RuntimeError("Service not ready")
        
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        model = self.models[model_name]
        
        # Preprocess
        data_processed = self.preprocess(data)
        
        # Convert to tensor if needed
        if isinstance(data_processed, np.ndarray):
            data_tensor = torch.FloatTensor(data_processed)
        else:
            data_tensor = data_processed
        
        # Reshape for model input
        if data_tensor.ndim == 1:
            data_tensor = data_tensor.unsqueeze(0)
        if data_tensor.ndim == 2:
            data_tensor = data_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            if model_name in ["autoencoder", "lstm"]:
                output = model(data_tensor)
            else:
                # sklearn model
                data_2d = data_processed.reshape(data_processed.shape[0], -1) if data_processed.ndim > 2 else data_processed
                if data_2d.ndim == 1:
                    data_2d = data_2d.reshape(1, -1)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(data_2d)
                    prediction = model.predict(data_2d)[0]
                    score = proba[0][1] if prediction == 1 else proba[0][0]
                else:
                    prediction = model.predict(data_2d)[0]
                    score = 0.5
                
                return {
                    "prediction": int(prediction),
                    "score": float(score),
                    "confidence": float(score),
                    "model_used": model_name
                }
        
        # For neural network models, use reconstruction error
        if model_name == "autoencoder":
            reconstruction_error = torch.mean((output - data_tensor) ** 2).item()
            threshold = 0.01  # Configurable threshold
            prediction = 1 if reconstruction_error > threshold else 0
            
            return {
                "prediction": prediction,
                "score": reconstruction_error,
                "confidence": min(1.0, reconstruction_error / threshold),
                "model_used": model_name
            }
        
        return {
            "prediction": 0,
            "score": 0.0,
            "confidence": 0.0,
            "model_used": model_name
        }

    def _predict_batch_impl(self, data: np.ndarray, model_name: str = "autoencoder") -> Dict[str, Any]:
        """
        Predict for batch of samples.

        Args:
            data: Input array (3D)
            model_name: Model to use

        Returns:
            Batch prediction dictionary
        """
        predictions = []
        scores = []
        confidences = []
        
        for i in range(data.shape[0]):
            result = self._predict_single_impl(data[i], model_name)
            predictions.append(result["prediction"])
            scores.append(result["score"])
            confidences.append(result["confidence"])
        
        return {
            "predictions": predictions,
            "scores": scores,
            "confidences": confidences,
            "model_used": model_name
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "is_ready": self.is_ready,
            "available_models": list(self.models.keys()),
            "has_preprocessor": self.preprocessor is not None,
            "has_feature_store": self.feature_store is not None
        }
        
        if self.feature_store:
            info["feature_schema"] = self.feature_store.get_schema()
            info["feature_stats"] = self.feature_store.get_statistics()
        
        return info

    @bentoml.api
    def predict(self, data: dict) -> dict:
        """API endpoint for single prediction.
        
        Args:
            data: Dictionary with 'data' (list of floats) and 'model_name' (str)
        
        Returns:
            Dictionary with prediction, score, confidence
        """
        try:
            if not self.is_ready:
                return {"error": "Service not ready"}
            
            input_data = np.array(data.get("data", []), dtype=np.float32)
            model_name = data.get("model_name", "RandomForest")
            
            result = self._predict_single_impl(input_data, model_name)
            return result
        except Exception as e:
            return {"error": str(e)}

    @bentoml.api
    def predict_batch(self, data: dict) -> dict:
        """API endpoint for batch prediction.
        
        Args:
            data: Dictionary with 'data' (list of lists) and 'model_name' (str)
        
        Returns:
            Dictionary with predictions lists
        """
        try:
            if not self.is_ready:
                return {"error": "Service not ready"}
            
            input_data = np.array(data.get("data", []), dtype=np.float32)
            model_name = data.get("model_name", "RandomForest")
            
            # Ensure 3D
            if input_data.ndim == 2:
                input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1])
            
            result = self._predict_batch_impl(input_data, model_name)
            return result
        except Exception as e:
            return {"error": str(e)}

    @bentoml.api
    def model_info(self) -> dict:
        """Get model information."""
        return self.get_model_info()

    @bentoml.api
    def health_check(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy" if self.is_ready else "not_ready",
            "service": "anomaly_detection_service"
        }
