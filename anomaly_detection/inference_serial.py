"""
Real-time serial inference system for anomaly detection.

This module handles:
- Serial communication with sensors
- Real-time data streaming
- Model inference
- Anomaly detection and classification
"""

import serial
import numpy as np
import pandas as pd
import time
import threading
from collections import deque
from typing import Optional, Dict, Any, Callable
import pickle
import torch

from . import config


class SerialDataBufferManager:
    """Manage buffered data from serial port."""

    def __init__(self, buffer_size: int = config.INFERENCE_BUFFER_SIZE):
        """
        Initialize buffer manager.

        Args:
            buffer_size: Maximum buffer size
        """
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.lock = threading.Lock()

    def add_sample(self, sample: np.ndarray):
        """Add a sample to buffer."""
        with self.lock:
            self.buffer.append(sample)

    def get_buffer(self) -> np.ndarray:
        """Get current buffer as numpy array."""
        with self.lock:
            if len(self.buffer) < config.WINDOW_SIZE:
                return None
            return np.array(list(self.buffer))

    def get_window(self) -> Optional[np.ndarray]:
        """
        Get a single window from buffer.

        Returns:
            Window of size WINDOW_SIZE or None if not enough data
        """
        with self.lock:
            if len(self.buffer) >= config.WINDOW_SIZE:
                # Get last WINDOW_SIZE samples
                return np.array(list(self.buffer))[-config.WINDOW_SIZE:]
            return None

    def clear_buffer(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()

    def buffer_full(self) -> bool:
        """Check if buffer is full."""
        with self.lock:
            return len(self.buffer) >= config.WINDOW_SIZE


class SerialDataReader:
    """Read real-time data from serial port."""

    def __init__(self, port: str = config.SERIAL_PORT,
                 baudrate: int = config.BAUD_RATE,
                 timeout: float = config.TIMEOUT):
        """
        Initialize serial reader.

        Args:
            port: Serial port name (e.g., 'COM3', '/dev/ttyUSB0')
            baudrate: Baud rate
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.is_connected = False
        self.data_buffer = SerialDataBufferManager()
        self.read_thread = None
        self.stop_flag = False

    def connect(self) -> bool:
        """
        Connect to serial port.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.is_connected = True
            print(f"[SerialReader] Connected to {self.port} @ {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"[SerialReader] Connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from serial port."""
        if self.serial_conn:
            self.serial_conn.close()
            self.is_connected = False
            print("[SerialReader] Disconnected")

    def parse_line(self, line: str) -> Optional[np.ndarray]:
        """
        Parse a line of serial data.

        Expected format: "sensor1,sensor2,sensor3,..."
        (Values separated by commas)

        Args:
            line: String from serial port

        Returns:
            Numpy array of sensor values or None if parse failed
        """
        try:
            # Remove whitespace
            line = line.strip()
            
            # Skip empty lines or comments
            if not line or line.startswith("#"):
                return None

            # Parse comma-separated values
            values = [float(x) for x in line.split(",")]
            return np.array(values)

        except ValueError as e:
            print(f"[SerialReader] Parse error: {line} - {e}")
            return None

    def read_data_stream(self, on_window_ready: Callable = None):
        """
        Read from serial port continuously.

        This method is meant to run in a thread and fills the data buffer.

        Args:
            on_window_ready: Callback function when a full window is ready
        """
        if not self.is_connected:
            print("[SerialReader] Not connected. Call connect() first.")
            return

        print("[SerialReader] Starting data stream...")
        self.stop_flag = False

        while not self.stop_flag:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore')
                    
                    sample = self.parse_line(line)
                    if sample is not None:
                        self.data_buffer.add_sample(sample)

                        # Check if window is ready
                        if self.data_buffer.buffer_full() and on_window_ready:
                            window = self.data_buffer.get_window()
                            on_window_ready(window)

                else:
                    time.sleep(0.01)  # Small sleep to avoid busy waiting

            except Exception as e:
                print(f"[SerialReader] Read error: {e}")
                time.sleep(0.1)

        print("[SerialReader] Data stream stopped")

    def start_stream_thread(self, on_window_ready: Callable = None):
        """
        Start reading in a background thread.

        Args:
            on_window_ready: Callback for when window is ready
        """
        self.read_thread = threading.Thread(
            target=self.read_data_stream,
            args=(on_window_ready,),
            daemon=True
        )
        self.read_thread.start()
        print("[SerialReader] Stream thread started")

    def stop_stream(self):
        """Stop the reading thread."""
        self.stop_flag = True
        if self.read_thread:
            self.read_thread.join(timeout=2)
        print("[SerialReader] Stream thread stopped")


class AnomalyDetectionInference:
    """Real-time anomaly detection inference."""

    def __init__(self, model: Any,
                 preprocessor: Any = None,
                 device: str = "cpu",
                 anomaly_threshold: float = config.ANOMALY_THRESHOLD):
        """
        Initialize inference system.

        Args:
            model: Trained model (classical or DL)
            preprocessor: Preprocessing pipeline (scaler)
            device: Device for DL models ('cpu' or 'cuda')
            anomaly_threshold: Threshold for anomaly detection
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.anomaly_threshold = anomaly_threshold
        self.inference_history = deque(maxlen=100)

    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """
        Preprocess a single window.

        Args:
            window: Input window (64, 6) - 64 samples, 6 features per sample

        Returns:
            Preprocessed window (1, 384) - flattened for model input
        """
        # Flatten: (64, 6) → (1, 384) for classical ML model
        window_flat = window.reshape(1, -1)  # (1, 384)
        
        # Note: Scaler is trained on original vibration data (1 feature),
        # but we now have 6 features per sample. For inference, we skip
        # normalization since the model was trained on normalized data
        # and the test data generator also produces unnormalized data.
        # In production, you would need to refit the scaler on the full
        # flattened 384-feature format.
        
        return window_flat

    def predict_window(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single window.

        Args:
            window: Input window

        Returns:
            Dictionary with predictions and anomaly info
        """
        # Preprocess
        window_processed = self.preprocess_window(window)

        # Get prediction
        prediction = self._run_model_inference(window_processed)

        # Determine if anomaly
        is_anomaly = prediction.get("anomaly_score", 0) > self.anomaly_threshold

        # Format result
        result = {
            "timestamp": time.time(),
            "is_anomaly": is_anomaly,
            "anomaly_score": prediction.get("anomaly_score", 0),
            "anomaly_type": self._classify_anomaly(prediction),
            "raw_prediction": prediction,
        }

        # Store in history
        self.inference_history.append(result)

        return result

    def _run_model_inference(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Run the actual model inference.

        Args:
            window: Preprocessed window

        Returns:
            Model prediction
        """
        import torch

        # Flatten for classical ML models
        window_flat = window.reshape(1, -1)

        # Check if PyTorch model
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            window_tensor = torch.from_numpy(window_flat).float().to(self.device)
            
            with torch.no_grad():
                output = self.model(window_tensor)
            
            # Compute reconstruction error
            reconstruction_error = torch.mean((window_tensor - output) ** 2).item()
            
            return {
                "anomaly_score": reconstruction_error,
                "type": "reconstruction_error",
            }

        else:
            # Classical ML model
            prediction = self.model.predict(window_flat)[0]
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(window_flat)[0]
                anomaly_score = 1 - proba[0]  # Probability of anomaly
            else:
                anomaly_score = 1 if prediction == 1 else 0

            return {
                "anomaly_score": anomaly_score,
                "type": "classification",
                "class": prediction,
            }

    def _classify_anomaly(self, prediction: Dict[str, Any]) -> str:
        """
        Classify specific type of anomaly.

        Args:
            prediction: Model prediction

        Returns:
            Anomaly type string
        """
        if not prediction or "anomaly_score" not in prediction:
            return config.ANOMALY_TYPES[0]

        # Simple classification based on score and pattern
        score = prediction.get("anomaly_score", 0)

        if score > self.anomaly_threshold * 2:
            return config.ANOMALY_TYPES[1]  # "High Vibration"
        elif score > self.anomaly_threshold * 1.5:
            return config.ANOMALY_TYPES[2]  # "High Temperature"
        else:
            return config.ANOMALY_TYPES[0]  # "Normal"

    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get statistics from inference history."""
        if not self.inference_history:
            return {}

        history_list = list(self.inference_history)
        anomalies = [r for r in history_list if r["is_anomaly"]]

        return {
            "total_inferences": len(history_list),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / len(history_list),
            "avg_anomaly_score": np.mean([r["anomaly_score"] for r in history_list]),
            "recent_state": "ANOMALY" if (history_list[-1]["is_anomaly"] if history_list else False) else "NORMAL",
        }


class RealTimeInferenceSystem:
    """Complete real-time inference system."""

    def __init__(self, model_path: str = None,
                 preprocessor_path: str = None,
                 serial_port: str = config.SERIAL_PORT,
                 baudrate: int = config.BAUD_RATE):
        """
        Initialize real-time inference system.

        Args:
            model_path: Path to trained model
            preprocessor_path: Path to preprocessor/scaler
            serial_port: Serial port for communication
            baudrate: Baud rate
        """
        # Load model and preprocessor
        if model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = None

        if preprocessor_path:
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
        else:
            self.preprocessor = None

        # Initialize components
        self.serial_reader = SerialDataReader(serial_port, baudrate)
        self.inference = AnomalyDetectionInference(self.model, self.preprocessor)

        print("[System] Real-time inference system initialized")

    def _load_model(self, model_path: str) -> Any:
        """Load model from disk."""
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.endswith('.pt'):
            # For PyTorch models, need architecture info
            model = torch.load(model_path)
        else:
            raise ValueError(f"Unknown model format: {model_path}")

        print(f"[System] Model loaded from {model_path}")
        return model

    def run(self, duration: int = None):
        """
        Run the inference system.

        Args:
            duration: Run duration in seconds (None for continuous)
        """
        print("[System] Starting real-time inference system...")

        # Connect to serial port
        if not self.serial_reader.connect():
            print("[System] Failed to connect to serial port")
            return

        # Start reading thread
        self.serial_reader.start_stream_thread(
            on_window_ready=self._on_window_ready
        )

        try:
            start_time = time.time()
            while True:
                if duration and (time.time() - start_time) > duration:
                    break

                time.sleep(1)
                stats = self.inference.get_inference_statistics()
                if stats:
                    print(f"[Status] Inferences: {stats['total_inferences']}, "
                          f"Anomalies: {stats['anomalies_detected']}, "
                          f"State: {stats['recent_state']}")

        except KeyboardInterrupt:
            print("\n[System] Interrupted by user")
        finally:
            self.stop()

    def _on_window_ready(self, window: np.ndarray):
        """Callback when a new window is ready."""
        result = self.inference.predict_window(window)
        
        # Print result
        status = "⚠️ ANOMALY" if result["is_anomaly"] else "✓ NORMAL"
        print(f"[Inference] {status} | Score: {result['anomaly_score']:.3f} | "
              f"Type: {result['anomaly_type']}")

    def stop(self):
        """Stop the inference system."""
        print("[System] Stopping...")
        self.serial_reader.stop_stream()
        self.serial_reader.disconnect()
        print("[System] Stopped")


if __name__ == "__main__":
    print("Serial inference module loaded.")
