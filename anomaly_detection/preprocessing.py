"""
Data preprocessing module for anomaly detection.

Handles:
- Missing value imputation
- Outlier handling
- Normalization / Standardization
- Time-series windowing
- Feature engineering (FFT, RMS, statistical features)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, List, Optional, Dict
import pickle

from . import config



class Preprocessor:
    """Preprocessing pipeline for sensor data."""

    def __init__(self, window_size: int = config.WINDOW_SIZE, 
                 normalize_method: str = config.NORMALIZE_METHOD):
        """
        Initialize Preprocessor.

        Args:
            window_size: Number of samples per window
            normalize_method: Normalization method ("standardize", "minmax", "robust")
        """
        self.window_size = window_size
        self.normalize_method = normalize_method
        self.scaler = None
        self.feature_cols = None
        self.label_col = None

    def fit(self, data: pd.DataFrame, feature_cols: List[str], label_col: Optional[str] = None):
        """
        Fit preprocessor on training data.

        Args:
            data: DataFrame with features
            feature_cols: List of feature column names
            label_col: Optional label column name
        """
        self.feature_cols = feature_cols
        self.label_col = label_col

        # Create scaler based on method
        if self.normalize_method == "standardize":
            self.scaler = StandardScaler()
        elif self.normalize_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.normalize_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")

        # Fit scaler on feature columns
        self.scaler.fit(data[feature_cols])

        print(f"[Preprocessor] Fitted with {len(feature_cols)} features")

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using configured method.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        data = data.copy()

        if config.MISSING_VALUE_METHOD == "interpolate":
            data = data.interpolate(method="linear", limit_direction="both")
        elif config.MISSING_VALUE_METHOD == "forward_fill":
            data = data.fillna(method="ffill").fillna(method="bfill")
        elif config.MISSING_VALUE_METHOD == "drop":
            data = data.dropna()

        return data

    def handle_outliers(self, data: pd.DataFrame, cols: List[str], 
                       method: str = "iqr") -> pd.DataFrame:
        """
        Handle outliers using IQR or Z-score.

        Args:
            data: Input DataFrame
            cols: Columns to process
            method: "iqr" or "zscore"

        Returns:
            DataFrame with outliers handled
        """
        data = data.copy()

        for col in cols:
            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap values instead of removing
                data[col] = np.clip(data[col], lower_bound, upper_bound)

            elif method == "zscore":
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                threshold = 3
                data[col] = np.where(z_scores > threshold, 
                                     data[col].mean(), data[col])

        return data

    def normalize_features(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Normalize features using fitted scaler.

        Args:
            data: Input DataFrame
            cols: Columns to normalize

        Returns:
            DataFrame with normalized features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        data = data.copy()
        data[cols] = self.scaler.transform(data[cols])
        return data

    def create_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        Generate statistical features (RMS, Peak, Kurtosis, Skewness).

        Args:
            data: 2D array of shape (n_samples, n_features) or
                  3D array of shape (n_windows, window_size, n_features)

        Returns:
            Array with statistical features
        """
        if data.ndim == 2:
            # Single window
            features = []
            
            # RMS (Root Mean Square)
            rms = np.sqrt(np.mean(data**2, axis=0))
            features.append(rms)
            
            # Peak value
            peak = np.max(np.abs(data), axis=0)
            features.append(peak)
            
            # Crest factor
            crest = peak / (np.sqrt(np.mean(data**2, axis=0)) + 1e-8)
            features.append(crest)
            
            # Kurtosis (peakedness)
            kurt = np.array([
                kurtosis(data[:, i]) for i in range(data.shape[1])
            ])
            features.append(kurt)
            
            # Skewness
            skewness = np.array([
                skew(data[:, i]) for i in range(data.shape[1])
            ])
            features.append(skewness)
            
            return np.concatenate(features)

        elif data.ndim == 3:
            # Multiple windows
            n_windows, window_size, n_features = data.shape
            features_list = []
            
            for w in range(n_windows):
                feat = self.create_statistical_features(data[w])
                features_list.append(feat)
            
            return np.array(features_list)

    def create_fft_features(self, data: np.ndarray, n_fft_bins: int = 10) -> np.ndarray:
        """
        Generate FFT-based frequency domain features.

        Args:
            data: 2D array of shape (n_samples, n_features)
            n_fft_bins: Number of FFT bins to use

        Returns:
            FFT features
        """
        features_list = []

        for feat_idx in range(data.shape[1]):
            # Compute FFT
            fft_vals = np.abs(np.fft.fft(data[:, feat_idx]))
            
            # Take first n_fft_bins frequencies
            fft_features = fft_vals[:n_fft_bins]
            features_list.append(fft_features)

        return np.concatenate(features_list)

    def create_windows(self, data: pd.DataFrame, 
                      feature_cols: List[str],
                      label_col: Optional[str] = None,
                      step_size: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding windows from time-series data.

        Args:
            data: Input DataFrame
            feature_cols: Feature columns
            label_col: Optional label column
            step_size: Step size for windows (if None, uses config)

        Returns:
            Tuple of (windows, labels)
        """
        if step_size is None:
            step_size = config.STEP_SIZE

        X_windows = []
        y_windows = []

        n_samples = len(data)
        
        for start_idx in range(0, n_samples - self.window_size + 1, step_size):
            end_idx = start_idx + self.window_size
            
            window = data.iloc[start_idx:end_idx][feature_cols].values
            X_windows.append(window)
            
            if label_col:
                # Use majority label in window
                window_label = data.iloc[start_idx:end_idx][label_col].mode()
                label = window_label[0] if len(window_label) > 0 else data.iloc[start_idx][label_col]
                y_windows.append(label)

        X = np.array(X_windows)
        y = np.array(y_windows) if label_col else None

        print(f"[Preprocessor] Created {len(X_windows)} windows of size {self.window_size}")

        return X, y

    def add_statistical_features_to_windows(self, X_windows: np.ndarray) -> np.ndarray:
        """
        Add statistical features (RMS, Peak, Crest, Kurtosis, Skewness) to each window.
        
        Args:
            X_windows: 3D array of shape (n_windows, window_size, 1)
        
        Returns:
            3D array of shape (n_windows, window_size, 6)
            - Feature 0: vibration (original value)
            - Feature 1: RMS (root mean square)
            - Feature 2: Peak (max absolute value)
            - Feature 3: Crest Factor (Peak / RMS)
            - Feature 4: Kurtosis (peakedness)
            - Feature 5: Skewness (asymmetry)
        """
        n_windows, window_size, n_features = X_windows.shape
        
        # Output array with 6 features
        X_with_features = np.zeros((n_windows, window_size, 6))
        
        for w_idx in range(n_windows):
            window = X_windows[w_idx]  # shape: (window_size, 1)
            window_data = window[:, 0]  # flatten to 1D array
            
            # Calculate statistical features for entire window
            rms = np.sqrt(np.mean(window_data**2))
            peak = np.max(np.abs(window_data))
            crest = peak / (rms + 1e-8)  # avoid division by zero
            kurt = kurtosis(window_data)
            skewness_val = skew(window_data)
            
            # Assign features to each sample in the window
            for s_idx in range(window_size):
                X_with_features[w_idx, s_idx, 0] = window_data[s_idx]       # vibration
                X_with_features[w_idx, s_idx, 1] = rms                      # RMS
                X_with_features[w_idx, s_idx, 2] = peak                     # Peak
                X_with_features[w_idx, s_idx, 3] = crest                    # Crest Factor
                X_with_features[w_idx, s_idx, 4] = kurt                     # Kurtosis
                X_with_features[w_idx, s_idx, 5] = skewness_val             # Skewness
        
        print(f"[Preprocessor] Added statistical features: {X_windows.shape} → {X_with_features.shape}")
        return X_with_features

    def preprocess_pipeline(self, data: pd.DataFrame, 
                           feature_cols: List[str],
                           label_col: Optional[str] = None,
                           fit_scaler: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Full preprocessing pipeline.

        Args:
            data: Input DataFrame
            feature_cols: Feature columns
            label_col: Optional label column
            fit_scaler: Whether to fit scaler (True for training data)

        Returns:
            Tuple of (windows with 6 features, labels)
            Returns shape: (n_windows, window_size, 6) for X and (n_windows,) for y
        """
        print("[Preprocessor] Starting preprocessing pipeline...")
        
        data = data.copy()

        # Step 1: Handle missing values
        print("  - Handling missing values...")
        data = self.handle_missing_values(data)

        # Step 2: Handle outliers
        print("  - Handling outliers...")
        data = self.handle_outliers(data, feature_cols, method="iqr")

        # Step 3: Fit and normalize
        if fit_scaler:
            print("  - Fitting scaler...")
            self.fit(data, feature_cols, label_col)
        
        print("  - Normalizing features...")
        data = self.normalize_features(data, feature_cols)

        # Step 4: Create windows
        print("  - Creating time-series windows...")
        X_windows, y_windows = self.create_windows(data, feature_cols, label_col)

        # Step 5: Add statistical features
        print("  - Adding statistical features (6 features per sample)...")
        X_windows = self.add_statistical_features_to_windows(X_windows)

        print("[Preprocessor] Preprocessing complete!")
        
        return X_windows, y_windows

    def save(self, filepath: str):
        """Save preprocessor (scaler) to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[Preprocessor] Saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Load preprocessor from disk."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"[Preprocessor] Loaded from {filepath}")
        return preprocessor


def preprocess_data(train_data: pd.DataFrame, 
                    val_data: pd.DataFrame,
                    test_data: pd.DataFrame,
                    feature_cols: List[str],
                    label_col: Optional[str] = None) -> Dict:
    """
    Preprocess all data splits.

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        feature_cols: Feature columns
        label_col: Optional label column

    Returns:
        Dictionary with preprocessed data and preprocessor
    """
    preprocessor = Preprocessor()

    # Process training data (fit on this)
    X_train, y_train = preprocessor.preprocess_pipeline(
        train_data, feature_cols, label_col, fit_scaler=True
    )

    # Process validation and test data (use fitted scaler)
    X_val, y_val = preprocessor.preprocess_pipeline(
        val_data, feature_cols, label_col, fit_scaler=False
    )

    X_test, y_test = preprocessor.preprocess_pipeline(
        test_data, feature_cols, label_col, fit_scaler=False
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    # Example usage
    print("Preprocessing module loaded.")
