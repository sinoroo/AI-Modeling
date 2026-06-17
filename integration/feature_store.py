"""
Feature Store management for anomaly detection models.

Manages:
- Feature schema definition and validation
- Feature statistics tracking
- Feature metadata storage
"""

from typing import Dict, List, Any, Optional
import json
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection import config


class FeatureStore:
    """Manage feature schemas and metadata."""

    def __init__(self, store_dir: str = "feature_store"):
        """
        Initialize Feature Store.

        Args:
            store_dir: Directory to store feature metadata
        """
        self.store_dir = store_dir
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        self.schema_file = os.path.join(store_dir, "feature_schema.json")
        self.stats_file = os.path.join(store_dir, "feature_stats.pkl")
        self.schema = {}
        self.stats = {}

    def create_schema(self, feature_cols: List[str], 
                     label_col: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create and store feature schema.

        Args:
            feature_cols: List of feature column names
            label_col: Label column name
            metadata: Additional metadata

        Returns:
            Schema dictionary
        """
        self.schema = {
            "features": feature_cols,
            "label": label_col,
            "feature_count": len(feature_cols),
            "window_size": config.WINDOW_SIZE,
            "normalize_method": config.NORMALIZE_METHOD,
            "missing_value_method": config.MISSING_VALUE_METHOD,
            "metadata": metadata or {},
            "version": "1.0"
        }
        
        self._save_schema()
        print(f"[FeatureStore] Schema created with {len(feature_cols)} features")
        return self.schema

    def compute_statistics(self, data: pd.DataFrame, 
                          feature_cols: List[str]) -> Dict[str, Any]:
        """
        Compute and store feature statistics.

        Args:
            data: Input DataFrame
            feature_cols: Feature column names

        Returns:
            Statistics dictionary
        """
        stats = {}
        
        for col in feature_cols:
            if col in data.columns:
                col_data = data[col].dropna()
                stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75)),
                    "null_count": int(data[col].isnull().sum()),
                    "dtype": str(data[col].dtype)
                }
        
        self.stats = {
            "features": stats,
            "computed_at": pd.Timestamp.now().isoformat(),
            "total_samples": len(data)
        }
        
        self._save_stats()
        print(f"[FeatureStore] Statistics computed for {len(stats)} features")
        return self.stats

    def get_schema(self) -> Dict[str, Any]:
        """Get feature schema."""
        if not self.schema and os.path.exists(self.schema_file):
            self._load_schema()
        return self.schema

    def get_statistics(self) -> Dict[str, Any]:
        """Get feature statistics."""
        if not self.stats and os.path.exists(self.stats_file):
            self._load_stats()
        return self.stats

    def validate_features(self, data: pd.DataFrame) -> bool:
        """
        Validate if data conforms to schema.

        Args:
            data: Input DataFrame

        Returns:
            True if valid, False otherwise
        """
        if not self.schema:
            print("[FeatureStore] Schema not loaded")
            return False

        required_cols = self.schema.get("features", [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            print(f"[FeatureStore] Missing columns: {missing_cols}")
            return False
        
        if len(data) == 0:
            print("[FeatureStore] Empty DataFrame")
            return False
        
        print(f"[FeatureStore] Data validation passed")
        return True

    def _save_schema(self):
        """Save schema to file."""
        with open(self.schema_file, 'w') as f:
            json.dump(self.schema, f, indent=2)

    def _load_schema(self):
        """Load schema from file."""
        if os.path.exists(self.schema_file):
            with open(self.schema_file, 'r') as f:
                self.schema = json.load(f)

    def _save_stats(self):
        """Save statistics to file."""
        with open(self.stats_file, 'wb') as f:
            pickle.dump(self.stats, f)

    def _load_stats(self):
        """Load statistics from file."""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'rb') as f:
                self.stats = pickle.load(f)

    def export_schema_json(self, filepath: str):
        """Export schema as JSON for external use."""
        with open(filepath, 'w') as f:
            json.dump(self.schema, f, indent=2)
        print(f"[FeatureStore] Schema exported to {filepath}")

    def export_stats_json(self, filepath: str):
        """Export statistics as JSON for external use."""
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"[FeatureStore] Statistics exported to {filepath}")
