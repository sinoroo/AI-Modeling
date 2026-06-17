"""
Data loading module for new CSV format.

Supports:
- New format: 9 lines of metadata + data from line 10 (time, vibration)
- Directory recursion with multiple CSV files
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-display backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import glob

from . import config


class DataLoader:
    """Load and analyze pump/motor sensor data from new format (metadata + vibration data)."""

    def __init__(self, data_path: str = None):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to CSV file or directory containing CSV files
        """
        self.data_path = data_path
        self.data = None
        self.feature_cols = None
        self.label_col = None
        self.metadata = None  # For new format files

    @staticmethod
    def find_csv_files(directory: str, recursive: bool = True) -> List[str]:
        """
        Find all CSV files in a directory.

        Args:
            directory: Directory path
            recursive: Whether to search subdirectories

        Returns:
            List of CSV file paths
        """
        if recursive:
            pattern = os.path.join(directory, "**", "*.csv")
            return sorted(glob.glob(pattern, recursive=True))
        else:
            pattern = os.path.join(directory, "*.csv")
            return sorted(glob.glob(pattern))

    def _parse_new_format_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Parse new format CSV file (9 lines metadata + data from line 10).

        Args:
            file_path: Path to CSV file

        Returns:
            Tuple of (data_df, metadata_dict)
        """
        metadata = {}
        
        # Read first 9 lines as metadata
        with open(file_path, 'r', encoding='utf-8') as f:
            for i in range(9):
                line = f.readline().strip()
                if ',' in line:
                    key, value = line.split(',', 1)
                    metadata[key.strip()] = value.strip()
        
        # Read data from line 10 onwards
        data_df = pd.read_csv(file_path, skiprows=9)
        
        # Rename columns to standard names if needed
        if len(data_df.columns) >= 2:
            data_df.columns = ['time', 'vibration'] + list(data_df.columns[2:])
        
        # Create label from Data Label metadata
        label = metadata.get('Data Label', '정상')
        data_df['label'] = 1 if label in config.ABNORMAL_LABELS else 0
        
        # Store metadata
        self.metadata = metadata
        
        return data_df, metadata

    def load_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a single CSV file (new format).

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        print(f" [DataLoader] Loading data from {file_path}...")

        try:
            self.data, self.metadata = self._parse_new_format_file(file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file in new format: {e}")

        print(f" [DataLoader] Data loaded: {self.data.shape}")
        return self.data

    def load_from_directory(self, directory: str) -> pd.DataFrame:
        """
        Load and combine all CSV files from a directory.

        Args:
            directory: Directory path

        Returns:
            Combined DataFrame from all CSV files
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")

        csv_files = self.find_csv_files(directory, recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {directory}")

        print(f" [DataLoader] Found {len(csv_files)} CSV files in {directory}")

        data_frames = []
        metadata_list = []

        for file_path in csv_files:
            try:
                print(f"  Loading: {os.path.basename(file_path)}")
                df = self.load_from_file(file_path)
                data_frames.append(df)
                if self.metadata:
                    metadata_list.append(self.metadata)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                continue

        if not data_frames:
            raise ValueError(f"No valid CSV files could be loaded from {directory}")

        # Combine all data frames
        self.data = pd.concat(data_frames, ignore_index=True)
        self.metadata = metadata_list  # Store list of metadata

        print(f" [DataLoader] Combined data shape: {self.data.shape}")
        return self.data

    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load data from file or directory (auto-detect).

        Args:
            data_path: Path to CSV file or directory. If None, uses self.data_path.

        Returns:
            DataFrame with loaded data
        """
        path = data_path or self.data_path
        
        if path is None:
            raise ValueError("data_path must be provided either in constructor or load_data() call")

        if os.path.isfile(path):
            return self.load_from_file(path)
        elif os.path.isdir(path):
            return self.load_from_directory(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def analyze_columns(self) -> Dict[str, object]:
        """
        Analyze columns and separate features from labels.

        Returns:
            Dictionary with column analysis
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(" [DataLoader] Analyzing columns...")

        # For new format, use 'label' column and 'vibration' feature
        self.label_col = 'label'
        self.feature_cols = ['vibration']

        analysis = {
            "total_rows": self.data.shape[0],
            "total_columns": self.data.shape[1],
            "num_features": len(self.feature_cols),
            "feature_cols": self.feature_cols,
            "label_col": self.label_col,
            "timestamp_col": 'time',
        }

        print(f"  Features: {len(self.feature_cols)} - {self.feature_cols}")
        print(f"  Label column: {self.label_col}")
        print(f"  Timestamp column: time")

        return analysis

    def get_eda_statistics(self) -> Dict[str, object]:
        """Compute EDA statistics."""
        if self.data is None or self.feature_cols is None:
            raise ValueError("Data not loaded. Call load_data() and analyze_columns() first.")

        stats = {
            "descriptive": self.data[self.feature_cols].describe().to_dict(),
            "missing_values": self.data[self.feature_cols].isnull().sum().to_dict(),
        }

        # Label distribution
        if self.label_col and self.label_col in self.data.columns:
            label_counts = self.data[self.label_col].value_counts()
            stats["label_distribution"] = label_counts.to_dict()

        # Correlations (for numeric features)
        numeric_data = self.data[self.feature_cols]
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            # Get top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_pairs.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j],
                        )
                    )
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            stats["top_correlations"] = [
                {"feature1": p[0], "feature2": p[1], "correlation": float(p[2])}
                for p in corr_pairs[:10]
            ]

        return stats

    def plot_eda_visualizations(self, output_dir: str = "eda_results"):
        """Generate and save EDA visualizations."""
        if self.data is None or self.feature_cols is None:
            raise ValueError("Data not loaded. Call load_data() and analyze_columns() first.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Distribution plot
        fig, axes = plt.subplots(1, len(self.feature_cols), figsize=(5 * len(self.feature_cols), 4))
        if len(self.feature_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(self.feature_cols):
            axes[idx].hist(self.data[col], bins=30, edgecolor="black")
            axes[idx].set_title(f"Distribution: {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "distribution.png"), dpi=100)
        plt.close()

        # Correlation heatmap (if multiple features)
        if len(self.feature_cols) > 1:
            corr = self.data[self.feature_cols].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
            plt.title("Feature Correlations")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation.png"), dpi=100)
            plt.close()

        # Label distribution (if label exists)
        if self.label_col and self.label_col in self.data.columns:
            label_counts = self.data[self.label_col].value_counts()
            plt.figure(figsize=(8, 6))
            label_counts.plot(kind="bar")
            plt.title(f"Label Distribution: {self.label_col}")
            plt.xlabel(self.label_col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "label_distribution.png"), dpi=100)
            plt.close()

        # Box plot for outlier detection
        if len(self.feature_cols) > 0:
            plt.figure(figsize=(10, 6))
            self.data[self.feature_cols].boxplot()
            plt.title("Box Plot: Outlier Detection")
            plt.ylabel("Values")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "boxplot.png"), dpi=100)
            plt.close()

    def print_eda_summary(self):
        """Print comprehensive EDA summary."""
        if self.data is None:
            raise ValueError("Data not loaded")

        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
        print("=" * 80)

        stats = self.get_eda_statistics()

        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 80)
        desc_df = pd.DataFrame(stats["descriptive"])
        print(desc_df)

        print("\n2. MISSING VALUES")
        print("-" * 80)
        missing = stats["missing_values"]
        if all(v == 0 for v in missing.values()):
            print("No missing values detected!")
        else:
            for col, count in missing.items():
                print(f"  {col}: {count}")

        if "label_distribution" in stats:
            print("\n3. LABEL DISTRIBUTION ('" + self.label_col + "')")
            print("-" * 80)
            for label, count in stats["label_distribution"].items():
                pct = 100 * count / self.data.shape[0]
                print(f"  {label}: {count} ({pct:.2f}%)")

        if "top_correlations" in stats:
            print("\n4. TOP FEATURE CORRELATIONS")
            print("-" * 80)
            for corr in stats["top_correlations"]:
                print(
                    f"  {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}"
                )
        
        if self.metadata:
            print("\n5. FILE METADATA (NEW FORMAT)")
            print("-" * 80)
            if isinstance(self.metadata, list):
                print(f"  Total files: {len(self.metadata)}")
                if self.metadata:
                    print(f"  First file metadata:")
                    for key, value in self.metadata[0].items():
                        print(f"    {key}: {value}")
            else:
                for key, value in self.metadata.items():
                    print(f"  {key}: {value}")

        print()
