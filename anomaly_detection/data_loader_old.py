"""
Data loading and exploratory data analysis (EDA) module.

This module handles:
- Loading CSV data
- Performing EDA (statistical summary, correlation, visualization)
- Time-series pattern analysis
- Feature importance assessment
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from . import config


class DataLoader:
    """Load and analyze pump/motor sensor data."""

    def __init__(self, data_path: str):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to CSV file
        """
        self.data_path = data_path
        self.data = None
        self.feature_cols = None
        self.label_col = None

    def load_data(self) -> pd.DataFrame:
        """Load CSV data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        print(f" [DataLoader] Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        print(f" [DataLoader] Data loaded: {self.data.shape}")
        return self.data

    def analyze_columns(self) -> Dict[str, object]:
        """
        Analyze columns and separate features from labels.

        Returns:
            Dictionary with column analysis
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(" [DataLoader] Analyzing columns...")

        # Identify label column (if exists)
        possible_label_cols = [
            col
            for col in self.data.columns
            if "label" in col.lower() or "anomaly" in col.lower() or "target" in col.lower()
        ]

        self.label_col = possible_label_cols[0] if possible_label_cols else None

        # Identify timestamp column
        timestamp_col = None
        possible_timestamp_cols = [
            col
            for col in self.data.columns
            if "time" in col.lower() or "timestamp" in col.lower() or "date" in col.lower()
        ]
        if possible_timestamp_cols:
            timestamp_col = possible_timestamp_cols[0]

        # Features are all numeric columns except label and timestamp
        exclude_cols = {self.label_col, timestamp_col} - {None}
        self.feature_cols = [
            col
            for col in self.data.columns
            if self.data[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            and col not in exclude_cols
        ]

        analysis = {
            "total_rows": self.data.shape[0],
            "total_columns": self.data.shape[1],
            "num_features": len(self.feature_cols),
            "feature_cols": self.feature_cols,
            "label_col": self.label_col,
            "timestamp_col": timestamp_col,
        }

        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Label column: {self.label_col}")
        print(f"  Timestamp column: {timestamp_col}")

        return analysis

    def get_eda_statistics(self) -> Dict:
        """Generate EDA statistics."""
        if self.feature_cols is None:
            self.analyze_columns()

        print(" [DataLoader] Computing EDA statistics...")

        stats = {
            "descriptive": self.data[self.feature_cols].describe().to_dict(),
            "missing_values": self.data[self.feature_cols].isnull().sum().to_dict(),
            "data_types": self.data[self.feature_cols].dtypes.to_dict(),
        }

        # Correlation matrix
        correlation = self.data[self.feature_cols].corr()
        stats["correlation"] = correlation

        # Label distribution if available
        if self.label_col:
            stats["label_distribution"] = self.data[self.label_col].value_counts().to_dict()

        return stats

    def plot_eda_visualizations(self, output_dir: str = None):
        """Generate and save EDA visualizations."""
        if output_dir is None:
            output_dir = config.EDA_OUTPUT_DIR

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if self.feature_cols is None:
            self.analyze_columns()

        print(f" [DataLoader] Generating EDA visualizations to {output_dir}...")

        # 1. Feature distributions
        fig, axes = plt.subplots(
            int(np.ceil(len(self.feature_cols) / 3)), 3, figsize=(15, 12)
        )
        axes = axes.flatten()

        for idx, col in enumerate(self.feature_cols):
            axes[idx].hist(self.data[col].dropna(), bins=30, edgecolor="black")
            axes[idx].set_title(f"Distribution: {col}")
            axes[idx].set_xlabel("Value")
            axes[idx].set_ylabel("Frequency")

        for idx in range(len(self.feature_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_feature_distributions.png"), dpi=100)
        plt.close()

        # 2. Correlation heatmap
        corr_matrix = self.data[self.feature_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_correlation_matrix.png"), dpi=100)
        plt.close()

        # 3. Box plots for outlier detection
        fig, axes = plt.subplots(
            int(np.ceil(len(self.feature_cols) / 3)), 3, figsize=(15, 12)
        )
        axes = axes.flatten()

        for idx, col in enumerate(self.feature_cols):
            axes[idx].boxplot(self.data[col].dropna())
            axes[idx].set_title(f"Box Plot: {col}")
            axes[idx].set_ylabel("Value")

        for idx in range(len(self.feature_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_box_plots.png"), dpi=100)
        plt.close()

        # 4. Time-series plot (if data is ordered)
        if len(self.data) <= 5000:
            fig, axes = plt.subplots(len(self.feature_cols), 1, figsize=(14, 3 * len(self.feature_cols)))
            if len(self.feature_cols) == 1:
                axes = [axes]

            for idx, col in enumerate(self.feature_cols):
                axes[idx].plot(self.data.index, self.data[col], linewidth=0.5)
                axes[idx].set_ylabel(col)
                axes[idx].set_title(f"Time Series: {col}")

            axes[-1].set_xlabel("Sample Index")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "04_time_series.png"), dpi=100)
            plt.close()

        # 5. Label distribution if available
        if self.label_col and self.data[self.label_col].nunique() < 10:
            plt.figure(figsize=(10, 6))
            self.data[self.label_col].value_counts().plot(kind="bar", color="steelblue")
            plt.title(f"Label Distribution: {self.label_col}")
            plt.xlabel(self.label_col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "05_label_distribution.png"), dpi=100)
            plt.close()

        print(f" [DataLoader] Visualizations saved to {output_dir}/")

    def print_eda_summary(self):
        """Print EDA summary to console."""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
        print("=" * 80)

        stats = self.get_eda_statistics()

        # Descriptive Statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 80)
        df_desc = pd.DataFrame(stats["descriptive"]).T
        print(df_desc)

        # Missing Values
        print("\n2. MISSING VALUES")
        print("-" * 80)
        missing = stats["missing_values"]
        if sum(missing.values()) == 0:
            print("No missing values detected!")
        else:
            for col, count in missing.items():
                if count > 0:
                    print(f"  {col}: {count} ({100*count/len(self.data):.2f}%)")

        # Label Distribution
        if self.label_col:
            print(f"\n3. LABEL DISTRIBUTION ('{self.label_col}')")
            print("-" * 80)
            label_dist = stats["label_distribution"]
            for label, count in label_dist.items():
                print(f"  {label}: {count} ({100*count/len(self.data):.2f}%)")

        # Top Correlations
        print("\n4. TOP FEATURE CORRELATIONS")
        print("-" * 80)
        corr = stats["correlation"]
        # Get top positive correlations (excluding self-correlation)
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, corr_val in corr_pairs[:10]:
            print(f"  {col1} <-> {col2}: {corr_val:.3f}")

        print("\n" + "=" * 80)


def perform_eda(data_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to perform EDA.

    Args:
        data_path: Path to CSV file

    Returns:
        Tuple of (data, eda_stats)
    """
    loader = DataLoader(data_path)
    data = loader.load_data()
    analysis = loader.analyze_columns()
    loader.print_eda_summary()
    loader.plot_eda_visualizations()

    stats = loader.get_eda_statistics()

    return data, stats


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = config.TRAIN_DATA_PATH

    perform_eda(data_path)
