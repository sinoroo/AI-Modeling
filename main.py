"""
Main orchestrator script for anomaly detection pipeline.

Usage:
    python main.py --train          # Run full training pipeline
    python main.py --infer          # Run inference system
    python main.py --eda            # Run EDA only
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anomaly_detection import (
    config,
    data_loader,
    preprocessing,
    model_training,
    evaluation,
    inference_serial,
)


def create_sample_data():
    """Create sample synthetic data for demonstration."""
    print("\n" + "="*80)
    print("CREATING SAMPLE DATA")
    print("="*80)

    # Create data directory
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Number of samples
    n_train = 1000
    n_val = 300
    n_test = 300

    def generate_normal_data(n_samples: int, n_features: int = 5):
        """Generate normal operating data."""
        data = {}
        for i in range(n_features):
            data[f"sensor_{i}"] = np.random.normal(loc=50, scale=10, size=n_samples)
        data["label"] = 0  # Normal
        return pd.DataFrame(data)

    def generate_anomaly_data(n_samples: int, n_features: int = 5):
        """Generate anomalous data."""
        data = {}
        for i in range(n_features):
            data[f"sensor_{i}"] = np.random.normal(loc=100, scale=20, size=n_samples)
        data["label"] = 1  # Anomaly
        return pd.DataFrame(data)

    # Generate splits
    train_normal = generate_normal_data(int(n_train * 0.8))
    train_anomaly = generate_anomaly_data(int(n_train * 0.2))
    train_data = pd.concat([train_normal, train_anomaly], ignore_index=True)
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    val_normal = generate_normal_data(int(n_val * 0.8))
    val_anomaly = generate_anomaly_data(int(n_val * 0.2))
    val_data = pd.concat([val_normal, val_anomaly], ignore_index=True)
    val_data = val_data.sample(frac=1).reset_index(drop=True)

    test_normal = generate_normal_data(int(n_test * 0.8))
    test_anomaly = generate_anomaly_data(int(n_test * 0.2))
    test_data = pd.concat([test_normal, test_anomaly], ignore_index=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    # Save to CSV
    train_data.to_csv(config.TRAIN_DATA_PATH, index=False)
    val_data.to_csv(config.VAL_DATA_PATH, index=False)
    test_data.to_csv(config.TEST_DATA_PATH, index=False)

    print(f"✓ Training data: {config.TRAIN_DATA_PATH} ({len(train_data)} samples)")
    print(f"✓ Validation data: {config.VAL_DATA_PATH} ({len(val_data)} samples)")
    print(f"✓ Test data: {config.TEST_DATA_PATH} ({len(test_data)} samples)")

    return train_data, val_data, test_data


def run_eda(data_path: str = None):
    """Run exploratory data analysis."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)

    if data_path is None:
        data_path = config.TRAIN_DATA_PATH

    # Load and analyze data
    loader = data_loader.DataLoader(data_path)
    data = loader.load_data()
    analysis = loader.analyze_columns()
    loader.print_eda_summary()
    loader.plot_eda_visualizations()

    print("✓ EDA complete. Results saved to:", config.EDA_OUTPUT_DIR)


def run_training_pipeline(
    train_path: str = None,
    val_path: str = None, 
    test_path: str = None,
    use_new_format: bool = None
):
    """
    Run complete training pipeline.
    
    Args:
        train_path: Path to training data (file or directory)
        val_path: Path to validation data (file or directory)
        test_path: Path to test data (file or directory)
        use_new_format: Whether to use new CSV format (9 lines metadata)
    """
    print("\n" + "="*80)
    print("ANOMALY DETECTION TRAINING PIPELINE")
    print("="*80)

    # Use provided paths or fall back to config defaults
    train_path = train_path or config.TRAIN_DATA_DIR or config.TRAIN_DATA_PATH
    val_path = val_path or config.VAL_DATA_DIR or config.VAL_DATA_PATH
    test_path = test_path or config.TEST_DATA_DIR or config.TEST_DATA_PATH
    use_new_format = use_new_format if use_new_format is not None else config.USE_NEW_CSV_FORMAT

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    loader_train = data_loader.DataLoader(use_new_format=use_new_format)
    train_df = loader_train.load_data(train_path)
    
    loader_val = data_loader.DataLoader(use_new_format=use_new_format)
    val_df = loader_val.load_data(val_path)
    
    loader_test = data_loader.DataLoader(use_new_format=use_new_format)
    test_df = loader_test.load_data(test_path)

    # Step 2: Analyze data and extract features
    print("\n[Step 2] Analyzing data structure...")
    loader = loader_train  # Use train loader for analysis
    analysis = loader.analyze_columns()
    feature_cols = analysis["feature_cols"]
    label_col = analysis["label_col"]

    # Step 3: Preprocess data
    print("\n[Step 3] Preprocessing data...")
    preprocessed = preprocessing.preprocess_data(
        train_df, val_df, test_df, feature_cols, label_col
    )

    X_train = preprocessed["X_train"]
    y_train = preprocessed["y_train"]
    X_val = preprocessed["X_val"]
    y_val = preprocessed["y_val"]
    X_test = preprocessed["X_test"]
    y_test = preprocessed["y_test"]
    preprocessor = preprocessed["preprocessor"]

    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Step 4: Train models
    print("\n[Step 4] Training models...")
    models, train_results = model_training.train_all_models(
        X_train, y_train, X_val, y_val,
        model_names=["RandomForest", "IsolationForest", "OneClassSVM", "Autoencoder", "LSTM"]
    )

    # Step 5: Save models
    print("\n[Step 5] Saving models...")
    for model_name, model in models.items():
        filepath = config.MODEL_FILES.get(model_name)
        if filepath:
            if isinstance(model, object) and hasattr(model, 'state_dict'):
                # PyTorch model
                model_training.DeepLearningTrainer.save_model(model, filepath)
            else:
                # Classical ML model
                model_training.ClassicalModelTrainer.save_model(model, filepath)

    # Save preprocessor
    preprocessor.save(config.MODEL_FILES["Preprocessor"])

    # Step 6: Evaluate models
    print("\n[Step 6] Evaluating models...")
    evaluation_results = {}

    for model_name, model in models.items():
        if model_name in ["RandomForest", "IsolationForest", "OneClassSVM"]:
            result = evaluation.evaluate_classical_model(model, X_test, y_test, model_name)
        else:
            result = evaluation.evaluate_deep_learning_model(
                model, X_test, y_test, model_name
            )
        evaluation_results[model_name] = result

    # Step 7: Compare models
    print("\n[Step 7] Model comparison...")
    comparison_df = evaluation.compare_models(evaluation_results)

    # Step 8: Generate report
    print("\n[Step 8] Generating report...")
    evaluation.generate_evaluation_report(evaluation_results, "evaluation_report.json")

    print("\n" + "="*80)
    print("✓ TRAINING PIPELINE COMPLETE")
    print("="*80)

    return models, preprocessor, evaluation_results


def run_inference_demo():
    """Run inference demonstration with sample serial data."""
    print("\n" + "="*80)
    print("REAL-TIME INFERENCE DEMO")
    print("="*80)

    # Load best model
    print("\nLoading trained model...")
    best_model = model_training.ClassicalModelTrainer.load_model(
        config.MODEL_FILES["RandomForest"]
    )
    
    preprocessor = preprocessing.Preprocessor.load(config.MODEL_FILES["Preprocessor"])

    # Create inference system
    inference_sys = inference_serial.AnomalyDetectionInference(
        best_model, preprocessor
    )

    # Generate test windows
    print("\nGenerating test data windows...")
    np.random.seed(42)

    for i in range(10):
        # Normal window
        window = np.random.normal(loc=50, scale=10, size=(config.WINDOW_SIZE, 5))
        result = inference_sys.predict_window(window)
        print(f"  Window {i+1}: {'⚠️ ANOMALY' if result['is_anomaly'] else '✓ NORMAL'} "
              f"(score: {result['anomaly_score']:.3f})")

    # Anomaly window
    print("\n  Simulating anomaly...")
    anomaly_window = np.random.normal(loc=150, scale=30, size=(config.WINDOW_SIZE, 5))
    result = inference_sys.predict_window(anomaly_window)
    print(f"  Anomaly window: {'⚠️ ANOMALY' if result['is_anomaly'] else '✓ NORMAL'} "
          f"(score: {result['anomaly_score']:.3f})")

    # Print statistics
    print("\nInference Statistics:")
    stats = inference_sys.get_inference_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("✓ INFERENCE DEMO COMPLETE")
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Pipeline for Pump/Motor Systems"
    )
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--eda", action="store_true", help="Run EDA only")
    parser.add_argument("--infer", action="store_true", help="Run inference demo")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample data")
    
    # Data path arguments
    parser.add_argument("--train-path", type=str, default=None, 
                       help="Path to training data (file or directory)")
    parser.add_argument("--val-path", type=str, default=None,
                       help="Path to validation data (file or directory)")
    parser.add_argument("--test-path", type=str, default=None,
                       help="Path to test data (file or directory)")
    parser.add_argument("--train-dir", type=str, default=None,
                       help="Directory with training CSV files (alternative to --train-path)")
    parser.add_argument("--val-dir", type=str, default=None,
                       help="Directory with validation CSV files (alternative to --val-path)")
    parser.add_argument("--test-dir", type=str, default=None,
                       help="Directory with test CSV files (alternative to --test-path)")
    
    # Format arguments
    parser.add_argument("--new-format", action="store_true", default=None,
                       help="Use new CSV format (9 lines metadata + data from line 10)")
    parser.add_argument("--legacy-format", action="store_true", 
                       help="Use legacy CSV format (single header line + data)")
    
    args = parser.parse_args()
    
    # Resolve data paths (dirs have priority over paths if both specified)
    train_path = args.train_dir or args.train_path
    val_path = args.val_dir or args.val_path
    test_path = args.test_dir or args.test_path
    
    # Resolve format
    use_new_format = None
    if args.new_format:
        use_new_format = True
    elif args.legacy_format:
        use_new_format = False

    # If no arguments, run full pipeline with defaults
    if not (args.train or args.eda or args.infer or args.generate_data):
        args.generate_data = True
        args.train = True
        args.infer = True
    
    # If no data paths specified and not generating data, generate it
    if not train_path and not args.generate_data and args.train:
        print("[INFO] No data paths specified. Generating sample data...")
        args.generate_data = True

    # Generate sample data if needed
    if args.generate_data:
        create_sample_data()

    # Run EDA
    if args.eda or (args.train and not train_path):
        # EDA is useful for generated data or when no training is specified
        run_eda()

    # Run training
    if args.train:
        models, preprocessor, evaluation_results = run_training_pipeline(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            use_new_format=use_new_format
        )

    # Run inference
    if args.infer:
        run_inference_demo()

    print("\n✓ Pipeline execution complete!")


if __name__ == "__main__":
    main()
