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
from integration import (
    MLflowTracker,
    FeatureStore,
)


def run_eda(data_path: str = None):
    """Run exploratory data analysis."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)

    if data_path is None:
        data_path = config.TRAIN_DATA_DIR

    if not data_path:
        raise ValueError("No data path provided. Specify data_path or configure TRAIN_DATA_DIR")

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
    use_mlflow: bool = True
):
    """
    Run complete training pipeline with MLflow tracking.
    
    Args:
        train_path: Path to training data directory
        val_path: Path to validation data directory
        test_path: Path to test data directory
        use_mlflow: Enable MLflow tracking
    """
    print("\n" + "="*80)
    print("ANOMALY DETECTION TRAINING PIPELINE")
    print("="*80)

    # Initialize MLflow
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MLflowTracker(
            experiment_name="anomaly_detection",
            tracking_uri="sqlite:///integration/mlflow.db"
        )
        run_id = mlflow_tracker.start_run(
            run_name="training_run",
            tags={
                "model_type": "ensemble",
                "task": "anomaly_detection",
                "dataset": "pump_motor_system"
            }
        )
        print(f"[MLflow] Experiment tracking started with run ID: {run_id}")

    # Initialize Feature Store
    fs = FeatureStore(store_dir="integration/feature_store")

    # Use provided paths or fall back to config defaults
    train_path = train_path or config.TRAIN_DATA_DIR
    val_path = val_path or config.VAL_DATA_DIR
    test_path = test_path or config.TEST_DATA_DIR

    if not all([train_path, val_path, test_path]):
        raise ValueError("Data paths not configured. Check config.TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR")

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    loader_train = data_loader.DataLoader()
    train_df = loader_train.load_data(train_path)
    
    loader_val = data_loader.DataLoader()
    val_df = loader_val.load_data(val_path)
    
    loader_test = data_loader.DataLoader()
    test_df = loader_test.load_data(test_path)

    # Step 2: Analyze data and extract features
    print("\n[Step 2] Analyzing data structure...")
    loader = loader_train  # Use train loader for analysis
    analysis = loader.analyze_columns()
    feature_cols = analysis["feature_cols"]
    label_col = analysis["label_col"]

    # Create and log feature schema
    print("\n[Step 2b] Creating feature schema...")
    schema = fs.create_schema(
        feature_cols=feature_cols,
        label_col=label_col,
        metadata={
            "data_source": "pump_motor_system",
            "window_size": config.WINDOW_SIZE,
            "normalize_method": config.NORMALIZE_METHOD
        }
    )
    
    # Compute and log feature statistics
    print("[Step 2c] Computing feature statistics...")
    stats = fs.compute_statistics(train_df, feature_cols)
    
    if mlflow_tracker:
        mlflow_tracker.log_feature_schema(schema, "feature_schema")
        mlflow_tracker.log_dict(stats, "feature_statistics")

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

    # Log preprocessing parameters
    if mlflow_tracker:
        mlflow_tracker.log_params({
            "window_size": config.WINDOW_SIZE,
            "normalize_method": config.NORMALIZE_METHOD,
            "missing_value_method": config.MISSING_VALUE_METHOD,
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "test_samples": X_test.shape[0],
            "feature_count": len(feature_cols)
        })

    # Step 4: Train models
    print("\n[Step 4] Training models...")
    models, train_results = model_training.train_all_models(
        X_train, y_train, X_val, y_val,
        model_names=["RandomForest", "IsolationForest", "OneClassSVM", "Autoencoder", "LSTM"],
        mlflow_tracker=mlflow_tracker
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
        
        # Log evaluation metrics to MLflow
        if mlflow_tracker and isinstance(result, dict):
            metrics_to_log = {
                f"{model_name}_accuracy": result.get("accuracy"),
                f"{model_name}_precision": result.get("precision"),
                f"{model_name}_recall": result.get("recall"),
                f"{model_name}_f1": result.get("f1")
            }
            mlflow_tracker.log_metrics({k: v for k, v in metrics_to_log.items() if v is not None})

    # Step 7: Compare models
    print("\n[Step 7] Model comparison...")
    comparison_df = evaluation.compare_models(evaluation_results)
    
    if mlflow_tracker:
        mlflow_tracker.log_dict(evaluation_results, "evaluation_results")

    # Step 8: Generate report
    print("\n[Step 8] Generating report...")
    evaluation.generate_evaluation_report(evaluation_results, "evaluation_report.json")

    # Find best model
    best_model_name = comparison_df.iloc[0]['Model'] if not comparison_df.empty else list(models.keys())[0]
    print(f"\n[Step 9] Best model: {best_model_name}")
    if mlflow_tracker:
        mlflow_tracker.log_params({"best_model": best_model_name})

    # End MLflow run
    if mlflow_tracker:
        mlflow_tracker.end_run()

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
    
    args = parser.parse_args()
    
    # Resolve data paths (dirs have priority over paths if both specified)
    train_path = args.train_dir or args.train_path
    val_path = args.val_dir or args.val_path
    test_path = args.test_dir or args.test_path

    # If no arguments, show help and exit
    if not (args.train or args.eda or args.infer):
        print("[INFO] No command specified. Use --help for usage instructions.")
        print("[INFO] Available commands:")
        print("  python main.py --train              # Run training pipeline")
        print("  python main.py --eda                # Run EDA only")
        print("  python main.py --infer              # Run inference demo")
        print("  python main.py --train --eda --infer  # Run all")
        print("\n[INFO] Data location: data_new_format/")
        print("       - data_new_format/train/  (training data)")
        print("       - data_new_format/val/    (validation data)")
        print("       - data_new_format/test/   (test data)")
        sys.exit(0)
    
    # Validate data paths for training
    if args.train and not all([train_path or config.TRAIN_DATA_DIR, 
                                val_path or config.VAL_DATA_DIR, 
                                test_path or config.TEST_DATA_DIR]):
        print("\n[ERROR] Training data not found!")
        print("[INFO] Please ensure data exists in data_new_format/ folder:")
        print("       - data_new_format/train/")
        print("       - data_new_format/val/")
        print("       - data_new_format/test/")
        sys.exit(1)

    # Run EDA
    if args.eda or (args.train and not train_path):
        # EDA is useful for generated data or when no training is specified
        run_eda()

    # Run training
    if args.train:
        models, preprocessor, evaluation_results = run_training_pipeline(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path
        )

    # Run inference
    if args.infer:
        run_inference_demo()

    print("\n✓ Pipeline execution complete!")


if __name__ == "__main__":
    main()
