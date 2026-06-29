"""
Main orchestrator script — 두 파이프라인 통합 진입점.

이 프로젝트에는 두 가지 파이프라인이 있으며, 모두 main.py 에서 시작합니다.

[A] legacy   : 윈도우 원시신호 기반 다중 모델 학습/평가 (model_training.py)
[B] standard : 표준 특징 테이블 기반 5클래스 고장 분류
               (build_feature_table.py + train_from_feature_table.py + feature_extraction.py)

사용법 (자세한 내용은 --help):
    # [B] 표준화 파이프라인 (권장)
    python main.py standard --all          # 특징생성 → 학습 → 추론 한번에
    python main.py standard --build        # 1) 원시 CSV → 특징 테이블
    python main.py standard --train        # 2) 특징 테이블 → 5클래스 학습
    python main.py standard --infer        # 3) 학습 모델로 실시간 추론 테스트
    python main.py standard --plot         # 4) 결과/특징 시각화(results/plots/)
    python main.py standard --fft-steps    # FFT 계산 단계별 그래프

    # [A] 레거시 파이프라인
    python main.py legacy --train          # 다중 모델 학습/평가
    python main.py legacy --eda            # EDA
    python main.py legacy --infer          # 추론 데모
"""

import argparse
import sys
import os
from pathlib import Path

# Add package/scripts to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "util"))

# config 는 가벼우므로 즉시 import (data 경로 등 공용)
from anomaly_detection import config

# 무거운 모듈(matplotlib/torch 의존)은 각 함수 내부에서 지연 import 하여
# 한 파이프라인의 의존성 문제가 다른 파이프라인을 막지 않도록 한다.


# ============================================================================
# [B] STANDARD PIPELINE — 표준 특징 테이블 기반 5클래스 분류
# ============================================================================

def standard_build(data_root=None, out_dir="feature_tables", regroup=True):
    """1) 원시 진동 CSV → 표준 특징 테이블 생성."""
    import build_feature_table as bft
    data_root = data_root or config.DATA_DIR
    print("\n" + "=" * 80)
    print("[STANDARD 1/3] 특징 테이블 생성 (build_feature_table)")
    print("=" * 80)
    return bft.build_feature_tables(
        data_root=data_root, out_dir=out_dir, regroup=regroup)


def standard_train(table_dir="feature_tables", model="both"):
    """2) 표준 특징 테이블 → 5클래스 분류(+이상탐지) 학습/평가."""
    import train_from_feature_table as tft
    print("\n" + "=" * 80)
    print("[STANDARD 2/3] 5클래스 분류 학습 (train_from_feature_table)")
    print("=" * 80)
    return tft.run_training(table_dir=table_dir, model=model)


def standard_infer(split="train", per_label=1, max_windows=3):
    """3) 학습된 5클래스 모델로 실데이터 실시간 분류 테스트."""
    import test_serial_inference as tsi
    print("\n" + "=" * 80)
    print("[STANDARD 3/3] 실시간 5클래스 추론 테스트 (test_serial_inference)")
    print("=" * 80)
    engine = tsi.test_realtime_classification_with_real_data(
        per_label_files=per_label, max_windows=max_windows, split=split)
    tsi.save_inference_results(engine)
    return engine


def standard_plot(split="test"):
    """4) 학습 결과/특징을 이미지로 시각화 (results/plots/)."""
    import visualize_feature_results as vfr
    print("\n" + "=" * 80)
    print("[STANDARD plot] 특징/모델 결과 시각화 (visualize_feature_results)")
    print("=" * 80)
    return vfr.run_visualization(split=split)


def standard_fft_steps(status="회전체불평형"):
    """FFT 특징 계산 과정을 단계별 그래프로 시각화 (results/plots/)."""
    import visualize_fft_steps as vfs
    print("\n" + "=" * 80)
    print("[STANDARD fft-steps] FFT 계산 단계별 시각화 (visualize_fft_steps)")
    print("=" * 80)
    return vfs.run(status)


def run_standard_pipeline(args):
    """[B] 표준화 파이프라인 오케스트레이션."""
    # 명시 플래그가 하나도 없으면 build/train/infer 를 기본 수행 (plot 은 제외)
    do_all = args.all or not (args.build or args.train or args.infer
                              or args.plot or args.fft_steps)
    if args.all:
        # --all 은 build/train/infer 모두 실행
        args.build = args.train = args.infer = True
        do_all = True

    if args.build or do_all:
        standard_build(data_root=args.data_root, out_dir=args.table_dir,
                       regroup=not args.no_regroup)
    if args.train or do_all:
        standard_train(table_dir=args.table_dir, model=args.model)
    if args.infer or do_all:
        try:
            standard_infer(split=args.split, per_label=args.per_label,
                           max_windows=args.max_windows)
        except Exception as e:
            print(f"[WARN] 추론 테스트 생략: {e}")
    if args.plot:
        try:
            standard_plot(split=args.split if args.split != "train" else "test")
        except Exception as e:
            print(f"[WARN] 시각화 생략: {e}")
    if args.fft_steps:
        try:
            standard_fft_steps(status=args.status)
        except Exception as e:
            print(f"[WARN] FFT 단계 시각화 생략: {e}")

    print("\n✓ [STANDARD] 파이프라인 완료!")


# ============================================================================
# [A] LEGACY PIPELINE — 윈도우 원시신호 기반 다중 모델
# ============================================================================

def run_eda(data_path: str = None):
    """Run exploratory data analysis."""
    from anomaly_detection import data_loader

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
    # 지연 import (matplotlib/torch 등 무거운 의존성)
    import numpy as np
    from anomaly_detection import (
        data_loader, preprocessing, model_training, evaluation,
    )
    from integration import MLflowTracker, FeatureStore

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
    import numpy as np
    from anomaly_detection import model_training, preprocessing, inference_serial

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


def _print_root_usage():
    """서브커맨드 없이 실행했을 때 보여줄 사용법."""
    print("""
================================================================================
  AI-Modeling-FFT  —  진동 이상탐지 / 고장분류 통합 진입점 (main.py)
================================================================================

두 가지 파이프라인을 제공합니다.

[B] standard  (권장) — 표준 특징 테이블 기반 5클래스 고장 분류
    python main.py standard --all       # 특징생성 → 학습 → 추론 한번에
    python main.py standard --build     # 1) 원시 CSV → 특징 테이블
    python main.py standard --train     # 2) 특징 테이블 → 5클래스 학습
    python main.py standard --infer     # 3) 학습 모델 실시간 추론 테스트

    옵션:
      --data-root <dir>    원시 데이터 루트 (기본: config.DATA_DIR)
      --table-dir <dir>    특징 테이블 폴더 (기본: feature_tables)
      --model both|classifier|anomaly   (기본: both)
      --split train|val|test            추론 테스트용 분할 (기본: train)
      --per-label N        라벨별 사용 파일 수 (기본: 1)
      --max-windows N      파일별 최대 윈도우 수 (기본: 3)
      --no-regroup         재분할(stratified split) 비활성화

[A] legacy — 윈도우 원시신호 기반 다중 모델 학습/평가
    python main.py legacy --train       # 다중 모델 학습/평가
    python main.py legacy --eda         # EDA
    python main.py legacy --infer       # 추론 데모

데이터 위치: data_new_format/{train,val,test}/
자세한 옵션: python main.py standard --help  /  python main.py legacy --help
================================================================================
""")


def main():
    """두 파이프라인 통합 진입점."""
    parser = argparse.ArgumentParser(
        description="진동 이상탐지/고장분류 통합 진입점 (standard | legacy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="pipeline")

    # ---- [B] standard 서브커맨드 ----
    sp = subparsers.add_parser(
        "standard", help="표준 특징 테이블 기반 5클래스 고장 분류 (권장)")
    sp.add_argument("--all", action="store_true",
                    help="build → train → infer 모두 실행")
    sp.add_argument("--build", action="store_true", help="원시 CSV → 특징 테이블")
    sp.add_argument("--train", action="store_true", help="특징 테이블 → 5클래스 학습")
    sp.add_argument("--infer", action="store_true", help="학습 모델 실시간 추론 테스트")
    sp.add_argument("--plot", action="store_true",
                    help="학습 결과/특징을 이미지로 시각화 (results/plots/)")
    sp.add_argument("--fft-steps", action="store_true", dest="fft_steps",
                    help="FFT 특징 계산 과정을 단계별 그래프로 시각화")
    sp.add_argument("--status", type=str, default="회전체불평형",
                    help="--fft-steps 대상 고장 상태(한글, 기본: 회전체불평형)")
    sp.add_argument("--data-root", type=str, default=None,
                    help="원시 데이터 루트 (기본: config.DATA_DIR)")
    sp.add_argument("--table-dir", type=str, default="feature_tables",
                    help="특징 테이블 폴더 (기본: feature_tables)")
    sp.add_argument("--model", type=str, default="both",
                    choices=["both", "classifier", "anomaly"],
                    help="학습 대상 (기본: both)")
    sp.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "test"],
                    help="추론 테스트용 분할 (기본: train)")
    sp.add_argument("--per-label", type=int, default=1,
                    help="라벨별 사용 파일 수 (기본: 1)")
    sp.add_argument("--max-windows", type=int, default=3,
                    help="파일별 최대 윈도우 수 (기본: 3)")
    sp.add_argument("--no-regroup", action="store_true",
                    help="재분할(stratified split) 비활성화")

    # ---- [A] legacy 서브커맨드 ----
    lp = subparsers.add_parser(
        "legacy", help="윈도우 원시신호 기반 다중 모델 학습/평가")
    lp.add_argument("--train", action="store_true", help="학습 파이프라인 실행")
    lp.add_argument("--eda", action="store_true", help="EDA 만 실행")
    lp.add_argument("--infer", action="store_true", help="추론 데모 실행")
    lp.add_argument("--train-path", type=str, default=None,
                    help="학습 데이터 경로 (파일 또는 폴더)")
    lp.add_argument("--val-path", type=str, default=None,
                    help="검증 데이터 경로 (파일 또는 폴더)")
    lp.add_argument("--test-path", type=str, default=None,
                    help="테스트 데이터 경로 (파일 또는 폴더)")
    lp.add_argument("--train-dir", type=str, default=None,
                    help="학습 CSV 폴더 (--train-path 대체)")
    lp.add_argument("--val-dir", type=str, default=None,
                    help="검증 CSV 폴더 (--val-path 대체)")
    lp.add_argument("--test-dir", type=str, default=None,
                    help="테스트 CSV 폴더 (--test-path 대체)")

    args = parser.parse_args()

    if args.pipeline is None:
        _print_root_usage()
        sys.exit(0)

    if args.pipeline == "standard":
        run_standard_pipeline(args)
        return

    # ---- legacy 라우팅 ----
    train_path = args.train_dir or args.train_path
    val_path = args.val_dir or args.val_path
    test_path = args.test_dir or args.test_path

    if not (args.train or args.eda or args.infer):
        print("[INFO] legacy: 명령이 없습니다. 사용 가능한 옵션:")
        print("  python main.py legacy --train   # 학습 파이프라인")
        print("  python main.py legacy --eda     # EDA")
        print("  python main.py legacy --infer   # 추론 데모")
        sys.exit(0)

    if args.train and not all([train_path or config.TRAIN_DATA_DIR,
                               val_path or config.VAL_DATA_DIR,
                               test_path or config.TEST_DATA_DIR]):
        print("\n[ERROR] 학습 데이터를 찾을 수 없습니다!")
        print("[INFO] data_new_format/ 폴더를 확인하세요:")
        print("       - data_new_format/train/")
        print("       - data_new_format/val/")
        print("       - data_new_format/test/")
        sys.exit(1)

    if args.eda or (args.train and not train_path):
        run_eda()

    if args.train:
        run_training_pipeline(
            train_path=train_path, val_path=val_path, test_path=test_path)

    if args.infer:
        run_inference_demo()

    print("\n✓ [LEGACY] 파이프라인 완료!")


if __name__ == "__main__":
    main()
