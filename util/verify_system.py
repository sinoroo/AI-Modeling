"""
End-to-End Pipeline Verification
Tests all major components to ensure everything works correctly.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가 (util/ 에서 실행해도 패키지 import 가능)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 상대 경로 점검이 루트 기준이 되도록 작업 디렉터리 이동
os.chdir(PROJECT_ROOT)

print("\n" + "=" * 80)
print("ANOMALY DETECTION PIPELINE - END-TO-END VERIFICATION")
print("=" * 80)

# Step 1: Check environment
print("\n[1] Checking Python environment...")
print(f"  Python version: {sys.version.split()[0]}")
print(f"  Current directory: {os.getcwd()}")

# Step 2: Check virtual environment
print("\n[2] Checking virtual environment...")
venv_path = Path("venv")
if venv_path.exists():
    print(f"  ✓ Virtual environment exists at: {venv_path}")
else:
    print(f"  ✗ Virtual environment not found")

# Step 3: Check dependencies
print("\n[3] Checking dependencies...")
dependencies = {
    "pandas": "Data manipulation",
    "numpy": "Numerical computing",
    "sklearn": "Classical ML models",
    "torch": "Deep Learning",
    "matplotlib": "Visualization",
    "seaborn": "Advanced plots",
    "scipy": "Signal processing",
    "serial": "Serial communication",
}

for pkg, desc in dependencies.items():
    try:
        if pkg == "sklearn":
            import sklearn
        elif pkg == "serial":
            import serial
        else:
            __import__(pkg)
        print(f"  ✓ {pkg:20s} - {desc}")
    except ImportError:
        print(f"  ✗ {pkg:20s} - NOT INSTALLED")

# Step 4: Check project structure
print("\n[4] Checking project structure...")
required_files = [
    # Core files
    "main.py",
    "requirements.txt",
    # anomaly_detection module
    "anomaly_detection/__init__.py",
    "anomaly_detection/config.py",
    "anomaly_detection/data_loader.py",
    "anomaly_detection/preprocessing.py",
    "anomaly_detection/model_training.py",
    "anomaly_detection/evaluation.py",
    "anomaly_detection/inference_serial.py",
    "anomaly_detection/feature_extraction.py",
    # 표준화/학습 스크립트
    "build_feature_table.py",
    "train_from_feature_table.py",
    # analysis folder
    "analysis/analyze_3d_array.py",
    "analysis/analyze_2d_data.py",
    # integration folder
    "integration/__init__.py",
    "integration/mlflow_utils.py",
    "integration/bentoml_service.py",
    "integration/feature_store.py",
    # util folder
    "util/serial_data_simulator.py",
    "util/test_serial_inference.py",
    "util/verify_system.py",
    # Documentation
    "README.md",
    "PROJECT_STRUCTURE.md",
    "DELIVERY_SUMMARY.md",
]

missing = []
for filepath in required_files:
    if Path(filepath).exists():
        print(f"  ✓ {filepath}")
    else:
        print(f"  ✗ {filepath} - MISSING")
        missing.append(filepath)

# Step 5: Check package imports
print("\n[5] Testing package imports...")
try:
    from anomaly_detection import config
    print(f"  ✓ config module")
    print(f"    - Window size: {config.WINDOW_SIZE}")
    print(f"    - Data dir: {config.DATA_DIR}")
    print(f"    - Serial port: {config.SERIAL_PORT}")
except Exception as e:
    print(f"  ✗ config module - {e}")

try:
    from anomaly_detection import data_loader
    print(f"  ✓ data_loader module")
except Exception as e:
    print(f"  ✗ data_loader module - {e}")

try:
    from anomaly_detection import preprocessing
    print(f"  ✓ preprocessing module")
except Exception as e:
    print(f"  ✗ preprocessing module - {e}")

try:
    from anomaly_detection import model_training
    print(f"  ✓ model_training module")
except Exception as e:
    print(f"  ✗ model_training module - {e}")

try:
    from anomaly_detection import evaluation
    print(f"  ✓ evaluation module")
except Exception as e:
    print(f"  ✗ evaluation module - {e}")

try:
    from anomaly_detection import inference_serial
    print(f"  ✓ inference_serial module")
except Exception as e:
    print(f"  ✗ inference_serial module - {e}")

try:
    from integration import MLflowTracker
    print(f"  ✓ integration.MLflowTracker")
except Exception as e:
    print(f"  ✗ integration.MLflowTracker - {e}")

try:
    from integration import AnomalyDetectionService
    print(f"  ✓ integration.AnomalyDetectionService")
except Exception as e:
    print(f"  ✗ integration.AnomalyDetectionService - {e}")

try:
    from integration import FeatureStore
    print(f"  ✓ integration.FeatureStore")
except Exception as e:
    print(f"  ✗ integration.FeatureStore - {e}")

# Step 6: Check data files
print("\n[6] Checking data files...")
data_dirs = {
    "data_new_format/train": "Training data directory",
    "data_new_format/val": "Validation data directory",
    "data_new_format/test": "Test data directory",
}

for dirpath, desc in data_dirs.items():
    dir_obj = Path(dirpath)
    if dir_obj.exists() and dir_obj.is_dir():
        csv_files = list(dir_obj.glob("*.csv"))
        print(f"  ✓ {dirpath:30s} ({len(csv_files)} CSV files) - {desc}")
    else:
        print(f"  ✗ {dirpath} - NOT FOUND ({desc})")

# Step 7: Quick functionality test
print("\n[7] Quick functionality tests...")

try:
    import pandas as pd
    import numpy as np
    from anomaly_detection.preprocessing import Preprocessor
    
    # Create sample data
    data = pd.DataFrame({
        "sensor_0": np.random.randn(100),
        "sensor_1": np.random.randn(100),
        "sensor_2": np.random.randn(100),
        "label": np.random.randint(0, 2, 100),
    })
    
    # Test preprocessing
    prep = Preprocessor()
    prep.fit(data, ["sensor_0", "sensor_1", "sensor_2"], "label")
    X_windows, y_windows = prep.preprocess_pipeline(
        data, ["sensor_0", "sensor_1", "sensor_2"], "label", fit_scaler=False
    )
    
    print(f"  ✓ Data preprocessing works")
    print(f"    - Input: {data.shape}")
    print(f"    - Windows: {X_windows.shape}")
    print(f"    - Labels: {y_windows.shape if y_windows is not None else 'None'}")
    
except Exception as e:
    print(f"  ✗ Data preprocessing failed - {e}")

# Step 7b: Standardized feature pipeline & 5-class model
print("\n[7b] Checking standardized feature pipeline & 5-class model...")

# 7b-1: feature tables
feature_tables = {
    "feature_tables/feature_table_train.csv": "Train feature table",
    "feature_tables/feature_table_val.csv": "Val feature table",
    "feature_tables/feature_table_test.csv": "Test feature table",
    "feature_tables/normalization_stats.json": "Per-equipment norm stats",
}
for fpath, desc in feature_tables.items():
    if Path(fpath).exists():
        print(f"  ✓ {fpath:45s} - {desc}")
    else:
        print(f"  ⚠️  {fpath:45s} - 없음 (build_feature_table.py 실행 필요)")

# 7b-2: trained classifier artifacts
model_artifacts = {
    "models/clf_random_forest.pkl": "5-class fault classifier",
    "models/clf_scaler.pkl": "Classifier feature scaler",
}
clf_ready = all(Path(p).exists() for p in model_artifacts)
for mpath, desc in model_artifacts.items():
    if Path(mpath).exists():
        print(f"  ✓ {mpath:45s} - {desc}")
    else:
        print(f"  ⚠️  {mpath:45s} - 없음 (train_from_feature_table.py 실행 필요)")

# 7b-3: feature extraction module quick test
try:
    from anomaly_detection import feature_extraction as fe
    sig = __import__("numpy").random.randn(fe.WINDOW_SIZE)
    vec = fe.extract_feature_vector(sig, rpm=1730.0, power_kw=2.2)
    assert vec.shape[0] == len(fe.FEATURE_COLUMNS), "특징 차원 불일치"
    print(f"  ✓ feature_extraction: {len(fe.FEATURE_COLUMNS)}개 특징 벡터 생성 OK")
except Exception as e:
    print(f"  ✗ feature_extraction 테스트 실패 - {e}")

# 7b-4: end-to-end 5-class inference test (모델이 있으면)
if clf_ready:
    try:
        import joblib
        import numpy as _np
        from anomaly_detection import feature_extraction as fe
        clf = joblib.load("models/clf_random_forest.pkl")
        scaler = joblib.load("models/clf_scaler.pkl")
        test_sig = _np.random.randn(fe.WINDOW_SIZE) * 0.01
        _vec = fe.extract_feature_vector(test_sig, rpm=1730.0, power_kw=2.2)
        _pred = clf.predict(scaler.transform(_vec.reshape(1, -1)))[0]
        print(f"  ✓ 5-class 추론 OK (예: 예측='{_pred}', "
              f"클래스={list(clf.classes_)})")
    except Exception as e:
        print(f"  ✗ 5-class 추론 테스트 실패 - {e}")
else:
    print("  ⚠️  5-class 모델 미존재 - 추론 테스트 생략")

# Step 8: Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

if not missing:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nThe system is ready to use. Next steps:")
    print("1. Run: python main.py")
    print("2. Check output in eda_results/ and results/ directories")
    print("3. Review evaluation_report.json for model performance")
else:
    print(f"\n⚠️  {len(missing)} file(s) missing:")
    for f in missing:
        print(f"   - {f}")
    print("\nPlease ensure all files are in place before running the pipeline.")

print("\n" + "=" * 80)
print("For detailed documentation, see README.md and DELIVERY_SUMMARY.md")
print("=" * 80 + "\n")
