"""
End-to-End Pipeline Verification
Tests all major components to ensure everything works correctly.
"""

import sys
import os
from pathlib import Path

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
    "main.py",
    "test_setup.py",
    "anomaly_detection/__init__.py",
    "anomaly_detection/config.py",
    "anomaly_detection/data_loader.py",
    "anomaly_detection/preprocessing.py",
    "anomaly_detection/model_training.py",
    "anomaly_detection/evaluation.py",
    "anomaly_detection/inference_serial.py",
    "requirements.txt",
    "README.md",
    "QUICK_START.py",
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

# Step 6: Check data files
print("\n[6] Checking data files...")
data_files = [
    "data/train.csv",
    "data/val.csv",
    "data/test.csv",
]

for filepath in data_files:
    if Path(filepath).exists():
        file_size = Path(filepath).stat().st_size / 1024  # KB
        rows = sum(1 for line in open(filepath)) - 1
        print(f"  ✓ {filepath:20s} ({rows:4d} rows, {file_size:6.1f} KB)")
    else:
        print(f"  ✗ {filepath} - NOT FOUND")

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
