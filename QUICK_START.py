#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║         ANOMALY DETECTION PIPELINE - QUICK START GUIDE v2.0               ║
║                    with Real-Time Serial Inference (NEW!)                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Interactive guide to get started quickly with the anomaly detection system.
Supports both batch training AND real-time inference!

Location: c:\\workspace\\python\\AI-Modeling
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Pretty print section header"""
    print(f"\n{'═' * 80}")
    print(f"  {title}")
    print(f"{'═' * 80}\n")

def main():
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║         ANOMALY DETECTION PIPELINE - QUICK START GUIDE v2.0               ║
║                  Real-Time Serial Inference Enabled                       ║
╚════════════════════════════════════════════════════════════════════════════╝

Choose what you want to do:

  [1] 🟢 NEW! Run Real-Time Inference Test (No model needed)
  [2] 🟠 Generate Synthetic Sensor Data  
  [3] 📊 Analyze 3D Array Structure
  [4] 📈 Analyze 2D Classical ML Data
  [5] 🟦 Run Full Training Pipeline
  [6] 📚 View Documentation

  [0] Exit

""")
    
    choice = input("Enter your choice (0-6): ").strip()
    
    if choice == '0':
        print("\n✅ Goodbye!")
        return
    
    elif choice == '1':
        print_header("🔴 REAL-TIME SERIAL INFERENCE TEST (NEW!)")
        print("""
This runs real-time inference on simulated sensor data:
  - Generates synthetic vibration signals (normal + anomalies)
  - Streams them through a real-time inference pipeline
  - Produces predictions with confidence scores
  - Saves results as JSON and visualization

Output files:
  ✓ inference_results.json (6,249 predictions)
  ✓ realtime_inference_results.png (3-plot visualization)
  ✓ synthetic_test_data.csv (160K samples for retraining)

Processing:
  - Windows processed: 6,249
  - Processing time: ~30 seconds
  - Throughput: ~208 windows/sec

""")
        input("Press Enter to start...")
        print("\n⏳ Running real-time inference test...\n")
        result = subprocess.run([sys.executable, "test_serial_inference.py"], 
                              cwd=os.getcwd())
        if result.returncode == 0:
            print("\n✅ Real-time inference test completed successfully!")
        else:
            print("\n❌ Test failed. Check error messages above.")
    
    elif choice == '2':
        print_header("🟠 GENERATE SYNTHETIC SENSOR DATA")
        print("""
This generates realistic synthetic sensor data with multiple anomaly types:
  
Signals Generated:
  ✓ Normal: Standard oscillation (20Hz + 50Hz)
  ✓ Spike: Bearing impact anomalies
  ✓ Harmonic: High-frequency damage signature (200Hz)
  ✓ Drift: Gradual degradation signal
  ✓ Discontinuity: Sudden fault detection

Output:
  - synthetic_test_data.csv (160,000 samples)
  - Signal statistics displayed in console
  - 50% normal, 50% anomaly samples

""")
        input("Press Enter to start...")
        print("\n⏳ Generating synthetic data...\n")
        result = subprocess.run([sys.executable, "serial_data_simulator.py"],
                              cwd=os.getcwd())
        if result.returncode == 0:
            print("\n✅ Synthetic data generated successfully!")
        else:
            print("\n❌ Generation failed.")
    
    elif choice == '3':
        print_header("📊 ANALYZE 3D ARRAY STRUCTURE")
        print("""
Analyzes the 3D array structure used in deep learning models:
  
3D Array Format:
  - Shape: (154, 64, 6)
  - 154 time windows
  - 64 samples per window
  - 6 features: vibration + RMS, Peak, Crest Factor, Kurtosis, Skewness

Output:
  ✓ 3D array statistics and distribution
  ✓ Feature visualization
  ✓ Temporal pattern analysis
  ✓ Boxplots for anomaly detection

Use: For understanding deep learning model inputs (Autoencoder, LSTM)

""")
        input("Press Enter to start...")
        print("\n⏳ Analyzing 3D array...\n")
        result = subprocess.run([sys.executable, "analyze_3d_array.py"],
                              cwd=os.getcwd())
        if result.returncode == 0:
            print("\n✅ 3D array analysis completed!")
        else:
            print("\n❌ Analysis failed.")
    
    elif choice == '4':
        print_header("📈 ANALYZE 2D CLASSICAL ML DATA")
        print("""
Analyzes the 2D array structure used in classical ML models:

2D Array Format:
  - Shape: (29, 384)
  - 29 time windows
  - 384 flattened features (64 samples × 6 features)

Analysis includes:
  ✓ Feature statistics and distribution
  ✓ PCA dimensionality reduction
  ✓ Feature correlation analysis
  ✓ Scatter plots and heatmaps
  ✓ Model training demo

Key Insight:
  - PCA can compress 384 → 25 dimensions (95% variance retained)

Output:
  ✓ ml_2d_analysis_visualization.png (6-subplot figure)
  ✓ Statistical summaries

Use: For understanding classical ML model inputs (RandomForest, SVM, etc.)

""")
        input("Press Enter to start...")
        print("\n⏳ Analyzing 2D data...\n")
        result = subprocess.run([sys.executable, "analyze_2d_data.py"],
                              cwd=os.getcwd())
        if result.returncode == 0:
            print("\n✅ 2D data analysis completed!")
        else:
            print("\n❌ Analysis failed.")
    
    elif choice == '5':
        print_header("🟦 RUN FULL TRAINING PIPELINE")
        print("""
Complete end-to-end machine learning pipeline:

Steps:
  1. Load training data (train.csv)
  2. Perform EDA (visualizations)
  3. Preprocess data (normalize, window, feature engineering)
  4. Train classical ML models:
     - Random Forest Classifier
     - Isolation Forest
     - One-Class SVM
  5. Train deep learning models:
     - Autoencoder
     - LSTM
  6. Evaluate all models
  7. Generate comparison report
  8. Save trained models

Requirements:
  - Training data: data/train.csv
  - Validation data: data/val.csv (optional)
  - Test data: data/test.csv (optional)

Output:
  ✓ Trained models in models/
  ✓ Evaluation report (evaluation_report.json)
  ✓ Visualizations in results/
  ✓ EDA plots in eda_results/

Expected Time: 2-5 minutes

""")
        input("Press Enter to start...")
        print("\n⏳ Running full training pipeline...\n")
        result = subprocess.run([sys.executable, "main.py"],
                              cwd=os.getcwd())
        if result.returncode == 0:
            print("\n✅ Training pipeline completed!")
        else:
            print("\n❌ Pipeline failed.")
    
    elif choice == '6':
        print_header("📚 DOCUMENTATION")
        print("""
Available Documentation Files:

  📖 README.md (Start here!)
     - Project overview
     - Feature summary
     - Installation instructions
     - Quick reference

  🔴 REALTIME_INFERENCE.md
     - Complete real-time inference guide
     - Architecture & data flow
     - Implementation examples
     - Production deployment
     - Troubleshooting

  🔷 3D_ARRAY_EXPLAINED.md
     - 3D array structure
     - Feature descriptions
     - Practical examples
     - Time resolution analysis

  🔷 3D_ARRAY_COMPLETE_GUIDE.md
     - Complete guide with examples
     - Statistical features
     - Use cases

  🟢 FEATURES_EXPLANATION.md
     - Feature descriptions
     - Feature importance ranking
     - Use cases for each feature

  📊 MODEL_IO_FORMAT.md
     - Input/output specifications
     - Data preprocessing details
     - Vector dimensions

═══════════════════════════════════════════════════════════════════════════════

Project Structure:

anomaly_detection/
  ├── config.py            Configuration & hyperparameters
  ├── preprocessing.py     Data preprocessing & feature engineering
  ├── model_training.py    Model training orchestrator
  ├── evaluation.py        Metrics & model comparison
  ├── inference_serial.py  Real-time serial inference
  └── data_loader.py       Data loading & EDA

Scripts:
  ├── main.py              Full training pipeline
  ├── test_serial_inference.py    Real-time inference test (NEW!)
  ├── serial_data_simulator.py    Synthetic data generation (NEW!)
  ├── analyze_3d_array.py         3D array analysis
  ├── analyze_2d_data.py          2D array analysis
  └── QUICK_START.py       This file

═══════════════════════════════════════════════════════════════════════════════

Quick Command Reference:

# Real-time inference (no model training needed)
python test_serial_inference.py

# Generate synthetic test data
python serial_data_simulator.py

# Analyze 3D array structure
python analyze_3d_array.py

# Analyze 2D array structure
python analyze_2d_data.py

# Full training pipeline
python main.py

═══════════════════════════════════════════════════════════════════════════════

Next Steps:

1. ✅ Read README.md for project overview
2. ✅ Run real-time inference test: python test_serial_inference.py
3. ✅ Check inference_results.json for output format
4. ✅ Review documentation for detailed information
5. ✅ Train models if you have custom data

═══════════════════════════════════════════════════════════════════════════════
""")
    
    else:
        print("❌ Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
