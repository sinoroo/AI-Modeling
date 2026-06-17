# Anomaly Detection Pipeline for Pump/Motor Systems

A production-ready, end-to-end anomaly detection system for industrial pump and motor monitoring. This system combines classical ML and deep learning approaches with **real-time serial inference** for live edge deployment.

## ✨ Key Features

- **Comprehensive Pipeline**: CSV → 3D Array → 2D Classical ML / 3D Deep Learning
- **Multiple Model Approaches**: Random Forest, Isolation Forest, One-Class SVM (classical ML); Autoencoders, LSTM (deep learning)
- **🔴 Real-Time Serial Inference**: Stream sensor data from serial ports for live anomaly detection
- **🟢 Synthetic Data Generation**: Generate realistic test signals (normal, spike, harmonic, drift, discontinuity)
- **Automatic Preprocessing**: Windowing, feature engineering (RMS, Peak, Crest, Kurtosis, Skewness)
- **Model Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices
- **Extensible Architecture**: Modular design for easy extension

## 🚀 Quick Start

### 1️⃣ Training Pipeline
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run full training with MLflow tracking
python main.py --train

# View MLflow results
mlflow ui --backend-store-uri file:./integration/mlruns
```

### 2️⃣ Data Analysis
```bash
# Analyze 2D classical ML features
python analysis/analyze_2d_data.py

# Analyze 3D deep learning features
python analysis/analyze_3d_array.py
```

### 3️⃣ Real-Time Inference (Utility)
```bash
# Generate synthetic sensor data
python util/serial_data_simulator.py

# Run real-time inference test
python util/test_serial_inference.py

# Verify system setup
python util/verify_system.py
```

## Project Structure

```
AI-Modeling/
├── 📌 Core Pipeline
│   ├── main.py                            # Full training pipeline orchestrator
│   ├── requirements.txt                   # Python dependencies
│   └── README.md                          # This file
│
├── 📦 Core Modules (anomaly_detection/)
│   ├── config.py                          # Configuration (data_new_format based)
│   ├── data_loader.py                     # Data I/O (new format only)
│   ├── preprocessing.py                   # Feature engineering pipeline
│   ├── model_training.py                  # Model training orchestrator
│   ├── evaluation.py                      # Metrics & validation
│   └── inference_serial.py                # Real-time serial inference
│
├── 📊 Data Analysis (analysis/)
│   ├── analyze_3d_array.py               # 3D array structure analysis
│   ├── analyze_2d_data.py                # 2D classical ML feature analysis
│   ├── ml_input_examples.py              # Input format examples
│   ├── MODEL_IO_QUICK_REF.py             # Model I/O quick reference
│   └── MODEL_IO_VISUALIZATION.md         # Model visualization guide
│
├── 🔧 Integration Modules (integration/)
│   ├── mlflow_utils.py                   # MLflow tracking integration
│   ├── bentoml_service.py                # BentoML REST API service
│   ├── feature_store.py                  # Feature schema management
│   ├── mlflow_bentoml_example.py         # Integration example
│   ├── bentofile.yaml                    # BentoML configuration
│   ├── mlflow.db                         # MLflow database
│   ├── mlruns/                           # MLflow experiment logs
│   ├── feature_store/                    # Feature schema & statistics
│   │
│   ├── 📖 Integration Guides
│   ├── INTEGRATION_SUMMARY.md            # Integration overview
│   ├── MLFLOW_BENTOML_GUIDE.md           # Detailed guide
│   ├── QUICK_START_MLFLOW_BENTOML.md     # 5-minute quickstart
│   └── MLFLOW_FILESTORE_MIGRATION.md     # SQLite migration
│
├── 🛠️ Utilities (util/)
│   ├── serial_data_simulator.py          # Generate synthetic vibration signals
│   ├── test_serial_inference.py          # Real-time processing test
│   ├── verify_system.py                  # System verification tool
│   ├── migrate_mlflow.py                 # MLflow migration utility
│   ├── QUICK_START.py                    # Interactive getting-started
│   ├── REALTIME_INFERENCE.md             # Detailed serial inference guide
│   ├── realtime_inference_results.png    # Inference visualization
│   └── synthetic_test_data.csv           # Sample test data
│
├── 📁 Data Storage
│   ├── data_new_format/                  # Main data format (REQUIRED)
│   │   ├── train/                        # Training CSV files
│   │   ├── val/                          # Validation CSV files
│   │   └── test/                         # Test CSV files
│   ├── data/                             # Empty (legacy - deprecated)
│   ├── models/                           # Trained models (.pkl, .pt)
│   ├── results/                          # Model evaluation outputs
│   ├── logs/                             # Training logs
│   └── eda_results/                      # EDA visualizations
│
├── 📚 Documentation
│   ├── PROJECT_STRUCTURE.md              # Detailed structure guide
│   ├── DELIVERY_SUMMARY.md               # Project delivery summary
│   ├── FEATURES_EXPLANATION.md           # Feature descriptions
│   ├── MODEL_IO_FORMAT.md                # Model I/O specification
│   ├── 3D_ARRAY_EXPLAINED.md             # 3D array details
│   └── 3D_ARRAY_COMPLETE_GUIDE.md        # 3D array usage guide
```

## Key Changes from Previous Version

- ✅ **New data format only**: `data_new_format/` is now the standard (9 lines metadata + data)
- ✅ **Organized structure**: 
  - `analysis/` for data analysis tools
  - `util/` for utilities and testing
  - `integration/` for MLflow/BentoML modules
- ✅ **Removed deprecated files**: `data_loader_old.py`, legacy CSV files
- ✅ **Updated imports**: All imports reference organized folders

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Create and activate virtual environment:**
   ```bash
   cd c:\workspace\python\AI-Modeling
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # On Windows
   source venv/bin/activate     # On Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Data

Place your CSV data files in `data_new_format/` directory:

```bash
data_new_format/
├── train/
│   ├── train_normal_000.csv
│   ├── train_anomaly_000.csv
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Or generate sample data:

```bash
python util/serial_data_simulator.py  # Creates synthetic test data
```

### 2. Run Full Training Pipeline (with MLflow)

```bash
python main.py --train
```

This executes all steps:
- Loads data from `data_new_format/`
- Parses metadata and sensor data
- Performs EDA with visualizations
- Preprocesses data (normalize, window, feature engineering)
- Trains all models with MLflow tracking
- Saves trained models
- Evaluates on test set
- Generates comparison report

### 3. Run EDA Only

```bash
python main.py --eda
```

Generates statistical summaries and visualizations:
- Feature distributions
- Correlation matrices
- Outlier detection (box plots)
- Time-series patterns
- Label distribution

### 4. Run Inference Demo

```bash
python main.py --infer
```

Demonstrates real-time anomaly detection using trained models.

### 5. Run Complete Pipeline

```bash
python main.py
```

Runs training and inference in sequence.

### 6. Monitor MLflow Experiments

```bash
# Start MLflow UI
mlflow ui

# Or with explicit database path
mlflow ui --backend-store-uri sqlite:///integration/mlflow.db

# Access at: http://localhost:5000
```

### 7. Deploy with BentoML

```bash
# Setup service
python integration/mlflow_bentoml_example.py --setup

# Start REST API server
bentoml serve anomaly_detection_service:latest

# API endpoint: http://localhost:3000
```

### 8. Migrate MLflow Backend (if needed)

```bash
# Automatic migration with validation
python util/migrate_mlflow.py

# Or manual migration
mlflow migrate-filestore -c sqlite:///integration/mlflow.db ./integration/mlruns
```

## Configuration

Edit `anomaly_detection/config.py` to customize:

- **Data paths**: `TRAIN_DATA_DIR`, `VAL_DATA_DIR`, `TEST_DATA_DIR` (directory paths with multiple CSV files)
- **CSV Format**: `USE_NEW_CSV_FORMAT = True` (metadata header + data)
- **Preprocessing**: Window size, normalization method, missing value handling
- **Models**: Hyperparameters for Random Forest, Autoencoder, LSTM, etc.
- **Evaluation**: Metrics to compute
- **Serial communication**: Port (`SERIAL_PORT`), baud rate (`BAUD_RATE`)
- **MLflow**: Backend store (SQLite), experiment naming
- **BentoML**: Service configuration

### Environment Variables

```bash
# Configure data directory
set TRAIN_DATA_DIR=c:\my_data\train
set VAL_DATA_DIR=c:\my_data\val
set TEST_DATA_DIR=c:\my_data\test

# Configure serial port
set SERIAL_PORT=COM3
set BAUD_RATE=9600

# MLflow SQLite backend
set MLFLOW_BACKEND_STORE_URI=sqlite:///integration/mlflow.db

# Debug mode
set DEBUG_MODE=True
```

## Data Format

### New CSV Format (Metadata + Data)

The project uses a new CSV format with metadata header lines followed by sensor data:

**File Structure:**
```
Line 1:  Date,2024-01-15 10:30:45
Line 2:  Filename,STFMK-ACHU-01L_2024-01-15_103045.csv
Line 3:  Data Label,정상
Line 4:  Label_No,00
Line 5:  Motor Spec,L-CAHU-01R,50Hz,2.2kW
Line 6:  Period,3SEC
Line 7:  Sample Rate,4000
Line 8:  RMS,0.010346
Line 9:  Data Length,12000
Line 10: (blank)
Line 11: time,vibration
Line 12: 0,45.2
Line 13: 1,46.1
...
```

**Configuration:**
```python
# In anomaly_detection/config.py
USE_NEW_CSV_FORMAT = True  # Enable new format with metadata (lines 0-8)
METADATA_MAPPING = {
    "Date": 0,           # Line 1: Timestamp
    "Filename": 1,       # Line 2: File identifier
    "Data Label": 2,     # Line 3: Condition label (정상, 축정렬불량, etc)
    "Label_No": 3,       # Line 4: Numeric label
    "Motor Spec": 4,     # Line 5: Equipment specification
    "Period": 5,         # Line 6: Measurement period
    "Sample Rate": 6,    # Line 7: Sampling rate (Hz)
    "RMS": 7,            # Line 8: RMS value
    "Data Length": 8,    # Line 9: Total samples
}
```

**Directory Structure:**
```
data_new_format/
├── train/
│   ├── train_normal_000.csv      # Training normal data
│   ├── train_anomaly_000.csv     # Training anomaly data
│   └── ...
├── val/
│   ├── val_normal_000.csv        # Validation normal data
│   └── ...
└── test/
    ├── test_normal_000.csv       # Test normal data
    └── ...
```

**Data Preprocessing:**
- Metadata is automatically parsed and logged
- Sensor data is extracted from line 11 onwards
- Windowing: Raw data → 3D Array (windows × samples × features)
- Feature Engineering: RMS, Peak, Crest Factor, Kurtosis, Skewness
- Normalization: StandardScaler, MinMaxScaler, or RobustScaler

## Models & Algorithms

### Classical ML Models

1. **Random Forest Classifier**
   - Ensemble method with multiple decision trees
   - Good for labeled data with mixed feature types
   - Fast inference
   - Hyperparameters: `n_estimators=100`, `max_depth=15`

2. **Isolation Forest**
   - Unsupervised anomaly detection
   - Isolates anomalies without computing distances
   - No label required
   - Hyperparameters: `n_estimators=100`, `contamination=0.1`

3. **One-Class SVM**
   - Unsupervised anomaly detection
   - Maps data to high-dimensional space
   - No label required
   - Hyperparameters: `kernel='rbf'`, `nu=0.05`

### Deep Learning Models

1. **Autoencoder**
   - Unsupervised representation learning
   - Reconstructs normal patterns; high error = anomaly
   - Architecture: Input → 128 → 64 → 32 → 64 → 128 → Output
   - Anomaly detection via reconstruction error

2. **LSTM (Long Short-Term Memory)**
   - Captures temporal dependencies
   - Predicts next timestep; deviation = anomaly
   - 2-layer LSTM with 64 hidden units
   - Sequences multiple sensor readings

## Output

### EDA Results

Located in `eda_results/`:
- `01_feature_distributions.png`: Histograms of each sensor
- `02_correlation_matrix.png`: Feature correlation heatmap
- `03_box_plots.png`: Outlier detection
- `04_time_series.png`: Time-series visualization
- `05_label_distribution.png`: Class frequency

### Trained Models

Located in `models/`:
- `.pkl` files: Classical ML models (sklearn)
- `.pt` files: Deep learning models (PyTorch)
- `scaler.pkl`: Data normalization scaler
- `preprocessor.pkl`: Full preprocessing pipeline

### Evaluation Report

`evaluation_report.json`:
```json
{
  "timestamp": "2024-06-10T15:30:45.123456",
  "models": {
    "RandomForest": {
      "metrics": {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.97,
        "f1": 0.95,
        "roc_auc": 0.98
      }
    },
    ...
  }
}
```

Confusion matrices and ROC curves saved as PNG files.

## Real-Time Inference

### Serial Communication Setup

1. **Connect serial device** (e.g., Arduino, sensor gateway):
   ```python
   from anomaly_detection.inference_serial import RealTimeInferenceSystem
   
   system = RealTimeInferenceSystem(
       model_path="models/random_forest_model.pkl",
       preprocessor_path="models/preprocessor.pkl",
       serial_port="COM3",
       baudrate=9600
   )
   system.run(duration=3600)  # Run for 1 hour
   ```

2. **Send sensor data** from device:
   - Format: Comma-separated values
   - Same order as training data
   - One sample per line, terminated with newline

3. **Receive predictions**:
   - Console output with anomaly status
   - Anomaly scores and types
   - Real-time statistics

### Example Serial Communication

**Device sends:**
```
45.2,98.3,102.1,52.3,88.4
46.1,99.1,103.2,51.8,87.9
...
```

**System outputs:**
```
✓ NORMAL | Score: 0.12 | Type: Normal
✓ NORMAL | Score: 0.15 | Type: Normal
⚠️ ANOMALY | Score: 0.87 | Type: High Vibration
```

## Performance Metrics

All models are evaluated on the test set using:

- **Accuracy**: Proportion of correct predictions
- **Precision**: Correctly predicted anomalies / All predicted anomalies
- **Recall**: Correctly predicted anomalies / All actual anomalies
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: True/False positives and negatives

## Extensibility

### Adding New Models

1. Create model class in `model_training.py`
2. Add training function to `ClassicalModelTrainer` or `DeepLearningTrainer`
3. Register in `train_all_models()` function
4. Update evaluation pipeline

### Adding New Features

1. Extend `Preprocessor.create_statistical_features()` or add new methods
2. Configure feature flags in `config.py`
3. Update windowing logic if needed

### Adding New Anomaly Types

1. Define new anomaly types in `config.ANOMALY_TYPES`
2. Implement classification logic in `inference_serial.AnomalyDetectionInference._classify_anomaly()`
3. Train model with labeled anomaly types

## Troubleshooting

### Issue: Serial Connection Failed
- Check port name (COM1-COM10 on Windows)
- Verify baud rate matches device configuration
- Ensure proper USB drivers installed
- Use COM port monitor to verify connectivity

### Issue: Low Model Accuracy
- Check data quality (missing values, outliers)
- Verify label correctness
- Increase training data size
- Tune hyperparameters
- Try different model architectures

### Issue: Memory Error with Large Datasets
- Reduce batch size in `config.py`
- Process data in chunks
- Use preprocessing to reduce dimensionality

## Best Practices

1. **Data Quality**: Clean and validate data before training
2. **Train-Val-Test Split**: Use separate datasets; no data leakage
3. **Preprocessing Consistency**: Use same preprocessor for training and inference
4. **Model Persistence**: Save trained models for reproducibility
5. **Hyperparameter Tuning**: Validate with held-out test set only
6. **Real-time Expectations**: Pre-compute models offline for low-latency inference
7. **Monitoring**: Track anomaly rates and model performance over time

## References

- Scikit-learn: https://scikit-learn.org/
- PyTorch: https://pytorch.org/
- Pandas: https://pandas.pydata.org/
- Time-series Anomaly Detection: https://arxiv.org/abs/1906.03821

## License

This project is provided as-is for educational and industrial use.

## Support

For issues or questions:
1. Check `logs/` directory for error messages
2. Enable `DEBUG_MODE=True` in config
3. Review console output and error traceback
4. Verify data format and path configurations

---

**Version**: 1.0.0  
**Last Updated**: June 2024
