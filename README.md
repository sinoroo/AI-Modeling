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

## 🚀 Quick Start (NEW - Real-Time Inference)

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Generate synthetic sensor data with anomalies
python serial_data_simulator.py

# Run real-time inference on simulated serial data
python test_serial_inference.py
```

**Output:** `inference_results.json` (6,249 predictions) + `realtime_inference_results.png` (visualization)

## Project Structure

```
AI-Modeling/
├── anomaly_detection/                      # Main ML package
│   ├── config.py                          # Hyperparameters & settings
│   ├── preprocessing.py                   # Feature engineering pipeline
│   ├── model_training.py                  # Model training orchestrator
│   ├── evaluation.py                      # Metrics & validation
│   ├── data_loader.py                     # Data I/O
│   └── inference_serial.py                # Real-time serial inference
│
├── 🔴 Real-Time Inference (NEW!)
│   ├── serial_data_simulator.py           # Generate synthetic vibration signals
│   ├── test_serial_inference.py           # Real-time processing engine
│   └── REALTIME_INFERENCE.md              # Detailed serial inference guide
│
├── 📊 Data Analysis & Visualization
│   ├── analyze_3d_array.py                # 3D array structure analysis
│   ├── analyze_2d_data.py                 # 2D classical ML feature analysis
│   ├── 3D_ARRAY_EXPLAINED.md              # 3D structure deep-dive
│   ├── FEATURES_EXPLANATION.md            # Feature engineering details
│   └── ml_input_examples.py               # Input format examples
│
├── 📁 Data Storage
│   ├── data/                              # Raw training data (CSV)
│   ├── data_new_format/                   # Preprocessed windowed data
│   ├── models/                            # Trained models (.pkl, .pt)
│   ├── results/                           # Model evaluation outputs
│   └── eda_results/                       # Analysis visualizations
│
└── 🔧 Configuration
    ├── requirements.txt                   # Python dependencies
    ├── QUICK_START.py                     # Interactive getting-started
    ├── main.py                            # Full pipeline orchestrator
    └── README.md                          # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Create and activate virtual environment:**
   ```bash
   cd c:\workspace\python\AI-Modeling
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # On Windows
   source venv/bin/activate     # On Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Sample Data

```bash
cd c:\workspace\python\AI-Modeling
python main.py --generate-data
```

This creates synthetic training, validation, and test datasets with simulated normal and anomalous behavior.

### 2. Run Full Training Pipeline

```bash
python main.py --train
```

This executes all steps:
- Loads data
- Performs EDA with visualizations
- Preprocesses data (normalize, window, feature engineering)
- Trains all models
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

Runs data generation, training, and inference in sequence.

## Configuration

Edit `anomaly_detection/config.py` to customize:

- **Data paths**: `DATA_DIR`, `TRAIN_DATA_PATH`, `VAL_DATA_PATH`, `TEST_DATA_PATH`
- **Preprocessing**: Window size, normalization method, missing value handling
- **Models**: Hyperparameters for Random Forest, Autoencoder, LSTM, etc.
- **Evaluation**: Metrics to compute
- **Serial communication**: Port (`SERIAL_PORT`), baud rate (`BAUD_RATE`)
- **Anomaly types**: Extensible anomaly classification

### Environment Variables

```bash
# Configure data directory
set DATA_DIR=c:\my_data

# Configure serial port
set SERIAL_PORT=COM3
set BAUD_RATE=9600

# Enable debug mode
set DEBUG_MODE=True
```

## Data Format

### Input CSV Format

Training/validation/test CSVs should have the following format:

```csv
timestamp,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,label
0,45.2,98.3,102.1,52.3,88.4,0
1,46.1,99.1,103.2,51.8,87.9,0
2,45.8,98.7,102.5,52.1,88.2,0
...
```

**Columns:**
- `timestamp` (optional): Time information
- `sensor_*`: Numeric sensor readings (vibration, temperature, current, pressure, etc.)
- `label` (optional): 0 = Normal, 1+ = Anomaly types

### Serial Input Format

For real-time inference via serial communication:

```
45.2,98.3,102.1,52.3,88.4
46.1,99.1,103.2,51.8,87.9
45.8,98.7,102.5,52.1,88.2
...
```

**Format:** Comma-separated sensor values (same order as training)

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
