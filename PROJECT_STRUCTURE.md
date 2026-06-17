# Project Structure & File Organization

Complete guide to understanding the project organization and file relationships.

## 📁 Directory Tree

```
AI-Modeling/
│
├── 📂 anomaly_detection/              ← 핵심 ML 패키지
│   ├── __init__.py                   
│   ├── config.py                     ← ⚙️ 중앙 설정 파일 (data_new_format 기본)
│   ├── data_loader.py                ← 📊 새 포맷 데이터 로딩 & EDA
│   ├── preprocessing.py              ← 🔧 데이터 전처리 & 피처 엔지니어링
│   ├── model_training.py             ← 🤖 모델 훈련 (Classical ML + DL)
│   ├── evaluation.py                 ← 📈 평가 & 메트릭
│   └── inference_serial.py           ← 🔴 실시간 시리얼 추론
│
├── 📂 analysis/                       ← 데이터 분석 도구
│   ├── analyze_3d_array.py           ← 3D 배열 분석
│   ├── analyze_2d_data.py            ← 2D 배열 분석  
│   ├── ml_input_examples.py          ← 입력 포맷 예제
│   ├── MODEL_IO_QUICK_REF.py         ← I/O 참조
│   └── MODEL_IO_VISUALIZATION.md     ← 모델 시각화
│
├── 📂 integration/                    ← MLflow + BentoML 통합
│   ├── mlflow_utils.py               ← MLflow 추적
│   ├── bentoml_service.py            ← BentoML REST API
│   ├── feature_store.py              ← Feature Store
│   ├── mlflow_bentoml_example.py     ← 통합 예제
│   ├── bentofile.yaml                ← BentoML 설정
│   ├── mlflow.db                     ← MLflow 데이터베이스
│   ├── mlruns/                       ← MLflow 실험 로그
│   ├── feature_store/                ← Feature 스토어
│   ├── INTEGRATION_SUMMARY.md        ← 통합 개요
│   ├── MLFLOW_BENTOML_GUIDE.md       ← 상세 가이드
│   ├── QUICK_START_MLFLOW_BENTOML.md ← 빠른 시작
│   └── MLFLOW_FILESTORE_MIGRATION.md ← 마이그레이션
│
├── 📂 util/                           ← 유틸리티 & 테스트
│   ├── serial_data_simulator.py      ← 신호 생성기
│   ├── test_serial_inference.py      ← 실시간 추론 테스트
│   ├── verify_system.py              ← 시스템 검증
│   ├── migrate_mlflow.py             ← MLflow 마이그레이션
│   ├── QUICK_START.py                ← 대화형 가이드
│   ├── REALTIME_INFERENCE.md         ← 시리얼 추론 가이드
│   ├── realtime_inference_results.png ← 추론 결과 이미지
│   └── synthetic_test_data.csv       ← 테스트 데이터
│
├── 📂 📖 Documentation
│   ├── README.md                     ← 📌 프로젝트 개요 (START HERE!)
│   ├── PROJECT_STRUCTURE.md          ← 이 파일
│   ├── DELIVERY_SUMMARY.md           ← 프로젝트 요약
│   ├── 3D_ARRAY_EXPLAINED.md         ← 3D 배열 설명
│   ├── 3D_ARRAY_COMPLETE_GUIDE.md    ← 3D 배열 가이드
│   ├── FEATURES_EXPLANATION.md       ← 피처 설명
│   └── MODEL_IO_FORMAT.md            ← I/O 포맷 명세
│
├── 📂 Data (data_new_format/)        ← 새 포맷 데이터 (메타데이터 + 센서)
│   ├── train/                         ← 훈련 데이터 CSV
│   ├── val/                           ← 검증 데이터 CSV
│   └── test/                          ← 테스트 데이터 CSV
│
├── 📂 data/                          ← 빈 폴더 (레거시 - 더 이상 사용 안 함)
│
├── 📂 Models & Results
│   ├── models/                       ← ✅ 훈련된 모델
│   │   ├── random_forest_model.pkl   (RandomForest)
│   │   ├── isolation_forest_model.pkl (IsolationForest)
│   │   ├── one_class_svm_model.pkl   (OneClassSVM)
│   │   ├── autoencoder_model.pt      (PyTorch)
│   │   ├── lstm_model.pt             (PyTorch)
│   │   ├── scaler.pkl                (StandardScaler)
│   │   └── preprocessor.pkl          (Preprocessor)
│   │
│   ├── results/                      ← 평가 결과
│   │   ├── model_*_evaluation.json   (메트릭)
│   │   └── evaluation_report.json    (전체 보고서)
│   │
│   ├── eda_results/                  ← EDA 시각화
│   ├── logs/                         ← 로깅
│   ├── inference_results.json        ← 추론 결과
│   └── evaluation_report.json        ← 평가 보고서
│
├── main.py                           ← 전체 파이프라인
├── requirements.txt                  ← 패키지 의존성
└── .venv/                            ← 파이썬 가상환경
```

### 주요 변경사항

✅ **폴더 정리**:
- `analysis/`: 데이터 분석 도구 모음
- `util/`: 유틸리티 및 테스트 스크립트
- `integration/`: MLflow/BentoML 통합 모듈

✅ **데이터 포맷**:
- `data_new_format/` 기본 사용 (메타데이터 9줄 + 센서 데이터)
- `data/` 폴더는 비어있음 (레거시)

✅ **제거된 파일**:
- `data_loader_old.py` (구식)
- `data/train.csv, val.csv, test.csv` (구식 포맷)

---

## 🔄 Data Flow & Processing Pipeline

### Complete Training Pipeline

```
raw_data.csv
    ↓ [data_loader.py]
    ├─→ EDA Visualization (eda_results/)
    ├─→ Statistical Analysis
    │
    ↓ [preprocessing.py::Preprocessor]
    ├─ Handle Missing Values
    ├─ Detect Outliers
    ├─ Normalize/Standardize
    ├─ Create Windows (size=64, step=32)
    ├─ Feature Engineering:
    │  ├─ Vibration (raw signal)
    │  ├─ RMS (energy)
    │  ├─ Peak (impulse)
    │  ├─ Crest Factor (sharpness)
    │  ├─ Kurtosis (⭐ most important)
    │  └─ Skewness (asymmetry)
    │
    ↓ Split into 3D & 2D formats
    │
    ├─→ 3D Array (154, 64, 6)      ├─→ 2D Array (29, 384)
    │   - 154 windows              │   - 29 windows
    │   - 64 samples/window        │   - 384 flattened features
    │   - 6 features               │
    │                              │
    ├─→ [model_training.py]        ├─→ [model_training.py]
    │   DL Models                  │   Classical ML Models
    │   │                          │   │
    │   ├─ Autoencoder            │   ├─ RandomForest
    │   │  (128→64→32→64→128)      │   │  (100 trees, depth=15)
    │   │                          │   │
    │   └─ LSTM                    │   ├─ IsolationForest
    │      (2 layers, 64 units)    │   │  (100 estimators)
    │                              │   │
    │                              │   └─ OneClassSVM
    │                              │      (RBF kernel, nu=0.05)
    │                              │
    ├─→ [evaluation.py]            ├─→ [evaluation.py]
    │   Metrics:                   │   Metrics:
    │   - Reconstruction Loss      │   - Accuracy
    │   - Anomaly Score            │   - Precision
    │   │                          │   - Recall
    │   │                          │   - F1-Score
    │   │                          │   - ROC-AUC
    │   │                          │   - Confusion Matrix
    │
    ↓ [model_training.py::save_models()]
    ├─ models/random_forest_model.pkl
    ├─ models/autoencoder_model.pt
    ├─ models/scaler.pkl
    └─ models/preprocessor.pkl
```

### Real-Time Inference Pipeline (NEW!)

```
Serial Port / Simulated Data
    ↓
serial_data_simulator.py
    ├─ generate_normal_signal()
    ├─ generate_anomaly_signal('spike'|'harmonic'|'drift'|'discontinuity')
    └─ generate_synthetic_dataset()
        ↓
        synthetic_test_data.csv (160K samples)
    
    ↓ OR ↓
    
Actual Serial Stream
    │
    ↓ [test_serial_inference.py::RealtimeDataBuffer]
    ├─ Collect samples
    ├─ Maintain 64-sample FIFO buffer
    ├─ Trigger every 32 samples (50% overlap)
    │
    ↓
    ├─ Window: (64, 1) raw samples
    │
    ↓ [test_serial_inference.py::RealtimeInferenceEngine]
    ├─ preprocess_window():
    │  ├─ Reshape: (64,1) → (1,64)
    │  ├─ Normalize: StandardScaler.transform()
    │  └─ Output: (1,64) normalized array
    │
    ├─ predict():
    │  ├─ Load trained model
    │  ├─ model.predict(X)
    │  ├─ Extract confidence
    │  └─ Generate label
    │
    ↓
    Results: {timestamp, prediction, confidence, label}
    │
    ├─→ JSON Log
    │   inference_results.json (748 KB, 6,249 entries)
    │
    ├─→ Visualization
    │   realtime_inference_results.png (3 subplots)
    │   - Original signal
    │   - Window statistics
    │   - Predictions over time
    │
    └─→ Console Output
        時間(s) 値 状態 予測 信頼度
        0.0000 -0.156 정상 정상 0.8000
        ...
```

---

## 📊 File Dependencies

### Circular Dependencies (None - Clean Architecture!)

```
main.py
├── anomaly_detection/config.py ✓
├── anomaly_detection/data_loader.py ✓
├── anomaly_detection/preprocessing.py ✓
├── anomaly_detection/model_training.py ✓
└── anomaly_detection/evaluation.py ✓
    └─ (all depend on config.py)

test_serial_inference.py ✓ (Independent) 
├── anomaly_detection/config.py
├── anomaly_detection/preprocessing.py
├── serial_data_simulator.py
└── sklearn, torch, etc.

serial_data_simulator.py ✓ (Independent)
└── numpy, scipy

analyze_*.py ✓ (Independent analysis tools)
└── Various visualization libraries
```

---

## 💾 Key Configuration Files

### anomaly_detection/config.py

**Hub for ALL settings:**

```python
# Paths
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Preprocessing
WINDOW_SIZE = 64                    # 윈도우 내 샘플 수
STEP_SIZE = 32                      # 슬라이딩 스텝
SAMPLE_RATE = 4000                  # Hz (시뮬레이션)
NORMALIZE_METHOD = "standardize"    # standardize, minmax, robust

# Features
N_FEATURES = 6                      # vibration + 5 statistical
CREATE_FFT_FEATURES = True
FFT_BINS = 10

# Classical ML
CLASSICAL_MODELS = {
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 15
    },
    "IsolationForest": { ... },
    "OneClassSVM": { ... }
}

# Deep Learning
DL_MODELS = {
    "Autoencoder": {
        "encoder_dims": [128, 64, 32],
        "learning_rate": 0.001
    },
    "LSTM": { ... }
}

# Serial Communication
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
TIMEOUT = 1.0
```

---

## 🚀 Execution Workflows

### Workflow 1: Full Training

```bash
python main.py

# Steps:
# 1. data_loader.py → EDA
# 2. preprocessing.py → Windows & Features
# 3. model_training.py → Train all models
# 4. evaluation.py → Metrics & comparison
# 5. Save models to models/
# 6. Save report to evaluation_report.json
```

### Workflow 2: Real-Time Inference (NEW!)

```bash
python test_serial_inference.py

# Steps:
# 1. serial_data_simulator.py → Generate signals
# 2. Create RealtimeDataBuffer & Engine
# 3. Stream samples through buffer
# 4. Every 32 samples: preprocess & predict
# 5. Log results to inference_results.json
# 6. Visualize to realtime_inference_results.png
```

### Workflow 3: Data Analysis

```bash
python analyze_3d_array.py    # 3D structure analysis
python analyze_2d_data.py     # 2D structure analysis
```

### Workflow 4: Quick Start (Interactive)

```bash
python QUICK_START.py

# Menu:
# [1] Real-time inference test
# [2] Generate synthetic data
# [3] Analyze 3D array
# [4] Analyze 2D array
# [5] Full training pipeline
# [6] View documentation
```

---

## 📊 Data Formats

### Input: Raw CSV

```csv
sensor_0,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,label
45.2,98.3,102.1,52.3,88.4,76.1,0
46.1,99.1,103.2,51.8,87.9,76.8,0
...
```

### 3D Array Format

Shape: `(n_windows, 64, 6)`

```python
array([
  # Window 0
  [[45.2, 0.391, 0.923, 2.361, -0.145, 0.234],  # Sample 0, 6 features
   [46.1, 0.392, 0.891, 2.355, -0.142, 0.231],  # Sample 1
   ...
   [45.8, 0.390, 0.910, 2.358, -0.144, 0.232]],  # Sample 63
  
  # Window 1
  [...],
  ...
])
```

### 2D Array Format

Shape: `(n_windows, 384)`

```python
# Each row = 64 samples × 6 features flattened
array([
  [45.2, 0.391, 0.923, ..., 45.8, 0.390, 0.910],  # 384 values
  [...],
])
```

### JSON Results Format

```json
{
  "timestamp": "2026-06-10 17:59:05",
  "total_predictions": 6249,
  "results": [
    {
      "timestamp": 0.01575,
      "prediction": 0,
      "confidence": 0.8,
      "label": "정상"
    },
    {
      "timestamp": 0.02375,
      "prediction": 0,
      "confidence": 0.5,
      "label": "정상"
    },
    ...
  ]
}
```

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | June 2026 | Added real-time serial inference, synthetic data generation |
| v1.5 | May 2026 | Comprehensive documentation, 3D/2D array analysis |
| v1.0 | April 2026 | Initial release with classical ML and DL models |

---

## 📈 Performance Specifications

| Aspect | Value | Unit |
|--------|-------|------|
| Sampling Rate | 4000 | Hz |
| Window Size | 64 | samples |
| Window Duration | 16 | ms |
| Step/Overlap | 32 / 50% | samples / % |
| Features per Window | 6 | count |
| Inference Latency | ~5-10 | ms |
| Throughput | ~200 | windows/sec |
| Model Count | 5 | (3 classical + 2 DL) |
| Memory Usage | ~150 | MB |

---

## 🎯 Best Practices

1. **Always start with README.md** for overview
2. **Use QUICK_START.py** for interactive guidance
3. **Check REALTIME_INFERENCE.md** for production deployment
4. **Maintain config.py** as single source of truth
5. **Document model architecture changes** in code comments
6. **Test with synthetic data** before hardware deployment
7. **Monitor inference_results.json** for prediction trends
8. **Back up trained models** in models/ directory

---

**Last Updated:** June 10, 2026  
**Maintained By:** AI-Modeling Team  
**Status:** ✅ Production Ready
