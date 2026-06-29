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
│   ├── feature_extraction.py         ← ⭐ NEW: 학습/추론 공용 특징 추출
│   ├── model_training.py             ← 🤖 모델 훈련 (Classical ML + DL) [레거시 A]
│   ├── evaluation.py                 ← 📈 평가 & 메트릭
│   └── inference_serial.py           ← 🔴 실시간 시리얼 추론
│
├── 📄 build_feature_table.py          ← ⭐ NEW: 원시 CSV → 표준 특징 테이블 [표준화 B]
├── 📄 train_from_feature_table.py     ← ⭐ NEW: 특징 테이블 → 5클래스 분류 [표준화 B]
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
├── 📂 Data (data_new_format/)        ← 새 포맷 데이터 (메타데이터 + 센서, 5클래스)
│   ├── train/                         ← 훈련 데이터 CSV ([동력]/[설비]/[상태] 트리)
│   ├── val/                           ← 검증 데이터 CSV
│   └── test/                          ← 테스트 데이터 CSV
│
├── 📂 feature_tables/                 ← ⭐ NEW: 표준화 특징 테이블 (윈도우=1행)
│   ├── feature_table_train.csv
│   ├── feature_table_val.csv
│   ├── feature_table_test.csv
│   └── normalization_stats.json       ← 설비별 정상 기준 통계
│
├── 📂 data/                          ← 빈 폴더 (레거시 - 더 이상 사용 안 함)
│
├── 📂 Models & Results
│   ├── models/                       ← ✅ 훈련된 모델
│   │   # [A] 레거시 파이프라인 (main.py / model_training.py)
│   │   ├── random_forest_model.pkl   (RandomForest)
│   │   ├── isolation_forest_model.pkl (IsolationForest)
│   │   ├── one_class_svm_model.pkl   (OneClassSVM)
│   │   ├── autoencoder_model.pt      (PyTorch)
│   │   ├── lstm_model.pt             (PyTorch)
│   │   ├── scaler.pkl                (StandardScaler)
│   │   ├── preprocessor.pkl          (Preprocessor)
│   │   # [B] 표준화 파이프라인 (train_from_feature_table.py) ⭐ NEW
│   │   ├── clf_random_forest.pkl     (5클래스 고장 분류기)
│   │   ├── clf_scaler.pkl            (분류기용 스케일러)
│   │   ├── anomaly_isolation_forest.pkl (이상탐지 One-Class)
│   │   └── anomaly_scaler.pkl        (이상탐지용 스케일러)
│   │
│   ├── results/                      ← 평가 결과
│   │   ├── model_*_evaluation.json   (레거시 모델별 메트릭)
│   │   ├── feature_table_evaluation.json (⭐ NEW: 5클래스/이상탐지)
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

> 이 프로젝트에는 **독립적인 두 파이프라인**이 있습니다.
> - **[A] 레거시**: `main.py` → `model_training.py` (윈도우 원시신호, 15개 모델)
> - **[B] 표준화**: `build_feature_table.py` → `train_from_feature_table.py` (5클래스 분류)
>
> `train_from_feature_table.py` 는 `model_training.py` 와 **연계되지 않으며**,
> sklearn 을 직접 사용합니다. 두 파이프라인의 공통 기반은 `feature_extraction.py` 입니다.

### Standardized Pipeline [B] (NEW ⭐)

```
data_new_format/**/*.csv  (원시 진동, 길이 12000/500/300 혼재, 5클래스)
    ↓ [build_feature_table.py --regroup]
    ├─ anomaly_detection/feature_extraction.py (공용 특징)
    │    ├─ make_windows: 4000샘플(1초) 고정 윈도우, 50% overlap
    │    ├─ normalize_amplitude: 설비별 정상 기준 Z-score
    │    ├─ time_features: rms/peak/crest/std/kurtosis/skewness/p2p
    │    └─ order_spectrum_features: 회전차수(order) 대역 + 스펙트럼 통계
    ↓
    feature_tables/feature_table_{train,val,test}.csv  (윈도우=1행, 20 특징)
    normalization_stats.json
    ↓ [train_from_feature_table.py]  (sklearn 직접)
    ├─ 과제1: RandomForestClassifier → 5클래스 (test acc ~99.5%)
    └─ 과제2: IsolationForest(정상만 학습) → 이상탐지
    ↓
    models/clf_random_forest.pkl, clf_scaler.pkl,
           anomaly_isolation_forest.pkl, anomaly_scaler.pkl
    results/feature_table_evaluation.json
    ↓ [util/test_serial_inference.py --mode classify]
    └─ FaultClassifierEngine
         └─ feature_extraction.py (동일 특징) → 실시간 5클래스 추론
```

### Complete Training Pipeline [A] (레거시)

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

# [B] 표준화 파이프라인 (NEW ⭐)
build_feature_table.py ✓
└── anomaly_detection/feature_extraction.py   (model_training 미사용)

train_from_feature_table.py ✓ (Independent)
├── feature_tables/*.csv (build_feature_table.py 산출물)
└── sklearn (RandomForest, IsolationForest) — model_training 미사용

test_serial_inference.py ✓
├── anomaly_detection/feature_extraction.py   (학습과 동일 특징)
├── serial_data_simulator.py (RealSignalStreamer)
└── models/clf_random_forest.pkl, clf_scaler.pkl

serial_data_simulator.py ✓ (Independent)
├── anomaly_detection/config.py
└── numpy, pandas

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

### Input: 진동 CSV (data_new_format, 새 포맷)

```
# 헤더 9줄(메타데이터) + 데이터 2열(time, vibration)
Date,2020-11-25 16:39:09
Filename,STFMK-...004.dat
Data Label,정상              ← 5클래스: 정상/축정렬불량/회전체불평형/베어링불량/벨트느슨함
Label_No,00
Motor Spec,L-DSF-01, 1730,2.2, 8.6,   ← 설비ID, RPM, 동력(kW), 전류(A)
Period,3SEC
Sample Rate,4000
RMS,0.005718,
Data Length,12000,
0,0.0031755939,             ← 이하 time, vibration
0.00025,0.0026699300,
...
```

### Standardized Feature Table (NEW ⭐, feature_tables/*.csv)

윈도우 1개 = 1행. `build_feature_table.py` 산출물.

```csv
source_file,equipment_id,power_kw,rpm,window_index,padded,
rms,peak,crest_factor,std,kurtosis,skewness,p2p,
spectral_centroid,spectral_entropy,spectral_energy,
band_1x,band_1x_ratio,band_2x,band_2x_ratio,band_3x,band_3x_ratio,band_high,band_high_ratio,
label_name,label_no,is_anomaly
```

### Legacy: Raw CSV (구 포맷, 참고용)

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
| v3.0 | June 2026 | ⭐ 표준화 파이프라인[B] 추가: feature_extraction 공용 모듈, build/train_from_feature_table, 5클래스 고장 분류, 회전차수(order) FFT 특징, 설비단위 재분할 |
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

**Last Updated:** June 22, 2026  
**Maintained By:** AI-Modeling Team  
**Status:** ✅ Production Ready
