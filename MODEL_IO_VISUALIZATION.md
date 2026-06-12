"""
모델 입출력 형식 시각적 가이드

각 모델의 데이터 흐름을 다이어그램으로 표현
"""

# ============================================================================
# 전체 데이터 파이프라인 흐름
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OVERALL DATA PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

         CSV 파일 (NEW FORMAT)
         ├─ 1-9줄: 메타데이터 (Date, Data Label, Motor Spec 등)
         └─ 10줄~: 실제 데이터 (time, vibration)

                        ↓

         DataLoader.load_data()
         ├─ 메타데이터 파싱
         ├─ 파일 자동 결합 (디렉토리 내 모든 CSV)
         └─ Output: DataFrame (4990, 3) [time, vibration, label]

                        ↓

         Preprocessor.preprocess_pipeline()
         ├─ Missing values 처리
         ├─ Outliers 처리
         ├─ Normalization (standardize)
         ├─ Windowing (64-size windows)
         └─ Output: 3D Array (154, 64, 1)

                        ↓
                   ┌─────────────────────────────────┐
                   │   모델별 처리                     │
                   └─────────────────────────────────┘
         ┌──────────────────┬──────────────────┬──────────────────┐
         │                  │                  │                  │
         ▼                  ▼                  ▼                  ▼
    RandomForest      IsolationForest      OneClassSVM       Autoencoder/LSTM
    (Supervised)      (Unsupervised)       (Unsupervised)    (Unsupervised)
    Flatten 2D        Flatten 2D           Flatten 2D          Keep 3D
    Input: (154, 64)  Input: (154, 64)     Input: (154, 64)    Input: (154, 64, 1)
         │                  │                  │                  │
         ▼                  ▼                  ▼                  ▼
    predict →            predict →           predict →       forward →
    Output:          Output:               Output:          Output:
    (27,)            (27,)                 (27,)            (27, 64) or (27, 1)

                        ↓

    predict_proba() / reconstruction_error()
         │
         ▼

    Evaluation & Metrics
    ├─ Accuracy, Precision, Recall, F1-Score
    ├─ ROC-AUC
    ├─ Confusion Matrix
    └─ JSON Report


┌─────────────────────────────────────────────────────────────────────────────┐
│                  DETAILED MODEL INPUT/OUTPUT SHAPES                        │
└─────────────────────────────────────────────────────────────────────────────┘

1. RANDOM FOREST (Classical ML - Supervised)
   ────────────────────────────────────────────────────────────────────────────

   INPUT (Training):
   ┌─────────────────┐
   │ (154, 64, 1)    │ 3D Array: 154 windows, 64 samples each, 1 feature
   │ + y_train       │ Labels: (154,) shape with [0, 1] values
   │ (154,)          │
   └─────────────────┘
           │
           ├─ Flatten: (154, 64, 1) → (154, 64)
           │
           ▼
   ┌─────────────────────────────────────┐
   │ RandomForestClassifier              │
   │ - n_estimators: 100                 │
   │ - max_depth: 15                     │
   │ - learning method: bootstrap+voting │
   └─────────────────────────────────────┘

   PREDICTION (Testing):
   ┌──────────────────┐     predict()      ┌────────────────┐
   │ (27, 64, 1)      │ ──────────────> │ (27,)          │
   │ 3D Test Array    │                 │ [0, 1, 0, 1, ...]
   └──────────────────┘                 └────────────────┘

   PROBABILITY:
   ┌──────────────────┐  predict_proba()   ┌────────────────┐
   │ (27, 64, 1)      │ ──────────────> │ (27, 2)        │
   │ 3D Test Array    │                 │ [[0.9, 0.1],   │
   └──────────────────┘                 │  [0.1, 0.9],   │
                                        │  ...]          │
                                        └────────────────┘

   OUTPUT (JSON):
   {
     "model_name": "RandomForest",
     "metrics": {
       "accuracy": 1.0,
       "precision": 1.0,
       "recall": 1.0,
       "f1": 1.0,
       "roc_auc": 1.0
     },
     "confusion_matrix": [[18, 0], [0, 9]],
     "predictions_count": 27,
     "probabilities": {"shape": [27, 2], "min": 0.0, "max": 1.0, "mean": 0.5}
   }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. AUTOENCODER (Deep Learning - Unsupervised)
   ────────────────────────────────────────────────────────────────────────────

   ARCHITECTURE:
   Input (64)
        │
        ▼
   Encoder: Linear(64 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 32)
        │
        │ (Bottleneck: 32 dimensions)
        │
        ▼
   Decoder: Linear(32 → 64) → ReLU → Linear(64 → 128) → ReLU → Linear(128 → 64)
        │
        ▼
   Output (64)

   INPUT (Training):
   ┌─────────────────┐
   │ (154, 64, 1)    │ 3D Array
   │ (flatten)       │ 
   │ (154, 64)       │ Flatten to 2D
   └─────────────────┘
           │
           ├─ Batch Processing: (16, 64)
           │ (batch_size=16)
           │
           ▼
   ┌─────────────────────────────────────┐
   │    Forward Pass                     │
   │ (16, 64) → Encoder → (16, 32)      │
   │         → Decoder → (16, 64)       │
   │                                     │
   │    Loss = MSE(input, output)       │
   │    Optimize: Adam, lr=0.001        │
   │    Epochs: 50                      │
   └─────────────────────────────────────┘

   PREDICTION (Testing):
   ┌──────────────────┐     forward()       ┌────────────────┐
   │ (27, 64, 1)      │ ──────────────> │ (27, 64)       │
   │ (flatten)        │                 │ Reconstructed  │
   │ (27, 64)         │                 │ Input          │
   └──────────────────┘                 └────────────────┘

   RECONSTRUCTION ERROR:
   ┌────────────────────┐    Calculation      ┌─────────────┐
   │ input:  (27, 64)   │   error =           │ (27,)       │
   │ output: (27, 64)   │   mean((inp-out)²)  │ Error scores│
   └────────────────────┘                     └─────────────┘
   Example: [0.26, 0.31, 3.40, 0.28, ...]
            (낮음)      (높음=이상)

   ANOMALY DETECTION:
   Threshold = 95-percentile = 3.04
   if error > 3.04: anomaly (1)
   else: normal (0)

   OUTPUT (JSON):
   {
     "model_name": "Autoencoder",
     "metrics": {
       "accuracy": 0.74,
       "precision": 0.81,
       "recall": 0.74,
       "f1": 0.68
     },
     "confusion_matrix": [[18, 0], [7, 2]],
     "predictions_count": 27,
     "reconstruction_error": {
       "threshold": 3.04,
       "min": 0.266,
       "max": 3.405,
       "mean": 1.026,
       "std": 0.976,
       "percentile_95": 3.04
     }
   }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. LSTM (Deep Learning - Unsupervised)
   ────────────────────────────────────────────────────────────────────────────

   ARCHITECTURE:
   Sequence Input (64, 1)
        │
        ▼              
   LSTM Layer 1 (input_size=1, hidden_size=64, num_layers=2)
   [Sequence Processing with 2 layers]
        │
        ▼
   Get Last Hidden State → FC Linear(64 → 1)
        │
        ▼
   Output (1,) [Last time step prediction]

   INPUT (Training):
   ┌─────────────────┐
   │ (154, 64, 1)    │ 3D Tensor: KEEP 3D (중요!)
   │                 │ 154 sequences, each 64 time steps, 1 feature
   └─────────────────┘
           │
           ├─ Batch Processing: (16, 64, 1)
           │ (batch_size=16)
           │
           ▼
   ┌────────────────────────────────────┐
   │    Forward Pass                    │
   │ (16, 64, 1) → LSTM → (16, 64, 64) │
   │             (lstm_out has 64 hidden dims)
   │            Take last: (16, 64)     │
   │            → FC: (16, 1)           │
   │                                    │
   │    Loss = MSE(input_mean, output) │
   │    Optimize: Adam, lr=0.001       │
   │    Epochs: 100                    │
   └────────────────────────────────────┘

   PREDICTION (Testing):
   ┌──────────────────┐     forward()       ┌────────────────┐
   │ (27, 64, 1)      │ ──────────────> │ (27, 1)        │
   │ Keep 3D!         │                 │ Predictions    │
   │ Sequence         │                 │ (Last time     │
   │ data             │                 │  step values)  │
   └──────────────────┘                 └────────────────┘

   RECONSTRUCTION ERROR:
   ┌────────────────────┐    Calculation      ┌─────────────┐
   │ input_mean: (27,1) │   error =           │ (27,)       │
   │ output:     (27,1) │   mean((mean-out)²) │ Error scores│
   └────────────────────┘                     └─────────────┘
   Example: [0.14, 0.22, 3.20, 0.15, ...]
            (낮음)       (높음=이상)

   ANOMALY DETECTION:
   Threshold = 95-percentile = 2.84
   if error > 2.84: anomaly (1)
   else: normal (0)

   OUTPUT (JSON):
   {
     "model_name": "LSTM",
     "metrics": {
       "accuracy": 0.74,
       "precision": 0.81,
       "recall": 0.74,
       "f1": 0.68
     },
     "confusion_matrix": [[18, 0], [7, 2]],
     "predictions_count": 27,
     "reconstruction_error": {
       "threshold": 2.84,
       "min": 0.142,
       "max": 3.201,
       "mean": 0.892,
       "std": 0.845,
       "percentile_95": 2.84
     }
   }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. ISOLATION FOREST & ONE-CLASS SVM (Classical ML - Unsupervised)
   ────────────────────────────────────────────────────────────────────────────

   [동일한 입력 처리]

   INPUT (Training):
   ┌─────────────────┐
   │ (154, 64, 1)    │ 3D Array (레이블 불필요)
   │ (flatten)       │ 
   │ (154, 64)       │ Flatten to 2D
   └─────────────────┘

   ISOLATION FOREST:
   - 알고리즘: Isolation Trees
   - contamination: 0.1 (예상 이상치율)
   - predict: [1, -1] → 변환 → [0, 1]

   ONE-CLASS SVM:
   - 알고리즘: Support Vector Machine (single class)
   - kernel: 'rbf'
   - nu: 0.05
   - predict: [1, -1] → 변환 → [0, 1]

   PREDICTION:
   ┌──────────────────┐     predict()       ┌────────────────┐
   │ (27, 64, 1)      │ ──────────────> │ (27,)          │
   │ (flatten)        │                 │ [1, 1, -1, 1...]
   │ (27, 64)         │                 │ Convert to     │
   └──────────────────┘                 │ [0, 0, 1, 0...]│
                                        └────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA SIZE COMPARISON TABLE                           │
└─────────────────────────────────────────────────────────────────────────────┘

Stage                    Operation              Train Size    Test Size
─────────────────────────────────────────────────────────────────────────────
1. CSV Files             Load from disk         Multiple      Multiple
                         files combined
                                                ↓              ↓
                         (4990, 3)              (897, 3)

2. Preprocessing         Time-series            ↓              ↓
                         windowing              (154, 64, 1)   (27, 64, 1)

3. Classical Models      Flatten                ↓              ↓
   (RF/IF/SVM)           2D shape               (154, 64)      (27, 64)
                                                ↓              ↓
                         Predictions            —              (27,)
                         Probabilities          —              (27, 2)

4. Autoencoder           Flatten                ↓              ↓
                         Batch shape            (16, 64)       (27, 64)
                                                ↓              ↓
                         Reconstructed          (16, 64)       (27, 64)
                                                ↓              ↓
                         Reconstruction         —              (27,)
                         Error

5. LSTM                  Keep 3D                ↓              ↓
                         Batch shape            (16, 64, 1)    (27, 64, 1)
                                                ↓              ↓
                         Predictions            (16, 1)        (27, 1)
                                                ↓              ↓
                         Reconstruction         —              (27,)
                         Error

6. Evaluation            Metrics                —              {dict}
                         JSON Report            —              .json


┌─────────────────────────────────────────────────────────────────────────────┐
│                            KEY TRANSFORMATIONS                             │
└─────────────────────────────────────────────────────────────────────────────┘

Transformation 1: CSV → DataFrame
┌─────────────┐
│ CSV Files   │
│ (NEW        │
│  FORMAT)    │
└──────┬──────┘
       │ load_data()
       ▼
┌─────────────┐
│ DataFrame   │
│ (4990, 3)   │ [time, vibration, label]
│ time: float │
│ vibration   │
│   : float   │
│ label: int  │
└─────────────┘

─────────────────────────────────────────────────────────────────────────────

Transformation 2: DataFrame → 3D Array (Windowing)
┌─────────────┐
│ DataFrame   │
│ (4990, 3)   │
└──────┬──────┘
       │ Preprocessing
       │ - Missing values
       │ - Outliers
       │ - Normalization
       │ - Windowing (window_size=64, overlap=50%)
       ▼
┌─────────────┐
│ 3D Numpy    │
│ Array       │
│ (154, 64, 1)│ [n_windows, window_size, n_features]
│ dtype       │ float32
│ range       │ [-3, 3] (normalized)
└─────────────┘

─────────────────────────────────────────────────────────────────────────────

Transformation 3: 3D → 2D (for Classical ML)
┌─────────────┐
│ 3D Array    │
│ (154, 64, 1)│
└──────┬──────┘
       │ reshape()
       │ (154, 64, 1) → (154, 64)
       ▼
┌─────────────┐
│ 2D Array    │
│ (154, 64)   │
│ (vectors    │
│  for each   │
│  window)    │
└─────────────┘

─────────────────────────────────────────────────────────────────────────────

Transformation 4: Predictions → Anomaly Scores
┌──────────────────┐
│ Model Output     │
│ (27,) or (27, 2) │
└──────┬───────────┘
       │ Evaluation
       │ - Metrics computation
       │ - Reconstruction error (if applicable)
       │ - Threshold detection
       │ - Anomaly classification
       ▼
┌──────────────────┐
│ JSON Report      │
│ - metrics        │
│ - confusion      │
│   matrix         │
│ - error stats    │
│ - predictions    │
└──────────────────┘
"""


# ============================================================================
# 모델 선택 가이드 (Which Model to Use?)
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 모델 선택 기준                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Random Forest
├─ 단점: 레이블 필요 (supervised)
├─ 장점: 빠른 학습, 높은 정확도, 해석 가능
├─ 추천: 레이블된 데이터 많을 때
└─ 성능: 뛰어남 (98%+ 정확도 가능)

Isolation Forest
├─ 단점: 성능 변동 크음
├─ 장점: 무감독, 빠름
├─ 추천: 빠른 탐지 필요할 때
└─ 성능: 중간 (60~70% 정확도)

One-Class SVM
├─ 단점: 성능 낮음 (튜닝 필요)
├─ 장점: 무감독, 확장성
├─ 추천: 정상 데이터만 있을 때
└─ 성능: 중간~낮음 (40~60%)

Autoencoder
├─ 단점: 학습 느림 (GPU 권장)
├─ 장점: 무감독, 복잡한 패턴 학습, 좋은 성능
├─ 추천: 충분한 계산 자원 있을 때
└─ 성능: 좋음 (70~80% 정확도)

LSTM
├─ 단점: 학습 매우 느림, 과적합 위험
├─ 장점: 무감독, 시계열 패턴 학습, 뛰어난 성능
├─ 추천: 시계열 데이터, 장기 의존성 있을 때
└─ 성능: 매우 좋음 (80~90% 정확도)
"""
