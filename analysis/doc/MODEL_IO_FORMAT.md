"""
모델 입출력 형식 분석 및 데이터 흐름 가이드

이 문서는 각 모델의 입력/출력 형식, 데이터 변환 과정을 상세히 설명합니다.
"""

# ============================================================================
# 전체 데이터 흐름 (Data Flow)
# ============================================================================

## 원본 데이터 형식 (새로운 CSV 포맷)
# 파일: train/val/test 디렉토리의 CSV 파일들
# 포맷: 
#   - 1-9줄: 메타데이터 (Date, Filename, Data Label, Motor Spec, RMS 등)
#   - 10줄부터: 실제 데이터 (time, vibration)
# 예제:
#   Date,2024-01-15 10:30:00
#   Filename,train_normal_000.csv
#   Data Label,정상
#   ...
#   Data Length,12000,
#   0.0,value1
#   0.00025,value2
#   ...

## DataLoader 거쳐서 로드된 데이터
# 형식: DataFrame
# 크기: (n_rows, 3)  
# 컬럼: ['time', 'vibration', 'label']
# 예제:
#       time   vibration  label
#   0    0.00    -0.0065      0
#   1    0.00    -0.0184      0
#   2    0.00    -0.0095      0
#   ...
#   4990 ...        ...      1

## 다중 파일 결합 후
# 훈련 데이터: (4990, 3)   # 10개 파일 결합
# 검증 데이터: (897, 3)    # 3개 파일 결합
# 테스트 데이터: (897, 3)   # 3개 파일 결합


# ============================================================================
# Step 1: 전처리 (Preprocessing)
# ============================================================================

## 입력 데이터 형식
# - 형식: DataFrame
# - 크기: (n_samples, n_features+1)
# - 예: (4990, 3) [time, vibration, label]

## 처리 과정
# 1. Missing values handling (보간법, forward fill 등)
# 2. Outlier handling (IQR 방법)
# 3. Feature normalization (Standardization, MinMax, RobustScaler)
# 4. Time-series windowing: (n_samples, features) → (n_windows, window_size, n_features)

## 출력 데이터 형식
# - 형식: Numpy array (3D)
# - 크기: (n_windows, window_size, n_features)
# 예:
#   - 훈련: (154, 64, 1)     # 154개 윈도우, 각 64개 샘플, 1개 특성
#   - 검증: (27, 64, 1)
#   - 테스트: (27, 64, 1)

## 레이블 변환
# 메타데이터 'Data Label' → 숫자 변환
# - "정상" → 0
# - "축정렬불량" 등 → 1


# ============================================================================
# Step 2: 모델별 입출력 형식
# ============================================================================

## ========================================
## A. Random Forest (Classical ML)
## ========================================

### 입력 형식
# - 원본 형식: (154, 64, 1) 3D array
# - 변환: 2D로 평탄화 (flatten)
# - 최종 입력: (154, 64) 2D array
# - 데이터 타입: float32/float64
# - 값 범위: 정규화된 범위 (standardized)

### 내부 처리
# sklearn RandomForestClassifier
# - n_estimators: 100개 트리
# - max_depth: 15
# - 분류 방식: 다중 트리 투표

### 출력 형식
# 1. predict() 반환:
#    - 타입: numpy array
#    - 크기: (n_test,)
#    - 값: [0, 1] (클래스 레이블)
#    예: [0, 0, 1, 0, 1, ...]

# 2. predict_proba() 반환:
#    - 타입: numpy array
#    - 크기: (n_test, 2)
#    - 값: 각 클래스의 확률 [0.0 ~ 1.0]
#    예: 
#      [[0.95, 0.05],
#       [0.88, 0.12],
#       [0.15, 0.85],
#       ...]

### 평가 메트릭

# 평가 지표:
# - Accuracy: 정확도
# - Precision: 정밀도 (이상으로 예측한 것 중 실제 이상인 비율)
# - Recall: 재현율 (실제 이상 중 이상으로 예측한 비율)
# - F1-Score: Precision과 Recall의 조화 평균
# - ROC-AUC: ROC 곡선 아래 면적

# 혼동 행렬 (Confusion Matrix):
#        실제 정상  실제 이상
# 예측 정상   TN        FN
# 예측 이상   FP        TP

# 예제 결과:
# {
#   "accuracy": 1.0,
#   "precision": 1.0,
#   "recall": 1.0,
#   "f1": 1.0,
#   "roc_auc": 1.0,
#   "confusion_matrix": [[18, 0], [0, 9]]
# }


## ========================================
## B. Isolation Forest (Classical ML - 비지도)
## ========================================

### 입력 형식
# - 원본 형식: (154, 64, 1) 3D array
# - 변환: 2D로 평탄화
# - 최종 입력: (154, 64) 2D array

### 내부 처리
# - 비지도 학습 (레이블 불필요)
# - 이상치 탐지: Isolation Trees
# - contamination: 0.1 (10%가 이상치로 예상)

### 출력 형식
# 1. predict() 반환:
#    - 타입: numpy array
#    - 크기: (n_test,)
#    - 값: [1, -1] (정상=1, 이상=-1)
#    - 변환: 내부에서 1→0, -1→1으로 변환

# 2. decision_function() 반환:
#    - 타입: numpy array
#    - 크기: (n_test,)
#    - 값: anomaly score (낮을수록 이상치)


## ========================================
## C. One-Class SVM (Classical ML - 비지도)
## ========================================

### 입력 형식
# - 원본 형식: (154, 64, 1) 3D array
# - 변환: 2D로 평탄화
# - 최종 입력: (154, 64) 2D array

### 내부 처리
# - 비지도 학습 (정상 데이터만 학습)
# - kernel: 'rbf' (Radial Basis Function)
# - gamma: 'auto'
# - nu: 0.05 (서포트 벡터 비율)

### 출력 형식
# 1. predict() 반환:
#    - 타입: numpy array
#    - 크기: (n_test,)
#    - 값: [1, -1] (정상=1, 이상=-1)
#    - 변환: 1→0, -1→1

# 2. decision_function() 반환:
#    - 타입: numpy array
#    - 크기: (n_test,)
#    - 값: decision score


## ========================================
## D. Autoencoder (Deep Learning - 비지도)
## ========================================

### 모델 구조
# Input → Encoder → Bottleneck → Decoder → Output
# Input (64) → 128 → 64 → 32 → 64 → 128 → Output (64)

### 입력 형식
# 훈련/검증/테스트 데이터:
# - 원본 형식: (n_windows, 64, 1) 3D array
# - 변환: 2D로 평탄화
# - 최종 입력: (n_windows, 64) 2D tensor
# - 배치 처리: (batch_size, 64) [예: (16, 64)]

### 내부 처리
# - forward pass: input → encoded → decoded
# - Loss: MSE (Mean Squared Error)
# - 학습: 입력과 출력의 차이 최소화
# - Epoch: 50
# - Learning rate: 0.001

### 출력 형식
# 1. forward() 반환:
#    - 타입: torch.Tensor
#    - 크기: (batch_size, 64)
#    - 값: 재구성된 입력 (float32)
#    예: 
#      tensor([[-0.0065, -0.0184, ...],
#              [ 0.0012,  0.0045, ...],
#              ...])

# 2. 재구성 오류 (Reconstruction Error):
#    - 계산: mean((input - output)^2)
#    - 타입: numpy array
#    - 크기: (n_test,)
#    - 값: 0.0 ~∞ (낮을수록 정상, 높을수록 이상)
#    예: [0.266, 0.315, 3.404, 0.289, ...]

# 3. 이상치 판정:
#    - 임계값 선택: 95 percentile
#    - 규칙: error > threshold → 이상(1), else → 정상(0)

### 평가 메트릭
# {
#   "accuracy": 0.74,
#   "precision": 0.81,
#   "recall": 0.74,
#   "f1": 0.68,
#   "reconstruction_error": {
#     "threshold": 3.04,
#     "min": 0.266,
#     "max": 3.405,
#     "mean": 1.026,
#     "std": 0.976
#   },
#   "confusion_matrix": [[18, 0], [7, 2]]
# }


## ========================================
## E. LSTM (Deep Learning - 비지도)
## ========================================

### 모델 구조
# Input → LSTM Layer 1 → LSTM Layer 2 → Last Hidden State → FC Layer → Output

### 입력 형식
# 훈련/검증/테스트 데이터:
# - 원본 형식: (n_windows, 64, 1) 3D array [그대로 유지]
# - 최종 입력: (n_windows, 64, 1) 3D tensor
# - 배치 처리: (batch_size, 64, 1) [예: (16, 64, 1)]
# - Explanation: batch_size=배치 크기, 64=시퀀스 길이, 1=특성 개수

### 내부 처리
# - LSTM 레이어: 2개
# - hidden_size: 64
# - batch_first=True (배치가 첫 번째 차원)
# - 처리: 시퀀스 전체 입력 → LSTM 상태 유지
# - 마지막 hidden state 취출
# - Fully Connected: 64 → 1

### 출력 형식
# 1. forward() 반환:
#    - 타입: torch.Tensor
#    - 크기: (batch_size, 1)
#    - 값: 마지막 시점 예측값 (float32)
#    예:
#      tensor([[-0.0145],
#              [ 0.0087],
#              [-0.0023],
#              ...])

# 2. 재구성 오류 (평균을 기준):
#    - 입력 평균: mean(input, dim=1) = (batch_size, 1)
#    - 오류: mean((input_mean - output)^2, dim=1)
#    - 타입: numpy array
#    - 크기: (n_test,)
#    - 값: 0.0 ~∞

### 평가 메트릭
# {
#   "accuracy": 0.74,
#   "precision": 0.81,
#   "recall": 0.74,
#   "f1": 0.68,
#   "reconstruction_error": {
#     "threshold": 2.84,
#     "min": 0.142,
#     "max": 3.201,
#     "mean": 0.892,
#     "std": 0.845
#   },
#   "confusion_matrix": [[18, 0], [7, 2]]
# }


# ============================================================================
# 데이터 크기 비교 표
# ============================================================================

# 단계별 데이터 변환:
# ┌──────────────────────────┬──────────────────┬────────────────────┐
# │        단계              │    입력 크기     │     출력 크기      │
# ├──────────────────────────┼──────────────────┼────────────────────┤
# │ 1. CSV 로드             │  N/A            │  (4990, 3)        │
# │ 2. 전처리               │  (4990, 3)      │  (154, 64, 1) 3D  │
# ├──────────────────────────┼──────────────────┼────────────────────┤
# │ 3a. RF/IF/SVM           │  (154, 64, 1)   │  (154, 64) 2D     │
# │     predict()           │  (27, 64) test  │  (27,) 클래스     │
# │     predict_proba()     │  (27, 64) test  │  (27, 2) 확률     │
# ├──────────────────────────┼──────────────────┼────────────────────┤
# │ 3b. Autoencoder         │  (154, 64, 1)   │  (154, 64) 2D     │
# │     forward()           │  (16, 64) batch │  (16, 64) 재구성  │
# │     reconstruction err  │  (27, 64)       │  (27,) 오류 점수  │
# ├──────────────────────────┼──────────────────┼────────────────────┤
# │ 3c. LSTM                │  (154, 64, 1)   │  (154, 64, 1) 3D  │
# │     forward()           │  (16, 64, 1)    │  (16, 1) 예측     │
# │     reconstruction err  │  (27, 64, 1)    │  (27,) 오류 점수  │
# └──────────────────────────┴──────────────────┴────────────────────┘


# ============================================================================
# 사용 예시
# ============================================================================

"""
from anomaly_detection import data_loader, preprocessing, model_training, evaluation

# 1. 데이터 로드
loader = data_loader.DataLoader(use_new_format=True)
train_df = loader.load_data("data_new_format/train")  # (4990, 3)

# 2. 전처리
preprocessor = preprocessing.Preprocessor()
X_train, y_train = preprocessor.preprocess_pipeline(train_df, feature_cols=['vibration'], label_col='label')
# 출력: X_train (154, 64, 1), y_train (154,)

# 3. 모델 학습
rf_model = model_training.ClassicalModelTrainer.train_random_forest(X_train, y_train)
ae_model = model_training.DeepLearningTrainer.train_autoencoder(X_train, X_val)

# 4. 예측
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # (27, 64)
y_pred_rf = rf_model.predict(X_test_flat)           # (27,)
y_proba_rf = rf_model.predict_proba(X_test_flat)    # (27, 2)

# 5. 평가
metrics = evaluation.ModelEvaluator.compute_metrics(y_test, y_pred_rf, y_proba_rf)
# 출력: {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}

# 6. 결과 저장 (JSON)
evaluation.generate_evaluation_report(evaluation_results)
# 생성: evaluation_report.json
# 생성: results/model_randomforest_evaluation.json
# 생성: results/model_autoencoder_evaluation.json
"""


# ============================================================================
# 중요 포인트
# ============================================================================

"""
1. 데이터 형식 변환
   - 결과: 3D → 2D 변환 필수 (Classical ML)
   - LSTM: 3D 유지
   - Autoencoder: 2D 변환

2. 입력 정규화
   - 모든 모델: StandardScaler 적용
   - 범위: [-3, 3] (대략)

3. 라벨 변환
   - 메타데이터 'Data Label'에서 자동 변환
   - "정상" → 0
   - 기타 → 1

4. 이상치 탐지 임계값
   - Autoencoder/LSTM: 95 percentile
   - 조정 가능: config.py ANOMALY_THRESHOLD

5. 배치 처리
   - PyTorch DataLoader 사용
   - 기본 batch_size: 32
   - 메모리 효율성 위해 조정 가능

6. 가중치 저장
   - 모든 모델: models/ 디렉토리에 저장
   - RF/IF/SVM: .pkl (pickle)
   - Autoencoder/LSTM: .pt (PyTorch)
"""
