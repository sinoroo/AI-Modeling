# MLflow와 BentoML 통합 가이드

## 개요

이 가이드는 이상 탐지(Anomaly Detection) 프로젝트에서 **MLflow**와 **BentoML**을 사용하여 모델과 피처를 관리하는 방법을 설명합니다.

### 주요 기능

- **MLflow**: 모델 학습 추적, 메트릭 로깅, 모델 레지스트리 관리
- **BentoML**: 학습된 모델을 REST API로 제공하기 위한 서빙 프레임워크
- **Feature Store**: 피처 스키마 및 통계 관리

---

## 설치

```bash
pip install -r requirements.txt
```

필수 패키지:
- `mlflow >= 2.0.0`
- `bentoml >= 1.1.0`
- `torch >= 2.0.0`
- `scikit-learn >= 1.3.0`
- `pandas >= 2.0.0`

---

## 1. 모델 학습 with MLflow

### 기본 사용법

```bash
python main.py --train
```

학습 중에 자동으로 다음이 수행됩니다:

1. **MLflow 실험 추적 시작**
   - 실험명: `anomaly_detection`
   - 로컬 저장소: `./mlruns`

2. **피처 스키마 생성 및 저장**
   - `feature_store/feature_schema.json`: 피처 정보
   - `feature_store/feature_stats.pkl`: 피처 통계

3. **모델 학습 및 메트릭 로깅**
   - 모든 모델의 파라미터와 메트릭 자동 로깅
   - 평가 결과 저장

4. **모델 아티팩트 저장**
   - 학습된 모델 파일
   - 피처 처리 전처리기

### MLflow UI 확인

학습 후 MLflow 대시보드에서 실험을 확인할 수 있습니다:

```bash
mlflow ui
```

브라우저에서 `http://localhost:5000` 접속

**주의:** MLflow 최신 버전(2.0+)은 SQLite 데이터베이스 백엔드를 사용합니다.
```bash
# SQLite 데이터베이스로 UI 실행
mlflow ui
# → 자동으로 ./mlflow.db 인식

# 또는 명시적으로 지정:
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**확인 가능한 정보:**
- 실행 메트릭 (정확도, 정밀도, 재현율, F1 점수)
- 모델 파라미터 설정
- 학습 시간 및 데이터 정보
- 아티팩트 (모델 파일, 피처 스키마)

---

## 2. 모델 비교 및 최적화

### MLflow에서 최고 성능 모델 찾기

```python
from anomaly_detection import mlflow_utils

# MLflow Tracker 초기화 (SQLite 데이터베이스)
tracker = mlflow_utils.MLflowTracker(
    experiment_name="anomaly_detection",
    tracking_uri="sqlite:///mlflow.db"  # 데이터베이스 백엔드
)

# 또는 URI 생략 시 기본값 사용:
tracker = mlflow_utils.MLflowTracker(
    experiment_name="anomaly_detection"
    # tracking_uri 생략 시 'sqlite:///mlflow.db' 자동 사용
)

# F1 스코어로 상위 5개 모델 비교
comparison = tracker.compare_runs(metric_name="f1", top_n=5)
print(comparison)

# 최고 성능 모델 가져오기
best_run = tracker.get_best_run(metric_name="f1", order_by="DESC")
print(f"Best Run ID: {best_run['run_id']}")
print(f"Best F1 Score: {best_run['metrics.f1']}")
```

### 모델 레지스트리에 등록

모델 학습 후 자동으로 등록됩니다:

```python
# main.py의 training pipeline에서 자동 수행 (Step 10)
# 각 모델이 MLflow Model Registry에 등록됨

# 예: AnomalyDetection-RandomForest-v1
#     AnomalyDetection-IsolationForest-v1
#     AnomalyDetection-Autoencoder-v1
```

**Manual Registration (필요시):**

```python
# 구체적인 model URI로 등록
version = tracker.register_model(
    model_uri="runs:/YOUR_RUN_ID/sklearn_models",
    model_name="AnomalyDetection-RandomForest-v2"
)
print(f"Registered model version: {version}")
```

---

## 3. Feature Store 활용

### 피처 스키마 관리

```python
from anomaly_detection import feature_store

# Feature Store 초기화
fs = feature_store.FeatureStore(store_dir="feature_store")

# 스키마 생성
schema = fs.create_schema(
    feature_cols=["sensor_0", "sensor_1", "sensor_2", "sensor_3", "sensor_4"],
    label_col="label",
    metadata={
        "data_source": "pump_motor_system",
        "window_size": 100,
        "normalize_method": "standardize"
    }
)

# 스키마 저장
fs.export_schema_json("feature_schema.json")
```

### 피처 통계 추적

```python
import pandas as pd

# 데이터 로드
data = pd.read_csv("data/train.csv")

# 피처 통계 계산
stats = fs.compute_statistics(data, feature_cols)

# 통계 저장
fs.export_stats_json("feature_stats.json")

print(stats)
# 출력 예:
# {
#   "features": {
#     "sensor_0": {
#       "mean": 50.1,
#       "std": 9.8,
#       "min": 20.5,
#       "max": 89.3,
#       ...
#     },
#     ...
#   },
#   "computed_at": "2024-01-15T10:30:00",
#   "total_samples": 1000
# }
```

### 데이터 검증

```python
# 새로운 데이터가 스키마를 따르는지 확인
new_data = pd.read_csv("new_data.csv")

if fs.validate_features(new_data):
    print("✓ Data is valid")
else:
    print("✗ Data validation failed")
```

---

## 4. BentoML로 모델 제공

### BentoML 서비스 설정

```python
from anomaly_detection import bentoml_service
import pickle
import numpy as np

# 서비스 인스턴스 생성
service = bentoml_service.AnomalyDetectionService()

# 모델 로드
with open("models/randomforest_model.pkl", 'rb') as f:
    model = pickle.load(f)
service.load_model("RandomForest", "models/randomforest_model.pkl")

# 전처리기 로드
service.load_preprocessor("models/preprocessor.pkl")

# 서비스 준비
service.set_ready(True)

# 단일 예측
test_data = np.random.normal(loc=50, scale=10, size=(100, 5))
result = service.predict_single(test_data, model_name="RandomForest")
print(result)  # {'prediction': 0, 'score': 0.05, 'confidence': 0.5, 'model_used': 'RandomForest'}

# 배치 예측
batch_data = np.random.normal(loc=50, scale=10, size=(10, 100, 5))
results = service.predict_batch(batch_data, model_name="RandomForest")
print(results)  # {'predictions': [...], 'scores': [...], 'confidences': [...], 'model_used': 'RandomForest'}
```

### BentoML 서비스 빌드

```bash
# bentofile.yaml이 프로젝트 루트에 있어야 함

# 1. 서비스 빌드
bentoml build -f bentofile.yaml

# 2. 로컬에서 서비스 실행
bentoml serve anomaly_detection_service:latest --port 3000

# 3. Docker 이미지로 빌드 (선택사항)
bentoml containerize anomaly_detection_service:latest
docker run -p 3000:3000 anomaly_detection_service:latest
```

### REST API 테스트

```bash
# 단일 예측
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[50, 51, 49, 52, 50]],
    "model_name": "RandomForest"
  }'

# 응답:
# {"prediction": 0, "score": 0.05, "confidence": 0.5, "model_used": "RandomForest"}

# 모델 정보
curl http://localhost:3000/model_info

# Health Check
curl http://localhost:3000/health_check
```

---

## 5. 통합 워크플로우 예제

### 전체 파이프라인 실행

```bash
# 1. 학습 실행 (MLflow 추적 + 자동 모델 등록 포함)
python main.py --train

# 2. MLflow UI에서 결과 확인
mlflow ui
# → http://localhost:5000 자동 접속

# 3. BentoML 서비스 시작
python -m bentoml serve anomaly_detection_service:latest --port 3000

# 4. API 테스트
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[50, 51, 49, 52, 50, 10.5]],
    "model_name": "RandomForest"
  }'
```

### Python 코드 예제

```python
from anomaly_detection import mlflow_utils, feature_store, bentoml_service
import pickle
import numpy as np

# 1. MLflow로 모델 학습 추적 (main.py에서 자동 수행)
tracker = mlflow_utils.MLflowTracker(
    "anomaly_detection",
    tracking_uri="sqlite:///mlflow.db"  # SQLite 데이터베이스 백엔드
)
run_id = tracker.start_run("training_run")

# 학습 로직...
tracker.log_params({"lr": 0.001, "epochs": 50})
tracker.log_metrics({"accuracy": 0.95, "f1": 0.92})
tracker.log_sklearn_model(model, "model")

# Step 10: 모델 등록 (MLflow Model Registry)
model_uri = f"runs:/{run_id}/sklearn_models"
version = tracker.register_model(model_uri, "AnomalyDetection-RandomForest-v1")
print(f"Registered version: {version}")

tracker.end_run()

# 2. 최고 성능 모델 찾기
best_run = tracker.get_best_run("f1")
print(f"Best model: {best_run['run_id']}")

# 3. 피처 스토어 설정
fs = feature_store.FeatureStore()
schema = fs.create_schema(feature_cols=["f1", "f2", "f3", "f4", "f5", "f6"])

# 4. BentoML 서비스 설정
service = bentoml_service.AnomalyDetectionService()
service.load_model("RandomForest", "models/rf_model.pkl")
service.set_ready(True)

# 5. 예측
test_data = np.random.normal(50, 10, (64, 6))
result = service.predict_single(test_data)
print(f"Prediction: {result}")
```

---

## 6. 디렉토리 구조

학습 후 생성되는 디렉토리 구조:

```
project_root/
├── anomaly_detection/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── inference_serial.py
│   ├── feature_store.py          # ✨ NEW
│   ├── mlflow_utils.py            # ✨ NEW
│   └── bentoml_service.py         # ✨ NEW
├── models/
│   ├── randomforest_model.pkl
│   ├── isolationforest_model.pkl
│   ├── oneClasssvm_model.pkl
│   ├── autoencoder_model.pt
│   ├── lstm_model.pt
│   └── preprocessor.pkl
├── feature_store/                 # ✨ NEW
│   ├── feature_schema.json
│   └── feature_stats.pkl
├── mlruns/                        # ✨ NEW (MLflow 실험 저장소)
│   └── 0/                        # Experiment ID
│       └── [run_ids]/            # 각 실행의 데이터
├── main.py
├── mlflow_bentoml_example.py      # ✨ NEW
├── bentofile.yaml                 # ✨ NEW
└── requirements.txt
```

---

## 7. 트러블슈팅

### MLflow 연결 문제

```python
# 로컬 저장소 대신 원격 서버 사용
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### BentoML 포트 충돌

```bash
# 다른 포트 사용
bentoml serve anomaly_detection_service:latest --port 3001
```

### 모델 로딩 실패

```python
# 모델 경로 확인
import os
print(os.path.exists("models/randomforest_model.pkl"))

# 피처 스키마 확인
fs = feature_store.FeatureStore()
print(fs.get_schema())
```

---

## 8. 추가 리소스

- [MLflow 공식 문서](https://mlflow.org/docs)
- [BentoML 공식 문서](https://docs.bentoml.org/)
- [Feature Store 개념](https://www.featurestore.org/)

---

## 변경 이력

| 날짜 | 버전 | 변경 사항 |
|------|------|---------|
| 2024-01-15 | 1.0 | MLflow, BentoML, Feature Store 통합 |
