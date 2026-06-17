# MLflow + BentoML 통합 완료

## 📦 추가된 기능

이제 프로젝트에 다음 기능들이 추가되었습니다:

### 1. **MLflow 통합** 🔍
- **모델 학습 추적**: 모든 모델 학습 중에 파라미터, 메트릭, 아티팩트 자동 로깅
- **실험 관리**: 학습 실행 비교 및 최고 성능 모델 찾기
- **모델 레지스트리**: 모델 버전 관리 및 등록

### 2. **Feature Store** 📊
- **피처 스키마**: 피처 정의, 타입, 메타데이터 관리
- **피처 통계**: 평균, 표준편차, 최소/최대값 등 추적
- **데이터 검증**: 입력 데이터가 스키마를 따르는지 확인

### 3. **BentoML 서빙** 🚀
- **REST API**: 학습된 모델을 HTTP API로 제공
- **배치 예측**: 단일 및 대량 데이터에 대한 예측
- **모델 정보**: 로드된 모델 및 피처 정보 조회
- **상태 확인**: Health check 엔드포인트

---

## 📂 추가된 파일

```
anomaly_detection/
├── feature_store.py          # Feature Store 관리
├── mlflow_utils.py           # MLflow 추적 유틸리티  
└── bentoml_service.py        # BentoML 서빙 서비스

프로젝트 루트/
├── bentofile.yaml            # BentoML 설정
├── mlflow_bentoml_example.py # 통합 예제
├── MLFLOW_BENTOML_GUIDE.md   # 상세 가이드
├── QUICK_START_MLFLOW_BENTOML.md  # 빠른 시작
└── requirements.txt          # 의존성 (mlflow, bentoml 추가)

feature_store/
├── feature_schema.json       # 피처 스키마 (생성됨)
└── feature_stats.pkl         # 피처 통계 (생성됨)

mlruns/                        # MLflow 실험 저장소 (생성됨)
```

---

## 🚀 빠른 시작

### 1️⃣ 의존성 설치
```bash
pip install -r requirements.txt
```

### 2️⃣ 모델 학습 (MLflow 추적 포함)
```bash
python main.py --train
```

### 3️⃣ MLflow 대시보드 확인
```bash
mlflow ui --backend-store-uri file:./mlruns
# 브라우저: http://localhost:5000
```

### 4️⃣ BentoML 서비스 시작
```bash
# 먼저 서비스 정보 확인
python mlflow_bentoml_example.py --setup

# 서비스 실행
bentoml serve anomaly_detection_service:latest
```

### 5️⃣ API 테스트
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[50, 51, 49, 52, 50]],
    "model_name": "RandomForest"
  }'
```

---

## 💻 주요 코드 예제

### MLflow 모델 추적

```python
from anomaly_detection import mlflow_utils, model_training

# MLflow 초기화
tracker = mlflow_utils.MLflowTracker("anomaly_detection")
run_id = tracker.start_run("training_run")

# 모델 학습 (자동 로깅)
models, results = model_training.train_all_models(
    X_train, y_train, X_val, y_val,
    mlflow_tracker=tracker
)

# 메트릭 로깅
tracker.log_metrics({"accuracy": 0.95, "f1": 0.92})

# 모델 등록
version = tracker.register_model(
    "runs:/RUN_ID/sklearn_models",
    "BestAnomalyDetectionModel"
)

tracker.end_run()
```

### Feature Store 사용

```python
from anomaly_detection import feature_store

# 초기화
fs = feature_store.FeatureStore()

# 스키마 생성
schema = fs.create_schema(
    feature_cols=["sensor_0", "sensor_1", "sensor_2"],
    label_col="label"
)

# 통계 계산
stats = fs.compute_statistics(data, feature_cols)

# 데이터 검증
if fs.validate_features(new_data):
    print("✓ Data is valid")
```

### BentoML API

```python
from anomaly_detection import bentoml_service
import numpy as np

# 서비스 초기화
service = bentoml_service.AnomalyDetectionService()
service.load_model("RandomForest", "models/randomforest_model.pkl")
service.set_ready(True)

# 단일 예측
test_data = np.random.normal(50, 10, (100, 5))
result = service.predict_single(test_data, "RandomForest")

# 배치 예측
batch_data = np.random.normal(50, 10, (10, 100, 5))
results = service.predict_batch(batch_data, "RandomForest")

# 모델 정보
info = service.get_model_info()
```

---

## 🎯 사용 시나리오

### 시나리오 1: 학습 및 모니터링
```bash
# 1. 모델 학습
python main.py --train

# 2. MLflow에서 결과 모니터링
mlflow ui

# 3. 최고 성능 모델 확인
python mlflow_bentoml_example.py --compare
```

### 시나리오 2: 프로덕션 배포
```bash
# 1. 모델 선택 및 서빙 준비
python mlflow_bentoml_example.py --setup

# 2. BentoML 서비스 구동
bentoml serve anomaly_detection_service:latest

# 3. API에서 예측 사용
# curl, Python requests, 또는 프론트엔드에서 호출
```

### 시나리오 3: 모델 버전 관리
```python
# MLflow에서 최고 모델의 아티팩트 다운로드
client = mlflow.tracking.MlflowClient()
client.download_artifacts(run_id="abc123", path="sklearn_models")

# 새로운 모델 레지스트리에 등록
tracker.register_model("runs:/NEW_RUN_ID/...", "v2-AnomalyModel")
```

---

## 📊 MLflow 대시보드 활용

![MLflow 기능]
- **실험 탭**: 모든 학습 실행 비교
- **메트릭 그래프**: 정확도, 정밀도, 재현율 등 시각화
- **파라미터**: 모델별 하이퍼파라미터 비교
- **아티팩트**: 모델 파일, 피처 스키마, 통계 다운로드

---

## 🔧 주요 엔드포인트

### BentoML API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/predict` | POST | 단일 예측 |
| `/predict_batch` | POST | 배치 예측 |
| `/model_info` | GET | 모델 정보 |
| `/health_check` | GET | 상태 확인 |

### MLflow API

```python
# 최고 run 조회
best_run = tracker.get_best_run("f1", order_by="DESC")

# 상위 N개 비교
comparison = tracker.compare_runs("f1", top_n=5)

# 모델 레지스트리 조회
client = mlflow.tracking.MlflowClient()
models = client.list_registered_models()
```

---

## 🔄 워크플로우 다이어그램

```
데이터 준비
    ↓
python main.py --train
    ↓
    ├→ [MLflow] 자동 모델 학습 추적
    ├→ [FeatureStore] 피처 스키마 생성
    ├→ [FeatureStore] 통계 계산
    └→ 모델 저장
    ↓
mlflow ui  (결과 확인)
    ↓
python mlflow_bentoml_example.py --setup
    ↓
bentoml serve
    ↓
REST API (예측 사용)
```

---

## 📚 문서

- [상세 가이드](./MLFLOW_BENTOML_GUIDE.md) - MLflow와 BentoML 사용법
- [빠른 시작](./QUICK_START_MLFLOW_BENTOML.md) - 5분 안에 시작하기
- [orig requirements](./requirements.txt) - 의존성

---

## ✨ 개선 사항

| 항목 | 이전 | 현재 |
|-----|------|------|
| 모델 추적 | ❌ | ✅ MLflow |
| 피처 관리 | ❌ | ✅ Feature Store |
| API 제공 | ❌ | ✅ BentoML |
| 모델 비교 | 수동 | ✅ 자동 |
| 메타데이터 | 관리 안함 | ✅ 자동 저장 |
| 버전 관리 | ❌ | ✅ MLflow Registry |

---

## 🎓 학습 자료

- MLflow: https://mlflow.org/docs
- BentoML: https://docs.bentoml.org
- Feature Store: https://www.featurestore.org

---

## 💡 다음 단계

1. **모니터링 추가**: MLflow에 모델 평가 자동화
2. **CI/CD 연동**: GitHub Actions로 자동 학습 및 배포
3. **실시간 모니터링**: 프로덕션 모델 성능 모니터링
4. **A/B 테스팅**: 신규 모델 vs 기존 모델 비교
5. **데이터 드리프트 감지**: 입력 데이터 분포 변화 감지

---

**축하합니다! 🎉 MLflow와 BentoML이 통합되었습니다.**

이제 모델을 효과적으로 관리하고 프로덕션 환경에서 제공할 수 있습니다.

시작하려면:
```bash
pip install -r requirements.txt
python main.py --train
mlflow ui
```
