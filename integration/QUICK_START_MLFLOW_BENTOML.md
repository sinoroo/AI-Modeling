# MLflow + BentoML 빠른 시작 가이드

> 모델 학습부터 배포까지 5분 안에 완료하기

## 📋 사전 조건

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데이터 확인 (또는 샘플 데이터 생성)
python main.py --generate-data --train
```

---

## 🎯 Step 1: 모델 학습 + MLflow 추적 (2분)

```bash
python main.py --train
```

**무슨 일이 일어나나요?**
- ✅ 모델 5개 학습 (RandomForest, IsolationForest, OneClassSVM, Autoencoder, LSTM)
- ✅ MLflow에 자동 로깅
- ✅ 피처 스키마 및 통계 저장
- ✅ 모델 평가 결과 저장

**출력 예:**
```
[MLflow] Started run: training_run (ID: a1b2c3d4e5f6...)
[FeatureStore] Schema created with 5 features
[FeatureStore] Statistics computed for 5 features
[MLflow] Logged 42 parameters
[MLflow] Logged 20 metrics
✓ TRAINING PIPELINE COMPLETE
```

---

## 📊 Step 2: MLflow 대시보드 확인 (1분)

```bash
# MLflow UI 시작
mlflow ui --backend-store-uri file:./mlruns
```

**브라우저에서 확인:**
1. `http://localhost:5000` 열기
2. "anomaly_detection" 실험 클릭
3. 최신 run 확인:
   - 📈 메트릭 그래프
   - ⚙️ 하이퍼파라미터
   - 📁 모델 아티팩트

### 빠른 비교

```python
python mlflow_bentoml_example.py --compare

# 출력:
# Rank  Model             Metric          Run ID
# 1     RandomForest      0.9421          a1b2c3d4e5f6...
# 2     Autoencoder       0.9156          b2c3d4e5f6g7...
# 3     LSTM              0.8932          c3d4e5f6g7h8...
```

---

## 🔧 Step 3: BentoML 서비스 설정 (1분)

```bash
# 방법 1: 미리 준비된 예제 스크립트 실행
python mlflow_bentoml_example.py --setup

# 방법 2: Python으로 수동 설정
python
```

```python
from anomaly_detection import bentoml_service, feature_store
import pickle

# 서비스 초기화
service = bentoml_service.AnomalyDetectionService()

# 모델 로드
service.load_model("RandomForest", "models/randomforest_model.pkl")
service.load_preprocessor("models/preprocessor.pkl")

# 피처 스토어 로드
fs = feature_store.FeatureStore()
service.load_feature_store(fs)

# 준비 완료
service.set_ready(True)
print("✅ Service ready!")
```

---

## 🚀 Step 4: 로컬 서버 시작 (1분)

```bash
# BentoML 서비스 빌드 및 실행
bentoml serve anomaly_detection_service:latest --port 3000
```

**출력:**
```
2024-01-15 10:30:00 INFO     Starting server process [12345]
2024-01-15 10:30:01 INFO     Uvicorn running on http://0.0.0.0:3000
```

---

## 🔮 Step 5: API 테스트 (1분)

### 터미널에서 테스트

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
```

### Python으로 테스트

```python
import numpy as np
import requests

# 테스트 데이터
test_data = np.random.normal(50, 10, (5)).tolist()

# API 호출
response = requests.post(
    "http://localhost:3000/predict",
    json={
        "data": [test_data],
        "model_name": "RandomForest"
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Score: {result['score']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")
```

---

## 📊 주요 엔드포인트

| 엔드포인트 | 메서드 | 설명 | 예제 |
|-----------|--------|------|------|
| `/predict` | POST | 단일 예측 | `{"data": [[50,51,49,52,50]], "model_name": "RandomForest"}` |
| `/predict_batch` | POST | 배치 예측 | `{"data": [[[50,51,49,52,50]]], "model_name": "RandomForest"}` |
| `/model_info` | GET | 모델 정보 | - |
| `/health_check` | GET | 상태 확인 | - |

---

## 🔄 전체 자동화 스크립트

한 줄로 모든 것을 실행하려면:

```bash
# 1. 학습
python main.py --train > /dev/null 2>&1 &

# 2. MLflow 시작
mlflow ui --backend-store-uri file:./mlruns &

# 3. BentoML 서비스 시작
bentoml serve anomaly_detection_service:latest --port 3000 &

echo "✅ All services started!"
echo "📊 MLflow UI: http://localhost:5000"
echo "🔮 API: http://localhost:3000"
echo "📝 Test: curl -X POST http://localhost:3000/predict ..."
```

또는 Python 스크립트로:

```python
# quick_start.py
import subprocess
import time

print("🚀 Starting MLflow + BentoML...")

# 학습
print("\n[1/3] Training model with MLflow...")
subprocess.run(["python", "main.py", "--train"], capture_output=True)

# MLflow UI
print("[2/3] Starting MLflow UI...")
mlflow_proc = subprocess.Popen(
    ["mlflow", "ui", "--backend-store-uri", "file:./mlruns"],
    stdout=subprocess.DEVNULL
)
time.sleep(2)

# BentoML
print("[3/3] Starting BentoML service...")
bentoml_proc = subprocess.Popen(
    ["bentoml", "serve", "anomaly_detection_service:latest", "--port", "3000"],
    stdout=subprocess.DEVNULL
)
time.sleep(2)

print("\n✅ All services ready!")
print("📊 MLflow UI: http://localhost:5000")
print("🔮 API: http://localhost:3000/docs")
print("\nPress Ctrl+C to stop all services...")

try:
    mlflow_proc.wait()
except KeyboardInterrupt:
    mlflow_proc.terminate()
    bentoml_proc.terminate()
    print("\n👋 Services stopped")
```

실행:
```bash
python quick_start.py
```

---

## 💡 유용한 팁

### MLflow에서 모델 내려받기

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.download_artifacts(
    run_id="a1b2c3d4e5f6...",
    path="sklearn_models",
    dst_path="./downloaded_models"
)
```

### 최고 성능 모델로 예측

```python
from anomaly_detection import mlflow_utils

tracker = mlflow_utils.MLflowTracker("anomaly_detection")
best_run = tracker.get_best_run("f1", order_by="DESC")

# 다운로드 및 로드
# ...
```

### 배포를 위한 도커 이미지

```bash
# Docker 이미지 빌드
bentoml containerize anomaly_detection_service:latest

# 실행
docker run -p 3000:3000 anomaly_detection_service:latest

# Docker Hub에 푸시
docker tag anomaly_detection_service:latest your-username/anomaly-detection:latest
docker push your-username/anomaly-detection:latest
```

---

## 🆘 문제 해결

### 포트가 이미 사용 중

```bash
# 다른 포트 사용
bentoml serve anomaly_detection_service:latest --port 3001

# 프로세스 확인 및 종료
lsof -i :3000
kill -9 <PID>
```

### 모델 캐시 문제

```bash
# BentoML 캐시 정리
rm -rf ~/bentoml

# MLflow 실행 삭제
rm -rf mlruns
```

### 메모리 부족

```python
# 작은 배치 크기로 예측
batch_size = 10  # 대신 100
for i in range(0, len(data), batch_size):
    results = service.predict_batch(data[i:i+batch_size])
```

---

## ✅ 체크리스트

- [ ] `pip install -r requirements.txt` 실행
- [ ] `python main.py --train` 학습 완료
- [ ] MLflow UI에서 결과 확인
- [ ] `python mlflow_bentoml_example.py --setup` 서비스 설정
- [ ] BentoML API 테스트
- [ ] 예측 정상 작동 확인

---

## 📚 더 알아보기

- [상세 가이드](./MLFLOW_BENTOML_GUIDE.md)
- [MLflow 공식 문서](https://mlflow.org)
- [BentoML 공식 문서](https://docs.bentoml.org)

---

**5분 완료! 🎉**

이제 REST API를 통해 이상 탐지 모델을 제공할 수 있습니다.

```python
# 사용자 지정 애플리케이션에서 사용
import requests

response = requests.post("http://localhost:3000/predict", json={
    "data": [[50, 51, 49, 52, 50]],
    "model_name": "RandomForest"
})
result = response.json()
print(f"Is Anomaly: {result['prediction'] == 1}")
```

Happy Detecting! 🎯
