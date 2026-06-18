"""
ANOMALY DETECTION MODELS EXPANSION GUIDE

이상탐지를 위한 모델 추가 및 확장 가이드
"""

# ============================================================================
# 1. 기존 이상탐지 모델 (4개)
# ============================================================================

## 1.1 Isolation Forest
- **특징**: 트리 기반 이상탐지 알고리즘
- **원리**: 이상치를 고립시키는 데 필요한 분할 수가 적다는 개념
- **장점**: 고차원 데이터에서도 효율적, 충돌에 강함
- **단점**: 데이터 분포에 민감할 수 있음
- **사용**: `CLASSICAL_MODELS["IsolationForest"]`

## 1.2 One-Class SVM
- **특징**: 서포트 벡터 머신 기반의 단일 클래스 분류
- **원리**: 정상 데이터를 초평면으로 분리하고 비정상 데이터를 이상치로 판정
- **장점**: 고차원 데이터에서 좋은 성능
- **단점**: 커널 선택과 하이퍼파라미터 튜닝이 중요
- **사용**: `CLASSICAL_MODELS["OneClassSVM"]`

## 1.3 Local Outlier Factor (LOF)
- **특징**: 밀도 기반 이상탐지 알고리즘
- **원리**: 각 점의 국소 밀도와 이웃들의 밀도를 비교
- **장점**: 국소적 이상치를 잘 감지
- **단점**: 계산 비용이 높을 수 있음
- **사용**: `CLASSICAL_MODELS["LocalOutlierFactor"]`

## 1.4 Elliptic Envelope
- **특징**: 공분산 행렬 추정 기반
- **원리**: 정상 데이터의 최소 공분산 결정(MCD) 추정
- **장점**: 로부스트하고 빠른 계산
- **단점**: 고차원 데이터에서는 성능 저하 가능
- **사용**: `CLASSICAL_MODELS["EllipticEnvelope"]`


# ============================================================================
# 2. 새로운 이상탐지 모델 (5개)
# ============================================================================

## 2.1 Robust Covariance
- **특징**: 견고한 공분산 행렬 추정
- **원리**: Elliptic Envelope와 유사하나 더 강건한 추정
- **장점**: 이상치에 덜 민감한 공분산 추정
- **단점**: 계산 복잡도가 높을 수 있음
- **사용**: `config.CLASSICAL_MODELS["RobustCovariance"]`
- **학습**: `ClassicalModelTrainer.train_robust_covariance()`

## 2.2 Minimum Covariance Determinant (MinCovDet)
- **특징**: 최소 공분산 행렬식 기반 이상탐지
- **원리**: 최소 행렬식을 가지는 부분집합을 찾아 이상치를 식별
- **장점**: 통계적으로 견고, 잘 알려진 방법
- **단점**: 고차원에서 계산 비용 증가
- **사용**: `config.CLASSICAL_MODELS["MinCovDet"]`
- **학습**: `ClassicalModelTrainer.train_min_cov_det()`

## 2.3 KMeans Anomaly Detection
- **특징**: K-Means 클러스터링 기반 이상탐지
- **원리**: 각 점과 가장 가까운 클러스터 중심까지의 거리로 이상치 판정
- **장점**: 빠른 계산, 해석이 쉬움
- **단점**: 클러스터 수 선택이 중요
- **사용**: `config.CLASSICAL_MODELS["KMeansAnomaly"]`
- **학습**: `ClassicalModelTrainer.train_kmeans_anomaly()`
- **파라미터**:
  - n_clusters: 클러스터 개수 (기본값: 5)
  - contamination: 이상치 비율 (기본값: 0.1)

## 2.4 PCA Anomaly Detection
- **특징**: PCA 재구성 오차 기반 이상탐지
- **원리**: PCA로 압축했을 때 복원 오차가 크면 이상치로 판정
- **장점**: 선형 구조 파악 용이, 차원 축소와 동시에 이상탐지
- **단점**: 비선형 패턴 감지에는 제한적
- **사용**: `config.CLASSICAL_MODELS["PCAAnomaly"]`
- **학습**: `ClassicalModelTrainer.train_pca_anomaly()`
- **파라미터**:
  - n_components: 유지할 분산 비율 (기본값: 0.95 = 95%)
  - contamination: 이상치 비율 (기본값: 0.1)

## 2.5 DBSCAN (Density-Based Spatial Clustering)
- **특징**: 밀도 기반 군집화로 이상치 식별
- **원리**: eps 거리 내에 최소 샘플 수가 없으면 noise(이상치)로 판정
- **장점**: 임의의 형태 클러스터 인식, 노이즈 자동 식별
- **단점**: 밀도 변화가 큰 데이터에서 성능 저하
- **사용**: `config.CLASSICAL_MODELS["DBSCAN"]`
- **학습**: `ClassicalModelTrainer.train_dbscan()`
- **파라미터**:
  - eps: 근방 거리 (기본값: 0.5)
  - min_samples: 최소 샘플 수 (기본값: 5)


# ============================================================================
# 3. 모든 이상탐지 모델 비교
# ============================================================================

| 모델 | 입력 | 학습 방식 | 특징 | 적합한 데이터 |
|------|------|---------|------|-------------|
| IsolationForest | 수치형 | Unsupervised | 트리 기반, 고차원 효율적 | 고차원, 이상치 희귀 |
| OneClassSVM | 수치형 | Unsupervised | 경계 학습 | 경계가 명확한 정상 데이터 |
| LocalOutlierFactor | 수치형 | Unsupervised | 밀도 기반 | 국소적 밀도 이상 감지 |
| EllipticEnvelope | 수치형 | Unsupervised | 공분산 기반 | 가우시안 분포 데이터 |
| **RobustCovariance** | 수치형 | Unsupervised | 견고한 공분산 | 이상치 영향 최소화 필요 |
| **MinCovDet** | 수치형 | Unsupervised | MCD 기반 | 통계적 견고성 필요 |
| **KMeansAnomaly** | 수치형 | Unsupervised | 거리 기반 | 구형 클러스터 데이터 |
| **PCAAnomaly** | 수치형 | Unsupervised | 재구성 오차 | 선형 구조가 있는 데이터 |
| **DBSCAN** | 수치형 | Unsupervised | 밀도 기반 | 임의의 형태 클러스터 |


# ============================================================================
# 4. 사용 예시
# ============================================================================

## 4.1 기본 비교 (3개 모델)
```python
from anomaly_detection_comparison import example_1_basic_anomaly_detection

result = example_1_basic_anomaly_detection()
```

## 4.2 기존 vs 신규 모델 비교
```python
from anomaly_detection_comparison import example_2_original_vs_new_models

result = example_2_original_vs_new_models()
```

## 4.3 모든 9개 모델 비교
```python
from anomaly_detection_comparison import example_3_all_anomaly_models

result = example_3_all_anomaly_models()
```

## 4.4 다양한 윈도우 크기로 비교
```python
from anomaly_detection_comparison import example_4_different_window_sizes

result = example_4_different_window_sizes()
```

## 4.5 빠른 테스트 (권장)
```python
from anomaly_detection_comparison import example_5_quick_test

result = example_5_quick_test()
```

## 4.6 커스텀 비교
```python
from anomaly_detection_comparison import run_anomaly_detection_comparison

result = run_anomaly_detection_comparison(
    models_to_test=["IsolationForest", "PCAAnomaly", "KMeansAnomaly"],
    window_size=128,
    output_dir="custom_comparison"
)
```


# ============================================================================
# 5. 출력 결과
# ============================================================================

## 5.1 생성되는 파일

anomaly_detection_results/
├── anomaly_detection_comparison.json      # 상세 결과 JSON
├── anomaly_detection_metrics.csv          # 메트릭 CSV
├── anomaly_detection_comparison.png       # 성능 비교 그래프
└── anomaly_detection_heatmap.png          # 성능 히트맵

## 5.2 평가 메트릭

- **Precision**: TP / (TP + FP) - 예측된 이상치 중 실제 이상치 비율
- **Recall**: TP / (TP + FN) - 실제 이상치 중 감지된 비율
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **ROC-AUC**: 다양한 임계값에서의 성능 종합 평가
- **Confusion Matrix**: TP, TN, FP, FN 분석


# ============================================================================
# 6. 모델 선택 가이드
# ============================================================================

### 빠른 성능이 필요한 경우
→ IsolationForest, KMeansAnomaly

### 통계적 견고성이 필요한 경우
→ MinCovDet, RobustCovariance

### 다양한 형태의 이상치를 감지해야 하는 경우
→ DBSCAN, LocalOutlierFactor

### 선형 구조를 가진 데이터
→ PCAAnomaly

### 경계가 명확한 정상 데이터
→ OneClassSVM

### 전반적으로 균형잡힌 성능
→ EllipticEnvelope, IsolationForest


# ============================================================================
# 7. 마이그레이션 가이드 (기존 코드 업데이트)
# ============================================================================

### 기존 코드
```python
from anomaly_detection.model_training import ClassicalModelTrainer

trainer = ClassicalModelTrainer()
model = trainer.train_isolation_forest(X_train)
```

### 새로운 모델 추가
```python
from anomaly_detection.model_training import ClassicalModelTrainer

trainer = ClassicalModelTrainer()

# 기존 모델들
original_model = trainer.train_isolation_forest(X_train)

# 새로운 모델들
new_model1 = trainer.train_robust_covariance(X_train)
new_model2 = trainer.train_pca_anomaly(X_train)
new_model3 = trainer.train_kmeans_anomaly(X_train)
```


# ============================================================================
# 8. 성능 최적화 팁
# ============================================================================

1. **데이터 정규화**: 모든 모델은 정규화된 입력값에서 더 잘 작동
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_data)
   ```

2. **하이퍼파라미터 튜닝**: 각 모델의 contamination 파라미터 조정
   ```python
   config.CLASSICAL_MODELS["IsolationForest"]["contamination"] = 0.05  # 5% 이상치
   ```

3. **앙상블**: 여러 모델의 결과를 조합
   ```python
   # 여러 모델의 이상 점수를 평균내기
   scores = []
   for model in [model1, model2, model3]:
       _, score = get_anomaly_predictions(model, X_test)
       scores.append(score)
   ensemble_score = np.mean(scores, axis=0)
   ```

4. **샘플 크기 실험**: FFT_SAMPLE_SIZES를 조정하여 최적의 윈도우 크기 찾기
   ```python
   results = run_anomaly_detection_comparison(
       window_size=128  # 다양한 값 시도: 32, 64, 128, 256
   )
   ```


# ============================================================================
# 9. 문제 해결
# ============================================================================

### 문제 1: 낮은 Recall 점수
→ 해결책: contamination 파라미터를 높이거나 다른 모델 사용

### 문제 2: 모든 데이터를 정상/이상으로 분류
→ 해결책: 데이터 정규화 확인, 샘플 크기 및 클러스터 개수 조정

### 문제 3: 메모리 부족
→ 해결책: 배치 처리 또는 데이터 다운샘플링 사용

### 문제 4: 느린 학습
→ 해결책: IsolationForest나 KMeansAnomaly 사용 (더 빠름)


# ============================================================================
# 10. 다음 스텝
# ============================================================================

1. ✅ 새로운 이상탐지 모델 5개 추가 완료
2. ✅ 비교 스크립트 및 시각화 추가 완료
3. 🔄 데이터로 모든 모델 실행/비교:
   ```bash
   python anomaly_detection_comparison.py
   ```
4. 🔄 최고 성능 모델 선정 및 프로덕션 배포
5. 🔄 실시간 inference에 적용


# ============================================================================
# 참고사항
# ============================================================================

- 모든 이상탐지 모델은 unsupervised learning 문제
- 훈련 데이터는 정상 데이터로만 구성되어야 함
- 평가 시에는 정상/이상 라벨이 필요
- 모델 성능은 데이터 특성에 따라 크게 달라질 수 있음
- 여러 모델 조합 (앙상블)이 단일 모델보다 좋을 수 있음

"""
