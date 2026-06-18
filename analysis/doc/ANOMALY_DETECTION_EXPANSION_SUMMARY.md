"""
ANOMALY DETECTION MODEL EXPANSION - SUMMARY

이상탐지 모델 확장 및 비교 분석 - 변경 요약
Date: 2024-12-19
"""

# ============================================================================
# 변경 사항 요약
# ============================================================================

## 1. 새로운 이상탐지 모델 추가 (5개)

기존: 4개 모델
- IsolationForest
- OneClassSVM
- LocalOutlierFactor
- EllipticEnvelope

→ 신규 추가: 5개 모델
- **RobustCovariance**: 견고한 공분산 행렬 추정
- **MinCovDet**: 최소 공분산 행렬식 기반
- **KMeansAnomaly**: K-Means 거리 기반
- **PCAAnomaly**: PCA 재구성 오차 기반
- **DBSCAN**: 밀도 기반 공간 군집화

**총 9개의 이상탐지 모델로 확장됨**


## 2. 수정된 파일

### 2.1 `anomaly_detection/config.py`
**변경 내용:**
- CLASSICAL_MODELS 딕셔너리 업데이트
- 새로운 5개 모델의 기본 파라미터 추가
- 각 모델별 하이퍼파라미터 설정:
  - RobustCovariance: contamination=0.1
  - MinCovDet: contamination=0.1, random_state
  - KMeansAnomaly: n_clusters=5, contamination=0.1
  - PCAAnomaly: n_components=0.95, contamination=0.1
  - DBSCAN: eps=0.5, min_samples=5

**코드 예시:**
```python
CLASSICAL_MODELS = {
    # ... 기존 모델들 ...
    
    # 새로운 이상탐지 모델들
    "RobustCovariance": {"contamination": 0.1},
    "MinCovDet": {"contamination": 0.1, "random_state": RANDOM_SEED},
    "KMeansAnomaly": {
        "n_clusters": 5,
        "n_init": 10,
        "random_state": RANDOM_SEED,
        "contamination": 0.1,
    },
    "PCAAnomaly": {
        "n_components": 0.95,
        "contamination": 0.1,
    },
    "DBSCAN": {
        "eps": 0.5,
        "min_samples": 5,
    },
}
```


### 2.2 `anomaly_detection/model_training.py`
**변경 내용:**
- 새로운 sklearn 모듈 임포트:
  - MinCovDet (from sklearn.covariance)
  - PCA (from sklearn.decomposition)
  - KMeans, DBSCAN (from sklearn.cluster)

- 5개의 새로운 학습 메서드 추가:
  1. `train_robust_covariance()`
  2. `train_min_cov_det()`
  3. `train_kmeans_anomaly()`
  4. `train_pca_anomaly()`
  5. `train_dbscan()`

**새로운 메서드 특징:**
- 모두 unsupervised learning
- 이상치 스코어 반환
- 임계값(threshold) 자동 계산
- 훈련된 모델에 추가 정보 저장 (anomaly_threshold, 등)


### 2.3 새 파일: `anomaly_detection_comparison.py` (생성)
**목적**: 모든 이상탐지 모델의 성능 비교 및 분석

**주요 기능:**
1. `run_anomaly_detection_comparison()`: 메인 비교 함수
   - 모든 모델 자동 훈련
   - 메트릭 계산: Precision, Recall, F1, ROC-AUC
   - 결과 시각화 및 저장

2. `get_anomaly_predictions()`: 모든 모델 유형 지원
   - 통일된 인터페이스로 예측 수행
   - 이상치 스코어 반환

3. `calculate_anomaly_detection_metrics()`: 표준화된 평가
   - Confusion Matrix 포함
   - ROC-AUC 계산

4. 5개의 예시 함수:
   - `example_1_basic_anomaly_detection()`: 3개 기본 모델
   - `example_2_original_vs_new_models()`: 기존 vs 신규 모델 비교
   - `example_3_all_anomaly_models()`: 9개 모든 모델
   - `example_4_different_window_sizes()`: 윈도우 크기별 비교
   - `example_5_quick_test()`: 빠른 테스트 (권장)

**출력 결과:**
- JSON 보고서: anomaly_detection_comparison.json
- CSV 메트릭: anomaly_detection_metrics.csv
- 비교 그래프: anomaly_detection_comparison.png
- 성능 히트맵: anomaly_detection_heatmap.png


### 2.4 새 파일: `ANOMALY_DETECTION_MODELS_GUIDE.md` (생성)
**목적**: 새로운 이상탐지 모델들의 상세 가이드

**포함 내용:**
1. 기존 4개 모델 상세 설명
2. 새로운 5개 모델 상세 설명
   - 특징, 원리, 장단점
   - 사용 방법, 파라미터

3. 모델 선택 가이드
4. 성능 비교 테이블
5. 사용 예시
6. 문제 해결 방법
7. 최적화 팁


# ============================================================================
# 모델 크기 비교 (기존 vs 신규)
# ============================================================================

| 분류 | 기존 모델 수 | 신규 모델 수 | 총합 |
|------|----------|----------|-----|
| 이상탐지 (Anomaly Detection) | 4 | 5 | **9** |
| 분류 (Classification) | 10 | 0 | 10 |
| 딥러닝 | 2 | 0 | 2 |
| **전체 모델** | **16** | **5** | **21** |

**이상탐지 모델 세부:**
- 기존: IsolationForest, OneClassSVM, LocalOutlierFactor, EllipticEnvelope
- 신규: RobustCovariance, MinCovDet, KMeansAnomaly, PCAAnomaly, DBSCAN


# ============================================================================
# 사용 방법
# ============================================================================

## 1. 빠른 시작 (권장)

```python
from anomaly_detection_comparison import example_5_quick_test

# 실행
result = example_5_quick_test()
```

결과는 `anomaly_detection_results_quick/` 디렉토리에 저장됨


## 2. 기존 vs 신규 모델 비교

```python
from anomaly_detection_comparison import example_2_original_vs_new_models

# 기존 4개 모델과 신규 5개 모델 비교
result = example_2_original_vs_new_models()
```

결과:
- `anomaly_detection_results_original/` - 기존 모델 결과
- `anomaly_detection_results_new/` - 신규 모델 결과


## 3. 모든 9개 모델 비교

```python
from anomaly_detection_comparison import example_3_all_anomaly_models

# 모든 이상탐지 모델 훈련 및 비교
result = example_3_all_anomaly_models()
```


## 4. 커스텀 비교

```python
from anomaly_detection_comparison import run_anomaly_detection_comparison

# 특정 모델만 선택하여 비교
result = run_anomaly_detection_comparison(
    models_to_test=["IsolationForest", "PCAAnomaly", "KMeansAnomaly"],
    window_size=64,
    output_dir="my_custom_comparison"
)
```


## 5. 명령줄 실행

```bash
# 기본 실행 (example_5_quick_test 실행)
python anomaly_detection_comparison.py
```


# ============================================================================
# 평가 메트릭 설명
# ============================================================================

생성되는 CSV 및 JSON에는 다음 메트릭 포함:

### Val_* (검증 데이터셋)
- Val_F1: 검증 데이터셋 F1 점수
- Val_Precision: 검증 데이터셋 정밀도
- Val_Recall: 검증 데이터셋 재현율
- Val_ROC_AUC: 검증 데이터셋 ROC-AUC

### Test_* (테스트 데이터셋)
- Test_F1: **테스트 데이터셋 F1 점수 (최종 성능)**
- Test_Precision: 테스트 데이터셋 정밀도
- Test_Recall: 테스트 데이터셋 재현율
- Test_ROC_AUC: 테스트 데이터셋 ROC-AUC

### Confusion Matrix 값
- TP (True Positive): 올바르게 감지된 이상치
- TN (True Negative): 올바르게 분류된 정상
- FP (False Positive): 잘못 감지된 이상치 (정상을 이상으로)
- FN (False Negative): 놓친 이상치 (이상을 정상으로)


# ============================================================================
# 주요 개선사항
# ============================================================================

### 1. 모델 다양성 증가
- 이전: 4개 이상탐지 모델
- 현재: 9개 이상탐지 모델 (125% 증가)

### 2. 표준화된 비교 인터페이스
- 모든 모델이 동일한 평가 프레임워크
- 일관된 메트릭 계산
- 자동 시각화 생성

### 3. 포괄적인 문서
- 각 모델별 상세 설명
- 모델 선택 가이드
- 성능 최적화 팁
- 문제 해결 가이드

### 4. 예제 시나리오
- 5개 다양한 사용 사례 제공
- 즉시 실행 가능한 코드
- 초보자~고급사용자 모두 적합


# ============================================================================
# 데이터 흐름
# ============================================================================

```
Data Loading
    ↓
Preprocessing (window_size 적용)
    ↓
Training & Evaluation
    ├── IsolationForest
    ├── OneClassSVM
    ├── LocalOutlierFactor
    ├── EllipticEnvelope
    ├── RobustCovariance ← NEW
    ├── MinCovDet ← NEW
    ├── KMeansAnomaly ← NEW
    ├── PCAAnomaly ← NEW
    └── DBSCAN ← NEW
    ↓
Metrics Calculation
    ├── Precision
    ├── Recall
    ├── F1 Score
    └── ROC-AUC
    ↓
Results Generation
    ├── JSON Report
    ├── CSV Metrics
    ├── Comparison Graph
    └── Performance Heatmap
```


# ============================================================================
# 성능 기준 (예상)
# ============================================================================

일반적으로 다음과 같은 범위의 성능을 기대할 수 있습니다:

| 메트릭 | 저성능 | 중간 | 고성능 |
|------|------|-----|------|
| Precision | < 0.5 | 0.5-0.8 | > 0.8 |
| Recall | < 0.5 | 0.5-0.8 | > 0.8 |
| F1 Score | < 0.5 | 0.5-0.8 | > 0.8 |
| ROC-AUC | < 0.6 | 0.6-0.8 | > 0.8 |

성능은 다음 요소에 따라 달라집니다:
- 데이터 특성 (분포, 노이즈, 차원)
- 이상치비율 (contamination)
- 윈도우 크기
- 모델 하이퍼파라미터


# ============================================================================
# 다음 스텝
# ============================================================================

### Phase 1: 테스트 및 비교 (지금)
1. ✅ 새로운 모델 5개 추가 완료
2. ✅ 비교 스크립트 생성 완료
3. 🔄 **지금 실행할 단계:**
   ```bash
   python anomaly_detection_comparison.py
   ```

### Phase 2: 결과 분석
1. 생성된 그래프 및 메트릭 검토
2. 최고 성능 모델 식별
3. 모델 조합 (앙상블) 고려

### Phase 3: 프로덕션 배포
1. 최고 성능 모델 선정
2. 실시간 inference 통합
3. 모니터링 및 재학습 계획


# ============================================================================
# 문의 사항
# ============================================================================

- 모델 선택 방법: ANOMALY_DETECTION_MODELS_GUIDE.md 참고
- 성능 최적화 팁: ANOMALY_DETECTION_MODELS_GUIDE.md 섹션 8 참고
- 문제 해결: ANOMALY_DETECTION_MODELS_GUIDE.md 섹션 9 참고
- 새로운 모델 추가: model_training.py에 새 메서드 작성


# ============================================================================
# 변경 파일 목록
# ============================================================================

**수정된 파일:**
1. anomaly_detection/config.py - 새 모델 파라미터 추가
2. anomaly_detection/model_training.py - 5개 새 학습 메서드 추가

**생성된 파일:**
1. anomaly_detection_comparison.py - 비교 및 분석 스크립트
2. ANOMALY_DETECTION_MODELS_GUIDE.md - 상세 가이드
3. ANOMALY_DETECTION_EXPANSION_SUMMARY.md - 이 파일

"""
