"""
📚 프로젝트 파일 & 폴더 완전 가이드

FFT 기반 이상탐지 시스템의 모든 파일과 폴더 설명
"""

# ============================================================================
# 📁 폴더 구조 개요
# ============================================================================

```
AI-Modeling-FFT/
├── 📂 anomaly_detection/          # 핵심 ML 모듈
├── 📂 data_new_format/             # 학습 데이터 (5클래스 진동 CSV)
├── 📂 feature_tables/              # ⭐ NEW: 표준화 특징 테이블 (윈도우=1행)
├── 📂 models/                      # 저장된 모델 파일
├── 📂 results/                     # 모델 평가 결과
├── 📂 util/                        # 유틸리티 스크립트
├── 📂 analysis/                    # 데이터 분석 스크립트
├── 📂 fft_comparison_results/      # FFT 비교 결과
├── 📂 anomaly_detection_results_quick/  # 이상탐지 비교 결과
├── 📂 integration/                 # MLflow/BentoML 통합
├── 📂 eda_results/                 # 탐색적 데이터 분석 결과
├── 📄 메인 스크립트들             # Python 파일들
└── 📄 가이드 문서들               # Markdown 파일들
```

## 🧭 두 가지 파이프라인 (중요)

이 프로젝트에는 **독립적인 두 갈래**의 파이프라인이 있습니다.

```
[A] 레거시 파이프라인 (윈도우 원시신호 기반, 15개 모델)
    main.py → anomaly_detection/model_training.py → evaluation.py

[B] 표준화 파이프라인 (특징 테이블 기반, 5클래스 분류) ⭐ NEW
    build_feature_table.py        (원시 CSV → 표준 특징 테이블)
        └─ anomaly_detection/feature_extraction.py (공용 특징 추출)
    train_from_feature_table.py   (특징 테이블 → 5클래스 분류 + 이상탐지)
        └─ sklearn 직접 사용 (model_training.py 와 무관)
    util/test_serial_inference.py (학습된 5클래스 모델로 실시간 추론)
        └─ anomaly_detection/feature_extraction.py (동일 특징 보장)
```

> ⚠️ `train_from_feature_table.py` 는 `anomaly_detection.model_training` 을
> **사용하지 않습니다**. sklearn 을 직접 호출하는 독립 학습 스크립트입니다.
> 두 파이프라인을 잇는 공통 기반은 `feature_extraction.py` 입니다.


# ============================================================================
# 🔧 핵심 모듈: anomaly_detection/
# ============================================================================

이상탐지 시스템의 모든 기능을 담당하는 Python 패키지입니다.

## 📄 anomaly_detection/__init__.py
**기능**: 패키지 초기화 및 모듈 임포트 관리
**역할**: 
- 모든 하위 모듈을 import하여 통일된 인터페이스 제공
- 패키지 초기화 시 자동으로 필요한 모듈 로드
**사용**: 
```python
from anomaly_detection import config, model_training, evaluation
```

---

## 📄 anomaly_detection/config.py ⭐ 핵심
**기능**: 전체 시스템의 설정 및 파라미터 관리
**주요 설정**:
```
✓ 데이터 경로 설정 (TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR)
✓ FFT 샘플 크기 (FFT_SAMPLE_SIZES = [32, 64, 128, 256])
✓ 전처리 설정 (WINDOW_SIZE, OVERLAP, NORMALIZE_METHOD)
✓ 15개 모델 설정 및 하이퍼파라미터
✓ 무작위 시드, 정규화 방식, 이상치 임곗값 등
```
**사용 방법**:
```python
from anomaly_detection import config

print(config.WINDOW_SIZE)           # 64
print(config.FFT_SAMPLE_SIZES)      # [32, 64, 128, 256]
print(config.CLASSICAL_MODELS)      # 모든 모델 파라미터
```
**언제 수정**: 하이퍼파라미터 조정, 데이터 경로 변경 시

---

## 📄 anomaly_detection/data_loader.py
**기능**: CSV 데이터 로드 및 분석
**주요 클래스**: `DataLoader`
**기능**:
```
✓ CSV 파일 로드 (메타데이터 + 진동 데이터)
✓ 데이터 분석 및 통계
✓ 정상/이상 라벨 분류
✓ 데이터 포맷 자동 감지
```
**사용 예시**:
```python
from anomaly_detection.data_loader import DataLoader

loader = DataLoader(config.TRAIN_DATA_DIR)
df = loader.load_data()
```

---

## 📄 anomaly_detection/preprocessing.py
**기능**: 데이터 전처리 파이프라인
**주요 클래스**: `Preprocessor`
**기능**:
```
✓ 결측값 처리 (보간, 앞당기기, 삭제)
✓ 이상치 처리 (IQR 방식, Z-score)
✓ 정규화 (StandardScaler, MinMaxScaler, RobustScaler)
✓ 시계열 윈도우 생성 (슬라이딩 윈도우)
✓ FFT 특성 생성
✓ 통계 특성 추출 (RMS, Peak, Kurtosis, Skewness)
```
**사용 예시**:
```python
from anomaly_detection.preprocessing import Preprocessor

preprocessor = Preprocessor(window_size=64)
X_train, y_train = preprocessor.preprocess_pipeline(
    train_data, feature_cols=['vibration'], label_col='label'
)
```

---

## 📄 anomaly_detection/model_training.py ⭐ 핵심
**기능**: 모든 ML 모델 훈련
**주요 클래스**: 
- `ClassicalModelTrainer`: 전통 ML 모델 (9개 이상탐지 + 6개 분류)
- `DeepLearningTrainer`: 딥러닝 모델 (Autoencoder, LSTM)

**구현된 모델**:
```
이상탐지 (9개):
├── IsolationForest
├── OneClassSVM
├── LocalOutlierFactor
├── EllipticEnvelope
├── RobustCovariance ⭐ NEW
├── MinCovDet ⭐ NEW
├── KMeansAnomaly ⭐ NEW
├── PCAAnomaly ⭐ NEW
└── DBSCAN ⭐ NEW

분류 (6개):
├── RandomForest
├── GradientBoosting
├── DecisionTree
├── KNearestNeighbors
├── LogisticRegression
└── SVM

딥러닝 (2개):
├── Autoencoder
└── LSTM
```

**사용 예시**:
```python
from anomaly_detection.model_training import ClassicalModelTrainer, DeepLearningTrainer

# 이상탐지 모델
trainer = ClassicalModelTrainer()
model = trainer.train_isolation_forest(X_train)

# 분류 모델
model = trainer.train_random_forest(X_train, y_train)

# 딜러닝
dl_trainer = DeepLearningTrainer()
model = dl_trainer.train_autoencoder(X_train, X_val)
```

---

## 📄 anomaly_detection/evaluation.py
**기능**: 모델 평가 및 메트릭 계산
**주요 클래스**: `ModelEvaluator`
**기능**:
```
✓ 정확도, 정밀도, 재현율, F1 점수 계산
✓ ROC-AUC 계산
✓ Confusion Matrix 분석
✓ 혼동행렬 시각화
✓ ROC 곡선 시각화
✓ 모델 비교 순위
```
**계산되는 메트릭**:
- Accuracy, Precision, Recall, F1, ROC-AUC
- True Positive, True Negative, False Positive, False Negative

**사용 예시**:
```python
from anomaly_detection.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.compute_metrics(y_test, y_pred)
```

---

## 📄 anomaly_detection/mlflow_utils.py
**기능**: MLflow를 이용한 실험 추적
**기능**:
```
✓ 모델 파라미터 로깅
✓ 메트릭 기록
✓ 모델 아티팩트 저장
✓ 실험 버전 관리
```

---

## 📄 anomaly_detection/inference_serial.py
**기능**: 실시간 직렬 포트를 통한 이상탐지
**기능**:
```
✓ 실시간 센서 데이터 수신
✓ 실시간 이상탐지 추론
✓ 기계 상태 모니터링
```

---

## 📄 anomaly_detection/feature_extraction.py ⭐ NEW 핵심
**기능**: 학습/추론 **공용** 특징 추출 모듈 (단일 진실원)
**역할**:
```
✓ build_feature_table.py(오프라인 학습)와 util 추론이 동일한 특징을 쓰도록 보장
✓ 고정 윈도우 분할(4000샘플=1초, 50% overlap)
✓ 시간영역 특징: rms, peak, crest_factor, std, kurtosis, skewness, p2p
✓ 주파수영역 특징: spectral_centroid/entropy/energy + 회전차수(order) 대역 에너지
   (band_1x/2x/3x/high 및 각 비율) → RPM 기반 설비 간 비교
✓ 진폭 Z-score 정규화(설비별 정상 기준)
✓ 5클래스 라벨 매핑 (LABEL_MAP / LABEL_EN_TO_KO)
```
**핵심 상수/함수**:
```python
from anomaly_detection import feature_extraction as fe

fe.FEATURE_COLUMNS          # 학습에 사용된 20개 특징 컬럼 순서(고정)
fe.WINDOW_SIZE              # 4000
vec = fe.extract_feature_vector(window, rpm=1730, power_kw=2.2)  # (20,) 벡터
sig = fe.normalize_amplitude(signal, mean, std)
windows = fe.make_windows(signal)   # [(window, padded), ...]
```
**연계**: `build_feature_table.py`, `util/test_serial_inference.py` 가 import

# ============================================================================
# 🚀 메인 스크립트: Root Python 파일들
# ============================================================================

## 📄 main.py
**목적**: 프로젝트의 메인 진입점 (레거시 파이프라인 [A])
**기능**: 폐쇄 루프 훈련 및 평가 파이프라인 (model_training.py 사용)
**실행**:
```bash
python main.py
```

---

## 📄 build_feature_table.py ⭐ NEW (표준화 파이프라인 [B] 1단계)
**목적**: 원시 진동 CSV → 머신러닝용 **표준 특징 테이블** 생성
**기능**:
```
✓ data_new_format 의 모든 CSV(길이 12000/500/300 혼재)를 고정 윈도우로 통일
✓ 설비별 정상 기준 진폭 Z-score 정규화 (동력/설비 차이 흡수)
✓ 회전차수(order) 기반 FFT 특징 추출 (RPM 사용)
✓ 윈도우=1행 형태의 feature_table_{train,val,test}.csv 생성
✓ --regroup: 설비×라벨 층화 후 파일단위 70/15/15 재분할(데이터 누수 방지)
✓ normalization_stats.json 저장(추론 시 재사용)
```
**연계**: `anomaly_detection/feature_extraction.py` 를 import (model_training 아님)
**실행**:
```bash
python build_feature_table.py --regroup
```

---

## 📄 train_from_feature_table.py ⭐ NEW (표준화 파이프라인 [B] 2단계)
**목적**: 표준 특징 테이블 → **5클래스 고장 분류** + 이상탐지 모델 학습/평가
**기능**:
```
✓ 과제1 다중분류(5클래스): 정상/축정렬불량/회전체불평형/베어링불량/벨트느슨함
   - RandomForestClassifier(class_weight='balanced')
   - test 정확도 ~99.5%, macro-F1 ~0.996
✓ 과제2 이상탐지(One-Class): 정상만 학습 → IsolationForest
✓ 특징 중요도 / classification report / confusion matrix 출력
✓ 결과: results/feature_table_evaluation.json
```
**중요**: sklearn 을 직접 사용하며 `anomaly_detection.model_training` 과 **무관**
**실행**:
```bash
python train_from_feature_table.py              # 분류 + 이상탐지
python train_from_feature_table.py --model classifier
```

---

## 📄 analysis/anomaly_detection_comparison.py
**위치**: `analysis/` 폴더 (루트가 아님 — 실행 시 `python analysis/anomaly_detection_comparison.py`)
**목적**: 모든 이상탐지 모델 자동 비교 및 분석 (탐색용, 핵심 파이프라인과 별개)
**기능**:
```
✓ 9개 모든 이상탐지 모델 훈련
✓ 자동 성능 평가
✓ 성능 순위 매김
✓ 결과 시각화 (그래프, 히트맵)
✓ JSON/CSV 보고서 생성
```
**5개 예시 함수**:
```python
from anomaly_detection_comparison import (
    example_5_quick_test,                  # 3개 모델, 1-2분 (권장)
    example_1_basic_anomaly_detection,     # 3개 모델
    example_2_original_vs_new_models,      # 기존(4개) vs 신규(5개)
    example_3_all_anomaly_models,          # 모든 9개 모델
    example_4_different_window_sizes,      # 다양한 윈도우 크기
)
```
**빠른 시작**:
```python
from anomaly_detection_comparison import example_5_quick_test
result = example_5_quick_test()
```

---

## 📄 fft_sample_comparison.py
**목적**: FFT 샘플 크기별 모델 성능 비교
**기능**:
```
✓ FFT_SAMPLE_SIZES = [32, 64, 128, 256]에서 성능 비교
✓ 15개 모델 모두 지원
✓ 최적 샘플 크기 찾기
✓ 비교 결과 시각화
```
**사용**:
```python
from fft_sample_comparison import run_fft_sample_comparison

result = run_fft_sample_comparison(
    models_to_test=["RandomForest", "Autoencoder"],
    fft_sample_sizes=[32, 64, 128]
)
```

---

## 📄 quick_start_fft_comparison.py
**목적**: FFT 비교의 빠른 시작 예시
**포함**: 7가지 다양한 비교 시나리오
**예시 함수**:
```python
example_1_basic()              # 기본 비교
example_2_ensemble_models()    # 앙상블 모델만
example_3_anomaly_models()     # 이상탐지 모델만
example_4_all_classical()      # 모든 전통 ML
example_5_all_models()         # 전체 15개 모델
example_6_custom_sizes()       # 커스텀 크기
example_7_quick_test()         # 빠른 테스트
```


# ============================================================================
# 📊 데이터 폴더: data_new_format/
# ============================================================================

**역할**: 학습/검증/테스트 데이터 저장소

```
data_new_format/
├── train/              # 훈련 데이터 (5개 정상 + 2개 이상)
├── val/                # 검증 데이터 (2개 정상 + 1개 이상)
└── test/               # 테스트 데이터 (2개 정상 + 1개 이상)
```

**데이터 포맷**:
```
CSV 포맷: 메타데이터 9줄 + 데이터
- 라인 1-9: 날짜, 파일명, 레이블, 샘플레이트 등
- 라인 10+: time, vibration 컬럼 (시계열 진동 데이터)
```

**라벨 (5클래스)**:
| 한글 | 영문 | label_no | 이상여부 |
|------|------|----------|----------|
| 정상 | NORMAL | 0 | 0 |
| 축정렬불량 | MISALIGN | 1 | 1 |
| 회전체불평형 | IMBALANCE | 2 | 1 |
| 베어링불량 | BEARING | 3 | 1 |
| 벨트느슨함 | BELT | 4 | 1 |

**동력별 구성**: 2.2kW / 3.7kW / 5.5kW 모터(설비 11종, RPM 상이)
→ 회전차수(order) 정규화로 설비 간 비교 가능


# ============================================================================
# 📁 결과 폴더들
# ============================================================================

## 📂 results/
**역할**: 모델 평가 결과 저장
```
results/
├── model_randomforest_evaluation.json    # 레거시 파이프라인[A] 모델별 결과
├── model_isolationforest_evaluation.json
├── model_*_evaluation.json
└── feature_table_evaluation.json         # ⭐ NEW: 표준화[B] 5클래스/이상탐지 결과
```

---

## 📂 feature_tables/ ⭐ NEW
**역할**: 표준화된 특징 테이블(윈도우=1행) 저장
```
feature_tables/
├── feature_table_train.csv      # 학습용 특징 테이블
├── feature_table_val.csv        # 검증용
├── feature_table_test.csv       # 테스트용
└── normalization_stats.json     # 설비별 정상 기준 통계(추론 재사용)
```
**생성**: `python build_feature_table.py --regroup`
**컬럼**: source_file, equipment_id, power_kw, rpm, window_index, padded,
rms~p2p(시간영역), spectral_*/band_*(주파수영역), label_name, label_no, is_anomaly

---

## 📂 fft_comparison_results/
**역할**: FFT 샘플 크기 비교 결과
```
fft_comparison_results/
├── fft_sample_comparison.json        # 비교 결과
├── fft_sample_metrics.csv            # 메트릭 테이블
├── fft_comparison.png                # 성능 비교 그래프
└── fft_heatmap.png                   # 성능 히트맵
```

---

## 📂 anomaly_detection_results_quick/
**역할**: 이상탐지 모델 비교 결과 (빠른 테스트)
```
anomaly_detection_results_quick/
├── anomaly_detection_comparison.json
├── anomaly_detection_metrics.csv
├── anomaly_detection_comparison.png
└── anomaly_detection_heatmap.png
```

---

## 📂 eda_results/
**역할**: 탐색적 데이터 분석 결과
- 데이터 분포 분석
- 통계 요약
- 시각화 그래프


# ============================================================================
# 🛠️ 유틸리티 폴더: util/
# ============================================================================

## 📄 util/serial_data_simulator.py ⭐ 업데이트
**목적**: 실시간 센서 데이터 시뮬레이션 및 실데이터 스트리밍
**기능**:
```
✓ 정상/이상 합성 신호 생성 (SerialDataSimulator)
✓ 실시간 스트리밍 시뮬레이션 / 직렬 포트 모의
✓ RealSignalStreamer: data_new_format 실제 CSV를 상태별 로드/스트리밍
  (설비ID·RPM·동력·라벨 메타 파싱) → 5클래스 모델 입력
```

---

## 📄 util/test_serial_inference.py ⭐ 업데이트
**목적**: 학습된 **5클래스 고장 분류 모델**로 실시간 추론 테스트
**주요 클래스**:
```
✓ FaultClassifierEngine  : models/clf_random_forest.pkl + clf_scaler.pkl +
  normalization_stats.json 로드 → feature_extraction 동일 특징으로 5클래스 예측
✓ RealtimeDataBuffer      : 4000샘플(1초) FIFO 버퍼
✓ RealtimeInferenceEngine : (레거시) 더미 이진 모델 데모
```
**실행**:
```bash
python util/test_serial_inference.py --mode classify --split train  # 5클래스(권장)
python util/test_serial_inference.py --mode legacy                  # 레거시 더미 데모
```
**연계**: `anomaly_detection/feature_extraction.py`, `util/serial_data_simulator.py`

---

## 📄 util/verify_system.py
**목적**: 시스템 설치 및 설정 검증
**기능**:
```
✓ 필요 라이브러리 확인
✓ 데이터 파일 확인
✓ 경로 설정 검증
```
**실행**:
```bash
python util/verify_system.py
```

---

## 📄 util/migrate_mlflow.py
**목적**: MLflow 데이터 마이그레이션
**기능**:
- 모델 메타데이터 마이그레이션
- 파라미터 마이그레이션

---

## 📄 util/QUICK_START.py
**목적**: 빠른 시작 가이드
**기능**: 기본 파이프라인 실행 예시

---

## 📄 util/synthetic_test_data.csv
**목적**: 테스트용 합성 데이터
**용도**: 공식 데이터 없을 때 테스트


# ============================================================================
# 📈 분석 폴더: analysis/
# ============================================================================

## 📄 analysis/analyze_2d_data.py
**목적**: 2D 이상탐지 데이터 분석
**기능**: 2차원 배열 데이터 분석

---

## 📄 analysis/analyze_3d_array.py
**목적**: 3D 배열 데이터 분석
**기능**:
```
✓ 3D 윈도우 배열 분석
✓ 특성 추출
✓ 시각화
```

---

## 📄 analysis/visualize_3d_array.py
**목적**: 3D 배열 시각화
**기능**: 윈도우 데이터의 3D 그래프 표현

---

## 📄 analysis/ml_input_examples.py
**목적**: ML 입력 포맷 예시
**기능**:
```
✓ 다양한 윈도우 크기 테스트
✓ FFT 변환 예시
✓ 입력 포맷 검증
```

---

## 📄 analysis/MODEL_IO_*.md
**목적**: 모델 입출력 포맷 설명
- `MODEL_IO_QUICK_REF.py`: 빠른 참조
- `MODEL_IO_VISUALIZATION.md`: 시각화
- `3D_ARRAY_*.md`: 3D 배열 가이드


# ============================================================================
# 📚 문서 파일: Markdown 가이드
# ============================================================================

## 🌟 필독 문서

### 1️⃣ RUN_ANOMALY_DETECTION_COMPARISON.md ⭐⭐⭐ 최우선
**내용**: 이상탐지 비교 실행 완전 가이드
**읽기 순서**: 1번
**시간**: 10분
**저장위치**: 프로젝트 루트

### 2️⃣ ANOMALY_DETECTION_QUICK_REFERENCE.md
**내용**: 9개 모델 한눈에 보기
**모델 선택 가이드**: 의사결정 트리
**읽기 순서**: 2번
**시간**: 15분

### 3️⃣ ANOMALY_DETECTION_MODELS_GUIDE.md
**내용**: 각 모델 상세 설명
**파라미터 튜닝**: 최적화 팁
**읽기 순서**: 3번
**시간**: 30분

### 4️⃣ ANOMALY_DETECTION_EXPANSION_SUMMARY.md
**내용**: 5개 신규 모델 기술 문서
**변경사항**: 상세 설명
**읽기 순서**: 필요시

---

## FFT 관련 문서

### FFT_SAMPLE_COMPARISON_GUIDE.md
**목적**: FFT 샘플 크기 비교 가이드
**내용**:
```
✓ FFT 개념 설명
✓ 샘플 크기의 영향
✓ 최적 크기 선택 방법
✓ 사용 예시
```

### FFT_MODIFICATIONS_SUMMARY.md
**목적**: FFT 기능 변경사항 정리

### FFT_COMPATIBILITY_*.md
**목적**: 전체 시스템 FFT 호환성 검증

---

## 프로젝트 개요 문서

### README.md
프로젝트 개요 및 시작하기

### PROJECT_STRUCTURE.md
프로젝트 구조 도표

### MODEL_IO_FORMAT.md
모델 입출력 포맷

### FEATURES_EXPLANATION.md
특성 엔지니어링 설명

### DELIVERY_SUMMARY.md
최종 배포 요약

---

## 데이터 가이드

### 3D_ARRAY_COMPLETE_GUIDE.md
완전한 3D 배열 가이드

### 3D_ARRAY_EXPLAINED.md
3D 배열 이해하기


# ============================================================================
# 💾 모델 저장소: models/
# ============================================================================

**역할**: 훈련된 모델 저장
```
models/
# [A] 레거시 파이프라인 (main.py / model_training.py)
├── random_forest_model.pkl         # 랜덤 포레스트
├── isolation_forest_model.pkl      # 이솔레이션 포레스트
├── one_class_svm_model.pkl         # One-Class SVM
├── autoencoder_model.pt            # Autoencoder (PyTorch)
├── lstm_model.pt                   # LSTM (PyTorch)
├── scaler.pkl                      # 데이터 정규화 스케일러
# [B] 표준화 파이프라인 (train_from_feature_table.py) ⭐ NEW
├── clf_random_forest.pkl           # 5클래스 고장 분류기
├── clf_scaler.pkl                  # 분류기용 특징 스케일러
├── anomaly_isolation_forest.pkl    # 이상탐지(One-Class) 모델
└── anomaly_scaler.pkl              # 이상탐지용 스케일러
```


# ============================================================================
# 🔗 통합 사용사례: integration/
# ============================================================================

## 목적: MLflow + BentoML 통합

**파일**:
- `mlflow_bentoml_example.py`: 통합 예시
- `bentoml_service.py`: BentoML 서비스
- `feature_store.py`: 특성 저장소
- `MLFLOW_BENTOML_GUIDE.md`: 상세 가이드

**기능**:
```
✓ 모델 실험 추적 (MLflow)
✓ 모델 서빙 (BentoML)
✓ 특성 저장소 관리
```


# ============================================================================
# 🎯 시작 가이드
# ============================================================================

## 초보자용 추천 순서

### Step 1: 개념 이해 (5분)
1. 이 가이드 읽기 (지금)
2. [RUN_ANOMALY_DETECTION_COMPARISON.md](RUN_ANOMALY_DETECTION_COMPARISON.md) 읽기

### Step 2: 빠른 실행 (5분)
```python
from anomaly_detection_comparison import example_5_quick_test
result = example_5_quick_test()
```

### Step 3: 결과 분석 (10분)
- `anomaly_detection_results_quick/` 폴더의 파일 확인
- CSV 메트릭 검토
- PNG 그래프 확인

### Step 4: 모델 선택 (10분)
- [ANOMALY_DETECTION_QUICK_REFERENCE.md](ANOMALY_DETECTION_QUICK_REFERENCE.md) 읽기
- 최고 성능 모델 확인

### Step 5: 상세 학습 (30분)
- [ANOMALY_DETECTION_MODELS_GUIDE.md](ANOMALY_DETECTION_MODELS_GUIDE.md) 읽기
- 각 모델의 장단점 학습

### Step 6: 고급 기능 활용
- FFT 샘플 크기 비교
- 커스텀 모델 추가
- MLflow 통합


## 경험자용 빠른 경로

```python
# 1. 특정 모델만 비교
from anomaly_detection_comparison import run_anomaly_detection_comparison
result = run_anomaly_detection_comparison(
    models_to_test=["IsolationForest", "PCAAnomaly", "KMeansAnomaly"],
    window_size=128
)

# 2. 커스텀 데이터 사용
from anomaly_detection.model_training import ClassicalModelTrainer
trainer = ClassicalModelTrainer()
model = trainer.train_isolation_forest(X_train)

# 3. 결과 분석
from anomaly_detection.evaluation import ModelEvaluator
metrics = ModelEvaluator.compute_metrics(y_test, y_pred)
```


# ============================================================================
# 🔍 파일 빠른 검색
# ============================================================================

### "어떻게 실행하지?"
→ [RUN_ANOMALY_DETECTION_COMPARISON.md](RUN_ANOMALY_DETECTION_COMPARISON.md)

### "9개 모델이 뭔데?"
→ [ANOMALY_DETECTION_QUICK_REFERENCE.md](ANOMALY_DETECTION_QUICK_REFERENCE.md)

### "어떤 모델을 써야 해?"
→ [ANOMALY_DETECTION_MODELS_GUIDE.md](ANOMALY_DETECTION_MODELS_GUIDE.md) 섹션 6

### "FFT가 뭐지?"
→ [FFT_SAMPLE_COMPARISON_GUIDE.md](FFT_SAMPLE_COMPARISON_GUIDE.md)

### "3D 배열이 뭔데?"
→ [3D_ARRAY_COMPLETE_GUIDE.md](analysis/3D_ARRAY_COMPLETE_GUIDE.md)

### "실시간 추론 어떻게?"
→ [util/REALTIME_INFERENCE.md](util/REALTIME_INFERENCE.md)

### "MLflow는?"
→ [integration/MLFLOW_BENTOML_GUIDE.md](integration/MLFLOW_BENTOML_GUIDE.md)

### "에러가 발생했어"
→ [ANOMALY_DETECTION_QUICK_REFERENCE.md](ANOMALY_DETECTION_QUICK_REFERENCE.md) 섹션 9

### "전체 구조 이해하고 싶어"
→ [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

### "특성 정보"
→ [FEATURES_EXPLANATION.md](FEATURES_EXPLANATION.md)


# ============================================================================
# 📞 파일 관계도
# ============================================================================

```
[A] 레거시 파이프라인 (윈도우 원시신호 기반, 15개 모델)

사용자
  ↓
main.py legacy --train|--eda|--infer   (통합 진입점)
  ├─→ anomaly_detection/config.py (설정)
  ├─→ anomaly_detection/data_loader.py (데이터 로드)
  ├─→ anomaly_detection/preprocessing.py (전처리)
  ├─→ anomaly_detection/model_training.py (모델 훈련) ⭐
  └─→ anomaly_detection/evaluation.py (평가)

  ※ analysis/anomaly_detection_comparison.py 는 탐색용 비교 스크립트(별개)

[B] 표준화 파이프라인 (특징 테이블 기반, 5클래스 분류) ⭐ NEW

사용자
  ↓
main.py standard --all|--build|--train|--infer   (통합 진입점)
  ↓
data_new_format/ (원시 진동 CSV)
  ↓  build_feature_table.py --regroup
  └─→ anomaly_detection/feature_extraction.py (공용 특징 추출) ⭐
  ↓
feature_tables/feature_table_{train,val,test}.csv + normalization_stats.json
  ↓  train_from_feature_table.py   (sklearn 직접, model_training 미사용)
  └─→ models/clf_random_forest.pkl, clf_scaler.pkl,
        anomaly_isolation_forest.pkl, anomaly_scaler.pkl
  ↓  util/test_serial_inference.py --mode classify
  └─→ anomaly_detection/feature_extraction.py (동일 특징) → 실시간 5클래스 추론

결과
  ├─→ results/feature_table_evaluation.json
  └─→ util/inference_results.json
```


# ============================================================================
# 🎓 학습 경로별 파일
# ============================================================================

### 🟢 완전 초보자 (하루)
1. 이 가이드 읽기
2. RUN_ANOMALY_DETECTION_COMPARISON.md
3. example_5_quick_test() 실행
4. ANOMALY_DETECTION_QUICK_REFERENCE.md 읽기
5. 결과 분석

### 🟡 중급자 (2-3일)
1. ANOMALY_DETECTION_MODELS_GUIDE.md 읽기
2. 다양한 예시 함수 실행
3. FFT_SAMPLE_COMPARISON_GUIDE.md 학습
4. 커스텀 데이터로 실험

### 🔴 고급자 (1주일)
1. 모든 모델 상세 분석
2. 파라미터 튜닝
3. 새로운 모델 추가 구현
4. MLflow/BentoML 통합
5. 실시간 배포


# ============================================================================
# 📋 체크리스트
# ============================================================================

### 첫 실행 요구사항
- [ ] Python 3.8+ 설치
- [ ] requirements.txt 의존성 설치
- [ ] 데이터 경로 확인 (data_new_format/)
- [ ] config.py 설정 확인

### 실행 전 확인
- [ ] RUN_ANOMALY_DETECTION_COMPARISON.md 읽음
- [ ] 데이터 파일 존재 확인
- [ ] 모델 저장 폴더 생성 (models/)
- [ ] 결과 저장 폴더 생성 (results/)

### 실행 후 확인
- [ ] example_5_quick_test() 성공
- [ ] 결과 파일 생성됨
- [ ] JSON/CSV/PNG 파일 확인
- [ ] 메트릭 값 확인

### 고급 기능 활성화
- [ ] MLflow 설치 (선택)
- [ ] BentoML 설치 (선택)
- [ ] 실시간 테스트 장비 준비 (선택)


# ============================================================================
# ⚠️ 주의사항
# ============================================================================

1. **메모리**: 전체 9개 모델 실행 시 높은 메모리 필요
   → 권장: example_5_quick_test() 부터 시작

2. **시간**: 모든 모델 훈련 시 10-15분 소요
   → 권장: 첫 실행은 5-10분 정도

3. **데이터**: 데이터 경로 반드시 확인
   → config.py에서 설정 가능

4. **의존성**: matplotlib numpy 호환성 문제 가능
   → 해결책: [RUN_ANOMALY_DETECTION_COMPARISON.md](RUN_ANOMALY_DETECTION_COMPARISON.md) 참고

5. **버전**: 모델 저장/로드 시 버전 일치 필요
   → sklearn, torch 버전 유지


# ============================================================================
# 🆘 도움말
# ============================================================================

### 자주 묻는 질문

**Q: 어디부터 시작해야 하나요?**
A: [RUN_ANOMALY_DETECTION_COMPARISON.md](RUN_ANOMALY_DETECTION_COMPARISON.md) 읽기 → example_5_quick_test() 실행

**Q: 어떤 모델을 써야 하나요?**
A: [ANOMALY_DETECTION_QUICK_REFERENCE.md](ANOMALY_DETECTION_QUICK_REFERENCE.md)의 의사결정 트리 참고

**Q: 에러가 발생했어요**
A: [ANOMALY_DETECTION_QUICK_REFERENCE.md](ANOMALY_DETECTION_QUICK_REFERENCE.md) 섹션 9 확인

**Q: 파일이 너무 많아요**
A: 이 가이드의 "🔍 파일 빠른 검색" 섹션 참고

**Q: 파라미터를 조정하고 싶어요**
A: anomaly_detection/config.py 수정 후 재실행

**Q: 새로운 모델을 추가하고 싶어요**
A: anomaly_detection/model_training.py에 train_* 메서드 추가

---

마지막 업데이트: 2026-06-22
문의: 각 파일의 주석과 docstring 참고
"""
