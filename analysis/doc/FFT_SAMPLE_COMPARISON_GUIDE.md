# FFT Sample Size Comparison Guide

## 개요

진동 데이터로부터 FFT를 생성하고, FFT 샘플 크기(윈도우 크기)를 다르게 하여 모델 정확도를 비교할 수 있도록 수정했습니다.

## 주요 수정 사항

### 1. **config.py 수정**
FFT 샘플 크기 관련 설정 추가:
```python
# FFT Sample Sizes for comparison and optimization
FFT_SAMPLE_SIZES = [32, 64, 128, 256]  # 테스트할 윈도우 크기
FFT_BINS_SIZES = [5, 10, 20, 32]       # 테스트할 FFT 빈 크기
```

### 2. **preprocessing.py 수정**
- `create_windows()` 메소드: `window_size` 파라미터 추가
- `preprocess_pipeline()` 메소드: `window_size` 파라미터 추가
- `preprocess_data()` 함수: `window_size` 파라미터 추가

이를 통해 동적으로 윈도우 크기를 변경하면서 데이터 전처리 가능

### 3. **fft_sample_comparison.py 작성**
새로운 스크립트로 FFT 샘플 크기별 모델 성능 비교:
- 여러 샘플 크기(32, 64, 128, 256)로 데이터 전처리
- 각 샘플 크기마다 RandomForest, Autoencoder 등 여러 모델 학습
- Accuracy, Precision, Recall, F1-Score 계산 및 저장
- 결과 JSON 형식 저장 및 표로 출력

## 사용 방법

### 1. FFT 샘플 크기 비교 실행

```bash
python fft_sample_comparison.py
```

**실행 결과:**
- `fft_comparison_results/fft_sample_comparison.json` 파일 생성
- 콘솔에 비교 표 출력

### 2. 결과 해석

출력 예시:
```
RandomForest:
Sample Size      Accuracy        Precision       Recall          F1 Score
32               0.8500          0.8400          0.8600          0.8500
64               0.8800          0.8750          0.8900          0.8800
128              0.8900          0.8850          0.9000          0.8900
256              0.8700          0.8650          0.8750          0.8700
```

**해석:**
- **Sample Size 64-128**에서 최고 성능 달성
- 더 큰 윈도우(256)에서는 성능 감소 가능성
- 모델별로 최적 샘플 크기 상이

### 3. 자신만의 샘플 크기로 테스트

#### Option A: config.py 수정
```python
# anomaly_detection/config.py에서 변경
FFT_SAMPLE_SIZES = [16, 32, 64, 128, 256, 512]  # 원하는 크기 지정
```

#### Option B: Python 스크립트에서 직접 호출
```python
from fft_sample_comparison import run_fft_sample_comparison

# 커스텀 샘플 크기로 실행
results = run_fft_sample_comparison(
    sample_sizes=[32, 64, 128],
    models_to_test=["RandomForest", "IsolationForest"]
)
```

## 데이터 흐름

```
원본 진동 데이터 (vibration)
    ↓
데이터 로드 & 전처리
    ↓
샘플 크기 1: 32
  ├─ 슬라이딩 윈도우 생성 (32 샘플)
  ├─ FFT 적용
  ├─ 모델 학습 (RandomForest, Autoencoder 등)
  └─ 성능 평가 (Accuracy, Precision, Recall, F1)
    ↓
샘플 크기 2: 64
  ├─ 슬라이딩 윈도우 생성 (64 샘플)
  ├─ FFT 적용
  ├─ 모델 학습
  └─ 성능 평가
    ↓
... 반복 ...
    ↓
결과 비교 및 시각화
```

## 주요 메트릭 설명

- **Accuracy (정확도)**: 전체 예측 중 맞은 비율
- **Precision (정밀도)**: 이상 감지 예측 중 실제 이상 비율
- **Recall (재현율)**: 실제 이상 중 감지된 비율
- **F1-Score**: Precision과 Recall의 조화 평균

## 결과 저장 위치

```
fft_comparison_results/
└── fft_sample_comparison.json
    ├─ sample_sizes: [32, 64, 128, 256]
    ├─ models: ["RandomForest", "Autoencoder"]
    └─ results_by_sample_size:
        ├─ 32: {RandomForest: {...}, Autoencoder: {...}}
        ├─ 64: {RandomForest: {...}, Autoencoder: {...}}
        └─ ... 이하 반복
```

## 최적 샘플 크기 선택 기준

1. **높은 Accuracy**: 가장 기본적인 성능 지표
2. **높은 Recall**: 이상을 놓치지 않는 것 중요
3. **균형**: Precision과 Recall의 균형 (F1-Score)
4. **연산 효율**: 더 큰 샘플 = 더 많은 연산

## 주의사항

- 샘플 크기가 클수록 **윈도우 개수 감소** → 학습 데이터 부족 가능
- 샘플 크기가 작을수록 **부족한 특성 정보** → 성능 저하 가능
- 각 데이터셋에 따라 최적 크기 다름 → 실험 필요

## 다음 단계

1. **FFT 빈 크기도 비교**: `FFT_BINS_SIZES` 활용
2. **다른 모델 추가**: config의 모델 목록 확장
3. **하이퍼파라미터 튜닝**: config.CLASSICAL_MODELS, config.DL_MODELS 조정
4. **실시간 추론**: 최적 샘플 크기 적용하여 inference_serial.py 업데이트

## 문제 해결

### 에러: "No data found"
- 데이터 경로 확인: `anomaly_detection/config.py`의 TRAIN/VAL/TEST_DATA_DIR 확인

### 에러: "Out of Memory"
- 샘플 크기 감소 or 배치 크기 감소: config.DL_MODELS의 batch_size 조정

### 낮은 성능
- 데이터 확인: EDA 실행 (`python main.py --eda`)
- 정규화 방법 변경: config.NORMALIZE_METHOD 조정
- 데이터 전처리 옵션 검토: missing values, outliers 처리
