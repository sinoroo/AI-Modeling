# FFT 샘플 크기 비교 - 수정 완료 요약

> **수정 완료**: 진동 데이터로부터 FFT를 생성하고, FFT 샘플 크기를 다르게 설정하여 모델 정확도를 비교할 수 있도록 구현했습니다.

---

## 📋 수정 사항 요약

### 1. **anomaly_detection/config.py**
✅ **FFT 샘플 크기 설정 추가**
```python
# FFT Sample Sizes for comparison and optimization
FFT_SAMPLE_SIZES = [32, 64, 128, 256]  # 테스트할 윈도우 크기
FFT_BINS_SIZES = [5, 10, 20, 32]       # 테스트할 FFT 빈 크기
```

### 2. **anomaly_detection/preprocessing.py**
✅ **윈도우 크기를 동적으로 변경 가능하도록 수정**

**수정된 메소드들:**
- `create_windows()`: `window_size` 파라미터 추가
- `preprocess_pipeline()`: `window_size` 파라미터 추가  
- `preprocess_data()`: `window_size` 파라미터 추가

**예시:**
```python
# 샘플 크기 64로 데이터 전처리
preprocessed = preprocess_data(
    train_df, val_df, test_df, 
    feature_cols, label_col,
    window_size=64  # 이제 동적으로 변경 가능
)
```

### 3. **fft_sample_comparison.py** (신규 생성)
✅ **FFT 샘플 크기별 모델 성능 비교 스크립트**

**주요 기능:**
- 설정된 모든 샘플 크기로 데이터 전처리
- 각 샘플 크기마다 여러 모델 학습
- Accuracy, Precision, Recall, F1-Score 계산
- 결과를 JSON으로 저장 및 표로 출력

**사용 방법:**
```bash
python fft_sample_comparison.py
```

### 4. **FFT_SAMPLE_COMPARISON_GUIDE.md** (신규 생성)
✅ **자세한 사용 가이드 문서**

### 5. **quick_start_fft_comparison.py** (신규 생성)
✅ **빠른 시작을 위한 예제 스크립트**

---

## 🚀 빠른 시작

### 방법 1: 기본 실행 (권장)
```bash
python fft_sample_comparison.py
```
- 샘플 크기: 32, 64, 128, 256
- 모델: RandomForest, Autoencoder
- 예상 시간: 5-10분

### 방법 2: 예제 스크립트 사용
```bash
python quick_start_fft_comparison.py
```
- 4가지 사전 정의된 예제 제공
- 빠른 테스트부터 상세한 비교까지 가능

### 방법 3: Python 코드에서 직접 호출
```python
from fft_sample_comparison import run_fft_sample_comparison

# 커스텀 샘플 크기로 실행
results = run_fft_sample_comparison(
    sample_sizes=[32, 64, 128],
    models_to_test=["RandomForest", "IsolationForest"]
)
```

---

## 📊 결과 해석

### 출력 예시
```
RandomForest:
Sample Size      Accuracy        Precision       Recall          F1 Score
32               0.8500          0.8400          0.8600          0.8500
64               0.8800          0.8750          0.8900          0.8800
128              0.8900          0.8850          0.9000          0.8900
256              0.8700          0.8650          0.8750          0.8700

Autoencoder:
Sample Size      Accuracy        Precision       Recall          F1 Score
32               0.8200          0.8100          0.8300          0.8200
64               0.8600          0.8550          0.8700          0.8600
128              0.8750          0.8700          0.8800          0.8750
256              0.8500          0.8450          0.8600          0.8500
```

### 분석 포인트
1. **최적 샘플 크기 찾기**: 가장 높은 정확도/F1-Score 확인
2. **모델별 차이**: 각 모델의 최적 크기가 다를 수 있음
3. **성능 곡선**: 샘플 크기 증가에 따른 성능 추이 관찰

---

## 💾 결과 저장

자동으로 저장되는 위치:
```
fft_comparison_results/
└── fft_sample_comparison.json
```

JSON 형식 구조:
```json
{
  "sample_sizes": [32, 64, 128, 256],
  "models": ["RandomForest", "Autoencoder"],
  "results_by_sample_size": {
    "32": {
      "RandomForest": {
        "accuracy": 0.85,
        "precision": 0.84,
        "recall": 0.86,
        "f1": 0.85
      },
      "Autoencoder": { ... }
    },
    "64": { ... },
    ...
  }
}
```

---

## ⚙️ 고급 사용법

### 1. config.py에서 기본 샘플 크기 변경
```python
# anomaly_detection/config.py
FFT_SAMPLE_SIZES = [16, 32, 64, 128, 256, 512]
```

### 2. 더 많은 모델 테스트
```python
results = run_fft_sample_comparison(
    models_to_test=[
        "RandomForest", 
        "IsolationForest",
        "OneClassSVM",
        "Autoencoder",
        "LSTM"
    ]
)
```

### 3. 최적 샘플 크기로 최종 모델 학습
```python
# 최적 크기 (예: 128)로 main.py 수정
# anomaly_detection/config.py
WINDOW_SIZE = 128  # 최적으로 찾은 크기로 변경

# 그 후 일반 학습 파이프라인 실행
python main.py --train
```

---

## 🔍 데이터 흐름

```
진동 데이터 (vibration column)
        ↓
    ┌───┴───┬───────┬───────┐
    ↓       ↓       ↓       ↓
  [32]    [64]   [128]   [256]  ← 샘플 크기
    ↓       ↓       ↓       ↓
  전처리 (같은 과정, 다른 윈도우 크기)
    ↓       ↓       ↓       ↓
  FFT 생성 (자동으로 계산)
    ↓       ↓       ↓       ↓
  특성 추출 (통계, FFT 빈 등)
    ↓       ↓       ↓       ↓
  [RF1]  [RF2]  [RF3]  [RF4]
  [AE1]  [AE2]  [AE3]  [AE4]  ← 각 샘플 크기마다 모델 학습
    ↓       ↓       ↓       ↓
  정확도 비교 및 분석
```

---

## 📌 주요 개선 사항

| 항목 | 이전 | 이후 |
|------|------|------|
| 윈도우 크기 | 고정 (64) | 동적 (32-256) |
| 샘플 크기 비교 | 불가능 | 자동으로 여러 크기 테스트 |
| 성능 평가 | 수동 | 자동으로 메트릭 생성/저장 |
| 결과 관리 | 없음 | JSON으로 체계적 저장 |

---

## ✅ 테스트 체크리스트

- [x] config.py에 FFT 샘플 크기 설정 추가
- [x] preprocessing.py에서 윈도우 크기 파라미터화
- [x] fft_sample_comparison.py 스크립트 작성
- [x] 모든 모델(RF, IF, SVM, AE, LSTM) 지원
- [x] 결과 JSON 저장 기능
- [x] 결과 표 출력 기능
- [x] 가이드 문서 작성
- [x] 빠른 시작 예제 작성

---

## 📞 문제 해결

### Q: 데이터가 없다고 나옴
**A:** `anomaly_detection/config.py`에서 데이터 경로 확인
```python
TRAIN_DATA_DIR = "data_new_format/train"
VAL_DATA_DIR = "data_new_format/val"
TEST_DATA_DIR = "data_new_format/test"
```

### Q: 실행이 너무 오래 걸림
**A:** 더 적은 샘플 크기 또는 모델로 테스트
```python
results = run_fft_sample_comparison(
    sample_sizes=[32, 64],              # 크기 2개만
    models_to_test=["RandomForest"]     # 모델 1개만
)
```

### Q: 메모리 부족 에러
**A:** config.py에서 배치 크기 감소
```python
DL_MODELS["Autoencoder"]["batch_size"] = 16  # 32에서 16으로 감소
```

---

## 🎯 다음 단계

1. **결과 분석**: `fft_comparison_results/fft_sample_comparison.json` 확인
2. **최적값 결정**: 성능이 가장 좋은 샘플 크기 선택
3. **최종 모델 학습**: 최적 크기로 `WINDOW_SIZE` 설정 후 `python main.py --train`
4. **실시간 추론**: 최적 크기 적용하여 `inference_serial.py` 업데이트

---

**작성일**: 2024년  
**버전**: 1.0  
**상태**: ✅ 완료
