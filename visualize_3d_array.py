"""
3D Array 시각화 및 실습 가이드
실제 배열에서 데이터를 조회하고 분석하는 방법
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. 3D Array 구조 이해
# ============================================================================

print("=" * 80)
print("1️⃣  3D Array 구조와 크기")
print("=" * 80)

# 예제 데이터 생성 (실제 전처리 결과와 같은 형태)
# 실제로는 preprocessing.py에서 이렇게 생성됨
example_array = np.random.randn(154, 64, 1)  # (n_windows, window_size, n_features)

print(f"\n배열 형태 (Shape): {example_array.shape}")
print(f"  → 154: 윈도우 개수")
print(f"  → 64:  각 윈도우의 샘플 개수 (window_size)")
print(f"  → 1:   특성 개수 (vibration만)")

print(f"\n배열 크기: {example_array.nbytes / 1024:.2f} KB")
print(f"배열 데이터 타입: {example_array.dtype}")

print(f"\n총 데이터 포인트: {154 * 64} = {154 * 64} 개의 진동 측정값")


# ============================================================================
# 2. 배열의 각 층(Level) 접근
# ============================================================================

print("\n" + "=" * 80)
print("2️⃣  배열의 다양한 수준에서 접근하기")
print("=" * 80)

print("\n[레벨 1] 전체 배열")
print(f"array.shape = {example_array.shape}")

print("\n[레벨 2] 하나의 윈도우 선택")
print(f"array[0].shape = {example_array[0].shape}")
print(f"array[50].shape = {example_array[50].shape}")
print(f"array[153].shape = {example_array[153].shape}")

print("\n[레벨 3] 윈도우 내 특정 샘플")
print(f"array[0][0].shape = {example_array[0][0].shape}")
print(f"array[0][0] = {example_array[0][0]} (첫 윈도우의 첫 샘플)")
print(f"array[0][63].shape = {example_array[0][63].shape}")
print(f"array[0][63] = {example_array[0][63]} (첫 윈도우의 마지막 샘플)")

print("\n[레벨 4] 실제 스칼라 값")
print(f"array[0][0][0] = {example_array[0][0][0]:.6f} (실제 진동값)")


# ============================================================================
# 3. 시간 축 이해하기
# ============================================================================

print("\n" + "=" * 80)
print("3️⃣  시간 축 이해하기 (Time Axis)")
print("=" * 80)

SAMPLE_RATE = 4000  # Hz
WINDOW_SIZE = 64    # samples
OVERLAP = 0.5       # 50%
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

dt = 1 / SAMPLE_RATE  # 샘플 간 시간 간격

print(f"\n샘플 레이트: {SAMPLE_RATE} Hz")
print(f"샘플 간 시간: {dt * 1000:.2f} ms")
print(f"윈도우 크기: {WINDOW_SIZE} 샘플")
print(f"윈도우 지속 시간: {WINDOW_SIZE * dt * 1000:.2f} ms")
print(f"윈도우 간 오버랩: {int(OVERLAP * 100)}%")
print(f"스텝 크기: {STEP_SIZE} 샘플 ({STEP_SIZE * dt * 1000:.2f} ms)")

print(f"\n각 윈도우의 시간 범위:")
for w in [0, 1, 2, 50, 100, 153]:
    start_time = w * STEP_SIZE * dt
    end_time = (w * STEP_SIZE + WINDOW_SIZE - 1) * dt
    print(f"  Window {w:3d}: {start_time*1000:7.2f} ~ {end_time*1000:7.2f} ms")


# ============================================================================
# 4. 3D Array에서 데이터 추출하기
# ============================================================================

print("\n" + "=" * 80)
print("4️⃣  3D Array에서 데이터 추출하는 방법들")
print("=" * 80)

print("\n[추출 1] 특정 윈도우의 모든 샘플")
window_0 = example_array[0]  # Shape: (64, 1)
print(f"Window 0: shape={window_0.shape}")
print(f"  첫 5개 샘플: {window_0[:5, 0]}")
print(f"  마지막 5개 샘플: {window_0[-5:, 0]}")

print("\n[추출 2] 특정 윈도우를 1D로 변환 (Classical ML용)")
window_0_flattened = example_array[0].flatten()  # Shape: (64,)
print(f"Window 0 (flattened): shape={window_0_flattened.shape}")
print(f"  값: {window_0_flattened[:10]} ...")

print("\n[추출 3] 여러 윈도우를 2D로 변환 (Classical ML 배치용)")
windows_batch = example_array[:10]  # Shape: (10, 64, 1)
windows_batch_2d = windows_batch.reshape(10, 64)  # Shape: (10, 64)
print(f"10개 윈도우 (원형): shape={windows_batch.shape}")
print(f"10개 윈도우 (2D): shape={windows_batch_2d.shape}")

print("\n[추출 4] 모든 윈도우를 2D로 변환")
all_windows_2d = example_array.reshape(154, 64)  # Shape: (154, 64)
print(f"전체 배열 (2D): shape={all_windows_2d.shape}")
print(f"  메모리: {all_windows_2d.nbytes / 1024:.2f} KB")

print("\n[추출 5] 특정 샘플 시점의 모든 윈도우 값")
all_at_sample_0 = example_array[:, 0, 0]  # Shape: (154,)
print(f"모든 윈도우의 첫 샘플: shape={all_at_sample_0.shape}")
print(f"  값: {all_at_sample_0[:5]} ... (5개만 표시)")

print("\n[추출 6] 특정 윈도우의 특정 샘플")
value = example_array[50, 32, 0]
print(f"Window 50, Sample 32의 값: {value:.6f}")


# ============================================================================
# 5. 데이터 통계 및 분석
# ============================================================================

print("\n" + "=" * 80)
print("5️⃣  데이터 통계 분석")
print("=" * 80)

print(f"\n전체 배열 통계:")
print(f"  평균 (Mean):     {example_array.mean():.6f}")
print(f"  표준편차 (Std):  {example_array.std():.6f}")
print(f"  최소값 (Min):    {example_array.min():.6f}")
print(f"  최대값 (Max):    {example_array.max():.6f}")
print(f"  중앙값 (Median): {np.median(example_array):.6f}")

print(f"\n개별 윈도우 통계 (샘플):")
for w in [0, 50, 100, 153]:
    window = example_array[w]
    print(f"\n  Window {w}:")
    print(f"    평균: {window.mean():.6f}")
    print(f"    표준편차: {window.std():.6f}")
    print(f"    최소값: {window.min():.6f}")
    print(f"    최대값: {window.max():.6f}")
    print(f"    범위 (Range): {window.max() - window.min():.6f}")

print(f"\n각 윈도우의 평균값")
window_means = example_array.mean(axis=1).flatten()  # Shape: (154,)
print(f"  모양: {window_means.shape}")
print(f"  첫 5개: {window_means[:5]}")
print(f"  마지막 5개: {window_means[-5:]}")


# ============================================================================
# 6. 정규화 이해하기 (StandardScaler)
# ============================================================================

print("\n" + "=" * 80)
print("6️⃣  정규화 (Normalization) 이해하기")
print("=" * 80)

print("""
원본 진동 데이터:
  범위: 다양함 (예: -0.5 ~ 15.3)
  평균: 특정 값 (예: 2.1)
  표준편차: 일정 (예: 3.2)

StandardScaler 정규화 수식:
  x_normalized = (x - mean) / std

정규화 후:
  평균: 0 (거의 정확히)
  표준편차: 1
  범위: [-3 ~ 3] (대략, 3-sigma)
  
의미:
  0: 평균 (정상)
  +1.0: 평균보다 1 표준편차 높음
  +2.0: 평균보다 2 표준편차 높음 (이상 신호)
  -1.0: 평균보다 1 표준편차 낮음
  -2.0: 평균보다 2 표준편차 낮음 (이상 신호)
  
이 배열의 정규화 상태:
""")

print(f"  mean ≈ {example_array.mean():.10f} (0에 매우 가까움)")
print(f"  std ≈ {example_array.std():.10f} (1에 매우 가까움)")
print(f"  대부분의 값이 [-3, 3] 범위에 있음? {np.sum((example_array >= -3) & (example_array <= 3)) / example_array.size * 100:.1f}%")


# ============================================================================
# 7. 실제 활용: 임계값 기반 이상 탐지
# ============================================================================

print("\n" + "=" * 80)
print("7️⃣  임계값 기반 이상 탐지 예시")
print("=" * 80)

THRESHOLD = 2.0  # 표준편차 2배 이상

print(f"\n이상 탐지 임계값: {THRESHOLD} (평균 ± 2σ)")

# 각 윈도우에서 이상치 샘플 개수
anomalous_samples = np.sum(np.abs(example_array) > THRESHOLD, axis=1)

print(f"\n각 윈도우별 이상 샘플 개수 (상위 10개):")
top_indices = np.argsort(anomalous_samples.flatten())[-10:][::-1]
for idx in top_indices:
    count = anomalous_samples[idx, 0]
    percentage = count / WINDOW_SIZE * 100
    print(f"  Window {idx:3d}: {count:2d} 샘플 ({percentage:5.1f}% 이상)")

# 전체 이상 샘플 통계
total_anomalous = np.sum(anomalous_samples)
print(f"\n전체 이상 샘플 개수: {total_anomalous} / {154 * 64} ({total_anomalous / (154*64) * 100:.2f}%)")


# ============================================================================
# 8. 배열 구조 다이어그램
# ============================================================================

print("\n" + "=" * 80)
print("8️⃣  배열 구조 다이어그램")
print("=" * 80)

print("""
전체 구조:
┌─────────────────────────────────────────────────────────────┐
│           3D Array: (154, 64, 1)                            │
│  전체 시계열 데이터를 154개 윈도우로 분할                  │
└─────────────────────────────────────────────────────────────┘
        ↓
        ├─ Window 0: (64, 1)
        │  ├─ [[ 0.0234],
        │  ├─  [-0.0667],
        │  ├─  [-0.0344],
        │  └─  ... (64개 행)]
        │
        ├─ Window 1: (64, 1)
        │  └─ ... (64개 진동값)
        │
        ├─ ...
        │
        └─ Window 153: (64, 1)
           └─ ... (64개 진동값)


시간 축 표현:
전체 1.2375 초 데이터
├─ [0.000 ~ 0.016초] Window 0   ──┐
├─ [0.008 ~ 0.024초] Window 1   ──┤ 8ms 간격
├─ [0.016 ~ 0.032초] Window 2   ──┤
├─ ...                            │
└─ [1.232 ~ 1.248초] Window 153 ──┘

각 윈도우는 16ms의 시계열 데이터를 64개 시점으로 세분화
윈도우 간 50% 겹침으로 연속성 보장
""")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ 3D Array 완전 이해!")
print("=" * 80)

print("""
전처리 후 3D Array (154, 64, 1)는:

📊 구조:
  - 154개 윈도우 (시간 구간)
  - 각 윈도우: 64개 연속 시점
  - 각 시점: 1개 정규화된 진동값

⏰ 시간 정보:
  - 전체 범위: 1.24 초
  - 윈도우 크기: 16 ms
  - 샘플 간격: 0.25 ms
  - 윈도우 간격: 8 ms (50% 겹침)

📈 데이터:
  - 정규화: StandardScaler (mean=0, std=1)
  - 범위: [-3, 3] (대부분)
  - 타입: float32

🔍 의미:
  - 0 근처: 정상 진동
  - |값| > 2: 이상 신호 가능성
  - 큰 변동: 불안정한 진동 신호

✨ 모델 입력 형식:
  - 고전 ML: (154, 64) - 2D로 평탄화
  - LSTM: (154, 64, 1) - 3D 시계열 유지
""")

print("\n모든 설명이 완료되었습니다! 🎯\n")
