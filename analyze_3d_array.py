"""
실제 전처리된 3D Array 로드 및 분석 도구

이 스크립트는 preprocessing.py에서 생성된 실제 3D Array를 로드하여
구조와 내용을 상세히 분석합니다.
"""

import sys
import os

# UTF-8 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# 경고 무시 (Deprecation 관련)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 한글 폰트 설정
try:
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
except:
    try:
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    except:
        pass

matplotlib.rcParams['axes.unicode_minus'] = False  # 음수 부호 표시

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent

# 경로 추가 (프로젝트 루트를 sys.path에 추가)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # 패키지로 import (상대 임포트 문제 해결)
    from anomaly_detection.config import *
    from anomaly_detection.data_loader import DataLoader
    from anomaly_detection.preprocessing import Preprocessor
    print("✅ 모듈 로드 성공\n")
except Exception as e:
    print(f"❌ 모듈 로드 실패: {e}")
    print("\n문제 해결 방법:")
    print("1. 프로젝트 루트에서 실행: cd C:\\workspace\\python\\AI-Modeling")
    print("2. 그 후 실행: python analyze_3d_array.py")
    print("\n또는 main.py 실행 후 데이터 생성: python anomaly_detection/main.py\n")
    sys.exit(1)


def analyze_3d_array_structure():
    """3D Array의 구조를 분석하고 출력"""
    
    print("=" * 80)
    print("1️⃣  데이터 로드 및 전처리")
    print("=" * 80)
    
    try:
        # 데이터 로드 (디렉토리 또는 파일)
        data_source = TRAIN_DATA_DIR if TRAIN_DATA_DIR else TRAIN_DATA_PATH
        
        if not data_source:
            raise ValueError("TRAIN_DATA_DIR 또는 TRAIN_DATA_PATH가 설정되지 않음")
        
        data_loader = DataLoader(
            data_path=str(data_source),
            use_new_format=USE_NEW_CSV_FORMAT
        )
        
        # load_data()는 DataFrame만 반환
        df = data_loader.load_data()
        print(f"✓ 훈련 데이터 로드 완료: {df.shape}")
        
        # X와 y 분리 (DataFrames 유지)
        X_train = df.iloc[:, :-1]
        y_train = df.iloc[:, -1]
        print(f"✓ 특성(X): {X_train.shape}, 레이블(y): {y_train.shape}")
        
        # 전처리
        from anomaly_detection.config import WINDOW_SIZE, NORMALIZE_METHOD
        
        preprocessor = Preprocessor(
            window_size=WINDOW_SIZE,
            normalize_method=NORMALIZE_METHOD
        )
        
        X_preprocessed, y_preprocessed = preprocessor.preprocess_pipeline(
            data=df,
            feature_cols=X_train.columns.tolist(),
            label_col=y_train.name,
            fit_scaler=True
        )
        print(f"✓ 전처리 완료: {X_preprocessed.shape}")
        
    except Exception as e:
        print(f"❌ 데이터 로드 또는 전처리 실패: {e}")
        print("\n✨ 문제 해결 방법:")
        print("1️⃣  먼저 main.py를 실행하여 초기 데이터 생성:")
        print("   python anomaly_detection/main.py")
        print("\n2️⃣  그 후 본 스크립트 재실행:")
        print("   python analyze_3d_array.py")
        print("\n💡 또는 test_data 디렉토리에 CSV 파일을 먼저 배치하세요.")
        return None
    
    return X_preprocessed, y_preprocessed


def display_3d_array_structure(X, y):
    """3D Array의 구조를 상세히 표시"""
    
    # config에서 가져온 값들
    from anomaly_detection.config import WINDOW_SIZE,OVERLAP
    SAMPLE_RATE = 4000  # Hz (데이터셋 표준 설정)
    
    print("\n" + "=" * 80)
    print("2️⃣  3D Array 구조 분석")
    print("=" * 80)
    
    print(f"\n전체 배열:")
    print(f"  형태 (Shape): {X.shape}")
    print(f"  데이터 타입: {X.dtype}")
    print(f"  메모리 크기: {X.nbytes / 1024 / 1024:.2f} MB")
    print(f"  총 데이터 포인트: {X.size:,}")
    
    n_windows, window_size, n_features = X.shape
    
    print(f"\n각 차원 의미:")
    print(f"  차원 0 (n_windows = {n_windows})")
    print(f"    → {n_windows}개의 시간 윈도우")
    print(f"    → 500샘플 데이터를 {window_size}개씩 겹치면서 추출")
    
    print(f"\n  차원 1 (window_size = {window_size})")
    print(f"    → 각 윈도우의 샘플 개수")
    print(f"    → 시간 축 길이 = {window_size} samples")
    print(f"    → 실제 시간 = {window_size / SAMPLE_RATE * 1000:.2f} ms")
    
    print(f"\n  차원 2 (n_features = {n_features})")
    print(f"    → 특성 개수 = {n_features}")
    
    extra_features = 0
    if USE_STATISTICAL_FEATURES:
        extra_features += 5  # RMS, Peak, Crest, Kurtosis, Skewness
    if USE_FFT_FEATURES:
        extra_features += 32  # FFT bins
    
    if extra_features > 0:
        print(f"    → {1} (vibration) + {extra_features} (engineered) = {n_features}")
    else:
        print(f"    → vibration (원본 신호)")


def display_array_statistics(X, y):
    """배열의 통계 정보 표시"""
    
    print("\n" + "=" * 80)
    print("3️⃣  통계 정보")
    print("=" * 80)
    
    print(f"\n전체 데이터 통계:")
    print(f"  평균 (Mean):     {X.mean():.6f}")
    print(f"  표준편차 (Std):  {X.std():.6f}")
    print(f"  최소값 (Min):    {X.min():.6f}")
    print(f"  최대값 (Max):    {X.max():.6f}")
    print(f"  중앙값 (Median): {np.median(X):.6f}")
    
    # 특성별 통계
    if X.shape[2] > 1:
        print(f"\n특성별 통계:")
        for f in range(min(X.shape[2], 5)):  # 최대 5개 특성 표시
            data = X[:, :, f]
            feature_name = ["vibration", "RMS", "Peak", "Crest", "Kurtosis", "Skewness"][f]
            print(f"  {feature_name}:")
            print(f"    평균: {data.mean():.6f}, 표준편차: {data.std():.6f}")
            print(f"    범위: [{data.min():.6f}, {data.max():.6f}]")
    
    # 윈도우별 통계
    print(f"\n윈도우별 평균값 통계:")
    window_means = X.mean(axis=1).mean(axis=1)  # 각 윈도우의 평균
    print(f"  평균 (Mean):    {window_means.mean():.6f}")
    print(f"  표준편차 (Std): {window_means.std():.6f}")
    print(f"  최소값 (Min):   {window_means.min():.6f}")
    print(f"  최대값 (Max):   {window_means.max():.6f}")
    
    print(f"\n윈도우 샘플 통계:")
    for w in [0, len(X)//4, len(X)//2, 3*len(X)//4, -1]:
        if w >= 0 and w < len(X):
            actual_idx = w
        elif w == -1:
            actual_idx = len(X) - 1
        else:
            continue
        
        window_data = X[actual_idx]
        print(f"  Window {actual_idx:3d}:")
        print(f"    평균: {window_data.mean():.6f}")
        print(f"    범위: [{window_data.min():.6f}, {window_data.max():.6f}]")


def display_sample_values(X, y):
    """샘플 값 표시"""
    
    print("\n" + "=" * 80)
    print("4️⃣  샘플 데이터 값")
    print("=" * 80)
    
    print(f"\nWindow 0 (첫 윈도우)의 첫 10개 샘플:")
    for i in range(min(10, X.shape[1])):
        value = X[0, i, 0]
        label_str = "정상" if y[0] == 0 else "이상"
        print(f"  Sample {i:2d}: {value:8.6f}")
    
    if X.shape[1] > 10:
        print(f"  ...(총 {X.shape[1]} 샘플)")
    
    print(f"\nWindow 0의 마지막 10개 샘플:")
    start_idx = max(0, X.shape[1] - 10)
    for i in range(start_idx, X.shape[1]):
        value = X[0, i, 0]
        print(f"  Sample {i:2d}: {value:8.6f}")
    
    print(f"\n다양한 윈도우의 중간 샘플:")
    mid_idx = X.shape[1] // 2
    for w in [0, X.shape[0]//4, X.shape[0]//2, 3*X.shape[0]//4, X.shape[0]-1]:
        value = X[w, mid_idx, 0]
        label_str = "정상" if y[w] == 0 else "이상"
        print(f"  Window {w:3d}, Sample {mid_idx}: {value:8.6f} [{label_str}]")


def calculate_time_information(X):
    """시간 정보 계산 및 표시"""
    
    # config에서 가져온 값들
    from anomaly_detection.config import WINDOW_SIZE, OVERLAP
    SAMPLE_RATE = 4000  # Hz
    
    print("\n" + "=" * 80)
    print("5️⃣  시간 정보")
    print("=" * 80)
    
    n_windows, window_size, _ = X.shape
    
    dt = 1 / SAMPLE_RATE
    window_duration = window_size * dt
    step_size = int(window_size * (1 - OVERLAP))
    step_duration = step_size * dt
    
    print(f"\n샘플링 정보:")
    print(f"  샘플 레이트: {SAMPLE_RATE} Hz")
    print(f"  샘플 간 시간: {dt * 1000:.2f} ms")
    print(f"  윈도우 크기: {window_size} 샘플")
    print(f"  윈도우 지속시간: {window_duration * 1000:.2f} ms")
    print(f"  오버랩: {int(OVERLAP * 100)}%")
    print(f"  스텝 크기: {step_size} 샘플 ({step_duration * 1000:.2f} ms)")
    
    total_time = (n_windows - 1) * step_duration + window_duration
    
    print(f"\n시간 범위:")
    print(f"  총 데이터 길이: 약 {total_time:.3f} 초")
    print(f"  첫 윈도우: 0.000 ~ {window_duration * 1000:.2f} ms")
    print(f"  마지막 윈도우: {(n_windows-1) * step_duration * 1000:.2f} ~ {total_time * 1000:.2f} ms")
    
    print(f"\n각 윈도우의 시간 범위 (샘플):")
    for w in [0, n_windows//4, n_windows//2, 3*n_windows//4, n_windows-1]:
        w_start_time = w * step_duration * 1000
        w_end_time = (w * step_duration + window_duration) * 1000
        w_start_sample = w * step_size
        w_end_sample = w * step_size + window_size - 1
        print(f"  Window {w:3d}: {w_start_sample:5d} ~ {w_end_sample:5d} samples | {w_start_time:7.2f} ~ {w_end_time:7.2f} ms")


def display_label_distribution(y):
    """레이블 분포 표시"""
    
    print("\n" + "=" * 80)
    print("6️⃣  레이블 분포")
    print("=" * 80)
    
    unique, counts = np.unique(y, return_counts=True)
    
    print(f"\n레이블 분포:")
    label_names = {0: "정상", 1: "이상"}
    
    for label, count in zip(unique, counts):
        percentage = count / len(y) * 100
        label_name = label_names.get(label, f"Label {label}")
        print(f"  {label_name:5s}: {count:3d} 윈도우 ({percentage:5.1f}%)")
    
    print(f"\n레이블별 데이터 통계:")
    for label, count in zip(unique, counts):
        label_name = label_names.get(label, f"Label {label}")
        mask = y == label
        X_subset = X[mask]
        print(f"\n  {label_name}:")
        print(f"    윈도우 개수: {count}")
        print(f"    평균값: {X_subset.mean():.6f}")
        print(f"    값 범위: [{X_subset.min():.6f}, {X_subset.max():.6f}]")


def create_visualization(X, y):
    """3D Array 시각화"""
    
    print("\n" + "=" * 80)
    print("7️⃣  시각화 생성")
    print("=" * 80)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("3D Array 분석 및 시각화", fontsize=16, fontweight='bold')
        
        # 1. 윈도우별 평균값
        window_means = X.mean(axis=1).mean(axis=1)
        ax = axes[0, 0]
        ax.plot(window_means, label='Window Mean', linewidth=1.5)
        ax.axhline(y=window_means.mean(), color='r', linestyle='--', 
                  label=f'Overall Mean: {window_means.mean():.3f}')
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Mean Value')
        ax.set_title('윈도우별 평균값')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. 첫 5개 윈도우 신호
        ax = axes[0, 1]
        for w in range(min(5, X.shape[0])):
            signal = X[w, :, 0]
            label_name = "정상" if y[w] == 0 else "이상"
            ax.plot(signal, label=f'Window {w} ({label_name})', alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Vibration (Normalized)')
        ax.set_title('처음 5개 윈도우의 신호')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # 3. 히스토그램
        ax = axes[1, 0]
        ax.hist(X.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=X.mean(), color='r', linestyle='--', 
                  label=f'Mean: {X.mean():.3f}')
        ax.set_xlabel('Vibration Value')
        ax.set_ylabel('Frequency')
        ax.set_title('값 분포 (전체)')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # 4. 레이블별 비교
        ax = axes[1, 1]
        normal_means = X[y == 0].mean(axis=1).mean(axis=1)
        abnormal_means = X[y == 1].mean(axis=1).mean(axis=1)
        
        box_data = [normal_means, abnormal_means]
        bp = ax.boxplot(box_data, tick_labels=['정상', '이상'], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Window Mean Value')
        ax.set_title('레이블별 평균값 비교')
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 저장
        output_path = PROJECT_ROOT / "3d_array_visualization.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"\n✓ 시각화 저장: {output_path}")
        
        # 화면에 표시
        plt.show()
        
    except Exception as e:
        print(f"❌ 시각화 생성 실패: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🔍 3D Array 상세 분석 도구")
    print("=" * 80)
    
    # 데이터 로드 및 전처리
    result = analyze_3d_array_structure()
    
    if result is not None:
        X, y = result
        
        # 분석 수행
        display_3d_array_structure(X, y)
        display_array_statistics(X, y)
        display_sample_values(X, y)
        calculate_time_information(X)
        display_label_distribution(y)
        
        # 시각화
        create_visualization(X, y)
        
        print("\n" + "=" * 80)
        print("✅ 분석 완료!")
        print("=" * 80)
        print("""
결론:
- 3D Array (n_windows=?, window_size=64, n_features=?)
- 시계열 데이터를 시간 윈도우로 분할
- 각 윈도우는 정규화된 진동 신호 포함
- 모델 학습에 바로 사용 가능
        """)
    else:
        print("\n프로젝트 설정 및 데이터 생성 후 다시 실행해주세요.")
