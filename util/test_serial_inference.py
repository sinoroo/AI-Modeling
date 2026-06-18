"""
시리얼 데이터 실시간 추론 테스트

훈련된 모델을 사용하여 시리얼 포트를 통해 들어오는 데이터를 실시간으로 분석합니다.
- 데이터 수집 및 버퍼링
- 실시간 전처리
- 모델 추론
- 결과 저장 및 시각화
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
from pathlib import Path
from collections import deque
import time
from sklearn.preprocessing import StandardScaler
import pickle
import json

# ✅ 경고 무시 및 한글 폰트 설정
warnings.filterwarnings('ignore')
try:
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent
PARENT_ROOT = PROJECT_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from anomaly_detection.config import *
    from anomaly_detection.preprocessing import Preprocessor
    from serial_data_simulator import SerialDataSimulator
    print("✅ 모듈 로드 성공\n")
except Exception as e:
    print(f"❌ 모듈 로드 실패: {e}\n")
    print(f"sys.path: {sys.path}\n")
    sys.exit(1)


# ============================================================================
# 1. 시리얼 데이터 읽음이 및 버퍼 관리
# ============================================================================

class RealtimeDataBuffer:
    """실시간 데이터 버퍼 관리"""
    
    def __init__(self, window_size: int = None):
        """
        버퍼 초기화
        
        Args:
            window_size: 윈도우 크기 (None이면 config에서 가져옴)
        """
        self.window_size = window_size or WINDOW_SIZE
        self.buffer = deque(maxlen=self.window_size)
        self.history = []  # 모든 데이터 저장
        self.timestamps = deque(maxlen=self.window_size)
        self.current_time = 0
        
        print(f"[RealtimeDataBuffer] 윈도우 크기: {self.window_size}")
        
    def add_sample(self, value: float, timestamp: float = None):
        """샘플 추가"""
        if timestamp is None:
            timestamp = self.current_time
            self.current_time += 1.0 / 4000  # 4000Hz 가정
        
        self.buffer.append(value)
        self.timestamps.append(timestamp)
        self.history.append((timestamp, value))
    
    def is_ready(self) -> bool:
        """윈도우 크기만큼 데이터 있는지 확인"""
        return len(self.buffer) >= self.window_size
    
    def get_window(self) -> np.ndarray:
        """현재 윈도우 반환"""
        if self.is_ready():
            return np.array(list(self.buffer)).reshape(-1, 1)
        return None
    
    def get_history_df(self) -> pd.DataFrame:
        """전체 이력을 DataFrame으로"""
        if not self.history:
            return pd.DataFrame()
        timestamps, values = zip(*self.history)
        return pd.DataFrame({
            'time': timestamps,
            'vibration': values
        })


# ============================================================================
# 2. 실시간 모델 로더
# ============================================================================

class RealtimeModelLoader:
    """훈련된 모델 로드"""
    
    @staticmethod
    def find_latest_model(model_dir: str = 'results/models') -> str:
        """최신 모델 파일 찾기"""
        model_path = Path(model_dir)
        if not model_path.exists():
            return None
        
        model_files = sorted(model_path.glob('*.pkl'), 
                            key=lambda p: p.stat().st_mtime, 
                            reverse=True)
        return str(model_files[0]) if model_files else None
    
    @staticmethod
    def load_model(model_path: str):
        """모델 로드"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ 모델 로드: {model_path}")
            return model
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return None
    
    @staticmethod
    def load_scaler(scaler_path: str = None):
        """Scaler 로드"""
        if scaler_path is None:
            scaler_path = Path(PROJECT_ROOT) / 'results' / 'models' / 'scaler.pkl'
        
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✓ Scaler 로드: {scaler_path}")
                return scaler
            except:
                pass
        
        # 기본 StandardScaler 생성
        print("⚠️  Scaler 로드 실패, 새로운 StandardScaler 생성")
        return StandardScaler()


# ============================================================================
# 3. 실시간 추론 엔진
# ============================================================================

class RealtimeInferenceEngine:
    """실시간 추론 엔진"""
    
    def __init__(self, model, scaler, preprocessor=None):
        self.model = model
        self.scaler = scaler
        self.preprocessor = preprocessor or Preprocessor()
        self.predictions = []
        self.probabilities = []
        self.results_log = []
    
    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """윈도우 전처리 - 64개 샘플 → 통계 피처 6개 → 정규화"""
        try:
            # 1. 통계 피처 생성 (64개 샘플 → 6개 피처)
            # window shape: (64, 1) → features shape: (6,)
            
            # 첮 번째: 직접 평탄화 (간단한 방법)
            # window.reshape(1, -1): (64, 1) → (1, 64)
            window_flat = window.reshape(1, -1)  # (1, 64)
            
            # 2. 정규화
            window_scaled = self.scaler.transform(window_flat)  # (1, 64)
            
            return window_scaled  # (1, 64)
        except Exception as e:
            print(f"❌ 전처리 오류: {e}")
            return None
    
    def predict(self, window_preprocessed: np.ndarray, timestamp: float = None) -> tuple:
        """
        예측 수행
        
        Returns:
            (prediction, confidence, anomaly_score)
        """
        try:
            if window_preprocessed is None or window_preprocessed.size == 0:
                return None, None, None
            
            # 모델 예측
            prediction = self.model.predict(window_preprocessed)[0]
            
            # 확률 (가능하면)
            confidence = 0.0
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(window_preprocessed)
                if proba.shape[1] > 1:
                    confidence = max(proba[0])
            elif hasattr(self.model, 'decision_function'):
                # SVM 등의 decision_function 사용
                confidence = abs(self.model.decision_function(window_preprocessed)[0])
            
            # 레이블
            label = "정상" if prediction == 0 else "이상"
            
            # 기록
            result = {
                'timestamp': timestamp or time.time(),
                'prediction': prediction,
                'confidence': confidence,
                'label': label
            }
            self.results_log.append(result)
            
            return prediction, confidence, label
        
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            return None, None, None


# ============================================================================
# 4. 실시간 테스트 시뮬레이션
# ============================================================================

def test_realtime_inference_with_simulated_data():
    """시뮬레이션 데이터로 실시간 추론 테스트"""
    
    print("=" * 80)
    print("🧪 실시간 추론 테스트 - 시뮬레이션 데이터")
    print("=" * 80)
    
    # 1. 데이터 시뮬레이터 준비
    simulator = SerialDataSimulator(sample_rate=4000, duration=5.0)
    
    # 정상 신호 5초 + 이상 신호 5초
    normal_signal = np.tile(simulator.generate_normal_signal(), 5)
    abnormal_signal = np.tile(simulator.generate_anomaly_signal('spike'), 5)
    
    combined_signal = np.concatenate([normal_signal, abnormal_signal])
    
    print(f"\n테스트 신호:")
    print(f"  정상: 샘플 0-{len(normal_signal)-1}")
    print(f"  이상: 샘플 {len(normal_signal)}-{len(combined_signal)-1}")
    print(f"  총 샘플: {len(combined_signal)}")
    
    # 2. 모델 준비
    print(f"\n모델 준비 중...")
    
    # 더미 모델 생성 (실제로는 훈련된 모델 로드)
    from sklearn.ensemble import RandomForestClassifier
    
    # 더미 훈련 데이터로 모델 학습
    X_dummy = np.random.randn(50, 64)
    y_dummy = np.random.randint(0, 2, 50)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    print(f"✓ RandomForest 모델 준비 완료")
    
    # 3. Scaler 준비
    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, 64))
    
    # 4. 추론 엔진 준비
    engine = RealtimeInferenceEngine(model, scaler)
    buffer = RealtimeDataBuffer(window_size=64)
    
    # 5. 스트리밍 추론
    print(f"\n실시간 추론 시작...")
    print(f"{'시간(s)':>10} {'값':>15} {'상태':>10} {'예측':>8} {'신뢰도':>10}")
    print("-" * 55)
    
    window_count = 0
    predictions_list = []
    timestamps_list = []
    
    for i, value in enumerate(combined_signal):
        # 버퍼에 추가
        timestamp = i / 4000.0
        buffer.add_sample(value, timestamp)
        
        # 윈도우 단위 처리 (32 샘플마다)
        if (i + 1) % 32 == 0 and buffer.is_ready():
            window = buffer.get_window()
            
            # 전처리 및 예측
            window_preprocessed = engine.preprocess_window(window)
            prediction, confidence, label = engine.predict(window_preprocessed, timestamp)
            
            if prediction is not None:
                window_count += 1
                predictions_list.append(prediction)
                timestamps_list.append(timestamp)
                
                # 상태
                actual_status = "정상" if i < len(normal_signal) else "이상"
                
                print(f"{timestamp:10.4f} {value:15.6f} {actual_status:>10} {label:>8} {confidence:10.4f}")
    
    # 6. 결과 요약
    print(f"\n결과 요약:")
    print(f"  처리된 윈도우: {window_count}")
    print(f"  총 추론: {len(engine.results_log)}")
    
    if window_count > len(normal_signal) / 32:
        print(f"\n⚠️  후반부 이상 신호 감지 여부 확인 필요")
    
    return engine, buffer, predictions_list, timestamps_list


# ============================================================================
# 5. 결과 시각화
# ============================================================================

def visualize_realtime_results(signal_data: np.ndarray, predictions: list, 
                               split_point: int = None):
    """실시간 추론 결과 시각화"""
    
    print(f"\n시각화 생성 중...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    time_axis = np.arange(len(signal_data)) / 4000.0
    
    # 1. 원본 신호
    ax = axes[0]
    ax.plot(time_axis, signal_data, linewidth=1.5, label='Signal')
    if split_point:
        ax.axvline(x=split_point/4000.0, color='red', linestyle='--', 
                  linewidth=2, label=f'Anomaly Start (t={split_point/4000.0:.2f}s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Original Signal')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. 신호 통계
    ax = axes[1]
    window_size = 64
    window_means = []
    window_times = []
    
    for i in range(0, len(signal_data) - window_size, 32):
        window = signal_data[i:i+window_size]
        window_means.append(window.mean())
        window_times.append((i + window_size/2) / 4000.0)
    
    ax.plot(window_times, window_means, 'o-', linewidth=2, markersize=6)
    if split_point:
        ax.axvline(x=split_point/4000.0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Window Mean')
    ax.set_title('Sliding Window Statistics')
    ax.grid(alpha=0.3)
    
    # 3. 예측 결과
    ax = axes[2]
    
    if predictions and window_times:
        colors = ['green' if p == 0 else 'red' for p in predictions[:len(window_times)]]
        ax.scatter(window_times[:len(predictions)], predictions[:len(window_times)], 
                  c=colors, s=100, alpha=0.6)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['정상', '이상'])
    
    if split_point:
        ax.axvline(x=split_point/4000.0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Prediction')
    ax.set_title('Model Predictions')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / "realtime_inference_results.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"✓ 시각화 저장: {output_path}")
    plt.show()


# ============================================================================
# 6. 결과 저장
# ============================================================================

def save_inference_results(engine: RealtimeInferenceEngine, output_file: str = None):
    """추론 결과 저장"""
    
    if output_file is None:
        output_file = PROJECT_ROOT / "inference_results.json"
    
    # 커스텀 JSON 인코더
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            return super().default(obj)
    
    # 결과 데이터 구성
    results_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_predictions': len(engine.results_log),
        'results': engine.results_log
    }
    
    # 커스텀 인코더로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"✓ 결과 저장: {output_file}")


# ============================================================================
# 메인
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("⚡ 실시간 시리얼 데이터 추론 테스트")
    print("=" * 80)
    
    # 1. 시뮬레이션 테스트
    engine, buffer, predictions, timestamps = test_realtime_inference_with_simulated_data()
    
    # 2. 결과 시각화
    buffer_df = buffer.get_history_df()
    if not buffer_df.empty:
        signal_data = buffer_df['vibration'].values
        split_point = len(signal_data) // 2
        visualize_realtime_results(signal_data, predictions, split_point)
    
    # 3. 결과 저장
    save_inference_results(engine)
    
    print("\n" + "=" * 80)
    print("✅ 실시간 추론 테스트 완료!")
    print("=" * 80)
    print(f"""
다음 단계:
1️⃣  실제 시리얼 포트 연결 (anomaly_detection/inference_serial.py 참조)
2️⃣  포트 번호 설정 (예: COM3, /dev/ttyUSB0)
3️⃣  센서 데이터 수집 및 실시간 분석

생성된 파일:
  - realtime_inference_results.png (예측 결과 시각화)
  - inference_results.json (상세 결과 로그)
    """)
