"""
시리얼 데이터 시뮬레이터

실제 센서 없이 시리얼 포트를 통해 들어오는 데이터를 시뮬레이션합니다.
- 정상 신호 생성 (사인파 + 노이즈)
- 이상 신호 생성 (스파이크, 고주파)
- 시리얼 포트로 전송
"""

import numpy as np
import time
import sys
from pathlib import Path

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from anomaly_detection import config
except:
    print("❌ 설정 로드 실패. 프로젝트 루트에서 실행해주세요.")
    sys.exit(1)


class SerialDataSimulator:
    """시리얼 데이터 시뮬레이터"""
    
    def __init__(self, sample_rate: int = 4000, duration: float = 2.0):
        """
        초기화
        
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            duration: 생성 시간 (초)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        
    def generate_normal_signal(self):
        """정상 신호 생성 (안정적인 진동)"""
        t = np.linspace(0, self.duration, self.n_samples)
        
        # 기본 신호: 저주파 사인파 + 고주파 노이즈
        signal = (
            0.5 * np.sin(2 * np.pi * 20 * t) +      # 20Hz 기본
            0.2 * np.sin(2 * np.pi * 50 * t) +      # 50Hz 배음
            0.1 * np.random.randn(self.n_samples)   # 노이즈
        )
        
        return signal
    
    def generate_anomaly_signal(self, anomaly_type: str = 'spike'):
        """이상 신호 생성"""
        t = np.linspace(0, self.duration, self.n_samples)
        
        # 기본 신호
        signal = (
            0.5 * np.sin(2 * np.pi * 20 * t) +
            0.1 * np.random.randn(self.n_samples)
        )
        
        if anomaly_type == 'spike':
            # 스파이크 (임펄스)
            spike_indices = np.random.choice(self.n_samples, size=5, replace=False)
            signal[spike_indices] += 3.0
            
        elif anomaly_type == 'harmonic':
            # 고주파 성분 (베어링 손상)
            signal += 1.5 * np.sin(2 * np.pi * 200 * t)
            
        elif anomaly_type == 'drift':
            # 신호 드리프트 (선형 증가)
            signal += np.linspace(0, 2.0, self.n_samples)
            
        elif anomaly_type == 'discontinuity':
            # 불연속 (충격)
            mid = self.n_samples // 2
            signal[mid:] += 1.5
        
        return signal
    
    def format_serial_message(self, value: float) -> str:
        """시리얼 메시지 형식"""
        return f"{value:.6f}\n"
    
    def print_signal_info(self, signal: np.ndarray, label: str):
        """신호 정보 출력"""
        print(f"\n{label}:")
        print(f"  평균: {signal.mean():.6f}")
        print(f"  표준편차: {signal.std():.6f}")
        print(f"  범위: [{signal.min():.6f}, {signal.max():.6f}]")
        print(f"  RMS: {np.sqrt(np.mean(signal**2)):.6f}")


class VirtualSerialPort:
    """가상 시리얼 포트 (테스트용)"""
    
    def __init__(self):
        self.buffer = []
        self.write_count = 0
        self.read_count = 0
    
    def write(self, data: bytes):
        """데이터 쓰기"""
        self.buffer.append(data)
        self.write_count += 1
    
    def readline(self) -> bytes:
        """데이터 읽기"""
        if self.buffer:
            self.read_count += 1
            return self.buffer.pop(0)
        return b''
    
    def close(self):
        """포트 닫기"""
        print(f"✓ 가상 포트 닫음 (쓰기: {self.write_count}, 읽기: {self.read_count})")


def test_data_generation():
    """데이터 생성 테스트"""
    
    print("=" * 80)
    print("시리얼 데이터 시뮬레이터 - 신호 생성 테스트")
    print("=" * 80)
    
    simulator = SerialDataSimulator(sample_rate=4000, duration=1.0)
    
    # 1. 정상 신호
    print("\n1️⃣  정상 신호 생성")
    normal_signal = simulator.generate_normal_signal()
    simulator.print_signal_info(normal_signal, "정상 신호")
    
    # 2. 여러 이상 신호
    anomaly_types = ['spike', 'harmonic', 'drift', 'discontinuity']
    
    for i, atype in enumerate(anomaly_types, 1):
        print(f"\n{i+1}️⃣  이상 신호: {atype}")
        anomaly_signal = simulator.generate_anomaly_signal(atype)
        simulator.print_signal_info(anomaly_signal, f"{atype} 신호")
    
    return simulator


def simulate_real_time_streaming():
    """실시간 데이터 스트리밍 시뮬레이션"""
    
    print("\n" + "=" * 80)
    print("실시간 데이터 스트리밍 시뮬레이션")
    print("=" * 80)
    
    simulator = SerialDataSimulator(sample_rate=4000, duration=3.0)
    
    # 신호 생성
    print("\n정상 신호 5초 →  이상 신호 5초")
    normal = simulator.generate_normal_signal()
    # 더 긴 신호를 위해 반복
    signal_normal = np.tile(normal, 5)  # 5초
    signal_abnormal = np.tile(simulator.generate_anomaly_signal('spike'), 5)  # 5초
    
    signal_combined = np.concatenate([signal_normal, signal_abnormal])
    
    # 시뮬레이션
    print("\n시뮬레이션 시작 (처음 100개 샘플만 표시)...")
    print(f"{'시간(s)':>10} {'값':>15} {'상태':>10}")
    print("-" * 35)
    
    sample_interval = 1.0 / 4000  # 4000Hz = 0.25ms
    
    for i, value in enumerate(signal_combined[:100]):
        timestamp = i * sample_interval
        status = "정상" if i < 500 else "이상"
        print(f"{timestamp:10.4f} {value:15.6f} {status:>10}")
        time.sleep(sample_interval * 10)  # 10배 빠르게
    
    print(f"... ({len(signal_combined)} 샘플 중 100개만 표시)")


def test_serial_data_format():
    """시리얼 데이터 형식 테스트"""
    
    print("\n" + "=" * 80)
    print("시리얼 데이터 형식 테스트")
    print("=" * 80)
    
    simulator = SerialDataSimulator()
    test_values = [-2.5, -1.0, 0.0, 1.5, 3.2]
    
    print("\n생성된 메시지 형식:")
    for value in test_values:
        message = simulator.format_serial_message(value)
        print(f"  값: {value:6.1f} → '{message.strip()}'")


def generate_synthetic_dataset(output_file: str = 'synthetic_test_data.csv'):
    """합성 테스트 데이터셋 생성"""
    
    print("\n" + "=" * 80)
    print("합성 테스트 데이터셋 생성")
    print("=" * 80)
    
    simulator = SerialDataSimulator(sample_rate=4000, duration=2.0)
    
    # 데이터 생성
    n_normal = 10
    n_abnormal = 10
    
    all_data = []
    
    # 정상 신호
    print(f"\n정상 신호 {n_normal}개 생성 중...")
    for i in range(n_normal):
        signal = simulator.generate_normal_signal()
        for j, value in enumerate(signal):
            all_data.append({
                'time': j / 4000.0,
                'vibration': value,
                'label': '정상'
            })
        if (i + 1) % 2 == 0:
            print(f"  {i+1}/{n_normal} 완료")
    
    # 이상 신호
    print(f"이상 신호 {n_abnormal}개 생성 중...")
    anomaly_types = ['spike', 'harmonic', 'drift', 'discontinuity']
    for i in range(n_abnormal):
        atype = anomaly_types[i % len(anomaly_types)]
        signal = simulator.generate_anomaly_signal(atype)
        for j, value in enumerate(signal):
            all_data.append({
                'time': j / 4000.0,
                'vibration': value,
                'label': '이상'
            })
        if (i + 1) % 2 == 0:
            print(f"  {i+1}/{n_abnormal} 완료")
    
    # 저장
    import pandas as pd
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ 저장 완료: {output_file}")
    print(f"  행 수: {len(df)}")
    print(f"  정상: {len(df[df['label'] == '정상'])}")
    print(f"  이상: {len(df[df['label'] == '이상'])}")
    
    return output_file


# ============================================================================
# 메인
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🔧 시리얼 데이터 시뮬레이터")
    print("=" * 80)
    
    # 1. 신호 생성 테스트
    simulator = test_data_generation()
    
    # 2. 데이터 형식 테스트
    test_serial_data_format()
    
    # 3. 실시간 스트리밍 시뮬레이션
    simulate_real_time_streaming()
    
    # 4. 합성 데이터셋 생성
    output_file = generate_synthetic_dataset()
    
    print("\n" + "=" * 80)
    print("✅ 시뮬레이터 테스트 완료!")
    print("=" * 80)
    print(f"""
다음 단계:
1️⃣  test_serial_inference.py를 사용하여 실시간 테스트
2️⃣  생성된 CSV 파일로 모델 재훈련 (선택사항)
3️⃣  실제 시리얼 포트 연결 및 테스트
    """)
