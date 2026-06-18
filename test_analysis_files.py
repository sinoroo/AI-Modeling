#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
분석 폴더의 파일들 실행 상태 확인 도구
"""

import os
import sys
import subprocess
from pathlib import Path

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

def run_script(script_name: str, description: str = ""):
    """Python 스크립트 실행 및 결과 확인"""
    script_path = ANALYSIS_DIR / script_name
    
    print(f"\n{'='*80}")
    print(f"📄 {script_name} 실행")
    print(f"{'='*80}")
    if description:
        print(f"설명: {description}\n")
    
    if not script_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {script_path}")
        return False
    
    try:
        # 현재 디렉토리를 프로젝트 루트로 변경
        os.chdir(PROJECT_ROOT)
        
        # 스크립트 실행 (타임아웃 60초)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        
        # 결과 출력
        if result.returncode == 0:
            print("✅ 실행 성공!")
            if result.stdout:
                # 처음 2000 글자만 출력
                output = result.stdout[:2000]
                print("\n[출력]")
                print(output)
                if len(result.stdout) > 2000:
                    print(f"\n... (더 많은 출력이 있습니다. 총 {len(result.stdout)} 글자)")
        else:
            print("❌ 실행 실패!")
            if result.stderr:
                error = result.stderr[:1000]
                print("\n[에러]")
                print(error)
                if len(result.stderr) > 1000:
                    print(f"\n... (더 많은 에러가 있습니다. 총 {len(result.stderr)} 글자)")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏱️  타임아웃 (60초 초과)")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    print("╔" + "═"*78 + "╗")
    print("║ 분석 폴더 파일 실행 상태 확인" + " "*42 + "║")
    print("╚" + "═"*78 + "╝")
    
    # 테스트할 파일들
    test_files = [
        ("visualize_3d_array.py", "3D Array 구조 시각화 및 설명 (외부 데이터 불필요)"),
        ("ml_input_examples.py", "ML 모델 입력 데이터 예제 (동적 데이터 생성)"),
        ("analyze_2d_data.py", "2D 데이터 분석 및 시각화 (실제 데이터 필요)"),
        ("analyze_3d_array.py", "3D Array 구조 분석 (실제 데이터 필요)"),
        ("fft_sample_comparison.py", "FFT 샘플 크기별 모델 성능 비교 (실제 데이터 필요)"),
        ("anomaly_detection_comparison.py", "이상치 탐지 모델 비교 (실제 데이터 필요)"),
    ]
    
    results = {}
    
    for script_name, description in test_files:
        success = run_script(script_name, description)
        results[script_name] = "✅ 성공" if success else "❌ 실패"
    
    # 결과 요약
    print(f"\n\n{'='*80}")
    print("📊 실행 결과 요약")
    print(f"{'='*80}\n")
    
    for script_name, status in results.items():
        print(f"{status} {script_name}")
    
    success_count = sum(1 for s in results.values() if "성공" in s)
    total_count = len(results)
    
    print(f"\n총 {success_count}/{total_count} 파일이 실행 가능합니다.")

if __name__ == "__main__":
    main()
