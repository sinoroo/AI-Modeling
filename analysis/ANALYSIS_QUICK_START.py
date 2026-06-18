#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║         DATA ANALYSIS & VISUALIZATION QUICK START GUIDE                   ║
║              3D Array 학습 및 데이터 분석 통합 도구                        ║
╚════════════════════════════════════════════════════════════════════════════╝

analysis 폴더의 모든 분석 도구를 쉽게 실행할 수 있는 메뉴 기반 스크립트
사용자의 학습 단계에 따라 추천하는 프로세스를 제시합니다.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title, subtitle=""):
    """보기 좋은 헤더 출력"""
    print(f"\n{'╔' + '═' * 78 + '╗'}")
    print(f"║ {title:<76} ║")
    if subtitle:
        print(f"║ {subtitle:<76} ║")
    print(f"{'╚' + '═' * 78 + '╝'}\n")

def print_menu_section(title):
    """메뉴 섹션 헤더"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}\n")

def main():
    # 분석 폴더를 현재 작업 디렉토리로 설정
    analysis_dir = Path(__file__).parent
    
    while True:
        print_header("📊 DATA ANALYSIS & VISUALIZATION", "analysis/ 통합 도구")
        
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                          어떤 작업을 하고 싶으신가요?                        │
└─────────────────────────────────────────────────────────────────────────────┘

🟢 [1] 🎓 처음 배우기 (추천: 초급자)
    → 3D Array 구조부터 모델 입출력까지 전체 흐름 학습
    
🟡 [2] 🔍 데이터 검증하기 (추천: 중급자)
    → 실제 전처리된 데이터의 통계 및 품질 확인
    
🟠 [3] 🛠️  개별 도구 실행 (추천: 고급자)
    → 필요한 분석 도구를 직접 선택해서 실행

🔵 [4] 📚 빠른 참조 보기
    → 모든 모델의 입출력 형식 확인

⚪ [0] 종료

""")
        
        choice = input("💬 선택 (0-4): ").strip()
        
        if choice == '0':
            print("\n✅ 분석 도구를 종료합니다.\n")
            break
        
        elif choice == '1':
            run_learning_path(analysis_dir)
        
        elif choice == '2':
            run_validation_path(analysis_dir)
        
        elif choice == '3':
            run_individual_tools(analysis_dir)
        
        elif choice == '4':
            run_quick_reference(analysis_dir)
        
        else:
            print("❌ 잘못된 선택입니다. 다시 시도해주세요.\n")
            input("Press Enter to continue...")


def run_learning_path(analysis_dir):
    """처음 배우기 프로세스"""
    print_header("🎓 처음 배우기 프로세스", "3D Array 구조 → 변환 방식 → 모델 스펙")
    
    steps = [
        {
            "num": 1,
            "name": "visualize_3d_array.py",
            "desc": "3D Array 구조 이해하기",
            "detail": """
    📌 이 단계에서 배우는 것:
    • 3D Array의 구조 (shape: 154, 64, 6)
    • 각 차원의 의미
    • 배열 슬라이싱 및 인덱싱 방법
    • 데이터에 접근하는 다양한 방법
    
    💡 소요 시간: 약 2-3분 (코드 읽기)
    """
        },
        {
            "num": 2,
            "name": "ml_input_examples.py",
            "desc": "3D → 2D 변환 방식 비교",
            "detail": """
    📌 이 단계에서 배우는 것:
    • 3D Array를 고전 ML에 입력하는 방법 (4가지)
    • 각 변환 방식의 장단점
    • 성능 비교 결과
    • 권장되는 입력 형식
    
    💡 소요 시간: 약 30초 (실행 시간)
    """
        },
        {
            "num": 3,
            "name": "MODEL_IO_QUICK_REF.py",
            "desc": "모든 모델의 입출력 형식 확인",
            "detail": """
    📌 이 단계에서 배우는 것:
    • CSV → DataFrame 변환 형식
    • 전처리 후 3D Array 형식
    • 각 모델별 입출력 스펙
    • 데이터 흐름 전체 맵
    
    💡 소요 시간: 약 1-2분 (정보 확인)
    """
        }
    ]
    
    print("📚 추천 학습 순서:\n")
    
    for step in steps:
        print(f"[단계 {step['num']}] {step['desc']}")
        print(f"└─ 파일: {step['name']}\n{step['detail']}\n")
    
    print("─" * 80)
    print("\n실행할 단계를 선택하세요:\n")
    for step in steps:
        print(f"  [{step['num']}] {step['desc']}")
    print(f"  [0] 돌아가기\n")
    
    step_choice = input("💬 단계 선택 (0-3): ").strip()
    
    if step_choice in ['1', '2', '3']:
        step_idx = int(step_choice) - 1
        script_name = steps[step_idx]['name']
        run_script(analysis_dir, script_name, steps[step_idx]['num'])
    elif step_choice != '0':
        print("❌ 잘못된 선택입니다.")


def run_validation_path(analysis_dir):
    """데이터 검증 프로세스"""
    print_header("🔍 데이터 검증 프로세스", "실제 데이터 분석 및 최적화")
    
    steps = [
        {
            "num": 1,
            "name": "analyze_3d_array.py",
            "desc": "3D 데이터 검증",
            "detail": """
    📌 분석 내용:
    • 실제 전처리된 데이터 로드
    • 데이터 통계 (평균, 표준편차, 최대/최소)
    • 특성별 분포 분석
    • 3D 배열 시각화
    • 데이터 품질 확인
    
    💡 소요 시간: 약 1-2분
    💾 입력: data_new_format/train/ CSV 파일
    📊 출력: 통계 분석 차트
    """
        },
        {
            "num": 2,
            "name": "analyze_2d_data.py",
            "desc": "2D 데이터 분석 & 최적화",
            "detail": """
    📌 분석 내용:
    • 2D 평탄화 데이터 상세 분석
    • 특성 상관관계 히트맵
    • PCA 차원 축소 효과 검증
    • t-SNE 군집 시각화
    • 고전 ML 모델 성능 비교
    
    💡 소요 시간: 약 2-3분
    💾 입력: 실제 전처리된 3D 데이터
    📊 출력: PCA, 상관관계, 모델 성능 차트
    """
        }
    ]
    
    print("🔍 권장 검증 순서:\n")
    
    for step in steps:
        print(f"[단계 {step['num']}] {step['desc']}")
        print(f"└─ 파일: {step['name']}\n{step['detail']}\n")
    
    print("─" * 80)
    print("\n검증할 단계를 선택하세요:\n")
    for step in steps:
        print(f"  [{step['num']}] {step['desc']}")
    print(f"  [0] 돌아가기\n")
    
    step_choice = input("💬 단계 선택 (0-2): ").strip()
    
    if step_choice in ['1', '2']:
        step_idx = int(step_choice) - 1
        script_name = steps[step_idx]['name']
        run_script(analysis_dir, script_name, steps[step_idx]['num'])
    elif step_choice != '0':
        print("❌ 잘못된 선택입니다.")


def run_individual_tools(analysis_dir):
    """개별 도구 실행 프로세스"""
    print_header("🛠️  개별 분석 도구 선택", "필요한 도구를 직접 선택해서 실행")
    
    tools = [
        {
            "name": "visualize_3d_array.py",
            "desc": "3D Array 시각화 & 실습",
            "use_case": "3D 배열 구조와 슬라이싱 방법 학습"
        },
        {
            "name": "ml_input_examples.py",
            "desc": "ML 입력 형식 비교",
            "use_case": "3D→2D 변환 방식별 성능 비교"
        },
        {
            "name": "analyze_3d_array.py",
            "desc": "3D 데이터 분석",
            "use_case": "실제 전처리 데이터 통계 분석"
        },
        {
            "name": "analyze_2d_data.py",
            "desc": "2D 데이터 분석",
            "use_case": "PCA, 상관관계, 모델 성능 분석"
        },
        {
            "name": "MODEL_IO_QUICK_REF.py",
            "desc": "모델 입출력 스펙",
            "use_case": "모든 모델의 입출력 형식 확인"
        }
    ]
    
    print("🛠️  사용 가능한 도구:\n")
    
    for idx, tool in enumerate(tools, 1):
        print(f"  [{idx}] {tool['desc']}")
        print(f"      └─ 파일: {tool['name']}")
        print(f"      └─ 사용 시점: {tool['use_case']}\n")
    
    print(f"  [0] 돌아가기\n")
    
    choice = input("💬 도구 선택 (0-5): ").strip()
    
    if choice in ['1', '2', '3', '4', '5']:
        tool_idx = int(choice) - 1
        script_name = tools[tool_idx]['name']
        run_script(analysis_dir, script_name, int(choice))
    elif choice != '0':
        print("❌ 잘못된 선택입니다.")


def run_quick_reference(analysis_dir):
    """빠른 참조 실행"""
    print_header("📚 모델 입출력 형식 빠른 참조")
    
    print("모든 모델의 입출력 형식을 확인합니다...\n")
    
    script_path = analysis_dir / "MODEL_IO_QUICK_REF.py"
    
    print("=" * 80)
    print("실행 결과:")
    print("=" * 80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=analysis_dir,
            capture_output=False
        )
        
        if result.returncode != 0:
            print("\n⚠️  스크립트 실행 중 경고가 발생했습니다.")
        else:
            print("\n✅ 완료!")
    
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
    
    input("\nPress Enter to continue...")


def run_script(analysis_dir, script_name, step_num=None):
    """스크립트 실행"""
    script_path = analysis_dir / script_name
    
    if not script_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {script_name}")
        input("Press Enter to continue...")
        return
    
    step_prefix = f"[단계 {step_num}] " if step_num else ""
    print_header(f"🚀 {step_prefix}{script_name} 실행 중...", "이 작업은 1-3분 정도 소요됩니다.")
    
    print("=" * 80)
    print("실행 결과:")
    print("=" * 80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=analysis_dir,
            capture_output=False
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("✅ 실행 완료!")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(f"⚠️  스크립트가 오류를 반환했습니다 (code: {result.returncode})")
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 중단됨")
    
    except Exception as e:
        print(f"\n❌ 실행 오류: {e}")
    
    print("\n다음 단계를 선택하세요:\n")
    print("  [1] 다른 분석 도구 실행")
    print("  [0] 메인 메뉴로 돌아가기\n")
    
    next_choice = input("💬 선택 (0-1): ").strip()
    
    if next_choice == '1':
        main()
    # '0' 또는 다른 입력은 이전 메뉴로 돌아감


def print_summary():
    """프로세스 요약"""
    print_header("📌 분석 도구 모음 개요")
    
    summary = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🎓 처음 배우기 순서                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: visualize_3d_array.py                                             │
│  ├─ 3D Array 구조 이해                                                     │
│  ├─ 배열 슬라이싱 방법                                                     │
│  └─ 예제 코드로 직접 실습 (2-3분)                                          │
│                                                                             │
│  Step 2: ml_input_examples.py                                              │
│  ├─ 3D→2D 변환의 4가지 방식                                                │
│  ├─ 각 방식의 성능 비교                                                   │
│  └─ 권장 방식 결정 (30초)                                                  │
│                                                                             │
│  Step 3: MODEL_IO_QUICK_REF.py                                             │
│  ├─ 모든 모델의 입출력 스펙                                                │
│  ├─ 데이터 흐름 전체 맵                                                    │
│  └─ 빠른 참고용 (1-2분)                                                    │
│                                                                             │
├─ 총 소요 시간: 약 5-10분 (개념 이해)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      🔍 실제 데이터 검증 순서                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: analyze_3d_array.py                                               │
│  ├─ 실제 데이터 로드 및 검증                                              │
│  ├─ 통계 분석 및 시각화                                                    │
│  └─ 데이터 품질 확인 (1-2분)                                               │
│                                                                             │
│  Step 2: analyze_2d_data.py                                                │
│  ├─ 2D 변환 데이터 분석                                                    │
│  ├─ PCA 차원 축소 검증                                                     │
│  ├─ 특성 상관관계 분석                                                     │
│  └─ 모델 성능 비교 (2-3분)                                                 │
│                                                                             │
├─ 총 소요 시간: 약 5-10분 (검증 + 최적화)                                   │
└─────────────────────────────────────────────────────────────────────────────┘

💡 팁:
  • 처음 사용자: [1] 처음 배우기로 시작
  • 데이터 관리자: [2] 데이터 검증으로 시작
  • 고급 사용자: [3] 개별 도구로 선택적 실행
  • 빠른 확인: [4] 빠른 참조만 보기
"""
    
    print(summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ 프로그램을 종료합니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)
