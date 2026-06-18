#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_3d_array.py 실행 테스트
외부 데이터 불필요, numpy와 matplotlib만 필요
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# 분석 파일 경로
script_path = ANALYSIS_DIR / "visualize_3d_array.py"

print("=" * 80)
print("visualize_3d_array.py 실행 테스트")
print("=" * 80)
print(f"\n파일 경로: {script_path}")
print(f"파일 존재: {script_path.exists()}")
print(f"파일 크기: {script_path.stat().st_size / 1024:.1f} KB\n")

if script_path.exists():
    try:
        # 파일 실행
        print("스크립트 실행 중...\n")
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 실행
        exec(code)
        
        print("\n✅ 스크립트가 정상적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {type(e).__name__}")
        print(f"메시지: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"❌ 파일을 찾을 수 없습니다: {script_path}")
