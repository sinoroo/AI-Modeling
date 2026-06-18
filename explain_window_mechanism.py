#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실시간 윈도우 생성 메커니즘 설명"""

print('='*80)
print('실시간 진동 데이터 윈도우 생성 메커니즘')
print('='*80)
print()
print('설정:')
print('  WINDOW_SIZE = 64')
print('  STEP_SIZE = 32 (50% 겹침)')
print('  샘플링 레이트 = 4000 Hz (0.25ms 간격)')
print()
print('시간 흐름 (처음 96개 샘플):')
print()
print('  시간(ms) | 버퍼크기 | 상태')
print('  ---------|----------|------------------------')

buffer = []
windows_created = 0

for t in range(96):
    buffer.append(t)
    if len(buffer) > 64:
        buffer = buffer[-64:]  # Keep only last 64 samples
    
    # Check if window is complete (happens every STEP_SIZE samples after initial 64)
    if len(buffer) == 64 and t >= 63:
        if (t - 63) % 32 == 0:
            windows_created += 1
            print(f'  {t*0.25:6.2f}    |   {len(buffer):>2d}    | ✅ 윈도우 #{windows_created} 완성 (샘플 {t-63}~{t})')
            print(f'            |        | → 즉시 6개 특성 계산 & AI 모델 예측')
        else:
            print(f'  {t*0.25:6.2f}    |   {len(buffer):>2d}    | 새 샘플 수신...')
    else:
        if len(buffer) < 64:
            print(f'  {t*0.25:6.2f}    |   {len(buffer):>2d}    | 초기 버퍼 채우는 중 ({len(buffer)}/64)...')

print()
print('='*80)
print('결론:')
print('='*80)
print('  ✓ 정확히 "64개씩 끊어서 만듭니다"')
print('  ✓ "계속 새로운 윈도우를 만듭니다" (32개 샘플마다 새 윈도우)')
print('  ✓ 각 윈도우가 완성되면:')
print('    1️⃣  64개 샘플 전체에서 6개 특성 계산')
print('    2️⃣  결과: (64, 6) 형태의 데이터')
print('    3️⃣  AI 모델에 즉시 입력하여 예측')
print()
print('컴퓨팅 방식:')
print('  • 버퍼: deque(maxlen=64)로 자동으로 최근 64개 유지')
print('  • 특성: 각 완성된 윈도우에서 한 번만 계산')
print('  • 처리량: 4000 Hz × 6 특성 = 24,000 특성값/초')
print('  • 예측주기: 32개 샘플마다 (= 8ms마다)')
print()
