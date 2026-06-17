"""
고전 ML 모델에 데이터 입력하기 - 실전 코드

3D Array를 고전 ML 모델에 입력하기 위해 2D로 변환하는 여러 방법들
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. 테스트 데이터 준비
# ============================================================================

# 실제 3D Array 형태
X = np.random.randn(29, 64, 6)  # (29 windows, 64 samples, 6 features)
y = np.random.randint(0, 2, 29)  # 레이블 (0: 정상, 1: 이상)

print("=" * 80)
print("원본 3D Array 정보")
print("=" * 80)
print(f"Shape: {X.shape}")
print(f"  - 29: 시간 윈도우")
print(f"  - 64: 각 윈도우의 샘플 개수")
print(f"  - 6: 특성 개수 (vibration, RMS, Peak, Crest, Kurtosis, Skewness)")


# ============================================================================
# 2. 방법 1: 모든 특성을 포함한 평탄화 (간단함)
# ============================================================================

print("\n" + "=" * 80)
print("방법 1️⃣ : 모든 샘플 × 모든 특성 평탄화")
print("=" * 80)

X_flat_all = X.reshape(X.shape[0], -1)
print(f"변환 결과: {X_flat_all.shape}")
print(f"  - 29: 윈도우 개수")
print(f"  - 384: 64 샘플 × 6 특성")

# 고전 ML 모델 학습
print("\n모델 학습 중...")
rf_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_1.fit(X_flat_all, y)
print(f"✓ RandomForest 학습 완료")

# 예측
y_pred_1 = rf_model_1.predict(X_flat_all)
accuracy_1 = np.mean(y_pred_1 == y)
print(f"정확도: {accuracy_1:.2%}")
print(f"특성 중요도 상위 5:")
top_features = np.argsort(rf_model_1.feature_importances_)[-5:][::-1]
for rank, feat_idx in enumerate(top_features, 1):
    importance = rf_model_1.feature_importances_[feat_idx]
    print(f"  {rank}. 특성 {feat_idx}: {importance:.4f}")


# ============================================================================
# 3. 방법 2: 진동값만 사용 (작은 입력)
# ============================================================================

print("\n" + "=" * 80)
print("방법 2️⃣ : 진동값만 추출 (특성 0)")
print("=" * 80)

X_vibration_only = X[:, :, 0]  # 첫 번째 특성만 추출
print(f"변환 결과: {X_vibration_only.shape}")
print(f"  - 29: 윈도우 개수")
print(f"  - 64: 샘플 개수 (진동값만)")

rf_model_2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_2.fit(X_vibration_only, y)
y_pred_2 = rf_model_2.predict(X_vibration_only)
accuracy_2 = np.mean(y_pred_2 == y)
print(f"✓ RandomForest 학습 완료")
print(f"정확도: {accuracy_2:.2%}")


# ============================================================================
# 4. 방법 3: 윈도우별 집계 특성 (권장! 효율적)
# ============================================================================

print("\n" + "=" * 80)
print("방법 3️⃣ : 윈도우별 집계 특성 (권장)")
print("=" * 80)

def aggregate_window_features(X):
    """
    각 윈도우의 통계 특성을 계산하여 집계
    
    Args:
        X: 3D Array (n_windows, window_size, n_features)
    
    Returns:
        2D Array (n_windows, n_features * 4)
        각 특성별 mean, std, max, min
    """
    n_windows, window_size, n_features = X.shape
    aggregated = []
    
    for w in range(n_windows):
        window = X[w]  # (64, 6)
        
        # 각 특성별로 평균, 표준편차, 최대, 최소 계산
        stats = []
        for f in range(n_features):
            feature_data = window[:, f]  # 64개 샘플
            
            stats.append(feature_data.mean())    # 평균
            stats.append(feature_data.std())     # 표준편차
            stats.append(feature_data.max())     # 최대
            stats.append(feature_data.min())     # 최소
        
        aggregated.append(stats)
    
    return np.array(aggregated)


X_agg = aggregate_window_features(X)
print(f"변환 결과: {X_agg.shape}")
print(f"  - 29: 윈도우 개수")
print(f"  - 24: 6 특성 × 4 통계 (mean, std, max, min)")

print("\n특성 구성:")
feature_names = ['vibration', 'RMS', 'Peak', 'Crest', 'Kurtosis', 'Skewness']
for f_idx, f_name in enumerate(feature_names):
    col_indices = [f_idx * 4 + i for i in range(4)]
    stats_names = ['mean', 'std', 'max', 'min']
    print(f"  {f_name}:")
    for col, stat in zip(col_indices, stats_names):
        print(f"    [{col:2d}] {stat}")

rf_model_3 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_3.fit(X_agg, y)
y_pred_3 = rf_model_3.predict(X_agg)
accuracy_3 = np.mean(y_pred_3 == y)
print(f"\n✓ RandomForest 학습 완료")
print(f"정확도: {accuracy_3:.2%}")

print(f"\n특성 중요도 상위 10:")
top_features = np.argsort(rf_model_3.feature_importances_)[-10:][::-1]
for rank, feat_idx in enumerate(top_features, 1):
    importance = rf_model_3.feature_importances_[feat_idx]
    feature_idx = feat_idx // 4
    stat_idx = feat_idx % 4
    stat_names = ['mean', 'std', 'max', 'min']
    feature_name = feature_names[feature_idx]
    stat_name = stat_names[stat_idx]
    print(f"  {rank:2d}. {feature_name} ({stat_name:3s}): {importance:.4f}")


# ============================================================================
# 5. 방법 4: 이상 탐지 모델 (비지도 학습)
# ============================================================================

print("\n" + "=" * 80)
print("방법 4️⃣ : 이상 탐지 모델 (비지도 학습)")
print("=" * 80)

# IsolationForest (이상 탐지용)
iso_model = IsolationForest(contamination=0.1, random_state=42)
anomaly_scores_1 = iso_model.fit_predict(X_flat_all)
print(f"✓ IsolationForest 학습 완료")
print(f"  입력: {X_flat_all.shape}")
print(f"  이상 탐지 비율: {(anomaly_scores_1 == -1).mean():.1%}")

# OneClassSVM (이상 탐지용)
ocsvm_model = OneClassSVM(nu=0.1)
anomaly_scores_2 = ocsvm_model.fit_predict(X_flat_all)
print(f"✓ OneClassSVM 학습 완료")
print(f"  입력: {X_flat_all.shape}")
print(f"  이상 탐지 비율: {(anomaly_scores_2 == -1).mean():.1%}")


# ============================================================================
# 6. 성능 비교 표
# ============================================================================

print("\n" + "=" * 80)
print("성능 비교")
print("=" * 80)

comparison = f"""
입력 방식              | 입력 크기    | 특성 수 | 정확도 | 장점         | 단점
─────────────────────┼─────────────┼────────┼────────┼─────────────┼──────────────
모든 특성 평탄화     | (29, 384)   | 384    | {accuracy_1:.2%}  | 모든 정보포함 | 차원 높음
진동값만             | (29, 64)    | 64     | {accuracy_2:.2%}  | 간단함      | 정보 부족
윈도우 집계 (권장)   | (29, 24)    | 24     | {accuracy_3:.2%}  | 효율적      | 약간의 정보손실

결론:
✓ 적은 샘플 수 (29개)에서는 방법 3️⃣ (윈도우 집계)이 가장 권장됨
✓ 진동값만 사용하는 것은 가능하지만, 다른 통계 특성들도 중요함
✓ 모든 384개 특성은 오버피팅 위험이 있음
"""

print(comparison)


# ============================================================================
# 7. 실제 사용 템플릿
# ============================================================================

print("\n" + "=" * 80)
print("실전 코드 템플릿")
print("=" * 80)

code_template = '''
# 1️⃣  데이터 준비 (전처리 후)
X = preprocessor.preprocess_pipeline(...)  # (n_windows, 64, 6)
y = labels

# 2️⃣  특성 엔지니어링 (선택)
if use_aggregation:
    X_input = aggregate_window_features(X)  # (n_windows, 24)
else:
    X_input = X.reshape(X.shape[0], -1)   # (n_windows, 384)

# 3️⃣  모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100)
model.fit(X_input, y)

# 4️⃣  예측
y_pred = model.predict(X_input)
'''

print(code_template)


# ============================================================================
# 8. 핵심 요약
# ============================================================================

print("\n" + "=" * 80)
print("📌 핵심 요약")
print("=" * 80)

summary = """
❓ 질문: 고전 ML 모델에서 진동값만 입력하면 되는 거야?

✅ 답:
1️⃣  기술적으로 가능: 진동값만 추출해서 사용 가능
2️⃣  하지만 비권장: 다른 통계 특성들도 중요한 정보 포함
3️⃣  권장 방식:
   - 진동값 + 통계 특성 6개 모두 사용 (최대 정보)
   - 또는 윈도우별 집계 특성 24개 사용 (효율적) ← 추천!
   - 진동값만 64개는 정보 손실

📊 각 특성의 역할:
- vibration    : 원본 신호 (진동값)
- RMS          : 진동 에너지 수준
- Peak         : 최대 진동 크기
- Crest Factor : 신호의 충돌성 (스파이크)
- Kurtosis     : 이상 임펄스 감지 (베어링 손상 등)
- Skewness     : 신호 비대칭성

🎯 권장 사용법:
X_input = aggregate_window_features(X)  # 효율적 (24 특성)
model.fit(X_input, y)
"""

print(summary)
