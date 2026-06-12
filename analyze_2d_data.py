"""
고전 ML 2D 데이터 분석 및 시각화 도구

3D Array를 2D로 평탄화한 데이터를 분석하고 시각화합니다.
- 특성 통계 분석
- 특성 분포 및 상관관계
- PCA 차원 축소 및 시각화
- 특성 중요도 분석
- 모델 성능 비교
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# ✅ 경고 무시 및 한글 폰트 설정
warnings.filterwarnings('ignore')
try:
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# UTF-8 인코딩
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from anomaly_detection.config import *
    from anomaly_detection.data_loader import DataLoader
    from anomaly_detection.preprocessing import Preprocessor
    print("✅ 모듈 로드 성공\n")
except Exception as e:
    print(f"❌ 모듈 로드 실패: {e}\n")
    sys.exit(1)


# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

def load_and_preprocess_data():
    """3D Array 로드 및 2D 변환"""
    
    print("=" * 80)
    print("1️⃣  데이터 로드 및 전처리")
    print("=" * 80)
    
    try:
        data_source = TRAIN_DATA_DIR if TRAIN_DATA_DIR else TRAIN_DATA_PATH
        
        data_loader = DataLoader(
            data_path=str(data_source),
            use_new_format=USE_NEW_CSV_FORMAT
        )
        
        df = data_loader.load_data()
        X_train = df.iloc[:, :-1]
        y_train = df.iloc[:, -1]
        
        print(f"✓ 훈련 데이터 로드: {df.shape}")
        
        preprocessor = Preprocessor(
            window_size=WINDOW_SIZE,
            normalize_method=NORMALIZE_METHOD
        )
        
        X_3d, y = preprocessor.preprocess_pipeline(
            data=df,
            feature_cols=X_train.columns.tolist(),
            label_col=y_train.name,
            fit_scaler=True
        )
        
        print(f"✓ 3D Array 생성: {X_3d.shape}")
        
        # 2D로 변환 (평탄화)
        X_2d = X_3d.reshape(X_3d.shape[0], -1)
        print(f"✓ 2D로 변환: {X_2d.shape}")
        print(f"  → {X_2d.shape[0]} 샘플 × {X_2d.shape[1]} 특성\n")
        
        return X_2d, y, X_3d
        
    except Exception as e:
        print(f"❌ 데이터 처리 실패: {e}\n")
        return None, None, None


# ============================================================================
# 2. 2D 데이터 통계 분석
# ============================================================================

def analyze_2d_statistics(X_2d, y):
    """2D 데이터의 통계 정보 분석"""
    
    print("=" * 80)
    print("2️⃣  2D 데이터 통계 분석")
    print("=" * 80)
    
    n_samples, n_features = X_2d.shape
    
    print(f"\n전체 데이터 통계:")
    print(f"  샘플 수: {n_samples}")
    print(f"  특성 수: {n_features}")
    print(f"  메모리: {X_2d.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\n값의 범위:")
    print(f"  최소값: {X_2d.min():.6f}")
    print(f"  최대값: {X_2d.max():.6f}")
    print(f"  평균: {X_2d.mean():.6f}")
    print(f"  표준편차: {X_2d.std():.6f}")
    print(f"  중앙값: {np.median(X_2d):.6f}")
    
    # 레이블별 통계
    print(f"\n레이블별 통계:")
    for label in np.unique(y):
        label_name = "정상" if label == 0 else "이상"
        mask = y == label
        X_label = X_2d[mask]
        print(f"  {label_name} ({np.sum(mask)} 샘플):")
        print(f"    평균: {X_label.mean():.6f}")
        print(f"    표준편차: {X_label.std():.6f}")
        print(f"    범위: [{X_label.min():.6f}, {X_label.max():.6f}]")


# ============================================================================
# 3. 특성 분석 및 상관관계
# ============================================================================

def analyze_features(X_2d, X_3d):
    """특성 수준의 분석"""
    
    print("\n" + "=" * 80)
    print("3️⃣  특성 분석")
    print("=" * 80)
    
    # 특성별 통계
    print(f"\n특성별 표준편차 (상위 10):")
    feature_stds = X_2d.std(axis=0)
    top_std_indices = np.argsort(feature_stds)[-10:][::-1]
    
    for rank, idx in enumerate(top_std_indices, 1):
        print(f"  {rank:2d}. 특성 {idx:3d}: std={feature_stds[idx]:.6f}")
    
    # 상관관계 분석
    print(f"\n상관관계 통계:")
    corr_matrix = np.corrcoef(X_2d.T)
    
    # 평균 상관관계
    mean_corr = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
    print(f"  평균 절대 상관도: {mean_corr:.6f}")
    
    # 높은 상관관계 쌍 찾기
    high_corr_pairs = []
    for i in range(X_2d.shape[1]):
        for j in range(i+1, X_2d.shape[1]):
            if abs(corr_matrix[i, j]) > 0.7:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    if high_corr_pairs:
        print(f"  높은 상관관계 쌍 (|r| > 0.7):")
        for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
            print(f"    특성 {i:3d} ↔ 특성 {j:3d}: {corr:.4f}")


# ============================================================================
# 4. PCA 분석
# ============================================================================

def analyze_pca(X_2d):
    """PCA를 이용한 차원 축소 분석"""
    
    print("\n" + "=" * 80)
    print("4️⃣  PCA 분석 (차원 축소)")
    print("=" * 80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # 누적 설명 분산
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    print(f"\n주요 성분별 설명 분산:")
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        var_pct = pca.explained_variance_ratio_[i] * 100
        cumsum_pct = cumsum_var[i] * 100
        print(f"  PC{i+1}: {var_pct:6.2f}% (누적: {cumsum_pct:6.2f}%)")
    
    # 필요한 성분 수
    n_components_90 = np.argmax(cumsum_var >= 0.9) + 1
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    
    print(f"\n차원 축소:")
    print(f"  전체 특성 수: {X_2d.shape[1]}")
    print(f"  90% 설명분산: {n_components_90} 개 (압축율: {n_components_90/X_2d.shape[1]:.1%})")
    print(f"  95% 설명분산: {n_components_95} 개 (압축율: {n_components_95/X_2d.shape[1]:.1%})")
    
    return pca, X_scaled


# ============================================================================
# 5. 모델 학습 및 성능 평가
# ============================================================================

def train_and_evaluate_models(X_2d, y):
    """고전 ML 모델 학습 및 성능 평가"""
    
    print("\n" + "=" * 80)
    print("5️⃣  고전 ML 모델 학습 및 평가")
    print("=" * 80)
    
    from sklearn.model_selection import cross_val_score
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # 학습
        model.fit(X_scaled, y)
        
        # 예측
        y_pred = model.predict(X_scaled)
        y_proba = None
        
        # predict_proba 처리 (클래스가 2개 이상일 때만)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)
            if proba.shape[1] >= 2:
                y_proba = proba[:, 1]
        
        # 평가
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0, labels=np.unique(y))
        recall = recall_score(y, y_pred, zero_division=0, labels=np.unique(y))
        f1 = f1_score(y, y_pred, zero_division=0, labels=np.unique(y))
        
        print(f"  정확도: {accuracy:.4f}")
        print(f"  정밀도: {precision:.4f}")
        print(f"  재현율: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        if y_proba is not None and len(np.unique(y)) == 2:
            auc = roc_auc_score(y, y_proba)
            print(f"  ROC-AUC: {auc:.4f}")
            results['auc'] = auc
        
        # 특성 중요도
        if hasattr(model, 'feature_importances_'):
            print(f"\n  특성 중요도 상위 10:")
            top_indices = np.argsort(model.feature_importances_)[-10:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                importance = model.feature_importances_[idx]
                print(f"    {rank:2d}. 특성 {idx:3d}: {importance:.6f}")
        
        results[name] = model
    
    return results, scaler


# ============================================================================
# 6. 시각화
# ============================================================================

def create_visualizations(X_2d, y, X_3d, pca_components_data):
    """2D 데이터 시각화"""
    
    print("\n" + "=" * 80)
    print("6️⃣  시각화 생성")
    print("=" * 80)
    
    pca, X_scaled = pca_components_data
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 특성 분포
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(X_2d.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=X_2d.mean(), color='r', linestyle='--', label=f'Mean: {X_2d.mean():.2f}')
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('All Features Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # 2. 레이블별 박스플롯
    ax2 = plt.subplot(2, 3, 2)
    normal_values = X_2d[y == 0].flatten()
    abnormal_values = X_2d[y == 1].flatten()
    bp = ax2.boxplot([normal_values, abnormal_values], 
                      tick_labels=['정상', '이상'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Normalized Value')
    ax2.set_title('Class Distribution')
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. PCA 누적 설명 분산
    ax3 = plt.subplot(2, 3, 3)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    ax3.plot(range(1, min(50, len(cumsum_var)+1)), 
             cumsum_var[:min(50, len(cumsum_var))], 'o-', linewidth=2)
    ax3.axhline(y=0.9, color='r', linestyle='--', label='90%')
    ax3.axhline(y=0.95, color='orange', linestyle='--', label='95%')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Cumulative Explained Variance')
    ax3.set_title('PCA Explained Variance')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. PCA 2D 시각화
    ax4 = plt.subplot(2, 3, 4)
    X_pca = pca.transform(X_scaled)[:, :2]
    colors = ['blue' if label == 0 else 'red' for label in y]
    scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=100)
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax4.set_title('PCA Visualization (2D)')
    ax4.grid(alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='정상'),
                       Patch(facecolor='red', label='이상')]
    ax4.legend(handles=legend_elements, loc='best')
    
    # 5. 특성별 표준편차
    ax5 = plt.subplot(2, 3, 5)
    feature_stds = X_2d.std(axis=0)
    top_indices = np.argsort(feature_stds)[-15:][::-1]
    top_stds = feature_stds[top_indices]
    bars = ax5.barh(range(len(top_stds)), top_stds)
    ax5.set_yticks(range(len(top_stds)))
    ax5.set_yticklabels([f'F{idx}' for idx in top_indices])
    ax5.set_xlabel('Standard Deviation')
    ax5.set_title('Top 15 Features by Std Dev')
    ax5.grid(alpha=0.3, axis='x')
    
    # Color bars
    for i, bar in enumerate(bars):
        if i < 5:
            bar.set_color('darkgreen')
        elif i < 10:
            bar.set_color('green')
        else:
            bar.set_color('lightgreen')
    
    # 6. 상관관계 히트맵 (상위 30개 특성)
    ax6 = plt.subplot(2, 3, 6)
    top_indices_corr = np.argsort(feature_stds)[-30:]
    X_subset = X_2d[:, top_indices_corr]
    corr_matrix = np.corrcoef(X_subset.T)
    
    im = ax6.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax6.set_title('Correlation Matrix\n(Top 30 Features)')
    ax6.set_xlabel('Feature Index')
    ax6.set_ylabel('Feature Index')
    plt.colorbar(im, ax=ax6, label='Correlation')
    
    plt.tight_layout()
    
    # 저장
    output_path = PROJECT_ROOT / "ml_2d_analysis_visualization.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"✓ 시각화 저장: {output_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# 7. 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🔍 고전 ML 2D 데이터 분석 및 시각화 도구")
    print("=" * 80 + "\n")
    
    # 1. 데이터 로드
    X_2d, y, X_3d = load_and_preprocess_data()
    
    if X_2d is not None:
        # 2. 통계 분석
        analyze_2d_statistics(X_2d, y)
        
        # 3. 특성 분석
        analyze_features(X_2d, X_3d)
        
        # 4. PCA 분석
        pca, X_scaled = analyze_pca(X_2d)
        
        # 5. 모델 학습
        model_results, scaler = train_and_evaluate_models(X_2d, y)
        
        # 6. 시각화
        create_visualizations(X_2d, y, X_3d, (pca, X_scaled))
        
        print("\n" + "=" * 80)
        print("✅ 분석 완료!")
        print("=" * 80)
        print("""
생성된 파일:
  - 3d_array_visualization.png (3D 데이터 시각화)
  - ml_2d_analysis_visualization.png (2D 데이터 시각화) ← 새로 생성
        """)
    else:
        print("데이터 처리 실패. 다시 시도해주세요.")
