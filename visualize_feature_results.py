# -*- coding: utf-8 -*-
"""
[B] standard 파이프라인 특징/모델 결과 시각화.

생성 이미지(results/plots/ 에 저장):
  1) classification_confusion_matrix.png   - 5클래스 분류 혼동행렬
  2) feature_importance.png                - RandomForest 특징 중요도 top-N
  3) order_band_by_class.png               - 고장유형별 FFT 차수 대역(1x/2x/3x/high) 에너지비율
  4) feature_distribution.png              - 주요 특징 클래스별 분포(box)
  5) pca_scatter.png                       - 20개 특징 PCA 2D 산점도(클래스별)
  6) anomaly_confusion_matrix.png          - 이상탐지 혼동행렬

사용:
  python visualize_feature_results.py
  python main.py standard --plot            (통합 진입점)
"""

import os
import sys
import json

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로 저장
try:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(PROJECT_ROOT, "feature_tables")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# 클래스 표시 순서/색상
CLASS_ORDER = ["NORMAL", "MISALIGN", "IMBALANCE", "BEARING", "BELT"]
CLASS_KO = {
    "NORMAL": "정상", "MISALIGN": "축정렬불량", "IMBALANCE": "회전체불평형",
    "BEARING": "베어링불량", "BELT": "벨트느슨함",
}
ORDER_BANDS = ["band_1x_ratio", "band_2x_ratio", "band_3x_ratio", "band_high_ratio"]
ORDER_LABELS = ["1x (불평형)", "2x (정렬)", "3x", "high (5~20x)"]


def _load_eval():
    path = os.path.join(RESULTS_DIR, "feature_table_evaluation.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_table(split="test"):
    path = os.path.join(TABLE_DIR, f"feature_table_{split}.csv")
    return pd.read_csv(path)


def _present_classes(df):
    return [c for c in CLASS_ORDER if c in set(df["label_name"])]


# ---------------------------------------------------------------------------
def plot_classification_confusion(ev):
    cm = np.array(ev["classification"]["test"]["confusion_matrix"])
    labels = ev["classification"]["test"]["labels"]
    acc = ev["classification"]["test"]["accuracy"]
    f1 = ev["classification"]["test"]["macro_f1"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("예측 (predicted)")
    ax.set_ylabel("실제 (true)")
    ax.set_title(f"5클래스 분류 혼동행렬 (test)\nacc={acc:.3f}  macro-F1={f1:.3f}")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "classification_confusion_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_importance(ev, top_n=12):
    items = ev["classification"].get("feature_importance_top", [])
    if not items:
        return None
    items = items[:top_n][::-1]
    names = [d["feature"] for d in items]
    vals = [d["importance"] for d in items]
    colors = ["#d62728" if n.startswith(("band_", "spectral_")) else "#1f77b4"
              for n in names]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("중요도 (importance)")
    ax.set_title("RandomForest 특징 중요도 (빨강=FFT/스펙트럼 특징)")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "feature_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_order_band_by_class(df):
    classes = _present_classes(df)
    means = df.groupby("label_name")[ORDER_BANDS].mean()

    x = np.arange(len(ORDER_BANDS))
    width = 0.8 / max(len(classes), 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        if c not in means.index:
            continue
        ax.bar(x + i * width, means.loc[c, ORDER_BANDS].values,
               width, label=f"{c} ({CLASS_KO[c]})", color=cmap(i))
    ax.set_xticks(x + width * (len(classes) - 1) / 2)
    ax.set_xticklabels(ORDER_LABELS)
    ax.set_ylabel("대역 에너지 비율 (평균)")
    ax.set_title("고장유형별 FFT 회전차수 대역 에너지 비율")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "order_band_by_class.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_distribution(df):
    feats = ["rms", "spectral_energy", "band_1x_ratio", "band_high_ratio"]
    classes = _present_classes(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feat in zip(axes.ravel(), feats):
        data = [df[df["label_name"] == c][feat].values for c in classes]
        ax.boxplot(data, tick_labels=classes, showfliers=False)
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
    fig.suptitle("주요 특징의 고장유형별 분포 (box)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "feature_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pca_scatter(df, ev):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    feature_cols = ev["feature_cols"]
    X = df[feature_cols].fillna(0.0).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    ev_ratio = pca.explained_variance_ratio_

    classes = _present_classes(df)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(classes):
        m = df["label_name"].values == c
        ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.6,
                   color=cmap(i), label=f"{c} ({CLASS_KO[c]})")
    ax.set_xlabel(f"PC1 ({ev_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev_ratio[1]*100:.1f}%)")
    ax.set_title("20개 특징 PCA 2D 투영 (고장유형별 군집)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "pca_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_anomaly_confusion(ev):
    if "anomaly_detection" not in ev:
        return None
    a = ev["anomaly_detection"]["test"]
    cm = np.array(a["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Oranges")
    ticks = ["정상(0)", "이상(1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(ticks); ax.set_yticklabels(ticks)
    ax.set_xlabel("예측"); ax.set_ylabel("실제")
    ax.set_title("이상탐지(IsolationForest) 혼동행렬\n"
                 f"F1={a['f1']:.3f}  ROC-AUC={a['roc_auc']:.3f}")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "anomaly_confusion_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_visualization(split="test"):
    """모든 시각화를 생성하고 저장된 경로 목록을 반환."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    ev = _load_eval()
    df = _load_table(split)

    saved = []
    print("=" * 70)
    print("특징/모델 결과 시각화 생성")
    print("=" * 70)
    for fn in (
        lambda: plot_classification_confusion(ev),
        lambda: plot_feature_importance(ev),
        lambda: plot_order_band_by_class(df),
        lambda: plot_feature_distribution(df),
        lambda: plot_pca_scatter(df, ev),
        lambda: plot_anomaly_confusion(ev),
    ):
        try:
            out = fn()
            if out:
                saved.append(out)
                print("  저장:", os.path.relpath(out, PROJECT_ROOT))
        except Exception as e:
            print("  [warn] 시각화 일부 실패:", e)

    print("=" * 70)
    print(f"완료. {len(saved)}개 이미지 -> {os.path.relpath(PLOT_DIR, PROJECT_ROOT)}")
    print("=" * 70)
    return saved


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    run_visualization(split)
