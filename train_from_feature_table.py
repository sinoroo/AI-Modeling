"""
표준 특징 테이블 기반 모델 학습/평가.

build_feature_table.py 가 생성한 feature_tables/*.csv 를 입력으로,
두 가지 과제를 학습한다.

1) 다중분류(5클래스): 정상/축정렬/불평형/베어링/벨트 고장 유형 분류
2) 이상탐지(One-Class): 정상만 학습 후 4개 고장을 이상으로 탐지

출력
----
- models/clf_random_forest.pkl, models/clf_scaler.pkl
- models/anomaly_isolation_forest.pkl
- results/feature_table_evaluation.json

사용법
------
    python train_from_feature_table.py
    python train_from_feature_table.py --table-dir feature_tables --model classifier
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)


# 메타/식별/라벨 컬럼은 특징에서 제외
NON_FEATURE_COLS = {
    "source_file", "equipment_id", "window_index", "padded",
    "label_name", "label_no", "is_anomaly",
}

MODEL_DIR = "models"
RESULTS_DIR = "results"


def load_split(table_dir, split):
    path = os.path.join(table_dir, f"feature_table_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"특징 테이블 없음: {path} "
                                f"(먼저 build_feature_table.py 실행)")
    return pd.read_csv(path)


def get_feature_columns(df):
    """수치형 특징 컬럼만 선택(메타/라벨 제외)."""
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def prepare_xy(df, feature_cols, target):
    X = df[feature_cols].to_numpy(dtype=float)
    # NaN(예: RPM 없는 설비의 order 대역) -> 0 으로 대치
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[target].to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# 1) 다중분류
# ---------------------------------------------------------------------------
def train_classifier(train, val, test, feature_cols):
    print("\n" + "=" * 70)
    print("[과제 1] 다중분류 (5클래스 고장 유형)")
    print("=" * 70)

    X_tr, y_tr = prepare_xy(train, feature_cols, "label_name")
    X_va, y_va = prepare_xy(val, feature_cols, "label_name")
    X_te, y_te = prepare_xy(test, feature_cols, "label_name")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = (scaler.transform(x) for x in (X_tr, X_va, X_te))

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=42,
    )
    clf.fit(X_tr_s, y_tr)

    def _metrics(X, y, name):
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        f1m = f1_score(y, pred, average="macro")
        print(f"  {name:5s}: acc={acc:.4f}  macro-F1={f1m:.4f}")
        return {"accuracy": float(acc), "macro_f1": float(f1m)}

    res = {
        "val": _metrics(X_va_s, y_va, "val"),
        "test": _metrics(X_te_s, y_te, "test"),
    }

    pred_te = clf.predict(X_te_s)
    labels_sorted = sorted(np.unique(y_tr).tolist())
    print("\n  [test] classification report:")
    print(classification_report(y_te, pred_te, labels=labels_sorted,
                                zero_division=0))
    res["test"]["confusion_matrix"] = confusion_matrix(
        y_te, pred_te, labels=labels_sorted).tolist()
    res["test"]["labels"] = labels_sorted

    # 특징 중요도 상위
    importances = sorted(
        zip(feature_cols, clf.feature_importances_),
        key=lambda x: x[1], reverse=True)
    res["feature_importance_top"] = [
        {"feature": f, "importance": float(i)} for f, i in importances[:10]]
    print("\n  상위 특징 중요도:")
    for f, i in importances[:8]:
        print(f"    {f:20s} {i:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "clf_random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "clf_scaler.pkl"))
    return res


# ---------------------------------------------------------------------------
# 2) 이상탐지 (One-Class)
# ---------------------------------------------------------------------------
def train_anomaly(train, val, test, feature_cols):
    print("\n" + "=" * 70)
    print("[과제 2] 이상탐지 (정상만 학습 -> 고장을 이상으로 탐지)")
    print("=" * 70)

    # 정상(is_anomaly==0)만 학습
    train_normal = train[train["is_anomaly"] == 0]
    X_tr, _ = prepare_xy(train_normal, feature_cols, "is_anomaly")
    X_te, y_te = prepare_xy(test, feature_cols, "is_anomaly")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    iso = IsolationForest(
        n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X_tr_s)

    # IsolationForest: -1=이상, 1=정상 -> 1(이상)/0(정상)으로 변환
    pred = (iso.predict(X_te_s) == -1).astype(int)
    # 이상점수(높을수록 이상): score_samples는 높을수록 정상 -> 부호 반전
    scores = -iso.score_samples(X_te_s)

    acc = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, zero_division=0)
    rec = recall_score(y_te, pred, zero_division=0)
    f1 = f1_score(y_te, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_te, scores)
    except ValueError:
        auc = float("nan")

    print(f"  [test] acc={acc:.4f}  precision={prec:.4f}  "
          f"recall={rec:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")
    print("\n  [test] confusion matrix (행=실제[정상,이상], 열=예측[정상,이상]):")
    print(confusion_matrix(y_te, pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(iso, os.path.join(MODEL_DIR, "anomaly_isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "anomaly_scaler.pkl"))

    return {
        "test": {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "roc_auc": float(auc),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
    }


def run_training(table_dir="feature_tables", model="both"):
    """
    표준 특징 테이블 기반 학습/평가 (main.py 및 CLI 공용).

    Args:
        table_dir: feature_tables 디렉터리
        model: "classifier" | "anomaly" | "both"

    Returns: results dict (results/feature_table_evaluation.json 에도 저장)
    """
    train = load_split(table_dir, "train")
    val = load_split(table_dir, "val")
    test = load_split(table_dir, "test")

    feature_cols = get_feature_columns(train)
    print(f"특징 컬럼 {len(feature_cols)}개 사용:", feature_cols)
    print(f"샘플 수  train={len(train)}  val={len(val)}  test={len(test)}")

    results = {"feature_cols": feature_cols,
               "n_samples": {"train": len(train), "val": len(val),
                             "test": len(test)}}

    if model in ("classifier", "both"):
        results["classification"] = train_classifier(
            train, val, test, feature_cols)

    if model in ("anomaly", "both"):
        results["anomaly_detection"] = train_anomaly(
            train, val, test, feature_cols)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "feature_table_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("완료. 결과 저장:", out_path)
    print("모델 저장:", MODEL_DIR)
    print("=" * 70)
    return results


def main():
    ap = argparse.ArgumentParser(description="표준 특징 테이블 기반 학습/평가")
    ap.add_argument("--table-dir", default="feature_tables")
    ap.add_argument("--model", choices=["classifier", "anomaly", "both"],
                    default="both")
    args = ap.parse_args()

    run_training(table_dir=args.table_dir, model=args.model)


if __name__ == "__main__":
    main()

