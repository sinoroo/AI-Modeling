"""
진동 데이터 표준화(정형화) 변환 스크립트.

data_new_format/ 의 원시 CSV(헤더 9줄 + time/vibration 2열)들을
머신러닝 입력용 "표준 특징 테이블"로 변환한다.

표준화 전략
-----------
1) 고정 길이 윈도우 분할
   - 모든 신호를 WINDOW_SECONDS(기본 1초 = 4000샘플)로 자르고 OVERLAP(기본 0.5) 적용
   - 길이가 12000/500/300 으로 제각각인 문제를 해소
2) 진폭 정규화
   - 설비(equipment)별 "정상" 데이터 기준 평균/표준편차로 Z-score (동력/설비 차이 흡수)
3) 주파수 정규화(회전차수, order)
   - RPM(Motor Spec)으로 회전주파수 f_r = RPM/60 계산
   - FFT 스펙트럼을 차수(order = f / f_r) 축에서 해석 → 설비 간 비교 가능
4) 윈도우 = 1행 형태의 특징 테이블 + 5클래스 영문 라벨로 저장

출력
----
- feature_table_train.csv / _val.csv / _test.csv
- normalization_stats.json (설비별 정상 기준 통계, 추론 시 재사용)

사용법
------
    python build_feature_table.py
    python build_feature_table.py --data-root data_new_format --out-dir feature_tables
"""

import os
import re
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# 공용 특징추출 모듈 (학습/추론 공통) — 동일 로직 보장
from anomaly_detection.feature_extraction import (
    SAMPLE_RATE, WINDOW_SECONDS, WINDOW_SIZE, OVERLAP, STEP_SIZE, MIN_USABLE,
    ORDER_BANDS, LABEL_MAP,
    time_features, order_spectrum_features, make_windows,
)

# 정상 윈도우 분할 기준(이 길이 이상이면 슬라이딩 분할; 정규화 기준 통계용)
MIN_LENGTH = WINDOW_SIZE


# ---------------------------------------------------------------------------
# 파싱 유틸
# ---------------------------------------------------------------------------
def parse_metadata(file_path, n_header=9):
    """헤더 9줄을 dict로 파싱."""
    meta = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in range(n_header):
            line = f.readline()
            if not line:
                break
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 2:
                meta[parts[0].strip()] = parts[1].strip()
    return meta


def parse_motor_spec(meta):
    """
    Motor Spec,L-EF-02, 1755,3.7, 7.6,  ->
    (equipment_id, rpm, power_kw, current_a)
    """
    raw = meta.get("Motor Spec", "")
    # 'Motor Spec' 키 다음의 값 + 추가 토큰들이 콤마로 이어짐
    # parse_metadata는 첫 콤마까지만 value로 저장하므로 원본 라인을 다시 읽지 않고
    # Filename/Date 등과 달리 여기서는 전체 라인을 별도 파싱한다.
    return raw


def parse_motor_spec_line(file_path):
    """Motor Spec 라인 전체를 직접 파싱하여 (equip, rpm, kw, current) 반환."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Motor Spec"):
                tokens = [t.strip() for t in line.rstrip("\n").split(",")]
                # tokens[0] = 'Motor Spec'
                vals = [t for t in tokens[1:] if t != ""]
                equip = vals[0] if len(vals) > 0 else "UNKNOWN"
                rpm = _to_float(vals[1]) if len(vals) > 1 else np.nan
                kw = _to_float(vals[2]) if len(vals) > 2 else np.nan
                cur = _to_float(vals[3]) if len(vals) > 3 else np.nan
                return equip, rpm, kw, cur
    return "UNKNOWN", np.nan, np.nan, np.nan


def _to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def load_signal(file_path, n_header=9):
    """데이터 본문(2열: time, vibration)을 numpy 배열로 로드."""
    df = pd.read_csv(file_path, skiprows=n_header, header=None,
                     usecols=[0, 1], names=["time", "vib"])
    vib = pd.to_numeric(df["vib"], errors="coerce").to_numpy()
    vib = vib[~np.isnan(vib)]
    return vib


def detect_power(path):
    """경로(절대/상대)에서 동력 폴더(예: 2.2kW) 추출, 없으면 ROOT."""
    for part in os.path.normpath(path).split(os.sep):
        if part.endswith("kW"):
            return part
    return "ROOT"


# 특징 추출 함수(time_features / order_spectrum_features / make_windows)는
# anomaly_detection.feature_extraction 에서 import 하여 학습/추론 일관성을 보장한다.


# ---------------------------------------------------------------------------
# 메인 변환
# ---------------------------------------------------------------------------
def fit_equip_stats(files):
    """설비별 정상 신호로부터 진폭 Z-score 기준(mean/std) 추정."""
    acc = defaultdict(list)
    for fp in files:
        meta = parse_metadata(fp)
        label = meta.get("Data Label", "?")
        if label not in LABEL_MAP or LABEL_MAP[label][2] != 0:
            continue  # 정상만 사용
        equip, *_ = parse_motor_spec_line(fp)
        sig = load_signal(fp)
        if len(sig) >= MIN_LENGTH:        # 정규화 기준은 실측 신호만 사용
            acc[equip].append(sig)
    stats = {}
    for equip, sigs in acc.items():
        allv = np.concatenate(sigs)
        stats[equip] = {
            "mean": float(np.mean(allv)),
            "std": float(np.std(allv) + 1e-12),
        }
    return stats


def process_files(files, equip_norm_stats):
    """파일 리스트 -> 특징 테이블(DataFrame) + 제외 개수."""
    rows = []
    skipped_short = 0
    for fp in files:
        meta = parse_metadata(fp)
        label = meta.get("Data Label", "?")
        if label not in LABEL_MAP:
            continue
        label_name, label_no, is_anom = LABEL_MAP[label]

        equip, rpm, kw, cur = parse_motor_spec_line(fp)
        power = detect_power(fp)

        sig = load_signal(fp)
        if len(sig) < MIN_USABLE:
            skipped_short += 1
            continue

        # 진폭 정규화(설비별 정상 기준 Z-score; 없으면 자체 표준화)
        stats = (equip_norm_stats or {}).get(equip)
        if stats is None:
            mean_, std_ = float(np.mean(sig)), float(np.std(sig) + 1e-12)
        else:
            mean_, std_ = stats["mean"], stats["std"]
        sig_norm = (sig - mean_) / std_

        for wi, (w, padded) in enumerate(make_windows(sig_norm)):
            feat = {
                "source_file": os.path.basename(fp),
                "equipment_id": equip,
                "power_kw": kw,
                "rpm": rpm,
                "window_index": wi,
                "padded": int(padded),
            }
            feat.update(time_features(w))
            feat.update(order_spectrum_features(w, rpm))
            feat["label_name"] = label_name
            feat["label_no"] = label_no
            feat["is_anomaly"] = is_anom
            rows.append(feat)

    return pd.DataFrame(rows), skipped_short


def build_table(data_root, split, equip_norm_stats=None, fit_stats=False):
    """하나의 split(train/val/test) 폴더를 처리."""
    root = os.path.join(data_root, split)
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if fit_stats:
        equip_norm_stats = fit_equip_stats(files)
    df, skipped_short = process_files(files, equip_norm_stats)
    return df, equip_norm_stats, skipped_short


def stratified_file_split(files, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    설비×라벨 층화 + 파일 단위 train/val/test 분할.
    동일 파일의 윈도우가 여러 split에 섮이지 않게 하여 누수 방지.
    """
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for fp in files:
        meta = parse_metadata(fp)
        label = meta.get("Data Label", "?")
        if label not in LABEL_MAP:
            continue
        equip, *_ = parse_motor_spec_line(fp)
        groups[(equip, label)].append(fp)

    tr, va, te = [], [], []
    for key, fps in groups.items():
        fps = list(fps)
        rng.shuffle(fps)
        n = len(fps)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        # 적은 그룹은 최소 1개씩 train에 포함되도록 보정
        n_tr = max(n_tr, 1) if n >= 1 else 0
        tr += fps[:n_tr]
        va += fps[n_tr:n_tr + n_va]
        te += fps[n_tr + n_va:]
    return tr, va, te


def build_feature_tables(data_root="data_new_format", out_dir="feature_tables",
                         regroup=True):
    """
    원시 CSV -> 표준 특징 테이블(train/val/test) 생성 후 저장.

    main.py 및 CLI 양쪽에서 호출 가능한 재사용 함수.

    Returns: dict 요약 (윈도우 수, 저장 경로, 클래스 분포)
    """
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("진동 데이터 표준화(정형화) 변환")
    print(f"  윈도우: {WINDOW_SIZE} 샘플 ({WINDOW_SECONDS}s), step={STEP_SIZE}, "
          f"overlap={OVERLAP}  | 모드: {'regroup' if regroup else 'folder'}")
    print("=" * 70)

    if regroup:
        # 전체 파일 수집 후 설비×라벨 층화 파일단위 분할
        all_files = sorted(glob.glob(
            os.path.join(data_root, "**", "*.csv"), recursive=True))
        tr_files, va_files, te_files = stratified_file_split(all_files)
        print(f"  파일 분할: train={len(tr_files)} val={len(va_files)} "
              f"test={len(te_files)}")
        norm_stats = fit_equip_stats(tr_files)
        df_train, sk_tr = process_files(tr_files, norm_stats)
        df_val, sk_va = process_files(va_files, norm_stats)
        df_test, sk_te = process_files(te_files, norm_stats)
        print(f"[train] {len(df_train)} windows (제외 {sk_tr})")
        print(f"[val]   {len(df_val)} windows (제외 {sk_va})")
        print(f"[test]  {len(df_test)} windows (제외 {sk_te})")
    else:
        # train: 통계 추정 + 변환
        print("\n[train] 설비별 정상 기준 통계 추정 및 변환 중...")
        df_train, norm_stats, sk_tr = build_table(
            data_root, "train", fit_stats=True)
        print(f"  -> {len(df_train)} windows, 짧은신호 제외 {sk_tr}개 파일")

        # val/test: train 통계 재사용
        print("[val] 변환 중...")
        df_val, _, sk_va = build_table(
            data_root, "val", equip_norm_stats=norm_stats)
        print(f"  -> {len(df_val)} windows, 짧은신호 제외 {sk_va}개 파일")

        print("[test] 변환 중...")
        df_test, _, sk_te = build_table(
            data_root, "test", equip_norm_stats=norm_stats)
        print(f"  -> {len(df_test)} windows, 짧은신호 제외 {sk_te}개 파일")

    # 저장
    paths = {
        "train": os.path.join(out_dir, "feature_table_train.csv"),
        "val": os.path.join(out_dir, "feature_table_val.csv"),
        "test": os.path.join(out_dir, "feature_table_test.csv"),
    }
    df_train.to_csv(paths["train"], index=False, encoding="utf-8-sig")
    df_val.to_csv(paths["val"], index=False, encoding="utf-8-sig")
    df_test.to_csv(paths["test"], index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "normalization_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)

    # 요약
    print("\n" + "=" * 70)
    print("완료. 저장 위치:", out_dir)
    print("=" * 70)
    class_dist = {}
    if len(df_train):
        class_dist = df_train["label_name"].value_counts().to_dict()
        print("\n[train] 클래스 분포(window 기준):")
        print(df_train["label_name"].value_counts().to_string())
        print(f"\n[train] 특징 컬럼 수: {df_train.shape[1]}")
        print("특징 컬럼:", [c for c in df_train.columns])

    return {
        "n_windows": {"train": len(df_train), "val": len(df_val),
                      "test": len(df_test)},
        "paths": paths,
        "class_distribution": class_dist,
    }


def main():
    ap = argparse.ArgumentParser(description="진동 데이터 표준 특징 테이블 생성")
    ap.add_argument("--data-root", default="data_new_format")
    ap.add_argument("--out-dir", default="feature_tables")
    ap.add_argument("--regroup", action="store_true",
                    help="기존 폴더 분할 대신 전체 실측 파일을 설비×라벨 층화 "
                         "파일단위 70/15/15로 재분할(권장)")
    args = ap.parse_args()

    build_feature_tables(data_root=args.data_root, out_dir=args.out_dir,
                         regroup=args.regroup)


if __name__ == "__main__":
    main()

