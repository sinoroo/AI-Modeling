# -*- coding: utf-8 -*-
"""
FFT 특징 계산 과정을 단계별 그래프로 시각화.

하나의 실제 윈도우에 대해 다음 단계를 순서대로 그린다.
  (1) 원시 진동 신호 (시간영역)
  (2) 해닝 창 적용 신호
  (3) 진폭 스펙트럼 (주파수영역, Hz)
  (4) 회전차수(order) 축으로 환산 + 차수 대역(1x/2x/3x/high) 표시
  (5) 대역별 에너지 / 에너지 비율 막대
그리고 고장유형별 평균 스펙트럼(차수축) 비교 그래프도 생성한다.

사용:
  python visualize_fft_steps.py                  # 기본: 회전체불평형 1윈도우
  python visualize_fft_steps.py 베어링불량        # 특정 상태
  python main.py standard --fft-steps            # 통합 진입점
"""

import os
import sys
import glob

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import matplotlib
matplotlib.use("Agg")
try:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from anomaly_detection.feature_extraction import (
    SAMPLE_RATE, WINDOW_SIZE, ORDER_BANDS, LABEL_MAP, LABEL_EN_TO_KO,
    make_windows, order_spectrum_features,
)
from build_feature_table import parse_motor_spec_line, load_signal

DATA_ROOT = os.path.join(PROJECT_ROOT, "data_new_format")
PLOT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")

BAND_COLORS = {
    "band_1x": "#d62728",     # 1x (불평형) - 빨강
    "band_2x": "#2ca02c",     # 2x (정렬)   - 초록
    "band_3x": "#9467bd",     # 3x          - 보라
    "band_high": "#ff7f0e",   # high        - 주황
}
BAND_KO = {
    "band_1x": "1x (불평형)", "band_2x": "2x (정렬불량)",
    "band_3x": "3x", "band_high": "high (5~20x)",
}


def _find_csv_for_status(status_ko):
    """data_new_format/train 아래에서 해당 상태(폴더명)의 CSV 1개 경로 반환."""
    # 폴더 구조: data_new_format/train/<power>/<equipment>/<status>/*.csv
    pattern = os.path.join(DATA_ROOT, "train", "**", status_ko, "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        # train 외 다른 split 도 탐색
        pattern2 = os.path.join(DATA_ROOT, "**", status_ko, "*.csv")
        files = glob.glob(pattern2, recursive=True)
    return files[0] if files else None


def _compute_spectrum(w, fs=SAMPLE_RATE):
    """order_spectrum_features 와 동일한 방식의 스펙트럼/주파수축 반환."""
    n = len(w)
    win = np.hanning(n)
    win_sig = w * win
    spectrum = np.abs(np.fft.rfft(win_sig)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return win_sig, spectrum, freqs


def visualize_steps(status_ko="회전체불평형"):
    """단일 윈도우에 대한 FFT 계산 5단계 그래프."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    csv = _find_csv_for_status(status_ko)
    if csv is None:
        print(f"[error] '{status_ko}' 상태의 CSV 를 찾지 못했습니다.")
        return None
    equip, rpm, kw, cur = parse_motor_spec_line(csv)
    sig = load_signal(csv)
    windows = make_windows(sig)
    if not windows:
        print("[error] 윈도우를 만들 수 없습니다(신호가 너무 짧음).")
        return None
    w, padded = windows[0]
    f_rot = rpm / 60.0

    win_sig, spectrum, freqs = _compute_spectrum(w)
    feats = order_spectrum_features(w, rpm)

    t = np.arange(len(w)) / SAMPLE_RATE

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        f"FFT 특징 계산 단계별 — {status_ko}  "
        f"(설비={equip}, RPM={rpm:.0f}, 회전주파수 f_rot={f_rot:.1f}Hz, {kw}kW)",
        fontsize=14)

    # (1) 원시 신호
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(t, w, lw=0.6, color="#1f77b4")
    ax1.set_title("(1) 원시 진동 신호 — 1초 윈도우 (4000샘플)")
    ax1.set_xlabel("시간 (s)"); ax1.set_ylabel("진폭")
    ax1.grid(True, alpha=0.3)

    # (2) 해닝 창 적용
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t, win_sig, lw=0.6, color="#1f77b4", label="창 적용 신호")
    ax2.plot(t, np.hanning(len(w)) * np.max(np.abs(w)), lw=1.2,
             color="#d62728", alpha=0.7, label="해닝 창(스케일)")
    ax2.set_title("(2) 해닝 창 적용 → 스펙트럼 누설 감소")
    ax2.set_xlabel("시간 (s)"); ax2.set_ylabel("진폭")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # (3) 진폭 스펙트럼 (Hz)
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(freqs, spectrum, lw=0.7, color="#333333")
    ax3.set_xlim(0, min(600, freqs[-1]))
    ax3.set_title("(3) 진폭 스펙트럼 (rfft) — 주파수영역")
    ax3.set_xlabel("주파수 (Hz)"); ax3.set_ylabel("정규화 진폭 |FFT|/n")
    ax3.axvline(f_rot, color="#d62728", ls="--", lw=1,
                label=f"1x = {f_rot:.1f}Hz")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # (4) 차수축 환산 + 대역
    ax4 = fig.add_subplot(3, 2, 4)
    orders = freqs / f_rot
    ax4.plot(orders, spectrum, lw=0.7, color="#333333")
    ax4.set_xlim(0, 22)
    for name, (lo, hi) in ORDER_BANDS.items():
        ax4.axvspan(lo, hi, color=BAND_COLORS[name], alpha=0.25,
                    label=BAND_KO[name])
    ax4.set_title("(4) 회전차수(order) 축 환산 — orders = freq / f_rot")
    ax4.set_xlabel("회전차수 (×rpm)"); ax4.set_ylabel("진폭")
    ax4.legend(fontsize=8, ncol=2); ax4.grid(True, alpha=0.3)

    # (5) 대역 에너지 / 비율
    ax5 = fig.add_subplot(3, 2, 5)
    names = list(ORDER_BANDS.keys())
    energies = [feats[n] for n in names]
    ax5.bar([BAND_KO[n] for n in names], energies,
            color=[BAND_COLORS[n] for n in names])
    ax5.set_title("(5a) 대역별 에너지  band_e = Σ spectrum²")
    ax5.set_ylabel("에너지"); ax5.tick_params(axis="x", rotation=15)
    ax5.grid(True, alpha=0.3, axis="y")

    ax6 = fig.add_subplot(3, 2, 6)
    ratios = [feats[n + "_ratio"] for n in names]
    ax6.bar([BAND_KO[n] for n in names], ratios,
            color=[BAND_COLORS[n] for n in names])
    ax6.set_title("(5b) 대역 에너지 비율  ratio = band_e / total_energy")
    ax6.set_ylabel("비율"); ax6.tick_params(axis="x", rotation=15)
    ax6.grid(True, alpha=0.3, axis="y")

    # 스펙트럼 통계 텍스트
    txt = (f"spectral_centroid = {feats['spectral_centroid']:.1f} Hz\n"
           f"spectral_entropy  = {feats['spectral_entropy']:.3f}\n"
           f"spectral_energy   = {feats['spectral_energy']:.3e}")
    fig.text(0.5, 0.005, txt, ha="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#999"))

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = os.path.join(PLOT_DIR, "fft_steps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  저장:", os.path.relpath(out, PROJECT_ROOT))
    return out


def visualize_class_spectra():
    """고장유형별 평균 스펙트럼(차수축) 비교."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    statuses = list(LABEL_MAP.keys())  # 한글 상태명

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    common_orders = np.linspace(0, 22, 440)

    plotted = 0
    for i, status in enumerate(statuses):
        csv = _find_csv_for_status(status)
        if csv is None:
            continue
        equip, rpm, kw, cur = parse_motor_spec_line(csv)
        if not (rpm and rpm > 0):
            continue
        sig = load_signal(csv)
        windows = make_windows(sig)
        if not windows:
            continue
        # 여러 윈도우 평균 스펙트럼
        specs = []
        f_rot = rpm / 60.0
        for w, _ in windows[:8]:
            _, spectrum, freqs = _compute_spectrum(w)
            orders = freqs / f_rot
            specs.append(np.interp(common_orders, orders, spectrum))
        mean_spec = np.mean(specs, axis=0)
        # 정규화(피크=1)로 형태 비교 용이
        mean_spec = mean_spec / (mean_spec.max() + 1e-12)
        ax.plot(common_orders, mean_spec, lw=1.3, color=cmap(i),
                label=f"{LABEL_MAP[status][0]} ({status})")
        plotted += 1

    for name, (lo, hi) in ORDER_BANDS.items():
        ax.axvspan(lo, hi, color="#cccccc", alpha=0.25)
    ax.set_xlim(0, 22)
    ax.set_title("고장유형별 평균 스펙트럼 (회전차수 축, 피크 정규화)")
    ax.set_xlabel("회전차수 (×rpm)"); ax.set_ylabel("정규화 진폭")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fft_class_spectra.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {os.path.relpath(out, PROJECT_ROOT)}  ({plotted}개 클래스)")
    return out


def run(status_ko="회전체불평형"):
    print("=" * 70)
    print("FFT 특징 계산 단계별 시각화")
    print("=" * 70)
    saved = []
    s = visualize_steps(status_ko)
    if s:
        saved.append(s)
    s = visualize_class_spectra()
    if s:
        saved.append(s)
    print("=" * 70)
    print(f"완료. {len(saved)}개 이미지 -> {os.path.relpath(PLOT_DIR, PROJECT_ROOT)}")
    print("=" * 70)
    return saved


if __name__ == "__main__":
    status = sys.argv[1] if len(sys.argv) > 1 else "회전체불평형"
    run(status)
