"""
공용 특징 추출 모듈 (학습/추론 공통).

build_feature_table.py(오프라인 학습 테이블 생성)와
util/ 의 실시간 추론 스크립트가 **동일한 특징**을 사용하도록 보장한다.

표준 특징 벡터(FEATURE_COLUMNS 순서)
------------------------------------
[power_kw, rpm,
 rms, peak, crest_factor, std, kurtosis, skewness, p2p,
 spectral_centroid, spectral_entropy, spectral_energy,
 band_1x, band_1x_ratio, band_2x, band_2x_ratio,
 band_3x, band_3x_ratio, band_high, band_high_ratio]

이 순서는 train_from_feature_table.py 가 학습에 사용한 컬럼 순서와 일치해야 한다.
"""

import numpy as np
from scipy.stats import kurtosis as _kurtosis, skew as _skew


# ---------------------------------------------------------------------------
# 설정 (build_feature_table.py 와 동일해야 함)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 4000               # Hz (전 파일 동일)
WINDOW_SECONDS = 1.0             # 윈도우 길이(초)
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)  # 4000 샘플
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
MIN_USABLE = 256                 # 이보다 짧으면 사용 불가

# 회전차수(order) 대역 — 회전주파수 배수 기준
ORDER_BANDS = {
    "band_1x": (0.8, 1.2),
    "band_2x": (1.8, 2.2),
    "band_3x": (2.8, 3.2),
    "band_high": (5.0, 20.0),
}

# 상태(한글) -> (영문 코드, 라벨 번호, 이상 여부)
LABEL_MAP = {
    "정상":       ("NORMAL",    0, 0),
    "축정렬불량":  ("MISALIGN",  1, 1),
    "회전체불평형": ("IMBALANCE", 2, 1),
    "베어링불량":  ("BEARING",   3, 1),
    "벨트느슨함":  ("BELT",      4, 1),
}

# 영문 코드 -> 한글
LABEL_EN_TO_KO = {v[0]: k for k, v in LABEL_MAP.items()}

# 학습 시 사용된 특징 컬럼 순서 (train_from_feature_table.py 와 동일)
FEATURE_COLUMNS = [
    "power_kw", "rpm",
    "rms", "peak", "crest_factor", "std", "kurtosis", "skewness", "p2p",
    "spectral_centroid", "spectral_entropy", "spectral_energy",
    "band_1x", "band_1x_ratio", "band_2x", "band_2x_ratio",
    "band_3x", "band_3x_ratio", "band_high", "band_high_ratio",
]


# ---------------------------------------------------------------------------
# 특징 함수
# ---------------------------------------------------------------------------
def time_features(w):
    """시간영역 특징."""
    w = np.asarray(w, dtype=float)
    rms = float(np.sqrt(np.mean(w ** 2)))
    peak = float(np.max(np.abs(w)))
    crest = float(peak / (rms + 1e-12))
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest,
        "std": float(np.std(w)),
        "kurtosis": float(_kurtosis(w)),
        "skewness": float(_skew(w)),
        "p2p": float(np.max(w) - np.min(w)),
    }


def order_spectrum_features(w, rpm, fs=SAMPLE_RATE):
    """FFT -> 회전차수(order) 기반 대역 에너지 + 스펙트럼 통계."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(w * win)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total_energy = float(np.sum(spectrum ** 2) + 1e-12)
    centroid = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12))
    p = spectrum / (np.sum(spectrum) + 1e-12)
    spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)))

    feats = {
        "spectral_centroid": centroid,
        "spectral_entropy": spectral_entropy,
        "spectral_energy": total_energy,
    }

    if rpm and rpm == rpm and rpm > 0:
        f_rot = rpm / 60.0
        orders = freqs / f_rot
        for name, (lo, hi) in ORDER_BANDS.items():
            mask = (orders >= lo) & (orders < hi)
            band_e = float(np.sum(spectrum[mask] ** 2))
            feats[name] = band_e
            feats[name + "_ratio"] = float(band_e / total_energy)
    else:
        for name in ORDER_BANDS:
            feats[name] = np.nan
            feats[name + "_ratio"] = np.nan

    return feats


def extract_feature_dict(window, rpm=np.nan, power_kw=np.nan, fs=SAMPLE_RATE):
    """단일 윈도우 -> 특징 dict (메타 power_kw/rpm 포함)."""
    feat = {"power_kw": power_kw, "rpm": rpm}
    feat.update(time_features(window))
    feat.update(order_spectrum_features(window, rpm, fs=fs))
    return feat


def extract_feature_vector(window, rpm=np.nan, power_kw=np.nan, fs=SAMPLE_RATE):
    """단일 윈도우 -> FEATURE_COLUMNS 순서의 1D numpy 벡터 (NaN은 0 대치)."""
    feat = extract_feature_dict(window, rpm=rpm, power_kw=power_kw, fs=fs)
    vec = np.array([feat.get(c, np.nan) for c in FEATURE_COLUMNS], dtype=float)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_amplitude(signal, mean=None, std=None):
    """진폭 Z-score 정규화. mean/std 미지정 시 신호 자체 통계 사용."""
    signal = np.asarray(signal, dtype=float)
    if mean is None:
        mean = float(np.mean(signal))
    if std is None or std == 0:
        std = float(np.std(signal) + 1e-12)
    return (signal - mean) / std


def make_windows(sig, window_size=WINDOW_SIZE, step=STEP_SIZE):
    """
    고정 길이 슬라이딩 윈도우 분할.
    반환: list of (window, padded:bool)
    """
    out = []
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    if n >= window_size:
        for start in range(0, n - window_size + 1, step):
            out.append((sig[start:start + window_size], False))
    elif n >= MIN_USABLE:
        padded = np.zeros(window_size, dtype=sig.dtype)
        padded[:n] = sig
        out.append((padded, True))
    return out
