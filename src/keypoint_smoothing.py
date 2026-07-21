"""
Temporal smoothing of per-person keypoint sequences.

Subtle-movement detection lives or dies on the noise floor: per-frame keypoint
jitter from the pose estimator looks like motion to a variance-based signal.
Smoothing the (T, K, 2) coordinate series along time suppresses that jitter
while preserving genuine motion.

The default filter is **OneEuro** (Casiez et al., 2012), the standard choice
for interactive pose work: it is adaptive — at low speed it smooths hard (quiet
still-baseline), at high speed it barely lags (real motion preserved).  This is
exactly the trade-off a subtle-movement detector needs.

Smoothing is gap-aware and confidence-gated: each keypoint is smoothed only over
contiguous runs of valid (score >= EPS) samples, and filter state resets across
holes.  Missing samples are never fabricated — their score stays 0 so downstream
code still ignores them.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from src.heatmap_volume import EPS


@dataclass
class SmoothConfig:
    """Configuration for keypoint temporal smoothing."""
    filter: str = "oneeuro"          # {"none", "oneeuro", "savgol", "ema"}
    min_cutoff: float = 1.0          # OneEuro: baseline cutoff frequency (Hz)
    beta: float = 0.3               # OneEuro: speed coefficient
    d_cutoff: float = 1.0           # OneEuro: derivative cutoff (Hz)
    savgol_window: int = 7          # savgol: window length (frames, odd)
    ema_alpha: float = 0.5          # ema: smoothing factor in (0, 1]


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _oneeuro_1d(x: np.ndarray, t: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
    """OneEuro filter over a 1-D series sampled at (possibly irregular) times t."""
    n = len(x)
    if n < 2:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    x_prev, dx_prev = x[0], 0.0
    for i in range(1, n):
        dt = float(t[i] - t[i - 1])
        if dt <= 0:
            dt = 1e-3
        dx = (x[i] - x_prev) / dt
        a_d = _alpha(cfg.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = cfg.min_cutoff + cfg.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x[i] + (1.0 - a) * x_prev
        out[i] = x_hat
        x_prev, dx_prev = x_hat, dx_hat
    return out


def _ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _savgol_1d(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window or window < 3:
        return x.copy()
    w = window if window % 2 == 1 else window + 1
    poly = min(3, w - 1)
    return savgol_filter(x, window_length=w, polyorder=poly)


def _smooth_run(x: np.ndarray, t: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
    if cfg.filter == "oneeuro":
        return _oneeuro_1d(x, t, cfg)
    if cfg.filter == "ema":
        return _ema_1d(x, cfg.ema_alpha)
    if cfg.filter == "savgol":
        return _savgol_1d(x, cfg.savgol_window)
    return x.copy()


def smooth_keypoint_tensor(
        keypoint: np.ndarray,        # (T, K, 2)
        keypoint_score: np.ndarray,  # (T, K)
        timestamps,                  # (T,) seconds
        cfg: Optional[SmoothConfig],
    ) -> np.ndarray:
    """
    Return a smoothed copy of `keypoint`.  Each keypoint's x and y are smoothed
    independently over contiguous valid runs (score >= EPS); holes are left
    untouched (their score is 0 downstream).  Returns the input unchanged when
    cfg is None or cfg.filter == "none".
    """
    if cfg is None or cfg.filter == "none" or keypoint.shape[0] < 2:
        return keypoint

    t = np.asarray(timestamps, dtype=np.float64)
    out = keypoint.copy()
    T, K, _ = keypoint.shape
    valid = keypoint_score >= EPS

    for k in range(K):
        v = valid[:, k]
        if v.sum() < 2:
            continue
        # Smooth each contiguous run of valid samples separately.
        idx = np.where(v)[0]
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for run in np.split(idx, splits):
            if len(run) < 2:
                continue
            for axis in (0, 1):
                out[run, k, axis] = _smooth_run(keypoint[run, k, axis], t[run], cfg)
    return out
