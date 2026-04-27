import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq

HEAD      = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
SHOULDERS = ['left_shoulder', 'right_shoulder']
ARMS      = ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
TORSO     = ['left_hip', 'right_hip']
UPPER     = HEAD + SHOULDERS + ARMS + TORSO


# ── helpers ──────────────────────────────────────────────────────────────────

def _smooth(s: pd.Series, window: int = 7) -> pd.Series:
    """Savitzky-Golay filter after linear interpolation over missing samples."""
    filled = s.interpolate(method='linear', limit_direction='both').ffill().bfill()
    if filled.isna().all() or len(filled) < window:
        return s
    smoothed = signal.savgol_filter(filled.values, window_length=window, polyorder=3)
    out = pd.Series(smoothed, index=s.index)
    out[s.isna()] = np.nan
    return out


def _diff(s: pd.Series, fps: float) -> pd.Series:
    return s.diff() * fps


def _speed(vx: pd.Series, vy: pd.Series) -> pd.Series:
    return np.sqrt(vx ** 2 + vy ** 2)


def _dominant_freq(s: pd.Series, fps: float) -> float:
    clean = s.dropna().values
    if len(clean) < 16:
        return 0.0
    detrended = signal.detrend(clean)
    freqs = fftfreq(len(detrended), d=1.0 / fps)
    spectrum = np.abs(fft(detrended))
    mask = freqs > 0
    return float(freqs[mask][np.argmax(spectrum[mask])]) if mask.any() else 0.0


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle (degrees) at joint B for points A–B–C, NaN when points are missing."""
    ba, bc = a - b, c - b
    na = np.linalg.norm(ba, axis=1)
    nc = np.linalg.norm(bc, axis=1)
    valid = (na > 1e-6) & (nc > 1e-6)
    cos_a = np.where(valid, np.sum(ba * bc, axis=1) / np.where(valid, na * nc, 1.0), np.nan)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return np.where(valid, np.degrees(np.arccos(cos_a)), np.nan)


# ── public API ────────────────────────────────────────────────────────────────

def poses_to_dataframe(frames: list[dict], person_idx: int = 0) -> pd.DataFrame:
    """Flatten per-frame pose data into a wide DataFrame (one row per sample)."""
    records = []
    for f in frames:
        row: dict = {'frame': f['frame_idx'], 'sample': f['sample_idx'], 'time': f['timestamp']}
        if person_idx < len(f['persons']):
            for kp, val in f['persons'][person_idx].items():
                if val:
                    row[f'{kp}_x'] = val['x']
                    row[f'{kp}_y'] = val['y']
                    row[f'{kp}_c'] = val['conf']
                else:
                    row[f'{kp}_x'] = np.nan
                    row[f'{kp}_y'] = np.nan
                    row[f'{kp}_c'] = 0.0
        records.append(row)
    return pd.DataFrame(records)


def normalize_to_torso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Express every keypoint position relative to the shoulder midpoint,
    scaled by shoulder width.  This removes global camera/body translation
    so that tiny relative movements become visible in the signal.
    """
    norm = df.copy()
    lx, ly = 'left_shoulder_x', 'left_shoulder_y'
    rx, ry = 'right_shoulder_x', 'right_shoulder_y'
    if not all(c in df.columns for c in [lx, ly, rx, ry]):
        return norm

    ref_x = (df[lx] + df[rx]) / 2
    ref_y = (df[ly] + df[ry]) / 2
    width = np.sqrt((df[lx] - df[rx]) ** 2 + (df[ly] - df[ry]) ** 2)
    scale = width.replace(0, np.nan).interpolate().ffill().bfill()
    scale = scale.where(scale > 1, other=scale.median())  # guard against zero

    for kp in [c.replace('_x', '') for c in df.columns if c.endswith('_x')]:
        norm[f'{kp}_x'] = (df[f'{kp}_x'] - ref_x) / scale
        norm[f'{kp}_y'] = (df[f'{kp}_y'] - ref_y) / scale

    return norm


def compute_features(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Compute frame-level kinematic features for every upper-body keypoint:
      speed, acceleration, per-axis range-of-motion (1-second window).
    Also computes elbow angles when landmarks are available.
    """
    feat = df[['frame', 'sample', 'time']].copy()

    for kp in UPPER:
        xc, yc = f'{kp}_x', f'{kp}_y'
        if xc not in df.columns:
            continue

        xs, ys = _smooth(df[xc]), _smooth(df[yc])
        vx, vy = _diff(xs, fps), _diff(ys, fps)
        ax, ay = _diff(vx, fps), _diff(vy, fps)

        win = max(1, int(fps))  # 1-second window for ROM
        feat[f'{kp}_speed'] = _speed(vx, vy)
        feat[f'{kp}_accel'] = _speed(ax, ay)
        feat[f'{kp}_rom_x'] = (xs.rolling(win, min_periods=1).max()
                                - xs.rolling(win, min_periods=1).min())
        feat[f'{kp}_rom_y'] = (ys.rolling(win, min_periods=1).max()
                                - ys.rolling(win, min_periods=1).min())

    # Joint angles
    for side in ('left', 'right'):
        s, e, w = f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist'
        cols = [f'{j}_x' for j in (s, e, w)] + [f'{j}_y' for j in (s, e, w)]
        if all(c in df.columns for c in cols):
            feat[f'{side}_elbow_angle'] = _joint_angle(
                df[[f'{s}_x', f'{s}_y']].values,
                df[[f'{e}_x', f'{e}_y']].values,
                df[[f'{w}_x', f'{w}_y']].values,
            )

    return feat


def compute_summary(feat: pd.DataFrame, raw_df: pd.DataFrame, fps: float) -> dict:
    """High-level statistics suitable for comparing engagement across participants."""
    summary: dict = {}

    for kp in UPPER:
        sc = f'{kp}_speed'
        if sc not in feat.columns:
            continue
        spd = feat[sc].dropna()
        if spd.empty:
            continue
        summary[f'{kp}_mean_speed']     = float(spd.mean())
        summary[f'{kp}_p95_speed']      = float(spd.quantile(0.95))
        # fraction of frames with detectable movement (> 1.5 px/s in normalised space)
        summary[f'{kp}_active_fraction'] = float((spd > 1.5).mean())
        if f'{kp}_x' in raw_df.columns:
            summary[f'{kp}_dominant_freq_hz'] = _dominant_freq(_smooth(raw_df[f'{kp}_x']), fps)

    # Head dynamics — most sensitive indicator of rhythmic engagement
    if 'nose_x' in raw_df.columns and 'nose_y' in raw_df.columns:
        nx = _smooth(raw_df['nose_x'])
        ny = _smooth(raw_df['nose_y'])
        summary['head_lateral_freq_hz']  = _dominant_freq(nx, fps)
        summary['head_vertical_freq_hz'] = _dominant_freq(ny, fps)
        summary['head_lateral_std']      = float(nx.std())
        summary['head_vertical_std']     = float(ny.std())

    # Shoulder sway
    lsx, rsx = 'left_shoulder_x', 'right_shoulder_x'
    if lsx in raw_df.columns and rsx in raw_df.columns:
        mid = (_smooth(raw_df[lsx]) + _smooth(raw_df[rsx])) / 2
        summary['shoulder_sway_freq_hz'] = _dominant_freq(mid, fps)
        summary['shoulder_sway_std']     = float(mid.std())

    # Overall upper-body movement energy
    speed_cols = [f'{kp}_speed' for kp in HEAD + SHOULDERS + ARMS
                  if f'{kp}_speed' in feat.columns]
    if speed_cols:
        summary['total_movement_energy'] = float(feat[speed_cols].fillna(0).mean(axis=1).mean())

    return summary
