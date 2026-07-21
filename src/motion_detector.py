"""
Unsupervised move / no-move detection on the PoseC3D representation.

The decision is grounded in the subject-centered crop from `heatmap_volume`:
after cropping and resizing keypoints into a fixed grid (removing camera
distance / framing confounds), motion is measured as the temporal variance of
keypoint positions.  This is Signal A, the primary decision.  Signal B (the
L1 temporal difference of the rendered joint heatmaps) is an optional,
heavier corroboration computed once per person.

This layer replaces the old heuristic `feature_extractor` + `movement_recognizer`
kinematics.  It emits a per-window DataFrame (schema below) that drives clip
splitting and plotting, and exports CNN-ready PYSKL annotations.
"""
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.heatmap_volume import (
    EPS,
    HEATMAP_SIZE,
    REGION_GROUPS,
    build_keypoint_tensor,
    pose_compact,
    resize_keypoints,
    render_volume,
    render_sequence_heatmaps,
    heatmap_diff_energy,
)
from src.utils.logger import get_logger

_LOGGER = get_logger(name="src.motion_detector", level="INFO")


# ==================== helpers ====================

def get_tracker_ids(frames: list[dict]) -> list[int]:
    """Return sorted list of unique tracker IDs seen across all frames."""
    seen: set[int] = set()
    for f in frames:
        for p in f['persons']:
            tid = p.get('tracker_id')
            if tid is not None:
                seen.add(int(tid))
    return sorted(seen)


def _weighted_variance(coords: np.ndarray, weights: np.ndarray) -> float:
    """
    Confidence-weighted temporal variance of a 1-D coordinate series.
    coords, weights: (n,).  Returns 0.0 when there is too little valid signal.
    """
    wsum = float(weights.sum())
    if wsum < EPS or len(coords) < 2:
        return 0.0
    mean = float((coords * weights).sum() / wsum)
    var = float((weights * (coords - mean) ** 2).sum() / wsum)
    return var


# ==================== Signal A: windowed motion energy ====================

def compute_motion_energy_windows(
        frames: list[dict],
        person_id: int,
        fps: float,
        window_seconds: float = 1.0,
        step_seconds: float = 0.5,
        use_heatmap: bool = True,
        heatmap_out_shape: tuple[int, int] = HEATMAP_SIZE,
        with_limb: bool = True,
        fuse_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        smooth=None,
        flow_series: Optional[tuple] = None,
        flow_weight: float = 0.0,
        region_energy: bool = True,
    ) -> pd.DataFrame:
    """
    Windowed motion energy from up to four signals, all derived from the same
    subject-centered crop (so they share framing):

      - energy_a     : Signal A — Σ_k [Var_t(x_k/W) + Var_t(y_k/H)] of the
                       crop-normalized keypoint coordinates (confidence-weighted).
      - energy_joint : L1 temporal diff of the joint heatmap volume in the window.
      - energy_limb  : L1 temporal diff of the limb heatmap volume in the window.
      - energy_flow  : mean dense-optical-flow magnitude in the window (only when
                       `flow_series` is supplied; genuinely independent of keypoints).

    When use_heatmap is True the signals are fused into a single `motion_energy`
    via `fuse_motion_energy` (each robust-normalized per person, then weighted).
    When False, only Signal A is computed and it becomes `motion_energy`.

    Args:
        smooth:       optional SmoothConfig for temporal keypoint smoothing.
        flow_series:  optional (times, magnitudes) arrays from optical_flow;
                      reduced to a per-window energy_flow and fused with weight
                      `flow_weight`.
        region_energy: also emit per-body-part energies (energy_head, ...).

    The heatmap volume is rendered ONCE for the whole sequence and sliced per
    window, so cost is one render per person regardless of window count.

    Returns the per-window DataFrame, or empty when the person has < 2 frames.
    """
    keypoint, keypoint_score, _, frame_indices, timestamps = build_keypoint_tensor(
        frames, person_id, smooth=smooth,
    )
    if keypoint.shape[0] < 2:
        return pd.DataFrame()

    # Crop + resize the whole sequence into the fixed grid (shared box over all T).
    cropped, crop_shape = pose_compact(keypoint, keypoint_score)
    grid_kp = resize_keypoints(cropped, crop_shape, HEATMAP_SIZE)
    out_h, out_w = HEATMAP_SIZE
    norm_x = grid_kp[..., 0] / out_w
    norm_y = grid_kp[..., 1] / out_h

    # Render the heatmap volume once (aligned 1:1 to the keypoint rows / timestamps).
    joint_vol = limb_vol = None
    if use_heatmap:
        vols = render_sequence_heatmaps(
            keypoint, keypoint_score, out_shape=heatmap_out_shape, with_limb=with_limb,
        )
        joint_vol, limb_vol = vols['joint'], vols['limb']

    # Optical-flow series aligned to its own timestamps (Stage 4).
    flow_times = flow_mags = None
    if flow_series is not None:
        flow_times, flow_mags = np.asarray(flow_series[0]), np.asarray(flow_series[1])

    # Real wall-clock timestamp per row (frames may be sparse / person absent).
    times = np.array(timestamps, dtype=np.float64)
    start_t, end_t = float(times.min()), float(times.max())
    step = max(step_seconds, 1.0 / max(fps, 1.0))
    window = max(window_seconds, step)

    records: list[dict] = []
    t = start_t
    while t <= end_t:
        w_end = min(t + window, end_t)
        mask = (times >= t) & (times <= w_end)
        n = int(mask.sum())
        if n >= 2:
            wx, wy = norm_x[mask], norm_y[mask]
            ws = keypoint_score[mask]
            valid_kp = (ws >= EPS).any(axis=0)

            # Per-joint contributions (summed for Signal A; grouped for regions).
            per_joint = np.zeros(keypoint.shape[1], dtype=np.float64)
            for k in np.where(valid_kp)[0]:
                per_joint[k] = (_weighted_variance(wx[:, k], ws[:, k])
                                + _weighted_variance(wy[:, k], ws[:, k]))
            energy_a = float(per_joint.sum())

            energy_joint = heatmap_diff_energy(joint_vol, mask) if joint_vol is not None else 0.0
            energy_limb = heatmap_diff_energy(limb_vol, mask) if limb_vol is not None else 0.0

            fi_win = np.array(frame_indices)[mask]
            valid_conf = ws[ws >= EPS]
            rec = {
                'person_id': int(person_id),
                'start_time': round(float(t), 3),
                'end_time': round(float(w_end), 3),
                'mid_time': round(float((t + w_end) / 2), 3),
                'start_frame': int(fi_win.min()),
                'end_frame': int(fi_win.max()),
                'energy_a': round(float(energy_a), 6),
                'energy_joint': round(float(energy_joint), 4),
                'energy_limb': round(float(energy_limb), 4),
                'mean_keypoint_conf': round(float(valid_conf.mean()) if valid_conf.size else 0.0, 3),
                'n_valid_kp': int(valid_kp.sum()),
            }

            if flow_mags is not None:
                fmask = (flow_times >= t) & (flow_times <= w_end)
                rec['energy_flow'] = round(float(flow_mags[fmask].mean()) if fmask.any() else 0.0, 4)

            if region_energy:
                for region, idxs in REGION_GROUPS.items():
                    rec[f'energy_{region}'] = round(float(per_joint[idxs].sum()), 6)

            records.append(rec)

        if w_end >= end_t:
            break
        t += step

    df = pd.DataFrame(records)
    if df.empty:
        return df

    if region_energy:
        _add_dominant_region(df)

    if use_heatmap:
        df = fuse_motion_energy(df, fuse_weights, with_limb=with_limb, flow_weight=flow_weight)
    else:
        # Signal A alone drives the decision.
        df['motion_energy'] = df['energy_a']
    return df


def _add_dominant_region(df: pd.DataFrame) -> None:
    """Add `dominant_region` = the region with the highest energy per window."""
    region_cols = [f'energy_{r}' for r in REGION_GROUPS if f'energy_{r}' in df.columns]
    if not region_cols:
        return
    stripped = [c[len('energy_'):] for c in region_cols]
    df['dominant_region'] = df[region_cols].to_numpy().argmax(axis=1)
    df['dominant_region'] = df['dominant_region'].map(dict(enumerate(stripped)))


# Absolute move/still floor on `energy_a` (variance of keypoints in the shared
# 0-1 normalized crop grid).  This is the pose-jitter noise floor: quiet windows
# of genuinely still people sit at ~1e-6..1e-5, while real movement reaches
# 1e-3..1e-1.  A fixed value near 1e-4 separates the two the SAME way for every
# person (unlike a per-person baseline).  Tune via --move-threshold for a given
# camera/resolution; it is a sensible default, not label-calibrated ground truth.
MOVE_ENERGY_THRESHOLD = 1e-4

# Detection-quality gate for the move/still decision.  An occluded or partially
# detected person has low-confidence keypoints (~0.65) that jitter frame-to-frame;
# that jitter inflates `energy_a` and reads as motion even when the body is still.
# A window is only allowed to be "moving" when its mean keypoint confidence clears
# this floor — otherwise its variance is detector uncertainty, not movement, and
# the window defaults to "still".  Measured on recording.mov: windows of real,
# well-detected people that cross the energy threshold sit at conf 0.85-0.98 (all
# 17 keypoints), while occluded people whose jitter fakes motion sit at 0.60-0.73
# (only 13-14 keypoints).  0.8 falls cleanly in that gap.
MOVE_CONF_FLOOR = 0.8


# ==================== signal fusion ====================

# Scale factor making MAD a consistent estimator of the standard deviation
# for normally-distributed data.  Used only for the fused DISPLAY magnitude
# (`motion_energy`), never for the move/still decision.
_MAD_TO_STD = 1.4826


def _robust_normalize(s: pd.Series) -> pd.Series:
    """
    Divide a signal by its own robust scale (median + 1.4826*MAD) so its
    per-person baseline maps to ~1.0 and different signals become comparable.
    Non-negative in, non-negative out.

    "Degenerate" means the signal carries no information (all values ~equal),
    which is judged RELATIVE to the signal's own magnitude — not against an
    absolute constant, since energies span very different scales (Signal A ~1e-4,
    heatmap L1 ~tens).  A genuinely flat signal collapses to 0.
    """
    clean = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    med = float(clean.median())
    mad = float((clean - med).abs().median())
    scale = med + _MAD_TO_STD * mad
    peak = float(clean.abs().max())
    if peak <= 0.0 or scale <= 1e-6 * peak:
        return pd.Series(np.zeros(len(clean)), index=s.index)
    return clean / scale


def fuse_motion_energy(
        df: pd.DataFrame,
        fuse_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        with_limb: bool = True,
        flow_weight: float = 0.0,
    ) -> pd.DataFrame:
    """
    Fuse Signal A, joint-heatmap, limb-heatmap and (optionally) optical-flow
    energies into one `motion_energy`.  Each signal is robust-normalized per
    person (so its own baseline sits at ~1.0), then combined as a weighted
    average.  Because every term is on a unit robust scale, the downstream
    `median + k*MAD` threshold keeps its meaning on the fused value.

    Adds z_a / z_joint / z_limb (+ z_flow when an energy_flow column is present)
    and sets `motion_energy`.  with_limb=False drops the limb term; flow_weight
    <= 0 drops the flow term.
    """
    out = df.copy()
    w_a, w_joint, w_limb = fuse_weights
    if not with_limb:
        w_limb = 0.0

    out['z_a'] = _robust_normalize(out['energy_a'])
    out['z_joint'] = _robust_normalize(out['energy_joint'])
    out['z_limb'] = _robust_normalize(out['energy_limb'])

    terms = [(w_a, out['z_a']), (w_joint, out['z_joint']), (w_limb, out['z_limb'])]

    if 'energy_flow' in out.columns:
        out['z_flow'] = _robust_normalize(out['energy_flow'])
        if flow_weight > 0:
            terms.append((flow_weight, out['z_flow']))

    wsum = sum(w for w, _ in terms)
    if wsum <= 0:
        # Degenerate weights -> fall back to Signal A.
        terms, wsum = [(1.0, out['z_a'])], 1.0
    fused = sum(w * z for w, z in terms) / wsum
    out['motion_energy'] = fused.round(6)
    return out


# ==================== decision ====================


def decide_move_no_move(
        motion_df: pd.DataFrame,
        threshold: Optional[float] = None,
        conf_floor: float = MOVE_CONF_FLOOR,
    ) -> pd.DataFrame:
    """
    Absolute move/no-move decision: a window is "moving" when its raw
    crop-normalized keypoint variance `energy_a` exceeds a FIXED threshold that
    means the same thing for every person AND the window's detection quality is
    high enough to trust that variance.

    This deliberately replaces the old per-person `median + k*MAD` baseline,
    which only flagged motion that spiked above a person's OWN normal (a
    rhythm/above-baseline detector).  Here the question is simply "did this
    person move", answered identically across people.

    The decision uses `energy_a` (not the fused `motion_energy`): `energy_a` is
    in shared, person-comparable units (variance in a 0-1 grid), whereas the
    fused `motion_energy` is per-person normalized and so not comparable in
    absolute terms.  `motion_energy` remains available as a display magnitude.

    Quality gate: a window whose `mean_keypoint_conf` is below `conf_floor` is
    forced "still".  Its keypoints are too uncertain (occlusion / partial
    detection) for the variance to mean movement rather than detector jitter.

    Adds: active_threshold (the constant used), is_moving (bool), state.
    """
    if motion_df.empty:
        return motion_df

    thr = MOVE_ENERGY_THRESHOLD if threshold is None else float(threshold)
    df = motion_df.copy()
    df['active_threshold'] = thr
    reliable = df['mean_keypoint_conf'] >= conf_floor if 'mean_keypoint_conf' in df.columns else True
    df['is_moving'] = (df['energy_a'] > thr) & reliable
    df['state'] = np.where(df['is_moving'], 'moving', 'still')
    return df


# ==================== Signal B: heatmap temporal difference ====================

def corroborate_signal_b(
        frames: list[dict],
        person_id: int,
        motion_df: pd.DataFrame,
        clip_len: int = 48,
    ) -> pd.DataFrame:
    """
    Signal B (corroborating, heavier).  Render the person's joint heatmap
    volume once and compute the mean over T of the L1 difference between
    consecutive frames.  Because heatmap peaks are soft Gaussians, this reacts
    smoothly to sub-pixel motion.  Attaches a per-person scalar
    `heatmap_l1_mean` (broadcast to every row) and an `ab_agreement` flag that
    is False when Signal B is near-zero while Signal A says the person moved
    (a hint of a crop/render problem).
    """
    if motion_df.empty:
        return motion_df

    vol = render_volume(frames, person_id, clip_len=clip_len, with_limb=False)['joint']  # (K, T, H, W)
    df = motion_df.copy()
    if vol.shape[1] < 2:
        df['heatmap_l1_mean'] = 0.0
        df['ab_agreement'] = True
        return df

    diffs = np.abs(np.diff(vol, axis=1)).sum(axis=(0, 2, 3))  # (T-1,)
    l1_mean = float(diffs.mean())
    df['heatmap_l1_mean'] = round(l1_mean, 4)
    signal_a_moved = bool(df['is_moving'].any()) if 'is_moving' in df.columns else False
    df['ab_agreement'] = not (signal_a_moved and l1_mean < EPS)
    if not df['ab_agreement'].iat[0]:
        _LOGGER.warning(
            f"  person{person_id}: Signal A reports motion but heatmap L1 ~ 0 "
            f"({l1_mean:.4f}) — check crop/render."
        )
    return df


# ==================== summary / output ====================

def summarize_motion(motion_df: pd.DataFrame) -> dict:
    """High-level per-person motion statistics."""
    if motion_df.empty:
        return {}

    moving = motion_df['is_moving'] if 'is_moving' in motion_df.columns else pd.Series(dtype=bool)
    return {
        'n_windows': int(len(motion_df)),
        'total_duration_s': round(float(motion_df['end_time'].max() - motion_df['start_time'].min()), 3),
        'moving_fraction': round(float(moving.mean()) if not moving.empty else 0.0, 3),
        'mean_motion_energy': round(float(motion_df['motion_energy'].mean()), 6),
        'peak_motion_energy': round(float(motion_df['motion_energy'].max()), 6),
        'active_threshold': round(float(motion_df['active_threshold'].iloc[0]), 6)
                            if 'active_threshold' in motion_df.columns else 0.0,
    }


def write_text_log(motion_df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    path = Path(output_path)
    with open(path, 'w') as f:
        for row in motion_df.itertuples(index=False):
            f.write(
                f"{row.start_time:>6.3f}-{row.end_time:>6.3f}s "
                f"person={row.person_id} "
                f"state={row.state} "
                f"energy={row.motion_energy:.6f} "
                f"threshold={row.active_threshold:.6f} "
                f"n_kp={row.n_valid_kp}\n"
            )


def write_summary_json(summary: dict, output_path: Union[str, Path]) -> None:
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
