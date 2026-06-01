import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def _safe_mean(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    value = frame[column].mean(skipna=True)
    return float(value) if pd.notna(value) else default


def _safe_max(frame: pd.DataFrame, columns: list[str], default: float = 0.0) -> float:
    available = [c for c in columns if c in frame.columns]
    if not available:
        return default
    value = frame[available].max(axis=1, skipna=True).mean(skipna=True)
    return float(value) if pd.notna(value) else default


def _safe_std(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    value = frame[column].std(skipna=True)
    return float(value) if pd.notna(value) else default


def _threshold(values: pd.Series, floor: float, quantile: float = 0.75, multiplier: float = 1.1) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return floor
    return float(max(floor, clean.quantile(quantile) * multiplier))


def _movement_confidence(labels: list[str], mean_keypoint_conf: float, energy: float, active_threshold: float) -> float:
    if labels == ['still']:
        base = 0.45
    else:
        base = 0.5 + min(0.25, energy / max(active_threshold, 1e-6) * 0.08)
    if len(labels) > 1:
        base += 0.08
    base += min(0.2, mean_keypoint_conf * 0.2)
    return float(np.clip(base, 0.05, 0.95))


def generate_movement_log(
        raw_df: pd.DataFrame,
        norm_df: pd.DataFrame,
        feat_df: pd.DataFrame,
        fps: float,
        person_id: Optional[int],
        window_seconds: float = 1.0,
        step_seconds: float = 0.5,
    ) -> pd.DataFrame:
    """
    Convert per-frame pose features into time-windowed movement labels.

    The labels are heuristic and pose-derived: YOLO provides the person boxes
    and keypoints; this layer interprets relative keypoint motion per person.
    """
    if feat_df.empty:
        return pd.DataFrame()

    speed_cols = [c for c in feat_df.columns if c.endswith('_speed')]
    energy_series = feat_df[speed_cols].mean(axis=1, skipna=True).fillna(0) if speed_cols else pd.Series(0, index=feat_df.index)
    active_threshold = _threshold(energy_series, floor=0.012, quantile=0.65, multiplier=1.15)
    wrist_threshold = _threshold(
        feat_df[[c for c in ['left_wrist_speed', 'right_wrist_speed'] if c in feat_df.columns]].max(axis=1),
        floor=0.025,
        quantile=0.7,
        multiplier=1.05,
    ) if any(c in feat_df.columns for c in ['left_wrist_speed', 'right_wrist_speed']) else 0.025

    start_t = float(feat_df['time'].min())
    end_t = float(feat_df['time'].max())
    step = max(step_seconds, 1.0 / max(fps, 1.0))
    window = max(window_seconds, step)

    records = []
    t = start_t
    while t <= end_t:
        w_end = min(t + window, end_t)
        feat_w = feat_df[(feat_df['time'] >= t) & (feat_df['time'] <= w_end)]
        raw_w = raw_df[(raw_df['time'] >= t) & (raw_df['time'] <= w_end)]
        norm_w = norm_df[(norm_df['time'] >= t) & (norm_df['time'] <= w_end)]
        if feat_w.empty:
            t += step
            continue

        energy = float(feat_w[speed_cols].mean(axis=1, skipna=True).mean(skipna=True)) if speed_cols else 0.0
        if not np.isfinite(energy):
            energy = 0.0
        head_speed = _safe_mean(feat_w, 'nose_speed')
        head_rom_x = _safe_mean(feat_w, 'nose_rom_x')
        head_rom_y = _safe_mean(feat_w, 'nose_rom_y')
        left_wrist_speed = _safe_mean(feat_w, 'left_wrist_speed')
        right_wrist_speed = _safe_mean(feat_w, 'right_wrist_speed')
        wrist_speed_max = max(left_wrist_speed, right_wrist_speed)
        elbow_std = max(
            _safe_std(feat_w, 'left_elbow_angle'),
            _safe_std(feat_w, 'right_elbow_angle'),
        )

        labels: list[str] = []
        if head_rom_x > max(0.006, head_rom_y * 1.25) and head_speed > active_threshold * 0.45:
            labels.append('head_shaking')
        if head_rom_y > max(0.006, head_rom_x * 1.15) and head_speed > active_threshold * 0.45:
            labels.append('nodding')
        if left_wrist_speed > wrist_threshold:
            labels.append('left_hand_moving')
        if right_wrist_speed > wrist_threshold:
            labels.append('right_hand_moving')
        if elbow_std > 2.0 and wrist_speed_max > wrist_threshold * 0.75:
            labels.append('arm_bending')
        if energy > active_threshold and not labels:
            labels.append('body_shift')
        if not labels:
            labels.append('still')

        primary = labels[0]
        if any(label.endswith('hand_moving') for label in labels):
            primary = 'gesturing'
        elif 'arm_bending' in labels:
            primary = 'arm_bending'
        elif 'head_shaking' in labels:
            primary = 'head_shaking'
        elif 'nodding' in labels:
            primary = 'nodding'
        elif 'body_shift' in labels:
            primary = 'body_shift'

        conf_cols = [c for c in raw_w.columns if c.endswith('_c')]
        mean_keypoint_conf = float(raw_w[conf_cols].replace(0, np.nan).mean(axis=1).mean(skipna=True)) if conf_cols else 0.0
        if not np.isfinite(mean_keypoint_conf):
            mean_keypoint_conf = 0.0
        bbox_conf = _safe_mean(raw_w, 'bbox_conf')
        confidence = _movement_confidence(labels, mean_keypoint_conf, energy, active_threshold)

        records.append({
            'person_id': person_id,
            'start_time': round(float(t), 3),
            'end_time': round(float(w_end), 3),
            'mid_time': round(float((t + w_end) / 2), 3),
            'start_frame': int(feat_w['frame'].min()),
            'end_frame': int(feat_w['frame'].max()),
            'primary_movement': primary,
            'movement_labels': ','.join(labels),
            'confidence': round(confidence, 3),
            'bbox_confidence': round(bbox_conf, 3),
            'mean_keypoint_confidence': round(mean_keypoint_conf, 3),
            'movement_energy': round(energy, 5),
            'active_threshold': round(active_threshold, 5),
            'head_speed_mean': round(head_speed, 5),
            'head_rom_x': round(head_rom_x, 5),
            'head_rom_y': round(head_rom_y, 5),
            'left_wrist_speed_mean': round(left_wrist_speed, 5),
            'right_wrist_speed_mean': round(right_wrist_speed, 5),
            'left_elbow_angle_std': round(_safe_std(feat_w, 'left_elbow_angle'), 5),
            'right_elbow_angle_std': round(_safe_std(feat_w, 'right_elbow_angle'), 5),
            'left_wrist_y_norm': round(_safe_mean(norm_w, 'left_wrist_y'), 5),
            'right_wrist_y_norm': round(_safe_mean(norm_w, 'right_wrist_y'), 5),
            'bbox_x1': round(_safe_mean(raw_w, 'bbox_x1'), 2),
            'bbox_y1': round(_safe_mean(raw_w, 'bbox_y1'), 2),
            'bbox_x2': round(_safe_mean(raw_w, 'bbox_x2'), 2),
            'bbox_y2': round(_safe_mean(raw_w, 'bbox_y2'), 2),
        })

        if w_end >= end_t:
            break
        t += step

    return pd.DataFrame(records)


def summarize_movement_log(log_df: pd.DataFrame) -> dict:
    if log_df.empty:
        return {}

    movement_counts = log_df['primary_movement'].value_counts(normalize=True)
    label_counts: dict[str, int] = {}
    for labels in log_df['movement_labels']:
        for label in str(labels).split(','):
            label_counts[label] = label_counts.get(label, 0) + 1

    n = len(log_df)
    return {
        'dominant_movement': str(movement_counts.index[0]),
        'movement_fractions': {str(k): round(float(v), 3) for k, v in movement_counts.items()},
        'label_fractions': {k: round(v / n, 3) for k, v in sorted(label_counts.items())},
        'n_windows': int(n),
        'total_duration_s': round(float(log_df['end_time'].max() - log_df['start_time'].min()), 3),
        'active_fraction': round(float((log_df['primary_movement'] != 'still').mean()), 3),
        'avg_confidence': round(float(log_df['confidence'].mean()), 3),
        'avg_bbox_confidence': round(float(log_df['bbox_confidence'].mean()), 3),
        'avg_movement_energy': round(float(log_df['movement_energy'].mean()), 5),
        'peak_movement_energy': round(float(log_df['movement_energy'].max()), 5),
    }


def write_text_log(log_df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    path = Path(output_path)
    with open(path, 'w') as f:
        for row in log_df.itertuples(index=False):
            f.write(
                f"{row.start_time:>6.3f}-{row.end_time:>6.3f}s "
                f"person={row.person_id} "
                f"movement={row.primary_movement} "
                f"labels={row.movement_labels} "
                f"confidence={row.confidence:.3f} "
                f"energy={row.movement_energy:.5f} "
                f"bbox=({row.bbox_x1:.1f},{row.bbox_y1:.1f},{row.bbox_x2:.1f},{row.bbox_y2:.1f})\n"
            )


def write_summary_json(summary: dict, output_path: Union[str, Path]) -> None:
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
