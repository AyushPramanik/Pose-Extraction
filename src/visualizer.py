import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.patches import Patch

from src.utils.pose_extractor import POSE_CONNECTIONS
from src.utils.logger import get_logger


_LOGGER = get_logger(name = "src.visualizer", level = "INFO")


# BGR color palette — one per tracker_id (mod 6)
_TRACKER_COLORS = [
    (80, 220,   0),   # green
    (255,  80,  80),  # blue
    (80,  80, 255),   # red
    (255, 200,   0),  # cyan
    (0,  140, 255),   # orange
    (200,   0, 255),  # magenta
]

_COLOR_GREY = (160, 160, 160)

_STATE_COLORS = {
    'moving': '#2ca02c',
    'still': '#c7c7c7',
}


def _person_color(tracker_id) -> tuple[int, int, int]:
    if tracker_id is None:
        return _COLOR_GREY
    return _TRACKER_COLORS[int(tracker_id) % len(_TRACKER_COLORS)]


# ==================== video annotation ====================

def annotate_frame(frame: np.ndarray, persons: list[dict], confidence: float = 0.15) -> np.ndarray:
    out = frame.copy()
    for person in persons:
        tracker_id = person.get('tracker_id')
        color = _person_color(tracker_id)
        bbox = person.get('bbox')
        if tracker_id is None and bbox and bbox.get('conf', 0.0) < 0.3:
            continue

        if bbox:
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"person {tracker_id}" if tracker_id is not None else "person"
            label += f" {bbox.get('conf', 0):.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_y = max(0, y1 - th - baseline - 4)
            cv2.rectangle(out, (x1, label_y), (x1 + tw + 8, label_y + th + baseline + 6), color, -1)
            cv2.putText(out, label, (x1 + 4, label_y + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Skeleton connections
        for a_name, b_name in POSE_CONNECTIONS:
            ka, kb = person.get(a_name), person.get(b_name)
            if ka and kb and ka['conf'] >= confidence and kb['conf'] >= confidence:
                cv2.line(out, (int(ka['x']), int(ka['y'])), (int(kb['x']), int(kb['y'])),
                         color, 2, cv2.LINE_AA)

        # Keypoint dots
        for name, kp in person.items():
            if name in ('tracker_id', 'bbox') or not isinstance(kp, dict):
                continue
            if kp and kp['conf'] >= confidence:
                is_extremity = name in ('left_wrist', 'right_wrist', 'nose')
                radius = 4 if is_extremity else 3
                cv2.circle(out, (int(kp['x']), int(kp['y'])), radius, color, -1, cv2.LINE_AA)

        # Tracker ID label above the nose (or first available upper keypoint)
        if tracker_id is not None:
            label_pt = None
            for kp_name in ('nose', 'left_eye', 'right_eye', 'left_shoulder', 'right_shoulder'):
                kp = person.get(kp_name)
                if kp and kp['conf'] >= confidence:
                    label_pt = (int(kp['x']), int(kp['y']) - 12)
                    break
            if label_pt:
                cv2.putText(out, f"#{tracker_id}", label_pt,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return out


def render_annotated_video(video_path: str, frames_data: list[dict], output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_map = {f['frame_idx']: f['persons'] for f in frames_data}
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated = annotate_frame(frame, frame_map.get(idx, []))
        writer.write(annotated)
        idx += 1

    cap.release()
    writer.release()
    _LOGGER.info(f"Annotated video -> {output_path}")


def render_person_clip(
    video_path: str,
    frames_data: list[dict],
    output_path: str,
    person_id: int,
    start_frame: int,
    end_frame: int,
) -> None:
    """
    Write a clip covering [start_frame, end_frame] of the source video, with the
    box + skeleton overlay drawn for one person only (the full frame is kept).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Persons for the target id only, keyed by frame index.
    frame_map = {
        f['frame_idx']: [p for p in f['persons'] if p.get('tracker_id') == person_id]
        for f in frames_data
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while cap.isOpened() and idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = annotate_frame(frame, frame_map.get(idx, []))
        writer.write(annotated)
        idx += 1

    cap.release()
    writer.release()
    _LOGGER.info(f"Clip -> {output_path}")


def _state_bands(ax, part: pd.DataFrame, y: float, height: float) -> None:
    """Draw move/still broken_barh bands for one person's sorted windows."""
    for row in part.itertuples(index=False):
        color = _STATE_COLORS.get(str(row.state), '#8c8c8c')
        ax.broken_barh(
            [(float(row.start_time), max(0.02, float(row.end_time) - float(row.start_time)))],
            (y - height / 2, height),
            facecolors=color,
            edgecolors='white',
            linewidth=0.35,
            alpha=0.9,
        )


def plot_person_motion(
    motion_df: pd.DataFrame,
    output_path: str,
    title: str = "Motion Analysis",
) -> None:
    """
    Per-person motion plot:
      1. motion_energy over time with the adaptive threshold and moving windows.
      2. move/still band across time.
      3. per-body-part energy (only when region columns are present).
    """
    region_cols = [c for c in motion_df.columns
                   if c.startswith('energy_') and c not in
                   ('energy_a', 'energy_joint', 'energy_limb', 'energy_flow')]
    has_regions = len(region_cols) > 0

    if has_regions:
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 2], hspace=0.4)
    else:
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.35)

    if motion_df.empty:
        fig.text(0.5, 0.5, "No motion data available", ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        _LOGGER.info(f"Motion plot -> {output_path}")
        return

    df = motion_df.sort_values('mid_time')

    # Panel 1: motion energy + threshold
    ax1 = fig.add_subplot(gs[0, 0])
    # Faint per-signal contributions (fused mode only), beneath the bold total.
    for col, lab, color in [('z_a', 'keypoint', 'tab:blue'),
                            ('z_joint', 'joint', 'tab:green'),
                            ('z_limb', 'limb', 'tab:orange'),
                            ('z_flow', 'flow', 'tab:purple')]:
        if col in df.columns and df[col].abs().sum() > 0:
            ax1.plot(df['mid_time'], df[col], color=color, linewidth=0.8, alpha=0.45, label=lab)
    ax1.fill_between(df['mid_time'], df['motion_energy'], alpha=0.45, color='steelblue')
    ax1.plot(df['mid_time'], df['motion_energy'], color='steelblue', linewidth=1.3, label='fused')
    if 'active_threshold' in df.columns:
        thr = float(df['active_threshold'].iloc[0])
        ax1.axhline(thr, color='crimson', linestyle='--', linewidth=1.2, label=f'threshold={thr:.3f}')
    moving = df[df['state'] == 'moving'] if 'state' in df.columns else df.iloc[0:0]
    ax1.scatter(moving['mid_time'], moving['motion_energy'], color=_STATE_COLORS['moving'],
                s=18, zorder=3, label='moving')
    fused = 'z_joint' in df.columns
    ax1.set_title('Motion Energy — fused keypoint + joint + limb' if fused
                  else 'Motion Energy (crop-normalized keypoint variance)')
    ax1.set_ylabel('motion energy')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis='y', alpha=0.25)
    ax1.legend(fontsize=8, loc='upper right')

    # Panel 2: move/still band
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    _state_bands(ax2, df.sort_values('start_time'), y=0.0, height=0.7)
    ax2.set_yticks([])
    ax2.set_title('Move / Still')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(-0.5, 0.5)
    ax2.legend(handles=[Patch(facecolor=_STATE_COLORS[s], label=s) for s in _STATE_COLORS],
               ncol=2, fontsize=8, loc='upper right')

    # Panel 3: per-body-part energy (stacked area), when region columns exist.
    if has_regions:
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        labels = [c[len('energy_'):] for c in region_cols]
        ax3.stackplot(df['mid_time'], *[df[c] for c in region_cols],
                      labels=labels, alpha=0.85)
        ax3.set_title('Motion Energy by Body Region')
        ax3.set_ylabel('energy')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylim(bottom=0)
        ax3.grid(True, axis='y', alpha=0.25)
        ax3.legend(fontsize=8, loc='upper right', ncol=min(len(labels), 4))

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _LOGGER.info(f"Motion plot -> {output_path}")


def plot_all_person_motion(
    motion_df: pd.DataFrame,
    output_path: str,
    title: str = "All-Person Motion Overview",
) -> None:
    """Combined move/no-move overview for every tracked person."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2.3, 2.3, 1.5], hspace=0.42)

    if motion_df.empty:
        fig.text(0.5, 0.5, "No motion data available", ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        _LOGGER.info(f"All-person motion graph -> {output_path}")
        return

    df = motion_df.copy()
    person_ids = sorted(df['person_id'].dropna().unique(), key=lambda x: int(x))
    labels = [f"person {int(pid)}" for pid in person_ids]
    person_colors = dict(zip(person_ids, plt.cm.tab10(np.linspace(0, 1, max(len(person_ids), 1)))))

    # Panel 1: motion energy over time
    ax1 = fig.add_subplot(gs[0, 0])
    for pid in person_ids:
        part = df[df['person_id'] == pid].sort_values('mid_time')
        ax1.plot(part['mid_time'], part['motion_energy'], marker='o', markersize=2.8,
                 linewidth=1.4, color=person_colors[pid], label=f"person {int(pid)}", alpha=0.9)
    ax1.set_title('Motion Energy Timeline')
    ax1.set_ylabel('motion energy')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis='y', alpha=0.25)
    ax1.legend(ncol=min(4, max(1, len(person_ids))), fontsize=8, loc='upper right')

    # Panel 2: move/still bands, one row per person
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for y, pid in enumerate(person_ids):
        part = df[df['person_id'] == pid].sort_values('start_time')
        _state_bands(ax2, part, y=y, height=0.72)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_title('Move / Still')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(-0.7, len(labels) - 0.3 if labels else 0.7)
    ax2.grid(True, axis='x', alpha=0.2)
    ax2.legend(handles=[Patch(facecolor=_STATE_COLORS[s], label=s) for s in _STATE_COLORS],
               ncol=2, fontsize=8, loc='upper right')

    # Panel 3: aggregate activity by person
    ax3 = fig.add_subplot(gs[2, 0])
    summary = (
        df.assign(moving=df['state'] == 'moving')
          .groupby('person_id', as_index=False)
          .agg(avg_energy=('motion_energy', 'mean'), moving_fraction=('moving', 'mean'))
          .sort_values('person_id')
    )
    x = np.arange(len(summary))
    bar_colors = [person_colors[pid] for pid in summary['person_id']]
    ax3.bar(x, summary['avg_energy'], color=bar_colors, alpha=0.82)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"person {int(pid)}" for pid in summary['person_id']])
    ax3.set_ylabel('avg motion energy')
    ax3.set_title('Aggregate Activity')
    ax3.grid(True, axis='y', alpha=0.25)

    ax3b = ax3.twinx()
    ax3b.plot(x, summary['moving_fraction'], color='black', marker='D', linewidth=1.4, label='moving fraction')
    ax3b.set_ylabel('moving fraction')
    ax3b.set_ylim(0, 1)
    ax3b.legend(fontsize=8, loc='upper right')

    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _LOGGER.info(f"All-person motion graph -> {output_path}")
