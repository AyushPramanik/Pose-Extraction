import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.patches import Patch

from src.pose_extractor import KEYPOINT_NAMES, POSE_CONNECTIONS
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

_MOVEMENT_COLORS = {
    'still': '#c7c7c7',
    'gesturing': '#d62728',
    'nodding': '#1f77b4',
    'head_shaking': '#ff7f0e',
    'arm_bending': '#2ca02c',
    'body_shift': '#9467bd',
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


def plot_all_person_movements(
    movement_df: pd.DataFrame,
    output_path: str,
    title: str = "All-Person Movement Overview",
) -> None:
    """Render a combined movement graph for every tracked person."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2.3, 2.3, 1.5], hspace=0.42)

    if movement_df.empty:
        fig.text(0.5, 0.5, "No movement data available", ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        _LOGGER.info(f"All-person movement graph -> {output_path}")
        return

    df = movement_df.copy()
    df['person_label'] = df['person_id'].apply(lambda x: f"person {int(x)}" if pd.notna(x) else "person")
    person_ids = sorted(df['person_id'].dropna().unique(), key=lambda x: int(x))
    labels = [f"person {int(pid)}" for pid in person_ids]
    person_colors = dict(zip(person_ids, plt.cm.tab10(np.linspace(0, 1, max(len(person_ids), 1)))))

    # Panel 1: energy over time
    ax1 = fig.add_subplot(gs[0, 0])
    for pid in person_ids:
        part = df[df['person_id'] == pid].sort_values('mid_time')
        ax1.plot(
            part['mid_time'],
            part['movement_energy'],
            marker='o',
            markersize=2.8,
            linewidth=1.4,
            color=person_colors[pid],
            label=f"person {int(pid)}",
            alpha=0.9,
        )
    ax1.set_title('Movement Energy Timeline')
    ax1.set_ylabel('mean speed')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis='y', alpha=0.25)
    ax1.legend(ncol=min(4, max(1, len(person_ids))), fontsize=8, loc='upper right')

    # Panel 2: movement label timeline
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for y, pid in enumerate(person_ids):
        part = df[df['person_id'] == pid].sort_values('start_time')
        for row in part.itertuples(index=False):
            movement = str(row.primary_movement)
            color = _MOVEMENT_COLORS.get(movement, '#8c8c8c')
            ax2.broken_barh(
                [(float(row.start_time), max(0.02, float(row.end_time) - float(row.start_time)))],
                (y - 0.36, 0.72),
                facecolors=color,
                edgecolors='white',
                linewidth=0.35,
                alpha=0.88,
            )
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_title('Primary Movement Labels')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(-0.7, len(labels) - 0.3 if labels else 0.7)
    ax2.grid(True, axis='x', alpha=0.2)
    used_movements = [m for m in _MOVEMENT_COLORS if m in set(df['primary_movement'].astype(str))]
    ax2.legend(
        handles=[Patch(facecolor=_MOVEMENT_COLORS[m], label=m) for m in used_movements],
        ncol=min(4, max(1, len(used_movements))),
        fontsize=8,
        loc='upper right',
    )

    # Panel 3: aggregate activity by person
    ax3 = fig.add_subplot(gs[2, 0])
    summary = (
        df.assign(active=df['primary_movement'] != 'still')
          .groupby('person_id', as_index=False)
          .agg(avg_energy=('movement_energy', 'mean'), active_fraction=('active', 'mean'))
          .sort_values('person_id')
    )
    x = np.arange(len(summary))
    bar_colors = [person_colors[pid] for pid in summary['person_id']]
    ax3.bar(x, summary['avg_energy'], color=bar_colors, alpha=0.82)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"person {int(pid)}" for pid in summary['person_id']])
    ax3.set_ylabel('avg movement energy')
    ax3.set_title('Aggregate Activity')
    ax3.grid(True, axis='y', alpha=0.25)

    ax3b = ax3.twinx()
    ax3b.plot(x, summary['active_fraction'], color='black', marker='D', linewidth=1.4, label='active fraction')
    ax3b.set_ylabel('active fraction')
    ax3b.set_ylim(0, 1)
    ax3b.legend(fontsize=8, loc='upper right')

    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _LOGGER.info(f"All-person movement graph -> {output_path}")


# ==================== movement analysis plot ====================

def plot_movement_features(
    feat_df: pd.DataFrame,
    summary: dict,
    output_path: str,
    title: str = "Movement Analysis — Music Listening Session",
) -> None:
    t = feat_df['time']

    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ==================== panel 1: head speed ====================
    ax1 = fig.add_subplot(gs[0, 0])
    for kp, col in [('nose', 'tab:blue'), ('left_ear', 'tab:orange'), ('right_ear', 'tab:green')]:
        sc = f'{kp}_speed'
        if sc in feat_df.columns:
            ax1.plot(t, feat_df[sc].fillna(0), label=kp, color=col, alpha=0.85, linewidth=0.9)
    ax1.set_title('Head Movement Speed')
    ax1.set_ylabel('shoulder-widths/s')
    ax1.set_xlabel('Time (s)')
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    # ==================== panel 2: wrist / elbow speed ====================
    ax2 = fig.add_subplot(gs[0, 1])
    palette = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for (kp, col) in zip(['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow'], palette):
        sc = f'{kp}_speed'
        if sc in feat_df.columns:
            ax2.plot(t, feat_df[sc].fillna(0), label=kp, color=col, alpha=0.85, linewidth=0.9)
    ax2.set_title('Arm Movement Speed')
    ax2.set_ylabel('shoulder-widths/s')
    ax2.set_xlabel('Time (s)')
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    # ==================== panel 3: vertical range-of-motion ====================
    ax3 = fig.add_subplot(gs[1, 0])
    for kp, col in [('nose', 'tab:blue'), ('left_shoulder', 'tab:cyan'), ('right_shoulder', 'tab:olive')]:
        yc = f'{kp}_rom_y'
        if yc in feat_df.columns:
            ax3.plot(t, feat_df[yc].fillna(0), label=kp, color=col, alpha=0.85, linewidth=0.9)
    ax3.set_title('Vertical Range of Motion (1 s window)')
    ax3.set_ylabel('shoulder-widths')
    ax3.set_xlabel('Time (s)')
    ax3.legend(fontsize=8)

    # ==================== panel 4: elbow angles ====================
    ax4 = fig.add_subplot(gs[1, 1])
    for col, label, color in [('left_elbow_angle', 'Left elbow', 'tab:blue'),
                               ('right_elbow_angle', 'Right elbow', 'tab:red')]:
        if col in feat_df.columns:
            ax4.plot(t, feat_df[col], label=label, color=color, alpha=0.85, linewidth=0.9)
    ax4.set_title('Elbow Angle')
    ax4.set_ylabel('degrees')
    ax4.set_xlabel('Time (s)')
    ax4.legend(fontsize=8)

    # ==================== panel 5: overall movement energy ====================
    ax5 = fig.add_subplot(gs[2, 0])
    speed_cols = [c for c in feat_df.columns if c.endswith('_speed')]
    if speed_cols:
        energy = feat_df[speed_cols].fillna(0).mean(axis=1)
        ax5.fill_between(t, energy, alpha=0.55, color='steelblue')
        ax5.plot(t, energy, color='steelblue', linewidth=0.8)
    ax5.set_title('Overall Body Movement Energy')
    ax5.set_ylabel('mean speed (shoulder-widths/s)')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylim(bottom=0)

    # ==================== panel 6: per-keypoint mean speed bar chart ====================
    ax6 = fig.add_subplot(gs[2, 1])
    kps = ['nose', 'left_ear', 'right_ear',
           'left_shoulder', 'right_shoulder',
           'left_elbow', 'right_elbow',
           'left_wrist', 'right_wrist']
    means  = [summary.get(f'{kp}_mean_speed', 0.0) for kp in kps]
    labels = [kp.replace('left_', 'L.').replace('right_', 'R.') for kp in kps]
    ax6.bar(labels, means, color='steelblue', alpha=0.8)
    ax6.set_title('Mean Speed per Keypoint')
    ax6.set_ylabel('shoulder-widths/s')
    ax6.tick_params(axis='x', rotation=45)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _LOGGER.info(f"Movement plot -> {output_path}")
