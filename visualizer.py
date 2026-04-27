import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from pose_extractor import KEYPOINT_NAMES, POSE_CONNECTIONS


# ── video annotation ──────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, persons: list[dict], confidence: float = 0.15) -> np.ndarray:
    out = frame.copy()
    for person in persons:
        for a_name, b_name in POSE_CONNECTIONS:
            ka, kb = person.get(a_name), person.get(b_name)
            if ka and kb and ka['conf'] >= confidence and kb['conf'] >= confidence:
                cv2.line(out, (int(ka['x']), int(ka['y'])), (int(kb['x']), int(kb['y'])),
                         (0, 220, 80), 2, cv2.LINE_AA)
        for name, kp in person.items():
            if kp and kp['conf'] >= confidence:
                is_extremity = name in ('left_wrist', 'right_wrist', 'nose')
                color = (0, 80, 255) if is_extremity else (255, 140, 0)
                cv2.circle(out, (int(kp['x']), int(kp['y'])), 4 if is_extremity else 3,
                           color, -1, cv2.LINE_AA)
    return out


def render_annotated_video(video_path: str, frames_data: list[dict], output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    print(f"Annotated video → {output_path}")


# ── movement analysis plot ────────────────────────────────────────────────────

def plot_movement_features(
    feat_df: pd.DataFrame,
    summary: dict,
    output_path: str,
    title: str = "Movement Analysis — Music Listening Session",
) -> None:
    t = feat_df['time']

    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ── panel 1: head speed ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for kp, col in [('nose', 'tab:blue'), ('left_ear', 'tab:orange'), ('right_ear', 'tab:green')]:
        sc = f'{kp}_speed'
        if sc in feat_df.columns:
            ax1.plot(t, feat_df[sc].fillna(0), label=kp, color=col, alpha=0.85, linewidth=0.9)
    ax1.set_title('Head Movement Speed')
    ax1.set_ylabel('px/s')
    ax1.set_xlabel('Time (s)')
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    # ── panel 2: wrist / elbow speed ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    palette = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for (kp, col) in zip(['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow'], palette):
        sc = f'{kp}_speed'
        if sc in feat_df.columns:
            ax2.plot(t, feat_df[sc].fillna(0), label=kp, color=col, alpha=0.85, linewidth=0.9)
    ax2.set_title('Arm Movement Speed')
    ax2.set_ylabel('px/s')
    ax2.set_xlabel('Time (s)')
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    # ── panel 3: vertical range-of-motion ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for kp, col in [('nose', 'tab:blue'), ('left_shoulder', 'tab:cyan'), ('right_shoulder', 'tab:olive')]:
        yc = f'{kp}_rom_y'
        if yc in feat_df.columns:
            ax3.plot(t, feat_df[yc].fillna(0), label=kp, color=col, alpha=0.85, linewidth=0.9)
    ax3.set_title('Vertical Range of Motion (1 s window)')
    ax3.set_ylabel('px')
    ax3.set_xlabel('Time (s)')
    ax3.legend(fontsize=8)

    # ── panel 4: elbow angles ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for col, label, color in [('left_elbow_angle', 'Left elbow', 'tab:blue'),
                               ('right_elbow_angle', 'Right elbow', 'tab:red')]:
        if col in feat_df.columns:
            ax4.plot(t, feat_df[col], label=label, color=color, alpha=0.85, linewidth=0.9)
    ax4.set_title('Elbow Angle')
    ax4.set_ylabel('degrees')
    ax4.set_xlabel('Time (s)')
    ax4.legend(fontsize=8)

    # ── panel 5: overall movement energy ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    speed_cols = [c for c in feat_df.columns if c.endswith('_speed')]
    if speed_cols:
        energy = feat_df[speed_cols].fillna(0).mean(axis=1)
        ax5.fill_between(t, energy, alpha=0.55, color='steelblue')
        ax5.plot(t, energy, color='steelblue', linewidth=0.8)
    ax5.set_title('Overall Body Movement Energy')
    ax5.set_ylabel('mean speed (px/s)')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylim(bottom=0)

    # ── panel 6: per-keypoint mean speed bar chart ─────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    kps = ['nose', 'left_ear', 'right_ear',
           'left_shoulder', 'right_shoulder',
           'left_elbow', 'right_elbow',
           'left_wrist', 'right_wrist']
    means  = [summary.get(f'{kp}_mean_speed', 0.0) for kp in kps]
    labels = [kp.replace('left_', 'L.').replace('right_', 'R.') for kp in kps]
    ax6.bar(labels, means, color='steelblue', alpha=0.8)
    ax6.set_title('Mean Speed per Keypoint')
    ax6.set_ylabel('px/s')
    ax6.tick_params(axis='x', rotation=45)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Movement plot → {output_path}")
