import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd

from pose_extractor import KEYPOINT_NAMES, POSE_CONNECTIONS

# Consistent colours for each emotion label
EMOTION_COLORS = {
    'excited':    '#f4a532',
    'engaged':    '#4caf8e',
    'calm':       '#6baed6',
    'disengaged': '#b0b0b0',
    'agitated':   '#e05252',
}


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


# ── emotion timeline plot ─────────────────────────────────────────────────────

def plot_emotion_timeline(
    results_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    summary: dict,
    output_path: str,
    title: str = "Behavioral State Analysis",
) -> None:
    """
    Three-panel figure:
      1. Emotion timeline (colour blocks per window)
      2. Movement energy with emotion colouring
      3. Action presence heatmap
    """
    if results_df.empty:
        print("No behavioral results to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1.2, 2, 2]})
    fig.subplots_adjust(hspace=0.45)

    t_min = float(results_df['start_time'].min())
    t_max = float(results_df['end_time'].max())

    # ── panel 1: emotion timeline ─────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_xlim(t_min, t_max)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Emotion Timeline', fontweight='bold')

    for _, row in results_df.iterrows():
        color = EMOTION_COLORS.get(row['emotion'], '#cccccc')
        ax1.axvspan(row['start_time'], row['end_time'], ymin=0, ymax=1,
                    alpha=0.75, color=color)
        mid = (row['start_time'] + row['end_time']) / 2
        ax1.text(mid, 0.5, row['emotion'], ha='center', va='center',
                 fontsize=7, rotation=90 if (row['end_time'] - row['start_time']) < 1.5 else 0,
                 clip_on=True)

    # legend
    patches = [mpatches.Patch(color=c, label=e)
               for e, c in EMOTION_COLORS.items()
               if e in summary.get('emotion_fractions', {})]
    ax1.legend(handles=patches, loc='upper right', fontsize=7,
               bbox_to_anchor=(1.0, 1.35), ncol=len(patches))

    # ── panel 2: movement energy coloured by emotion ──────────────────────────
    ax2 = axes[1]
    speed_cols = [c for c in feat_df.columns if c.endswith('_speed')]
    if speed_cols:
        t = feat_df['time']
        energy = feat_df[speed_cols].mean(axis=1).fillna(0)
        ax2.fill_between(t, energy, alpha=0.25, color='steelblue')
        ax2.plot(t, energy, color='steelblue', linewidth=0.8, alpha=0.8)

        # shade background by emotion windows
        for _, row in results_df.iterrows():
            color = EMOTION_COLORS.get(row['emotion'], '#cccccc')
            ax2.axvspan(row['start_time'], row['end_time'], alpha=0.12, color=color)

    ax2.set_xlim(t_min, t_max)
    ax2.set_ylim(bottom=0)
    ax2.set_title('Movement Energy (background = inferred emotion)', fontweight='bold')
    ax2.set_ylabel('mean speed')
    ax2.set_xlabel('Time (s)')

    # ── panel 3: action heatmap ───────────────────────────────────────────────
    ax3 = axes[2]
    action_labels = ['still', 'nodding', 'head_shaking', 'gesturing', 'arm_raised',
                     'active', 'moving']
    present = [a for a in action_labels
               if a in summary.get('action_fractions', {})]
    if present:
        n_actions = len(present)
        mat = np.zeros((n_actions, len(results_df)))
        for j, row in enumerate(results_df.itertuples()):
            acts = row.actions.split(',')
            for i, a in enumerate(present):
                mat[i, j] = 1.0 if a in acts else 0.0

        mid_times = results_df['mid_time'].values
        ax3.pcolormesh(mid_times, np.arange(n_actions), mat,
                       cmap='YlOrRd', vmin=0, vmax=1, shading='nearest')
        ax3.set_yticks(np.arange(n_actions) + 0.5)
        ax3.set_yticklabels(present, fontsize=9)
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Detected Actions', fontweight='bold')
        ax3.set_xlim(t_min, t_max)
    else:
        ax3.set_visible(False)

    # ── annotation box ────────────────────────────────────────────────────────
    dom = summary.get('dominant_emotion', 'unknown')
    dur = summary.get('total_duration_s', 0)
    fracs = summary.get('emotion_fractions', {})
    frac_str = '  '.join(f"{e}: {v:.0%}" for e, v in sorted(fracs.items(), key=lambda x: -x[1]))
    info = f"Dominant: {dom}   |   Duration: {dur:.1f}s   |   {frac_str}"
    fig.text(0.5, 0.01, info, ha='center', fontsize=9, style='italic', color='#444444')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Emotion timeline → {output_path}")
