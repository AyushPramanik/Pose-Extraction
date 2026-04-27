"""
Subtle movement extraction for seated subjects listening to music.

Usage:
    python main.py <video> [options]

Outputs (all written to --output-dir):
    <stem>_poses.json          raw keypoints per frame
    <stem>_keypoints.csv       flat CSV of raw pixel coordinates
    <stem>_keypoints_norm.csv  torso-normalised coordinates (use these for subtle-movement analysis)
    <stem>_features.csv        frame-level kinematics (speed, acceleration, ROM, joint angles)
    <stem>_summary.json        aggregate statistics and dominant movement frequencies
    <stem>_movement_plot.png   6-panel visualisation
    <stem>_annotated.mp4       (optional, --annotate-video)
"""
import argparse
import json
from pathlib import Path

from pose_extractor import PoseExtractor
from feature_extractor import (
    poses_to_dataframe,
    normalize_to_torso,
    compute_features,
    compute_summary,
)
from visualizer import render_annotated_video, plot_movement_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract subtle movement features from video using OpenPose (openpifpaf).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Directory for all output files')
    parser.add_argument('--checkpoint', '-c', default='yolov8x-pose.pt',
                        help='YOLO-Pose model. '
                             'Higher accuracy: yolo11x-pose.pt. '
                             'Faster: yolov8n-pose.pt.')
    parser.add_argument('--confidence', type=float, default=0.15,
                        help='Keypoint detection threshold. '
                             'Lower catches subtler poses but adds noise.')
    parser.add_argument('--person', type=int, default=0,
                        help='Person index when multiple people are detected.')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Process every (N+1)th frame. 0 = every frame.')
    parser.add_argument('--annotate-video', action='store_true',
                        help='Write a copy of the video with pose skeleton overlaid.')
    parser.add_argument('--load-poses', metavar='JSON',
                        help='Skip extraction and load a previously saved _poses.json.')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem

    # ── 1. pose extraction ────────────────────────────────────────────────────
    if args.load_poses:
        print(f"\n[1/4] Loading saved poses from '{args.load_poses}'...")
        frames, fps = PoseExtractor.load_from_json(args.load_poses)
        print(f"      {len(frames)} samples at {fps:.2f} fps")
    else:
        print(f"\n[1/4] Extracting poses from '{args.video}'...")
        poses_path = out / f'{stem}_poses.json'
        extractor = PoseExtractor(checkpoint=args.checkpoint, confidence=args.confidence)
        frames, fps = extractor.extract_from_video(
            args.video,
            output_path=str(poses_path),
            skip_frames=args.skip_frames,
        )
        print(f"      {len(frames)} samples at {fps:.2f} fps → {poses_path.name}")

    # ── 2. keypoint DataFrames ────────────────────────────────────────────────
    print("\n[2/4] Building keypoint DataFrames...")
    raw_df  = poses_to_dataframe(frames, person_idx=args.person)
    norm_df = normalize_to_torso(raw_df)

    raw_df.to_csv(out / f'{stem}_keypoints.csv', index=False)
    norm_df.to_csv(out / f'{stem}_keypoints_norm.csv', index=False)
    print(f"      {len(raw_df)} rows written (raw + torso-normalised)")

    # ── 3. movement features ──────────────────────────────────────────────────
    print("\n[3/4] Computing movement features...")
    # Use normalised coordinates so features reflect relative movement only
    feat_df = compute_features(norm_df, fps)
    summary = compute_summary(feat_df, norm_df, fps)

    feat_df.to_csv(out / f'{stem}_features.csv', index=False)
    with open(out / f'{stem}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"      {feat_df.shape[1] - 3} feature columns computed")

    # ── 4. visualisation ──────────────────────────────────────────────────────
    print("\n[4/4] Generating outputs...")
    plot_movement_features(feat_df, summary, str(out / f'{stem}_movement_plot.png'))

    if args.annotate_video:
        render_annotated_video(args.video, frames, str(out / f'{stem}_annotated.mp4'))

    # ── summary print ─────────────────────────────────────────────────────────
    print("\n=== Movement Summary ===")
    keys_to_show = [
        'head_lateral_freq_hz', 'head_vertical_freq_hz',
        'head_lateral_std',     'head_vertical_std',
        'shoulder_sway_freq_hz','shoulder_sway_std',
        'total_movement_energy',
    ]
    for k in keys_to_show:
        if k in summary:
            print(f"  {k:<36} {summary[k]:.5f}")

    print(f"\nAll outputs in: {out}/")


if __name__ == '__main__':
    main()
