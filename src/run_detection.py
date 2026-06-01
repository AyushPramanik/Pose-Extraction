"""
Subtle movement extraction for seated subjects.

Usage:
    uv run python -m src.run_detection assets/recording.mov -o output --annotate-video

Outputs per tracked person (all written to --output-dir):
    <stem>_person<id>_keypoints.csv        raw pixel coordinates
    <stem>_person<id>_keypoints_norm.csv   torso-normalised coordinates
    <stem>_person<id>_features.csv         frame-level kinematics
    <stem>_person<id>_summary.json         aggregate statistics
    <stem>_person<id>_movement_log.csv     windowed movement labels
    <stem>_person<id>_movement_log.txt     readable movement log
    <stem>_person<id>_movement_summary.json aggregate movement labels
    <stem>_person<id>_movement_plot.png    6-panel visualisation

Shared outputs (one per video):
    <stem>_poses.json          raw keypoints for all persons, all frames
    <stem>_all_person_movement_log.csv combined movement log for all persons
    <stem>_all_person_movements.png combined movement graph for all persons
    <stem>_annotated.mp4       video with color-coded boxes/skeletons (--annotate-video)
"""
import argparse, json
from pathlib import Path

import pandas as pd

from src.pose_extractor import PoseExtractor
from src.feature_extractor import (
    get_tracker_ids,
    poses_to_dataframe,
    normalize_to_torso,
    compute_features,
    compute_summary,
)
from src.movement_recognizer import (
    generate_movement_log,
    summarize_movement_log,
    write_text_log,
    write_summary_json,
)
from src.visualizer import (
    render_annotated_video,
    plot_movement_features,
    plot_all_person_movements,
)
from src.utils.logger import get_logger


_LOGGER = get_logger(name = "src.run_detection", level = "INFO")


class PoseExtractionPipeline():
    def __init__(self) -> None:
        self.parse_args()
        self._extractor = None

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description='Extract subtle movement features from video using YOLO-Pose + box tracking.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument('video', help='Path to input video file')                                           # 1st unnamed argument
        parser.add_argument('--output-dir', '-o', default='output',
                            help='Directory for all output files')
        parser.add_argument('--checkpoint', '-c', default='yolov8x-pose.pt',
                            help='YOLO-Pose model. Higher accuracy: yolo11x-pose.pt. Faster: yolov8n-pose.pt.')
        parser.add_argument('--confidence', type=float, default=0.15,
                            help='Keypoint detection threshold.')
        parser.add_argument('--skip-frames', type=int, default=0,
                            help='Process every (N+1)th frame. 0 = every frame.')
        parser.add_argument('--annotate-video', action='store_true',
                            help='Write a copy of the video with color-coded pose skeletons.')
        parser.add_argument('--load-poses', metavar='JSON',
                            help='Skip detection and load a previously saved _poses.json.')
        parser.add_argument('--person-ids', metavar='IDS',
                            help='Comma-separated tracker IDs to analyse (e.g. "1,3"). '
                                'Default: all detected persons.')
        parser.add_argument('--person-idx', type=int, default=0,
                            help='(Legacy) Array index when loading old pose JSON without tracker IDs.')
        parser.add_argument('--movement-window-seconds', type=float, default=1.0,
                            help='Window size for per-person movement recognition logs.')
        parser.add_argument('--movement-step-seconds', type=float, default=0.5,
                            help='Step size between movement-recognition windows.')
        # Tracker tuning
        parser.add_argument('--track-activation-threshold', type=float, default=0.3,
                            help='Min detection confidence to activate a new track.')
        parser.add_argument('--lost-track-buffer', type=int, default=60,
                            help='Frames to keep a lost track alive (occlusion tolerance).')
        parser.add_argument('--minimum-matching-threshold', type=float, default=0.7,
                            help='IoU threshold for matching detections to existing tracks.')
        self.args = parser.parse_args()

    @staticmethod
    def _parse_person_ids(raw: str) -> list[int]:
        return [int(x.strip()) for x in raw.split(',') if x.strip()]
    
    def create_output_dir(self) -> Path:
        self.output_path = Path(self.args.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.video_stem = Path(self.args.video).stem

    def run(self) -> None:
        self.create_output_dir()

        # ==================== 1. pose extraction ====================
        if self.args.load_poses:
            _LOGGER.info(f"[1/4] Loading saved poses from '{self.args.load_poses}'")
            frames, fps = PoseExtractor.load_from_json(self.args.load_poses)
            _LOGGER.info(f"      {len(frames)} samples at {fps:.2f} fps")
        else:
            _LOGGER.info(f"[1/4] Extracting poses from '{self.args.video}'")
            poses_path = self.output_path / f'{self.video_stem}_poses.json'
            self._extractor = PoseExtractor(
                checkpoint=self.args.checkpoint,
                confidence=self.args.confidence,
                track_activation_threshold=self.args.track_activation_threshold,
                lost_track_buffer=self.args.lost_track_buffer,
                minimum_matching_threshold=self.args.minimum_matching_threshold,
            )
            frames, fps = self._extractor.extract_from_video(
                video_path=self.args.video,
                output_path=str(poses_path),
                skip_frames=self.args.skip_frames,
            )
            _LOGGER.info(f"      {len(frames)} samples at {fps:.2f} fps -> {poses_path.name}")

        # ==================== determine persons to analyse ====================
        tracker_ids = get_tracker_ids(frames)

        if not tracker_ids:
            _LOGGER.warning("\n  No tracker IDs found.")
            tracker_ids = None

        if tracker_ids is not None and self.args.person_ids:
            requested = self._parse_person_ids(self.args.person_ids)
            tracker_ids = [tid for tid in tracker_ids if tid in requested]
            if not tracker_ids:
                _LOGGER.error(f"  None of the requested IDs {requested} were detected.")
                return

        # ==================== 2–4. per-person analysis ====================
        ids_to_run = tracker_ids if tracker_ids is not None else [None]
        n = len(ids_to_run)
        _LOGGER.info(f"\nStart analysing {n} person(s): {ids_to_run if tracker_ids is not None else ['(index fallback)']}")

        all_summaries = {}
        all_movement_summaries = {}
        all_movement_logs = []
        for person_id in ids_to_run:
            label = f"person{person_id}" if person_id is not None else f"person_idx{self.args.person_idx}"
            prefix = f'{self.video_stem}_{label}'
            _LOGGER.info(f"\n-- {prefix} --")

            # [2/4] DataFrames
            _LOGGER.info("Building keypoint DataFrames...")
            raw_df = poses_to_dataframe(
                frames=frames,
                person_id=person_id,
                person_idx=self.args.person_idx,
            )
            norm_df = normalize_to_torso(raw_df)
            raw_df.to_csv(self.output_path / f'{prefix}_keypoints.csv', index=False)
            norm_df.to_csv(self.output_path / f'{prefix}_keypoints_norm.csv', index=False)
            _LOGGER.info(f"        {len(raw_df)} rows written")

            # [3/4] Features
            _LOGGER.info("Computing movement features...")
            feat_df = compute_features(norm_df, fps)
            summary = compute_summary(feat_df, norm_df, fps)
            feat_df.to_csv(self.output_path / f'{prefix}_features.csv', index=False)
            with open(self.output_path / f'{prefix}_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            _LOGGER.info(f"        {feat_df.shape[1] - 3} feature columns computed")

            # Movement labels
            _LOGGER.info("Recognizing windowed movement labels...")
            movement_log = generate_movement_log(
                raw_df=raw_df,
                norm_df=norm_df,
                feat_df=feat_df,
                fps=fps,
                person_id=person_id,
                window_seconds=self.args.movement_window_seconds,
                step_seconds=self.args.movement_step_seconds,
            )
            movement_log.to_csv(self.output_path / f'{prefix}_movement_log.csv', index=False)
            write_text_log(movement_log, self.output_path / f'{prefix}_movement_log.txt')
            movement_summary = summarize_movement_log(movement_log)
            write_summary_json(movement_summary, self.output_path / f'{prefix}_movement_summary.json')
            _LOGGER.info(f"        {len(movement_log)} movement windows written")
            all_movement_logs.append(movement_log)

            # [4/4] Plot
            _LOGGER.info("Generating movement plot...")
            plot_movement_features(
                feat_df, 
                summary,
                str(self.output_path / f'{prefix}_movement_plot.png'),
                title=f"Movement Analysis — {label}",
            )

            all_summaries[label] = summary
            all_movement_summaries[label] = movement_summary

        if all_movement_logs:
            _LOGGER.info("Generating all-person movement graph...")
            combined_movement = pd.concat(all_movement_logs, ignore_index=True)
            combined_movement.to_csv(
                self.output_path / f'{self.video_stem}_all_person_movement_log.csv',
                index=False,
            )
            plot_all_person_movements(
                combined_movement,
                str(self.output_path / f'{self.video_stem}_all_person_movements.png'),
                title=f"All-Person Movement Overview — {self.video_stem}",
            )

        # ==================== annotated video (shared across all persons) ====================
        if self.args.annotate_video:
            _LOGGER.info("Rendering annotated video...")
            render_annotated_video(self.args.video, frames, str(self.output_path / f'{self.video_stem}_annotated.mp4'))

        # ==================== summary ====================
        _LOGGER.info("\n=== Movement Summary ===")
        keys_to_show = [
            'head_lateral_freq_hz', 'head_vertical_freq_hz',
            'head_lateral_std',     'head_vertical_std',
            'shoulder_sway_freq_hz', 'shoulder_sway_std',
            'total_movement_energy',
        ]
        for label, summary in all_summaries.items():
            _LOGGER.info(f"\n  {label}:")
            for k in keys_to_show:
                if k in summary:
                    _LOGGER.info(f"    {k:<36} {summary[k]:.5f}")
            movement_summary = all_movement_summaries.get(label, {})
            if movement_summary:
                _LOGGER.info(f"    {'dominant_movement':<36} {movement_summary.get('dominant_movement')}")
                _LOGGER.info(f"    {'active_fraction':<36} {movement_summary.get('active_fraction'):.3f}")

        _LOGGER.info(f"\nAll outputs in: {self.output_path}/")


if __name__ == '__main__':
    PoseExtractionPipeline().run()
