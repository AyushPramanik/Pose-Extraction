import argparse, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.utils.pose_extractor import PoseExtractor
from src.motion_detector import (
    MOVE_ENERGY_THRESHOLD,
    get_tracker_ids,
    compute_motion_energy_windows,
    decide_move_no_move,
    corroborate_signal_b,
    summarize_motion,
    write_text_log,
    write_summary_json,
)
from src.heatmap_volume import to_pyskl_annotation, save_pyskl_pickle
from src.keypoint_smoothing import SmoothConfig
from src.optical_flow import compute_person_flow_series
from src.track_diagnostics import track_report, find_duplicate_tracks
from src.visualizer import (
    render_annotated_video,
    plot_person_motion,
    plot_all_person_motion,
)
from src.video_splitter import split_video
from src.utils.logger import get_logger


_LOGGER = get_logger(name = "src.run_detection", level = "INFO")


def analyze_person(args: dict) -> dict:
    """
    Run the full per-person analysis (DataFrames, features, movement log, plot)
    for one tracker ID.  Module-level + picklable so it can run in a worker
    process.  Returns the data the parent needs for the combined outputs.
    """
    import matplotlib
    matplotlib.use('Agg')  # headless backend, required inside worker processes

    frames        = args['frames']
    fps           = args['fps']
    person_id     = args['person_id']
    stats_path    = Path(args['stats_path'])
    stem          = args['stem']
    window_secs   = args['window_seconds']
    step_secs     = args['step_seconds']
    move_threshold = args['move_threshold']
    signal_b      = args['signal_b']
    num_frames    = args['num_frames']
    use_heatmap   = args['use_heatmap']
    heatmap_shape = args['heatmap_out_shape']
    with_limb     = args['with_limb']
    fuse_weights  = args['fuse_weights']
    smooth        = args['smooth']
    smooth_pyskl  = args['smooth_pyskl']
    region_energy = args['region_energy']
    flow          = args['flow']
    flow_weight   = args['flow_weight']
    flow_preset   = args['flow_preset']
    video_path    = args['video_path']

    label = f"person{person_id}"
    prefix = f'{stem}_{label}'
    _LOGGER.info(f"-- {prefix} --")

    # Optional orthogonal optical-flow series (second decode pass, present frames only).
    flow_series = None
    if flow:
        flow_series = compute_person_flow_series(video_path, frames, person_id, preset=flow_preset)

    # [2/4] Windowed motion energy + move/no-move decision
    motion_log = compute_motion_energy_windows(
        frames=frames,
        person_id=person_id,
        fps=fps,
        window_seconds=window_secs,
        step_seconds=step_secs,
        use_heatmap=use_heatmap,
        heatmap_out_shape=heatmap_shape,
        with_limb=with_limb,
        fuse_weights=fuse_weights,
        smooth=smooth,
        flow_series=flow_series,
        flow_weight=flow_weight,
        region_energy=region_energy,
    )
    motion_log = decide_move_no_move(motion_log, threshold=move_threshold)

    # [3/4] Optional Signal B corroboration (heavier: renders the heatmap volume)
    if signal_b:
        motion_log = corroborate_signal_b(frames, person_id, motion_log, clip_len=num_frames)

    motion_log.to_csv(stats_path / f'{prefix}_motion_log.csv', index=False)
    write_text_log(motion_log, stats_path / f'{prefix}_motion_log.txt')
    motion_summary = summarize_motion(motion_log)
    write_summary_json(motion_summary, stats_path / f'{prefix}_motion_summary.json')

    # CNN-ready PYSKL annotation (raw pixel keypoints by default; smoothed only if asked).
    annotation = to_pyskl_annotation(
        frames, person_id, frame_dir=prefix, smooth=smooth if smooth_pyskl else None,
    )

    # [4/4] Plot
    plot_person_motion(
        motion_log,
        str(stats_path / f'{prefix}_motion_plot.png'),
        title=f"Motion Analysis — {label}",
    )

    return {
        'label': label,
        'motion_summary': motion_summary,
        'motion_log': motion_log,
        'annotation': annotation,
    }


class PoseExtractionPipeline():
    def __init__(self) -> None:
        self.parse_args()
        self._extractor = None

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description='Extract subtle movement features from video using YOLO-Pose + ByteTrack.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Input/output
        parser.add_argument('input_video', help='input video path')
        parser.add_argument('--output-dir', '-o', default='output',
                            help='output directory (statistics/ and videos/ are created inside)')
        parser.add_argument('--annotate-video', action='store_true',
                            help='Write a full annotated copy of the video with all pose skeletons.')
        parser.add_argument('--split-clips', action='store_true',
                            help='Write per-person, per-movement-bout clips to videos/.')
        parser.add_argument('--load-poses', metavar='JSON',
                            help='Skip detection and load a previously saved _poses.json.')
        parser.add_argument('--workers', type=int, default=0,
                            help='Worker processes for per-person analysis. '
                                 '0 = auto (min(cpu, n_persons)); 1 = no multiprocessing.')

        # Pose extraction parameters
        parser.add_argument('--checkpoint', '-c', default='yolo11x-pose.pt',
                            help='YOLO-Pose model (17 COCO keypoints). yolo11x-pose = best accuracy.')
        parser.add_argument('--confidence', type=float, default=0.15,
                            help='Keypoint detection threshold.')
        parser.add_argument('--skip-frames', type=int, default=10,
                            help='Process every (N+1)th frame. Primary speed/detail lever.')
        parser.add_argument('--person-ids', metavar='IDS',
                            help='Comma-separated tracker IDs to analyse (e.g. "1,3"). '
                                'Default: all detected persons.')
        parser.add_argument('--movement-window-seconds', type=float, default=1.0,
                            help='Window size for the per-person move/no-move motion log.')
        parser.add_argument('--movement-step-seconds', type=float, default=0.5,
                            help='Step size between motion-detection windows.')

        # Motion detection (PoseC3D representation)
        parser.add_argument('--motion-mode', choices=['fused', 'keypoint'], default='fused',
                            help='fused = combine keypoint-variance + joint + limb heatmap motion '
                                 '(uses the paper representation); keypoint = Signal A only.')
        parser.add_argument('--heatmap-size', type=int, choices=[56, 28], default=56,
                            help='Heatmap grid for the fused motion signal. 28 is ~4x faster.')
        parser.add_argument('--no-limb', action='store_true',
                            help='Drop the limb-heatmap term from the fused motion signal.')
        parser.add_argument('--fuse-weights', metavar='A,J,L', default='1,1,1',
                            help='Weights for keypoint,joint,limb energies in fused mode.')
        parser.add_argument('--signal-b', action='store_true',
                            help='Also compute the per-person heatmap-diff corroboration flag '
                                 '(Signal B diagnostics). Slower.')
        parser.add_argument('--num-frames', type=int, default=48,
                            help='Frames uniformly sampled per person for Signal B heatmap rendering.')

        # Keypoint smoothing (jitter floor)
        parser.add_argument('--smooth-filter', choices=['none', 'oneeuro', 'savgol', 'ema'],
                            default='oneeuro',
                            help='Temporal keypoint smoothing to suppress detector jitter.')
        parser.add_argument('--smooth-min-cutoff', type=float, default=1.0,
                            help='OneEuro baseline cutoff frequency (Hz). Lower = smoother.')
        parser.add_argument('--smooth-beta', type=float, default=0.3,
                            help='OneEuro speed coefficient. Higher = less lag on fast motion.')
        parser.add_argument('--smooth-pyskl', action='store_true',
                            help='Also smooth the exported PYSKL keypoints (default: raw, for CNN).')

        # Per-body-part output
        parser.add_argument('--no-region-energy', action='store_true',
                            help='Do not emit per-body-part energy columns.')

        # Optical-flow signal (orthogonal motion; needs a second video decode pass)
        parser.add_argument('--flow', action='store_true',
                            help='Add a dense optical-flow motion signal inside each person bbox.')
        parser.add_argument('--flow-weight', type=float, default=1.0,
                            help='Fusion weight for the optical-flow signal (only with --flow).')
        parser.add_argument('--flow-preset', choices=['fast', 'medium'], default='fast',
                            help='DIS optical-flow preset.')

        # Track diagnostics
        parser.add_argument('--no-track-report', action='store_true',
                            help='Skip writing the per-track quality report CSV.')

        # Duplicate-track suppression (post-tracking; excludes phantom tracks from analysis)
        parser.add_argument('--no-dedup', action='store_true',
                            help='Do not exclude duplicate tracks (one person detected twice).')
        parser.add_argument('--dedup-dist', type=float, default=0.06,
                            help='Two tracks are duplicates when their upper-body keypoints coincide '
                                 'within this fraction of body height on shared frames (true dup ~0.03). '
                                 'Lower = stricter (less likely to merge distinct people).')

        # Tracker tuning
        parser.add_argument('--track-activation-threshold', type=float, default=0.3,
                            help='Min detection confidence to activate a new track.')
        parser.add_argument('--lost-track-buffer', type=int, default=60,
                            help='Frames to keep a lost track alive (occlusion tolerance).')
        parser.add_argument('--minimum-matching-threshold', type=float, default=0.7,
                            help='IoU threshold for matching detections to existing tracks.')

        # Move/no-move decision (absolute)
        parser.add_argument('--move-threshold', type=float, default=MOVE_ENERGY_THRESHOLD,
                            help='Absolute move/still floor on crop-normalized keypoint variance '
                                 '(energy_a). Same for every person: moving = energy_a > threshold. '
                                 'Lower = more sensitive. Tune per camera/resolution.')

        # Clip splitting tuning
        parser.add_argument('--min-track-seconds', type=float, default=3.0,
                            help='Ignore tracks shorter than this (filters pass-through people).')
        parser.add_argument('--clip-bridge-seconds', type=float, default=1.5,
                            help='Merge active bouts separated by a still-gap shorter than this.')
        parser.add_argument('--clip-min-seconds', type=float, default=2.0,
                            help='Drop bouts shorter than this.')
        self.args = parser.parse_args()

    @staticmethod
    def _parse_person_ids(raw: str) -> list[int]:
        return [int(x.strip()) for x in raw.split(',') if x.strip()]

    @staticmethod
    def _parse_fuse_weights(raw: str) -> tuple[float, float, float]:
        parts = [float(x.strip()) for x in raw.split(',') if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"--fuse-weights expects 'A,J,L' (three numbers), got '{raw}'")
        return (parts[0], parts[1], parts[2])

    def create_output_dir(self) -> None:
        self.output_path = Path(self.args.output_dir)
        self.stats_path = self.output_path / 'statistics'
        self.videos_path = self.output_path / 'videos'
        for p in (self.output_path, self.stats_path, self.videos_path):
            p.mkdir(parents=True, exist_ok=True)
        self.stem = Path(self.args.input_video).stem

    def extract_poses(self) -> tuple[list[dict], float]:
        if self.args.load_poses:
            _LOGGER.info(f"[1/4] Loading saved poses from '{self.args.load_poses}'")
            frames, fps = PoseExtractor.load_from_json(self.args.load_poses)
            _LOGGER.info(f"      {len(frames)} samples at {fps:.2f} fps")
        else:
            _LOGGER.info(f"[1/4] Extracting poses from '{self.args.input_video}'")
            poses_path = self.videos_path / f'{self.stem}_poses.json'
            self._extractor = PoseExtractor(
                checkpoint=self.args.checkpoint,
                confidence=self.args.confidence,
                track_activation_threshold=self.args.track_activation_threshold,
                lost_track_buffer=self.args.lost_track_buffer,
                minimum_matching_threshold=self.args.minimum_matching_threshold,
            )
            frames, fps = self._extractor.extract_from_video(
                video_path=self.args.input_video,
                output_path=str(poses_path),
                skip_frames=self.args.skip_frames,
            )
            _LOGGER.info(f"      {len(frames)} samples at {fps:.2f} fps -> {poses_path.name}")
        return frames, fps

    def run(self) -> None:
        self.create_output_dir()

        # ==================== 1. pose extraction ====================
        frames, fps = self.extract_poses()

        # ==================== determine persons to analyse ====================
        tracker_ids = get_tracker_ids(frames)
        if not tracker_ids:
            _LOGGER.warning("\n  No tracker IDs found.")
            return

        if self.args.person_ids:
            requested = self._parse_person_ids(self.args.person_ids)
            tracker_ids = [tid for tid in tracker_ids if tid in requested]
            if not tracker_ids:
                _LOGGER.error(f"  None of the requested IDs {requested} were detected.")
                return

        # ==================== track-quality diagnostic ====================
        if not self.args.no_track_report:
            report = track_report(frames, source_fps=fps)
            report_path = self.stats_path / f'{self.stem}_track_report.csv'
            report.to_csv(report_path, index=False)
            _LOGGER.info(f"Track report -> {report_path.name} "
                         f"({len(report)} tracks, "
                         f"{int(report['id_switch_suspects'].sum()) if not report.empty else 0} switch suspects)")

        # ==================== duplicate-track suppression (post-tracking) ====================
        # One person split into two overlapping detections becomes two tracks; drop
        # the duplicate from analysis. Runs AFTER tracking so ByteTrack identity is
        # untouched; conservative so genuinely distinct people are never merged.
        if not self.args.no_dedup:
            dup_of = find_duplicate_tracks(frames, confidence=self.args.confidence,
                                           dist_thresh=self.args.dedup_dist)
            dups = [d for d in dup_of if d in tracker_ids]
            if dups:
                for d in dups:
                    _LOGGER.info(f"  duplicate track: person{d} -> same as person{dup_of[d]}; "
                                 f"excluded from analysis.")
                tracker_ids = [tid for tid in tracker_ids if tid not in dup_of]

        # ==================== 2–4. per-person analysis ====================
        n = len(tracker_ids)
        _LOGGER.info(f"\nStart analysing {n} person(s): {tracker_ids}")

        smooth_cfg = SmoothConfig(
            filter=self.args.smooth_filter,
            min_cutoff=self.args.smooth_min_cutoff,
            beta=self.args.smooth_beta,
        )

        jobs = [{
            'frames': frames,
            'fps': fps,
            'person_id': pid,
            'stats_path': str(self.stats_path),
            'stem': self.stem,
            'window_seconds': self.args.movement_window_seconds,
            'step_seconds': self.args.movement_step_seconds,
            'move_threshold': self.args.move_threshold,
            'signal_b': self.args.signal_b,
            'num_frames': self.args.num_frames,
            'use_heatmap': self.args.motion_mode == 'fused',
            'heatmap_out_shape': (self.args.heatmap_size, self.args.heatmap_size),
            'with_limb': not self.args.no_limb,
            'fuse_weights': self._parse_fuse_weights(self.args.fuse_weights),
            'smooth': smooth_cfg,
            'smooth_pyskl': self.args.smooth_pyskl,
            'region_energy': not self.args.no_region_energy,
            'flow': self.args.flow,
            'flow_weight': self.args.flow_weight,
            'flow_preset': self.args.flow_preset,
            'video_path': self.args.input_video,
        } for pid in tracker_ids]

        workers = self.args.workers or min(os.cpu_count() or 1, n)
        if workers > 1 and n > 1:
            _LOGGER.info(f"Running per-person analysis on {workers} workers...")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(analyze_person, jobs))
        else:
            results = [analyze_person(job) for job in jobs]

        all_motion_summaries = {r['label']: r['motion_summary'] for r in results}
        all_motion_logs = [r['motion_log'] for r in results if not r['motion_log'].empty]

        # CNN-ready PYSKL annotations for later PoseC3D training.
        annotations = [r['annotation'] for r in results if r['annotation']['total_frames'] > 0]
        if annotations:
            pkl_path = self.videos_path / f'{self.stem}_pyskl.pkl'
            save_pyskl_pickle(annotations, str(pkl_path))
            _LOGGER.info(f"PYSKL annotations -> {pkl_path.name}")

        combined_motion = pd.DataFrame()
        if all_motion_logs:
            _LOGGER.info("Generating all-person motion graph...")
            combined_motion = pd.concat(all_motion_logs, ignore_index=True)
            combined_motion.to_csv(
                self.stats_path / f'{self.stem}_all_person_motion_log.csv',
                index=False,
            )
            plot_all_person_motion(
                combined_motion,
                str(self.stats_path / f'{self.stem}_all_person_motions.png'),
                title=f"All-Person Motion Overview — {self.stem}",
            )

        # ==================== per-person bout clips ====================
        if self.args.split_clips and not combined_motion.empty:
            _LOGGER.info("Splitting per-person movement clips...")
            split_video(
                video_path=self.args.input_video,
                frames=frames,
                movement_log=combined_motion,
                output_dir=self.videos_path,
                stem=self.stem,
                fps=fps,
                min_track_seconds=self.args.min_track_seconds,
                move_threshold=self.args.move_threshold,
                bridge_seconds=self.args.clip_bridge_seconds,
                min_clip_seconds=self.args.clip_min_seconds,
            )

        # ==================== full annotated video ====================
        if self.args.annotate_video:
            _LOGGER.info("Rendering annotated video...")
            render_annotated_video(
                self.args.input_video, frames,
                str(self.videos_path / f'{self.stem}_annotated.mp4'),
            )

        # ==================== summary ====================
        _LOGGER.info("\n=== Motion Summary ===")
        keys_to_show = ['moving_fraction', 'mean_motion_energy', 'peak_motion_energy', 'active_threshold']
        for label, summary in all_motion_summaries.items():
            if not summary:
                continue
            _LOGGER.info(f"\n  {label}:")
            for k in keys_to_show:
                if k in summary:
                    _LOGGER.info(f"    {k:<24} {summary[k]:.6f}")

        _LOGGER.info(f"\nStatistics in: {self.stats_path}/")
        _LOGGER.info(f"Videos in:     {self.videos_path}/")


if __name__ == '__main__':
    import time
    start_time = time.time()
    PoseExtractionPipeline().run()
    elapsed = time.time() - start_time
    _LOGGER.info(f"\nTotal elapsed time: {elapsed:.2f} seconds")