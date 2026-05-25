"""
OpenPose-compatible pose extraction using YOLO-Pose (ultralytics)
with ByteTrack multi-person tracking for stable person IDs across frames.

Model selection:
  yolov8n-pose  – fastest, least accurate
  yolov8s-pose  – small
  yolov8m-pose  – medium
  yolov8l-pose  – large
  yolov8x-pose  – extra-large, best for subtle movement detection (default)
  yolo11x-pose  – latest architecture, highest accuracy
"""
import numpy as np
from ultralytics import YOLO
import supervision as sv

import cv2, json, warnings
from typing import Optional

from src.utils.logger import get_logger


_LOGGER = get_logger(name = "src.pose_extractor", level = "INFO")
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
POSE_CONNECTIONS = [
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
]


class PoseExtractor:
    """YOLO-Pose wrapper with ByteTrack multi-person tracking."""

    def __init__(
            self,
            checkpoint: str = 'yolov8x-pose.pt',
            confidence: float = 0.15,
            track_activation_threshold: float = 0.3,
            lost_track_buffer: int = 60,
            minimum_matching_threshold: float = 0.7,
        ):
        self.confidence = confidence
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        _LOGGER.info(f"Loading pose model ({checkpoint})...")
        self.model = YOLO(checkpoint)
        _LOGGER.info("Model ready.")

    def _build_tracker(self, fps: float) -> sv.ByteTrack:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return sv.ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                frame_rate=fps,
            )

    @staticmethod
    def _match_tracked_to_orig(orig_bboxes: np.ndarray, tracked_bboxes: np.ndarray) -> dict[int, int]:
        """
        Match each tracked bbox to the closest original detection bbox.
        Returns {orig_idx: position_in_tracked_array}.

        ByteTrack returns a filtered subset of the input bboxes; matching by L1
        distance finds which original detection each tracked entry came from.
        """
        mapping: dict[int, int] = {}
        used_orig: set[int] = set()
        for t_i, t_bbox in enumerate(tracked_bboxes):
            diffs = np.sum(np.abs(orig_bboxes - t_bbox), axis=1)
            orig_i = int(np.argmin(diffs))
            if diffs[orig_i] < 2.0 and orig_i not in used_orig:
                mapping[orig_i] = t_i
                used_orig.add(orig_i)
        return mapping

    def _detect_and_track(self, frame: np.ndarray, tracker: sv.ByteTrack) -> list[dict]:
        """Run YOLO on one frame, assign stable tracker IDs via ByteTrack."""
        results = self.model(frame, verbose=False)
        persons: list[dict] = []

        for r in results:
            if r.keypoints is None or len(r.keypoints.data) == 0:                            # No people detected
                continue

            kps_tensor = r.keypoints.data.cpu().numpy()  # .shape <-> (n, 17, 3)
            if len(kps_tensor) == 0:
                continue

            orig_bboxes = r.boxes.xyxy.cpu().numpy()     # .shape <-> (n, 4)

            # Run ByteTrack on YOLO bounding boxes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                detections = sv.Detections.from_ultralytics(r)                               # YOLO format → ByteTrack-compatible format
                tracked = tracker.update_with_detections(detections)                         # Compares the current detection with previous frame tracks (stored internally) to assign IDs

            # Map each original detection to its tracker_id
            tid_by_orig: dict[int, int] = {}
            if tracked.tracker_id is not None and len(tracked) > 0:
                orig_to_t = self._match_tracked_to_orig(orig_bboxes, tracked.xyxy)
                for orig_i, t_i in orig_to_t.items():
                    tid_by_orig[orig_i] = int(tracked.tracker_id[t_i])

            for idx, person_kps in enumerate(kps_tensor):
                person: dict = {'tracker_id': tid_by_orig.get(idx)}
                for kp_i, name in enumerate(KEYPOINT_NAMES):
                    x = float(person_kps[kp_i][0])
                    y = float(person_kps[kp_i][1])
                    conf = float(person_kps[kp_i][2])
                    person[name] = {'x': x, 'y': y, 'conf': conf} if conf >= self.confidence else None
                persons.append(person)

        return persons

    def extract_from_video(
            self,
            video_path: str,
            output_path: Optional[str] = None,
            skip_frames: int = 0,
        ) -> tuple[list[dict], float]:
        """
        Extract tracked pose keypoints from every (or every Nth) frame.

        Returns (frames_data, effective_fps).
        Each frame entry: {frame_idx, sample_idx, timestamp, persons}.
        Each person entry: {tracker_id, nose: {x,y,conf}, left_eye: ..., ...}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, skip_frames + 1)
        effective_fps = source_fps / stride

        tracker = self._build_tracker(effective_fps)

        frames: list[dict] = []
        frame_idx = 0
        sample_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                persons = self._detect_and_track(frame, tracker)
                frames.append({
                    'frame_idx': frame_idx,
                    'sample_idx': sample_idx,
                    'timestamp': frame_idx / source_fps,
                    'persons': persons,
                })
                sample_idx += 1
                if sample_idx % 50 == 0:
                    pct = 100 * frame_idx / total if total else 0
                    _LOGGER.info(f"  {frame_idx}/{total} frames ({pct:.0f}%)")

            frame_idx += 1

        cap.release()
        _LOGGER.info(f"Done - {sample_idx} samples from {frame_idx} frames ({effective_fps:.2f} effective fps)")

        if output_path:
            with open(output_path, 'w') as f:
                json.dump({'source_fps': source_fps, 'effective_fps': effective_fps, 'frames': frames}, f)

        return frames, effective_fps

    @staticmethod
    def load_from_json(path: str) -> tuple[list[dict], float]:
        with open(path) as f:
            data = json.load(f)
        return data['frames'], data['effective_fps']