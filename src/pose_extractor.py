"""
OpenPose-compatible pose extraction using YOLO-Pose (ultralytics)
with lightweight box tracking for stable person IDs across frames.

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

import cv2, json
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
    """YOLO-Pose wrapper with multi-person box tracking."""

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

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
        area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _bbox_center(box: np.ndarray) -> np.ndarray:
        return np.array([(float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2])

    @staticmethod
    def _bbox_diag(box: np.ndarray) -> float:
        return float(np.linalg.norm([float(box[2]) - float(box[0]), float(box[3]) - float(box[1])]))

    def _build_tracker(self, fps: float) -> "SimpleBoxTracker":
        return SimpleBoxTracker(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
        )

    def _detect_and_track(self, frame: np.ndarray, tracker: "SimpleBoxTracker") -> list[dict]:
        """Run YOLO on one frame, assign stable tracker IDs using box matching."""
        results = self.model(frame, verbose=False, conf=max(0.01, self.confidence))
        detections: list[dict] = []

        for r in results:
            if r.keypoints is None or len(r.keypoints.data) == 0:                            # No people detected
                continue

            kps_tensor = r.keypoints.data.cpu().numpy()  # .shape <-> (n, 17, 3)
            if len(kps_tensor) == 0:
                continue

            orig_bboxes = r.boxes.xyxy.cpu().numpy()     # .shape <-> (n, 4)
            orig_conf = r.boxes.conf.cpu().numpy()
            for idx, person_kps in enumerate(kps_tensor):
                detections.append({
                    'bbox_xyxy': orig_bboxes[idx].astype(float),
                    'detection_confidence': float(orig_conf[idx]),
                    'keypoints': person_kps,
                })

        tracker_ids = tracker.update(detections)
        persons: list[dict] = []
        for det, tracker_id in zip(detections, tracker_ids):
            x1, y1, x2, y2 = [float(v) for v in det['bbox_xyxy']]
            person: dict = {
                'tracker_id': tracker_id,
                'bbox': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'conf': float(det['detection_confidence']),
                },
            }
            for kp_i, name in enumerate(KEYPOINT_NAMES):
                x = float(det['keypoints'][kp_i][0])
                y = float(det['keypoints'][kp_i][1])
                conf = float(det['keypoints'][kp_i][2])
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


class SimpleBoxTracker:
    """
    Lightweight person tracker for mostly static multi-person videos.

    It matches YOLO person boxes frame-to-frame using IoU first, then centre
    distance as a fallback for jittery boxes.  This avoids an extra runtime
    dependency while still producing stable IDs for seated subjects.
    """

    def __init__(
            self,
            track_activation_threshold: float = 0.3,
            lost_track_buffer: int = 60,
            minimum_matching_threshold: float = 0.7,
        ):
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        return PoseExtractor._bbox_iou(a, b)

    @staticmethod
    def _bbox_center(box: np.ndarray) -> np.ndarray:
        return PoseExtractor._bbox_center(box)

    @staticmethod
    def _bbox_diag(box: np.ndarray) -> float:
        return PoseExtractor._bbox_diag(box)

    def _match_score(self, track_box: np.ndarray, det_box: np.ndarray) -> tuple[float, float, float]:
        iou = self._bbox_iou(track_box, det_box)
        centre_dist = float(np.linalg.norm(self._bbox_center(track_box) - self._bbox_center(det_box)))
        diag = max(1.0, (self._bbox_diag(track_box) + self._bbox_diag(det_box)) / 2)
        centre_score = max(0.0, 1.0 - centre_dist / diag)
        score = iou + 0.25 * centre_score
        return score, iou, centre_dist / diag

    def update(self, detections: list[dict]) -> list[Optional[int]]:
        assignments: list[Optional[int]] = [None] * len(detections)
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            if track['lost'] > self.lost_track_buffer:
                continue
            for det_idx, det in enumerate(detections):
                score, iou, norm_dist = self._match_score(track['bbox'], det['bbox_xyxy'])
                min_iou = self.minimum_matching_threshold
                if iou >= min_iou or norm_dist <= 0.35:
                    candidates.append((score, track_id, det_idx))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, det_idx in sorted(candidates, reverse=True):
            if track_id in used_tracks or det_idx in used_detections:
                continue
            assignments[det_idx] = track_id
            used_tracks.add(track_id)
            used_detections.add(det_idx)

        for det_idx, det in enumerate(detections):
            if assignments[det_idx] is None and det['detection_confidence'] >= self.track_activation_threshold:
                assignments[det_idx] = self.next_id
                self.next_id += 1

        matched_ids = {track_id for track_id in assignments if track_id is not None}
        for track_id in list(self.tracks):
            if track_id in matched_ids:
                continue
            self.tracks[track_id]['lost'] += 1
            if self.tracks[track_id]['lost'] > self.lost_track_buffer:
                del self.tracks[track_id]

        for det_idx, track_id in enumerate(assignments):
            if track_id is None:
                continue
            self.tracks[track_id] = {
                'bbox': detections[det_idx]['bbox_xyxy'],
                'lost': 0,
            }

        return assignments
