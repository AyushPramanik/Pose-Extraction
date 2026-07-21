"""
OpenPose-compatible pose extraction using YOLO-Pose (ultralytics)
with ByteTrack multi-person tracking for stable person IDs across frames.

Model selection:
  yolov8n-pose  – fastest, least accurate
  yolov8s-pose  – small
  yolov8m-pose  – medium
  yolov8l-pose  – large
  yolov8x-pose  – extra-large, strong accuracy
  yolo11x-pose  – latest architecture, highest accuracy (default)
"""
import cv2, json, warnings
import numpy as np
from ultralytics import YOLO
import supervision as sv
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

# How many frames to feed YOLO per inference call. Batching cuts per-frame
# overhead (especially on GPU) without changing tracking order.
_INFER_BATCH = 16


class PoseExtractor:
    """YOLO-Pose wrapper with ByteTrack multi-person tracking."""

    def __init__(
            self,
            checkpoint: str = 'yolo11x-pose.pt',
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
    def _iou(a: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """IoU of one xyxy box `a` against an array of xyxy `boxes` -> (n,)."""
        ix1 = np.maximum(a[0], boxes[:, 0])
        iy1 = np.maximum(a[1], boxes[:, 1])
        ix2 = np.minimum(a[2], boxes[:, 2])
        iy2 = np.minimum(a[3], boxes[:, 3])
        iw = np.clip(ix2 - ix1, 0, None)
        ih = np.clip(iy2 - iy1, 0, None)
        inter = iw * ih
        area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
        union = area_a + area_b - inter
        return np.where(union > 0, inter / union, 0.0)

    @staticmethod
    def _match_tracked_to_orig(orig_bboxes: np.ndarray, tracked_bboxes: np.ndarray) -> dict[int, int]:
        """
        Match each tracked bbox to its original detection bbox by IoU.
        Returns {orig_idx: position_in_tracked_array}.

        ByteTrack returns a filtered subset of the input detections; IoU is the
        tracker's own association metric and is robust to sub-pixel box changes
        (the previous exact-L1 match dropped detections to tracker_id=None on any
        tiny perturbation).  One-to-one, greedy on highest IoU >= 0.5.
        """
        mapping: dict[int, int] = {}
        used_orig: set[int] = set()
        for t_i, t_bbox in enumerate(tracked_bboxes):
            ious = PoseExtractor._iou(t_bbox, orig_bboxes)
            order = np.argsort(ious)[::-1]
            for orig_i in order:
                orig_i = int(orig_i)
                if ious[orig_i] < 0.5:
                    break
                if orig_i not in used_orig:
                    mapping[orig_i] = t_i
                    used_orig.add(orig_i)
                    break
        return mapping

    def _result_to_persons(self, r, tracker: sv.ByteTrack) -> list[dict]:
        """Turn one YOLO result into tracked person dicts, updating the tracker."""
        if r.keypoints is None or len(r.keypoints.data) == 0:                            # No people detected
            return []

        kps_tensor = r.keypoints.data.cpu().numpy()  # .shape <-> (n, 17, 3)
        if len(kps_tensor) == 0:
            return []

        orig_bboxes = r.boxes.xyxy.cpu().numpy()     # .shape <-> (n, 4)
        orig_conf = r.boxes.conf.cpu().numpy()

        # Run ByteTrack on YOLO bounding boxes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            detections = sv.Detections.from_ultralytics(r)                               # YOLO format → ByteTrack-compatible format
            tracked = tracker.update_with_detections(detections)                         # Compares the current detection with previous-frame tracks to assign IDs

        # Map each original detection to its tracker_id
        tid_by_orig: dict[int, int] = {}
        if tracked.tracker_id is not None and len(tracked) > 0:
            orig_to_t = self._match_tracked_to_orig(orig_bboxes, tracked.xyxy)
            for orig_i, t_i in orig_to_t.items():
                tid_by_orig[orig_i] = int(tracked.tracker_id[t_i])

        persons: list[dict] = []
        for idx, person_kps in enumerate(kps_tensor):
            x1, y1, x2, y2 = [float(v) for v in orig_bboxes[idx]]
            person: dict = {
                'tracker_id': tid_by_orig.get(idx),
                'bbox': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'conf': float(orig_conf[idx]),
                },
            }
            for kp_i, name in enumerate(KEYPOINT_NAMES):
                x = float(person_kps[kp_i][0])
                y = float(person_kps[kp_i][1])
                conf = float(person_kps[kp_i][2])
                person[name] = {'x': x, 'y': y, 'conf': conf} if conf >= self.confidence else None
            persons.append(person)

        return persons

    @staticmethod
    def load_from_json(path: str) -> tuple[list[dict], float]:
        with open(path) as f:
            data = json.load(f)
        try:
            frames, effective_fps = data['frames'], data['effective_fps']
        except KeyError as e:
            raise ValueError(f"Invalid JSON format: missing key {e}")
        return frames, effective_fps

    def extract_from_video(
            self,
            video_path: str,
            output_path: Optional[str] = None,
            skip_frames: int = 0,
        ) -> tuple[list[dict], float]:
        """
        Extract tracked pose keypoints from every (or every Nth) frame.

        Returns (frames_data, effective_fps).
        Each frame entry : {frame_idx, sample_idx, timestamp, persons}.
        Each person entry: {tracker_id, bbox, nose: {x,y,conf}, left_eye: ..., ...}

        YOLO inference is batched for speed; ByteTrack is then fed the batch's
        results in strict frame order so tracking stays correct.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 31.0
        total: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, skip_frames + 1)
        effective_fps = source_fps / stride

        tracker = self._build_tracker(effective_fps)

        frames: list[dict] = []
        frame_idx = 0
        sample_idx = 0
        batch_frames: list[np.ndarray] = []
        batch_indices: list[int] = []

        def flush_batch() -> None:
            nonlocal sample_idx
            if not batch_frames:
                return
            results = self.model(batch_frames, verbose=False, conf=max(0.01, self.confidence))
            for fidx, r in zip(batch_indices, results):
                persons = self._result_to_persons(r, tracker)
                frames.append({
                    'frame_idx': fidx,
                    'sample_idx': sample_idx,
                    'timestamp': fidx / source_fps,
                    'persons': persons,
                })
                sample_idx += 1
                if sample_idx % 50 == 0:
                    pct = 100 * fidx / total if total else 0
                    _LOGGER.info(f"  {fidx}/{total} frames ({pct:.0f}%)")
            batch_frames.clear()
            batch_indices.clear()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                if len(batch_frames) >= _INFER_BATCH:
                    flush_batch()

            frame_idx += 1

        flush_batch()
        cap.release()
        _LOGGER.info(f"Done - {sample_idx} samples from {frame_idx} frames ({effective_fps:.2f} effective fps)")

        if output_path:
            with open(output_path, 'w') as f:
                json.dump({'source_fps': source_fps, 'effective_fps': effective_fps, 'frames': frames}, f)

        return frames, effective_fps
