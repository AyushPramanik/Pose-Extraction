"""
OpenPose-compatible pose extraction using YOLO-Pose (ultralytics).

YOLO-Pose outputs the same 17-keypoint COCO format as OpenPose-body and uses
the same Part Affinity Field concept modernised for transformer-based detection.
Model selection:
  yolov8n-pose  – fastest, least accurate
  yolov8s-pose  – small
  yolov8m-pose  – medium
  yolov8l-pose  – large
  yolov8x-pose  – extra-large, best for subtle movement detection (default)
  yolo11x-pose  – latest architecture, highest accuracy
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

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
    """YOLO-Pose wrapper with OpenPose-compatible output format."""

    def __init__(self, checkpoint: str = 'yolov8x-pose.pt', confidence: float = 0.15):
        self.confidence = confidence
        print(f"Loading OpenPose model ({checkpoint})...")
        self.model = YOLO(checkpoint)
        print("Model ready.")

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Return a list of person dicts mapping keypoint name → {x, y, conf}."""
        results = self.model(frame, verbose=False)
        persons: list[dict] = []

        for r in results:
            if r.keypoints is None:
                continue
            kps_tensor = r.keypoints.data  # (n_people, 17, 3)
            for person_kps in kps_tensor.cpu().numpy():
                kp_dict: dict = {}
                for i, name in enumerate(KEYPOINT_NAMES):
                    x, y, conf = float(person_kps[i][0]), float(person_kps[i][1]), float(person_kps[i][2])
                    kp_dict[name] = {'x': x, 'y': y, 'conf': conf} if conf >= self.confidence else None
                persons.append(kp_dict)

        return persons

    def extract_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 0,
    ) -> tuple[list[dict], float]:
        """
        Extract pose keypoints from every (or every Nth) frame of a video.

        Returns (frames_data, effective_fps).
        frames_data entries: {frame_idx, sample_idx, timestamp, persons}.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, skip_frames + 1)
        effective_fps = source_fps / stride

        frames: list[dict] = []
        frame_idx = 0
        sample_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                persons = self.process_frame(frame)
                frames.append({
                    'frame_idx': frame_idx,
                    'sample_idx': sample_idx,
                    'timestamp': frame_idx / source_fps,
                    'persons': persons,
                })
                sample_idx += 1
                if sample_idx % 50 == 0:
                    pct = 100 * frame_idx / total if total else 0
                    print(f"  {frame_idx}/{total} frames ({pct:.0f}%)")

            frame_idx += 1

        cap.release()
        print(f"  Done — {sample_idx} samples from {frame_idx} frames ({effective_fps:.2f} effective fps)")

        if output_path:
            with open(output_path, 'w') as f:
                json.dump({'source_fps': source_fps, 'effective_fps': effective_fps, 'frames': frames}, f)

        return frames, effective_fps

    @staticmethod
    def load_from_json(path: str) -> tuple[list[dict], float]:
        with open(path) as f:
            data = json.load(f)
        return data['frames'], data['effective_fps']
