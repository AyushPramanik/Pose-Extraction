"""
Dense optical-flow motion signal on extremity keypoint patches.

Whole-bbox flow just echoes the gross body motion the skeleton already captures
(measured corr ~0.94 with keypoint variance).  To be a genuinely INDEPENDENT
cue, flow is instead computed on small fixed-size patches centered on the
extremity keypoints (wrists, head) — the regions where subtle micro-motion
lives and where the coarse 17-joint skeleton is least expressive.

Anchoring each patch on the (per-frame) keypoint location cancels gross
translation, so the flow measures *residual* local motion (finger/hand fidget,
head tremor) rather than the whole body moving — the part keypoint variance
under-represents.

Needs the original video pixels, which the analysis stage does not otherwise
load, so this runs a targeted second decode pass over the frames where the
person is present.
"""
import cv2
import numpy as np

from src.utils.logger import get_logger

_LOGGER = get_logger(name="src.optical_flow", level="INFO")

# Keypoints whose local motion the skeleton captures poorly — flow focuses here.
_FLOW_KEYPOINTS = ('left_wrist', 'right_wrist', 'nose')
_PATCH = 48        # pixel side of each keypoint patch (in the original frame)
_PATCH_OUT = 64    # patch resized to this square before flow (scale-invariance)


def _make_flow(preset: str):
    p = (cv2.DISOPTICAL_FLOW_PRESET_FAST if preset == "fast"
         else cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return cv2.DISOpticalFlow_create(p)


def _patch(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Fixed-size grayscale patch centered on (cx, cy), edge-clamped + resized."""
    h, w = frame.shape[:2]
    half = _PATCH // 2
    x1 = int(np.clip(cx - half, 0, max(0, w - _PATCH)))
    y1 = int(np.clip(cy - half, 0, max(0, h - _PATCH)))
    x2, y2 = min(w, x1 + _PATCH), min(h, y1 + _PATCH)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((_PATCH_OUT, _PATCH_OUT), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (_PATCH_OUT, _PATCH_OUT))


def _keypoint_patches(frame: np.ndarray, person: dict) -> np.ndarray:
    """
    Horizontal strip of extremity patches for one person, or None if none of the
    focus keypoints are present.  Patches are anchored on the keypoint locations,
    so gross translation between frames is cancelled.
    """
    strips = []
    for name in _FLOW_KEYPOINTS:
        kp = person.get(name)
        if kp:
            strips.append(_patch(frame, kp['x'], kp['y']))
        else:
            strips.append(np.zeros((_PATCH_OUT, _PATCH_OUT), dtype=np.uint8))
    return np.hstack(strips)


def compute_person_flow_series(
        video_path: str,
        frames: list[dict],
        person_id: int,
        preset: str = "fast",
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-present-frame optical-flow magnitude on the person's extremity patches.

    Returns (times, magnitudes): mean residual flow magnitude between each
    consecutive pair of present frames, timestamped at the later frame.  Empty
    arrays when the person appears in < 2 frames or the video can't open.
    """
    present = []
    for f in frames:
        p = next((q for q in f['persons'] if q.get('tracker_id') == person_id), None)
        if p is not None:
            present.append((f['frame_idx'], f['timestamp'], p))
    if len(present) < 2:
        return np.array([]), np.array([])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _LOGGER.warning(f"optical flow: cannot open {video_path}; skipping person {person_id}")
        return np.array([]), np.array([])

    flow = _make_flow(preset)
    times: list[float] = []
    mags: list[float] = []
    prev_strip = None
    prev_present = False

    for frame_idx, ts, person in present:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            prev_strip, prev_present = None, False
            continue
        strip = _keypoint_patches(frame, person)
        if prev_strip is not None and prev_present and prev_strip.shape == strip.shape:
            f_uv = flow.calc(prev_strip, strip, None)
            mag = np.sqrt(f_uv[..., 0] ** 2 + f_uv[..., 1] ** 2)
            times.append(float(ts))
            mags.append(float(mag.mean()))
        prev_strip, prev_present = strip, True

    cap.release()
    return np.array(times), np.array(mags)
