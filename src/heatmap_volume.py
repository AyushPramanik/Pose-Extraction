"""
PoseC3D-style 3D heatmap-volume representation.

This is the data-preparation stage borrowed from PoseC3D ("Revisiting
Skeleton-based Action Recognition", Duan et al. CVPR 2022) as used by the
MiGA micro-gesture paper.  It turns per-frame 2D keypoints into a
subject-centered Gaussian heatmap volume of shape (C, T, H, W).

The pipeline order matches the PYSKL reference implementation:

    build_keypoint_tensor -> pose_compact -> uniform_sample_indices
        -> resize_keypoints -> generate_joint/limb_heatmaps

This module is pure representation: no pandas, no windowing, no move/no-move
decision logic.  It is reusable unchanged by a future PoseC3D CNN trainer,
which consumes the PYSKL annotation dicts written by `to_pyskl_annotation`.
"""
import pickle
from typing import Optional

import numpy as np

from src.utils.pose_extractor import KEYPOINT_NAMES, POSE_CONNECTIONS

# PoseC3D / PYSKL GeneratePoseTarget defaults.
SIGMA: float = 0.6
EPS: float = 1e-3
HEATMAP_SIZE: tuple[int, int] = (56, 56)   # (H, W)
DEFAULT_NUM_FRAMES: int = 48
PADDING: float = 0.25

# COCO skeleton edges as (index_a, index_b) pairs into KEYPOINT_NAMES,
# derived from the same connections used for skeleton rendering (16 limbs).
_NAME_TO_IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
LIMB_CONNECTIONS: list[tuple[int, int]] = [
    (_NAME_TO_IDX[a], _NAME_TO_IDX[b]) for a, b in POSE_CONNECTIONS
]


def _region_groups(names: list[str]) -> dict[str, list[int]]:
    """
    Group keypoint indices into body regions by name, so per-region motion
    energy can be reported.  Keyed off `names` (not hardcoded indices) so it
    extends automatically if a richer keypoint set (hands/face) is used later.
    """
    def idxs(*substrings: str) -> list[int]:
        return [i for i, n in enumerate(names)
                if any(s in n for s in substrings)]

    groups = {
        'head':  idxs('nose', 'eye', 'ear'),
        'arms':  idxs('shoulder', 'elbow', 'wrist', 'hand', 'finger', 'thumb'),
        'torso': idxs('shoulder', 'hip'),
        'legs':  idxs('hip', 'knee', 'ankle', 'foot', 'heel', 'toe'),
    }
    # Include a face region only when a rich keypoint set actually provides it.
    face = idxs('face')
    if face:
        groups['face'] = face
    return {k: v for k, v in groups.items() if v}


# Body-region -> keypoint indices, for per-part motion output.
REGION_GROUPS: dict[str, list[int]] = _region_groups(KEYPOINT_NAMES)


# ==================== keypoint tensor ====================

def build_keypoint_tensor(
        frames: list[dict],
        person_id: int,
        smooth=None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[int], list[float]]:
    """
    Extract one person's pose sequence into PoseC3D tensors (pixel space).

    Only frames in which the target person is present contribute a row; the
    row order follows the frames list.

    Args:
        smooth: optional keypoint_smoothing.SmoothConfig.  When set, the
                assembled (T, K, 2) tensor is temporally smoothed (gap-aware,
                confidence-gated) before returning, so every consumer sees the
                same smoothed coordinates.  None = raw keypoints.

    Returns:
        keypoint:       (T, K, 2) float32 pixel coordinates.  Missing keypoints
                        are 0.0 (and flagged via keypoint_score == 0.0).
        keypoint_score: (T, K)    float32 confidence in [0, 1].
        img_shape:      (H, W) inferred from the largest bbox seen (best-effort;
                        the poses JSON does not store frame size directly).
        frame_indices:  source frame_idx for each of the T rows.
        timestamps:     source timestamp (seconds) for each of the T rows.
    """
    K = len(KEYPOINT_NAMES)
    kp_rows: list[np.ndarray] = []
    sc_rows: list[np.ndarray] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    max_x, max_y = 0.0, 0.0

    for f in frames:
        person = next((p for p in f['persons'] if p.get('tracker_id') == person_id), None)
        if person is None:
            continue

        kp = np.zeros((K, 2), dtype=np.float32)
        sc = np.zeros((K,), dtype=np.float32)
        for k, name in enumerate(KEYPOINT_NAMES):
            val = person.get(name)
            if val:
                kp[k, 0] = val['x']
                kp[k, 1] = val['y']
                sc[k] = val['conf']
                max_x = max(max_x, val['x'])
                max_y = max(max_y, val['y'])

        bbox = person.get('bbox')
        if bbox:
            max_x = max(max_x, bbox.get('x2', 0.0))
            max_y = max(max_y, bbox.get('y2', 0.0))

        kp_rows.append(kp)
        sc_rows.append(sc)
        frame_indices.append(int(f['frame_idx']))
        timestamps.append(float(f['timestamp']))

    if not kp_rows:
        empty_kp = np.zeros((0, K, 2), dtype=np.float32)
        empty_sc = np.zeros((0, K), dtype=np.float32)
        return empty_kp, empty_sc, (0, 0), [], []

    keypoint = np.stack(kp_rows, axis=0)
    keypoint_score = np.stack(sc_rows, axis=0)

    if smooth is not None:
        # Lazy import avoids a circular dependency (keypoint_smoothing imports EPS).
        from src.keypoint_smoothing import smooth_keypoint_tensor
        keypoint = smooth_keypoint_tensor(keypoint, keypoint_score, timestamps, smooth)

    # img_shape is (H, W); ceil the observed extent as a best-effort frame size.
    img_shape = (int(np.ceil(max_y)) or 1, int(np.ceil(max_x)) or 1)
    return keypoint, keypoint_score, img_shape, frame_indices, timestamps


# ==================== subject-centered crop (PoseCompact) ====================

def pose_compact(
        keypoint: np.ndarray,
        keypoint_score: np.ndarray,
        padding: float = PADDING,
        hw_ratio: float = 1.0,
        threshold: int = 10,
    ) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Subject-centered square crop, following PYSKL PoseCompact.

    Computes a single union bounding box over all keypoints with score >= EPS
    across all T frames, expands it by `padding`, forces the aspect ratio to
    `hw_ratio` (1.0 => square for PoseC3D), and shifts keypoints into the box
    frame.  Coordinates stay in crop-pixel space (no resize here).

    Returns:
        keypoint:   (T, K, 2) shifted into the crop frame.
        crop_shape: (crop_h, crop_w) size of the crop box in pixels.

    When the visible extent is degenerate (< threshold px on a side) the
    keypoints are returned unchanged with their observed extent as crop_shape.
    """
    valid = keypoint_score >= EPS
    if not valid.any():
        return keypoint.copy(), (1.0, 1.0)

    xs = keypoint[..., 0][valid]
    ys = keypoint[..., 1][valid]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    if (max_x - min_x) < threshold or (max_y - min_y) < threshold:
        return keypoint.copy(), (max(max_y - min_y, 1.0), max(max_x - min_x, 1.0))

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    half_w = (max_x - min_x) / 2 * (1 + padding)
    half_h = (max_y - min_y) / 2 * (1 + padding)

    # Enforce aspect ratio h/w == hw_ratio by growing the smaller side.
    half_h = max(hw_ratio * half_w, half_h)
    half_w = max(half_h / hw_ratio, half_w)

    box_min_x = center_x - half_w
    box_min_y = center_y - half_h
    crop_w = 2 * half_w
    crop_h = 2 * half_h

    shifted = keypoint.copy()
    shifted[..., 0] = np.where(valid, keypoint[..., 0] - box_min_x, 0.0)
    shifted[..., 1] = np.where(valid, keypoint[..., 1] - box_min_y, 0.0)
    return shifted, (crop_h, crop_w)


def resize_keypoints(
        keypoint: np.ndarray,
        crop_shape: tuple[float, float],
        out_shape: tuple[int, int] = HEATMAP_SIZE,
    ) -> np.ndarray:
    """Scale crop-space keypoints into the (H, W) heatmap grid."""
    crop_h, crop_w = crop_shape
    out_h, out_w = out_shape
    scaled = keypoint.copy()
    scaled[..., 0] = keypoint[..., 0] * (out_w / max(crop_w, 1e-6))
    scaled[..., 1] = keypoint[..., 1] * (out_h / max(crop_h, 1e-6))
    return scaled


# ==================== uniform sampling ====================

def uniform_sample_indices(
        num_frames: int,
        clip_len: int = DEFAULT_NUM_FRAMES,
        test_mode: bool = True,
        seed: Optional[int] = None,
    ) -> np.ndarray:
    """
    UniformSampleFrames: pick `clip_len` frame indices from a variable-length
    clip by splitting into `clip_len` equal segments and drawing one per
    segment.  Deterministic mid-segment sampling in test_mode (the unsupervised
    default); random within-segment offset otherwise.  Short clips (num_frames
    < clip_len) are loop-padded.

    Returns an int array of length clip_len with values in [0, num_frames).
    """
    if num_frames <= 0:
        return np.zeros((clip_len,), dtype=np.int64)

    if num_frames < clip_len:
        inds = np.arange(clip_len) % num_frames
        return inds.astype(np.int64)

    bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
    seg_lengths = bids[1:] - bids[:-1]
    if test_mode:
        offsets = seg_lengths // 2
    else:
        rng = np.random.default_rng(seed)
        offsets = np.array([rng.integers(0, max(1, s)) for s in seg_lengths])
    inds = bids[:-1] + offsets
    return np.clip(inds, 0, num_frames - 1).astype(np.int64)


# ==================== Gaussian heatmap rendering ====================

def _gaussian_patch(
        arr: np.ndarray,
        mu_x: float,
        mu_y: float,
        amplitude: float,
        sigma: float,
    ) -> None:
    """Add a joint Gaussian into `arr` (H, W) in place, using np.maximum merge."""
    h, w = arr.shape
    st_x = max(int(mu_x - 3 * sigma), 0)
    ed_x = min(int(mu_x + 3 * sigma) + 1, w)
    st_y = max(int(mu_y - 3 * sigma), 0)
    ed_y = min(int(mu_y + 3 * sigma) + 1, h)
    if st_x >= ed_x or st_y >= ed_y:
        return
    x = np.arange(st_x, ed_x, dtype=np.float32)
    y = np.arange(st_y, ed_y, dtype=np.float32)[:, None]
    patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2)) * amplitude
    arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)


def generate_joint_heatmaps(
        keypoint: np.ndarray,
        keypoint_score: np.ndarray,
        out_shape: tuple[int, int] = HEATMAP_SIZE,
        sigma: float = SIGMA,
    ) -> np.ndarray:
    """
    Joint-modality volume of shape (K, T, H, W), float32.

    Each keypoint contributes a Gaussian exp(-((i-x)^2+(j-y)^2)/(2*sigma^2))
    scaled by its confidence.  Keypoints with score < EPS are skipped.  The
    kernel is only evaluated within a +/-3*sigma window for speed.
    """
    T, K, _ = keypoint.shape
    h, w = out_shape
    vol = np.zeros((K, T, h, w), dtype=np.float32)
    for t in range(T):
        for k in range(K):
            conf = float(keypoint_score[t, k])
            if conf < EPS:
                continue
            _gaussian_patch(vol[k, t], keypoint[t, k, 0], keypoint[t, k, 1], conf, sigma)
    return vol


def _limb_patch(
        arr: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        amplitude: float,
        sigma: float,
    ) -> None:
    """Add a limb (segment) Gaussian into `arr` (H, W) in place."""
    h, w = arr.shape
    min_x = max(int(min(start[0], end[0]) - 3 * sigma), 0)
    max_x = min(int(max(start[0], end[0]) + 3 * sigma) + 1, w)
    min_y = max(int(min(start[1], end[1]) - 3 * sigma), 0)
    max_y = min(int(max(start[1], end[1]) + 3 * sigma) + 1, h)
    if min_x >= max_x or min_y >= max_y:
        return

    x = np.arange(min_x, max_x, dtype=np.float32)
    y = np.arange(min_y, max_y, dtype=np.float32)[:, None]

    d2_ab = float((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    if d2_ab < 1e-6:
        # Degenerate segment: fall back to a point Gaussian at the start.
        _gaussian_patch(arr, float(start[0]), float(start[1]), amplitude, sigma)
        return

    d2_start = (x - start[0]) ** 2 + (y - start[1]) ** 2
    d2_end = (x - end[0]) ** 2 + (y - end[1]) ** 2
    coeff = (d2_start - d2_end + d2_ab) / (2 * d2_ab)
    coeff = np.clip(coeff, 0.0, 1.0)
    # Distance^2 from each pixel to its projection on the segment.
    proj_x = start[0] + coeff * (end[0] - start[0])
    proj_y = start[1] + coeff * (end[1] - start[1])
    d2_seg = (x - proj_x) ** 2 + (y - proj_y) ** 2

    patch = np.exp(-d2_seg / (2 * sigma ** 2)) * amplitude
    arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)


def generate_limb_heatmaps(
        keypoint: np.ndarray,
        keypoint_score: np.ndarray,
        connections: list[tuple[int, int]] = LIMB_CONNECTIONS,
        out_shape: tuple[int, int] = HEATMAP_SIZE,
        sigma: float = SIGMA,
    ) -> np.ndarray:
    """
    Limb-modality volume of shape (E, T, H, W), float32, where E = number of
    skeleton edges.  Each limb is a segment Gaussian between its two endpoint
    joints, scaled by min(conf_a, conf_b).  A limb is skipped when either
    endpoint has score < EPS.
    """
    T, _, _ = keypoint.shape
    h, w = out_shape
    E = len(connections)
    vol = np.zeros((E, T, h, w), dtype=np.float32)
    for t in range(T):
        for e, (a, b) in enumerate(connections):
            ca, cb = float(keypoint_score[t, a]), float(keypoint_score[t, b])
            if ca < EPS or cb < EPS:
                continue
            _limb_patch(vol[e, t], keypoint[t, a], keypoint[t, b], min(ca, cb), sigma)
    return vol


# ==================== full pipeline / export ====================

def render_volume(
        frames: list[dict],
        person_id: int,
        clip_len: int = DEFAULT_NUM_FRAMES,
        with_limb: bool = True,
        smooth=None,
    ) -> dict:
    """
    Run the full representation pipeline for one person:
    build -> pose_compact -> uniform_sample -> resize -> render.

    Returns a dict:
        joint:                  (K, T, H, W) float32 joint volume.
        limb:                   (E, T, H, W) float32 limb volume, or None.
        sampled_keypoint:       (T, K, 2) grid-space keypoints used to render.
        sampled_keypoint_score: (T, K) confidences for the sampled frames.
        img_shape:              (H, W) original frame size (best-effort).
    Empty arrays are returned when the person has no detected frames.
    """
    keypoint, keypoint_score, img_shape, _, _ = build_keypoint_tensor(frames, person_id, smooth=smooth)
    if keypoint.shape[0] == 0:
        return {'joint': np.zeros((len(KEYPOINT_NAMES), 0, *HEATMAP_SIZE), dtype=np.float32),
                'limb': None, 'sampled_keypoint': keypoint,
                'sampled_keypoint_score': keypoint_score, 'img_shape': img_shape}

    idx = uniform_sample_indices(keypoint.shape[0], clip_len=clip_len, test_mode=True)
    kp = keypoint[idx]
    sc = keypoint_score[idx]

    cropped, crop_shape = pose_compact(kp, sc)
    grid_kp = resize_keypoints(cropped, crop_shape, HEATMAP_SIZE)

    joint = generate_joint_heatmaps(grid_kp, sc, HEATMAP_SIZE, SIGMA)
    limb = generate_limb_heatmaps(grid_kp, sc, LIMB_CONNECTIONS, HEATMAP_SIZE, SIGMA) if with_limb else None
    return {
        'joint': joint,
        'limb': limb,
        'sampled_keypoint': grid_kp,
        'sampled_keypoint_score': sc,
        'img_shape': img_shape,
    }


def render_sequence_heatmaps(
        keypoint: np.ndarray,
        keypoint_score: np.ndarray,
        out_shape: tuple[int, int] = HEATMAP_SIZE,
        sigma: float = SIGMA,
        with_limb: bool = True,
    ) -> dict:
    """
    Render per-frame heatmap volumes for a full keypoint sequence, WITHOUT
    uniform sampling — so each heatmap frame maps 1:1 to the input rows (and
    thus to their timestamps).  This is what a per-window motion signal needs:
    the caller renders once, then slices the volume per time window.

    Uses the same subject-centered crop as Signal A (`pose_compact` over all T),
    so the coordinate-variance and heatmap signals see the same framing.

    Args:
        keypoint:       (T, K, 2) pixel-space keypoints (from build_keypoint_tensor).
        keypoint_score: (T, K) confidences.
        out_shape:      (H, W) heatmap grid; (28, 28) is a ~4x-cheaper option.

    Returns:
        joint:   (K, T, H, W) float32.
        limb:    (E, T, H, W) float32, or None when with_limb is False.
        grid_kp: (T, K, 2) crop-resized keypoints used to render.
    """
    if keypoint.shape[0] == 0:
        K = len(KEYPOINT_NAMES)
        return {'joint': np.zeros((K, 0, *out_shape), dtype=np.float32),
                'limb': None, 'grid_kp': keypoint}

    cropped, crop_shape = pose_compact(keypoint, keypoint_score)
    grid_kp = resize_keypoints(cropped, crop_shape, out_shape)
    joint = generate_joint_heatmaps(grid_kp, keypoint_score, out_shape, sigma)
    limb = (generate_limb_heatmaps(grid_kp, keypoint_score, LIMB_CONNECTIONS, out_shape, sigma)
            if with_limb else None)
    return {'joint': joint, 'limb': limb, 'grid_kp': grid_kp}


def heatmap_diff_energy(vol: np.ndarray, mask: np.ndarray) -> float:
    """
    Motion energy from a heatmap volume over a window: the mean over
    consecutive-frame L1 differences.

    Args:
        vol:  (C, T, H, W) float32 heatmap volume (joint or limb modality).
        mask: (T,) bool selecting the frames inside the window.

    Returns the mean over the (n-1) time steps of Σ_{C,H,W} |Δ|, or 0.0 when
    fewer than 2 frames fall inside the window.  Because it averages over time
    steps it is invariant to how many (sparse) frames a window contains.
    """
    if vol.shape[1] == 0:
        return 0.0
    sub = vol[:, mask]                       # (C, n, H, W)
    if sub.shape[1] < 2:
        return 0.0
    diff = np.abs(sub[:, 1:] - sub[:, :-1])  # (C, n-1, H, W)
    per_step = diff.sum(axis=(0, 2, 3))      # (n-1,)
    return float(per_step.mean())


def to_pyskl_annotation(
        frames: list[dict],
        person_id: int,
        frame_dir: str,
        label: int = 0,
        smooth=None,
    ) -> dict:
    """
    Build a CNN-ready PYSKL annotation dict for one person.

    Stores the full-length keypoint sequence in ORIGINAL pixel space (not the
    rendered volume): a PoseC3D data pipeline re-runs crop/resize/render at
    train time, so we must feed raw pixels + img_shape.  `label` is a
    placeholder (0) in the unsupervised setting.

    By default `smooth` is None: a future CNN should learn from unfiltered
    detector output (temporal smoothing is a downstream analysis choice, not
    ground truth).  Pass a SmoothConfig only if smoothed export is explicitly
    wanted.

    Shapes follow PYSKL: keypoint (M, T, K, 2), keypoint_score (M, T, K),
    with M = 1 person.
    """
    keypoint, keypoint_score, img_shape, _, _ = build_keypoint_tensor(frames, person_id, smooth=smooth)
    T, K = keypoint.shape[0], keypoint.shape[1] if keypoint.shape[0] else len(KEYPOINT_NAMES)
    return {
        'frame_dir': frame_dir,
        'label': int(label),
        'img_shape': img_shape,
        'original_shape': img_shape,
        'total_frames': int(T),
        'keypoint': keypoint[None].astype(np.float32),        # (1, T, K, 2)
        'keypoint_score': keypoint_score[None].astype(np.float32),  # (1, T, K)
    }


def save_pyskl_pickle(annotations: list[dict], output_path: str) -> None:
    """Pickle a list of per-person annotation dicts (PYSKL list-of-dicts)."""
    with open(output_path, 'wb') as f:
        pickle.dump(annotations, f)
