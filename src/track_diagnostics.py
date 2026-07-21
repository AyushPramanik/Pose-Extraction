"""
Per-track quality diagnostics for multi-person runs.

Subtle-movement detection needs stable tracker IDs: if a person's ID flickers,
their per-person motion baseline is computed on fragmented data.  This module
summarizes each track so ID switches / gaps are visible, and identifies
duplicate tracks (one person detected twice — e.g. an occluded subject split
into an "upper-only" and an "upper+lower" detection), all WITHOUT touching the
detection or tracking itself.
"""
import itertools

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

_LOGGER = get_logger(name="src.track_diagnostics", level="INFO")

# Duplicate-track detection (post-tracking).
# When two tracks are the SAME person, their upper-body keypoints coincide on the
# frames they share.  Measured on real duplicates: mean upper-body keypoint
# distance ~3% of body height.  Box IoU is unreliable across occlusions
# (0.3-0.6), so keypoint coincidence is primary and IoU is only a loose gate.
_UPPER_BODY = ('nose', 'left_shoulder', 'right_shoulder',
               'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')
DUP_DIST_THRESH = 0.06        # normalized (dist / body-height) per shared frame; true dup ~0.03
DUP_MIN_SHARED_KPS = 3        # shared confident upper-body kps required to compare a frame
DUP_MIN_OVERLAP_FRAC = 0.5    # fraction of the shorter track's frames the pair must co-occur on
DUP_MIN_COINCIDE_FRAC = 0.6   # fraction of co-occurring frames that must be coincident to call it a dup
DUP_IOU_FLOOR = 0.1           # loose per-frame box-overlap gate. An "upper-only" split of an
                              # occluded person has a smaller/offset box (IoU ~0.18 vs the full
                              # upper+lower detection) even when upper-body joints coincide at
                              # ~0.03, so this is only a sanity gate — keypoint coincidence is
                              # the real discriminator.


def _bbox_center(bbox: dict) -> tuple[float, float]:
    return ((bbox['x1'] + bbox['x2']) / 2.0, (bbox['y1'] + bbox['y2']) / 2.0)


def _bbox_diag(bbox: dict) -> float:
    return float(np.hypot(bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']))


def track_report(frames: list[dict], source_fps: float) -> pd.DataFrame:
    """
    One row per tracker_id summarizing track health:

      first_frame, last_frame, duration_s, n_present, n_gaps, longest_gap_s,
      mean_bbox_conf, id_switch_suspects

    `id_switch_suspects` counts consecutive present frames where the bbox center
    jumped further than its own diagonal — a heuristic for an ID reassigned to a
    different body.
    """
    per_track: dict[int, list[dict]] = {}
    for f in frames:
        for p in f['persons']:
            tid = p.get('tracker_id')
            if tid is None:
                continue
            per_track.setdefault(int(tid), []).append({
                'frame_idx': f['frame_idx'],
                'timestamp': f['timestamp'],
                'bbox': p.get('bbox'),
            })

    rows: list[dict] = []
    for tid, entries in sorted(per_track.items()):
        entries.sort(key=lambda e: e['frame_idx'])
        frame_idxs = [e['frame_idx'] for e in entries]
        times = [e['timestamp'] for e in entries]
        confs = [e['bbox']['conf'] for e in entries if e['bbox']]

        # Gaps between consecutive present frames (in sampled frames).
        gaps = np.diff(frame_idxs)
        n_gaps = int((gaps > 1).sum()) if len(gaps) else 0
        longest_gap_s = float(gaps.max() / source_fps) if len(gaps) and source_fps else 0.0

        # ID-switch suspects: center jump > bbox diagonal between present frames.
        suspects = 0
        for a, b in zip(entries[:-1], entries[1:]):
            if a['bbox'] and b['bbox']:
                ca, cb = _bbox_center(a['bbox']), _bbox_center(b['bbox'])
                jump = np.hypot(cb[0] - ca[0], cb[1] - ca[1])
                if jump > max(_bbox_diag(a['bbox']), 1.0):
                    suspects += 1

        rows.append({
            'tracker_id': tid,
            'first_frame': frame_idxs[0],
            'last_frame': frame_idxs[-1],
            'duration_s': round(times[-1] - times[0], 3),
            'n_present': len(entries),
            'n_gaps': n_gaps,
            'longest_gap_s': round(longest_gap_s, 3),
            'mean_bbox_conf': round(float(np.mean(confs)), 3) if confs else 0.0,
            'id_switch_suspects': suspects,
        })

    return pd.DataFrame(rows)


def _box_iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    ix2, iy2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    bb = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def find_duplicate_tracks(frames: list[dict], confidence: float = 0.15,
                          dist_thresh: float = DUP_DIST_THRESH) -> dict[int, int]:
    """
    Identify tracks that are duplicate detections of the SAME person, WITHOUT
    merging IDs or touching detection/tracking.  Runs after tracking, so
    ByteTrack identity continuity is preserved.

    Two tracks are duplicates when, on the frames where both are present
    (temporal overlap >= DUP_MIN_OVERLAP_FRAC of the shorter track), their
    upper-body keypoints coincide (mean normalized distance < dist_thresh, box
    IoU >= DUP_IOU_FLOOR) on most (>= DUP_MIN_COINCIDE_FRAC) of those frames.

    Conservative by design — the temporal-overlap + majority-coincidence
    requirements make it very hard to flag two genuinely distinct people who
    merely pass close for a moment.

    Returns {duplicate_tracker_id: primary_tracker_id}: the duplicate should be
    excluded from analysis; the primary (longer-lived, then higher-confidence,
    then more complete) is kept.
    """
    # Gather per-track frame -> (person dict) and simple stats.
    per_track: dict[int, dict] = {}
    for f in frames:
        for p in f['persons']:
            tid = p.get('tracker_id')
            if tid is None:
                continue
            per_track.setdefault(int(tid), {})[f['frame_idx']] = p

    def stats(tid: int) -> tuple:
        entries = per_track[tid]
        confs = [e['bbox']['conf'] for e in entries.values() if e.get('bbox')]
        completeness = np.mean([_n_valid_kp(e, confidence) for e in entries.values()])
        return (len(entries), float(np.mean(confs)) if confs else 0.0, completeness)

    dup_of: dict[int, int] = {}
    tids = sorted(per_track)
    for a, b in itertools.combinations(tids, 2):
        fa, fb = per_track[a], per_track[b]
        shared_frames = set(fa) & set(fb)
        shorter = min(len(fa), len(fb))
        if shorter == 0 or len(shared_frames) < DUP_MIN_OVERLAP_FRAC * shorter:
            continue

        coincident = 0
        comparable = 0
        for fi in shared_frames:
            pa, pb = fa[fi], fb[fi]
            nd = _mean_upper_dist(pa, pb, confidence)
            if nd is None:
                continue
            comparable += 1
            if nd < dist_thresh and _box_iou(pa['bbox'], pb['bbox']) >= DUP_IOU_FLOOR:
                coincident += 1
        if comparable == 0 or coincident < DUP_MIN_COINCIDE_FRAC * comparable:
            continue

        # Duplicate pair -> keep the primary (longer, higher conf, more complete).
        sa, sb = stats(a), stats(b)
        primary, dup = (a, b) if sa >= sb else (b, a)
        dup_of[dup] = primary

    # Resolve chains so every duplicate maps to a surviving primary.
    for dup in list(dup_of):
        root = dup_of[dup]
        while root in dup_of:
            root = dup_of[root]
        dup_of[dup] = root
    return dup_of


def _kp_conf(p: dict, name: str) -> float:
    v = p.get(name)
    return float(v['conf']) if v else 0.0


def _n_valid_kp(p: dict, confidence: float) -> int:
    return sum(1 for name, v in p.items()
               if isinstance(v, dict) and 'conf' in v and v['conf'] >= confidence)


def _mean_upper_dist(pa: dict, pb: dict, confidence: float):
    """Mean upper-body keypoint distance between two detections, normalized by
    body height.  None when too few shared confident upper-body keypoints."""
    shared = [k for k in _UPPER_BODY
              if _kp_conf(pa, k) >= confidence and _kp_conf(pb, k) >= confidence]
    if len(shared) < DUP_MIN_SHARED_KPS:
        return None
    d = np.mean([np.hypot(pa[k]['x'] - pb[k]['x'], pa[k]['y'] - pb[k]['y']) for k in shared])
    ha = pa['bbox']['y2'] - pa['bbox']['y1']
    hb = pb['bbox']['y2'] - pb['bbox']['y1']
    scale = max(ha, hb)
    return (d / scale) if scale > 0 else None
