"""
Split a long video into per-person, per-bout clips.

A "bout" is a continuous stretch where one tracked person moves notably more
than their own baseline.  The goal is to capture the beginning and end of a
real movement by a single person, while ignoring:

  - people who only pass through / transient false detections (min track duration)
  - chronic fidgeting (per-person adaptive energy threshold)
  - brief pauses inside one continuous performance (short-gap bridging)

Each clip keeps the full original frame but overlays the box + skeleton for the
target person only (see visualizer.render_person_clip).
"""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.motion_detector import MOVE_ENERGY_THRESHOLD
from src.visualizer import render_person_clip
from src.utils.logger import get_logger


_LOGGER = get_logger(name = "src.video_splitter", level = "INFO")


def _person_active_mask(person_log: pd.DataFrame, threshold: float = MOVE_ENERGY_THRESHOLD) -> pd.Series:
    """
    A window is "active" when it was labelled moving by the absolute move/no-move
    decision.  Reuses the per-window `state` column directly so clip bouts always
    agree with the motion log; falls back to the same absolute rule on `energy_a`
    (or the recorded `active_threshold`) if `state` is missing.
    """
    if 'state' in person_log.columns:
        return person_log['state'] == 'moving'
    if 'energy_a' in person_log.columns:
        thr = (float(person_log['active_threshold'].iloc[0])
               if 'active_threshold' in person_log.columns else threshold)
        return person_log['energy_a'] > thr
    return pd.Series(False, index=person_log.index)


def _assemble_bouts(
        person_log: pd.DataFrame,
        active: pd.Series,
        bridge_seconds: float,
        min_seconds: float,
    ) -> list[dict]:
    """
    Group active windows into bouts, bridging still-gaps shorter than
    bridge_seconds, then drop bouts shorter than min_seconds.
    """
    rows = person_log.reset_index(drop=True)
    active = active.reset_index(drop=True)

    bouts: list[dict] = []
    current: list[int] = []      # row indices in the current bout

    def close_current() -> None:
        if not current:
            return
        sub = rows.iloc[current]
        bouts.append({
            'start_time': float(sub['start_time'].min()),
            'end_time': float(sub['end_time'].max()),
            'start_frame': int(sub['start_frame'].min()),
            'end_frame': int(sub['end_frame'].max()),
            'movement': sub['state'].mode().iat[0] if not sub.empty else 'moving',
            'peak_energy': float(sub['motion_energy'].max()),
            'mean_energy': float(sub['motion_energy'].mean()),
        })

    last_active_end = None
    for i in range(len(rows)):
        if active.iat[i]:
            if current and last_active_end is not None:
                gap = float(rows['start_time'].iat[i]) - last_active_end
                if gap > bridge_seconds:
                    close_current()
                    current = []
            current.append(i)
            last_active_end = float(rows['end_time'].iat[i])
        # inactive windows are simply skipped; the gap check above bridges them
    close_current()

    return [b for b in bouts if (b['end_time'] - b['start_time']) >= min_seconds]


def split_video(
        video_path: str,
        frames: list[dict],
        movement_log: pd.DataFrame,
        output_dir: Union[str, Path],
        stem: str,
        fps: float,
        min_track_seconds: float = 3.0,
        move_threshold: float = MOVE_ENERGY_THRESHOLD,
        bridge_seconds: float = 1.5,
        min_clip_seconds: float = 2.0,
    ) -> pd.DataFrame:
    """
    Render one clip per detected movement bout per person and return an index
    DataFrame describing every clip written.

    movement_log is the combined all-person log (one row per window per person).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if movement_log.empty:
        _LOGGER.warning("No movement windows; nothing to split.")
        return pd.DataFrame()

    index_rows: list[dict] = []
    for person_id, person_log in movement_log.groupby('person_id'):
        if pd.isna(person_id):
            continue
        person_id = int(person_id)
        person_log = person_log.sort_values('start_time')

        # 1. Min track duration — skip pass-through people / transient detections.
        track_span = float(person_log['end_time'].max() - person_log['start_time'].min())
        if track_span < min_track_seconds:
            _LOGGER.info(f"  person{person_id}: track span {track_span:.1f}s < "
                         f"{min_track_seconds:.1f}s — skipped.")
            continue

        # 2. Absolute move/still mask (reuses the per-window decision).
        active = _person_active_mask(person_log, move_threshold)

        # 3. Bout assembly with gap bridging + minimum length.
        bouts = _assemble_bouts(person_log, active, bridge_seconds, min_clip_seconds)
        if not bouts:
            _LOGGER.info(f"  person{person_id}: no clip-worthy movement bouts.")
            continue

        for n, bout in enumerate(bouts, start=1):
            fname = (f"{stem}_person{person_id}_bout{n}_{bout['movement']}_"
                     f"{bout['start_time']:.1f}-{bout['end_time']:.1f}s.mp4")
            out_path = output_dir / fname
            render_person_clip(
                video_path=video_path,
                frames_data=frames,
                output_path=str(out_path),
                person_id=person_id,
                start_frame=bout['start_frame'],
                end_frame=bout['end_frame'],
            )
            index_rows.append({
                'person_id': person_id,
                'bout': n,
                'movement': bout['movement'],
                'start_time': round(bout['start_time'], 3),
                'end_time': round(bout['end_time'], 3),
                'duration_s': round(bout['end_time'] - bout['start_time'], 3),
                'start_frame': bout['start_frame'],
                'end_frame': bout['end_frame'],
                'peak_energy': round(bout['peak_energy'], 5),
                'mean_energy': round(bout['mean_energy'], 5),
                'file': fname,
            })

    index_df = pd.DataFrame(index_rows)
    index_path = output_dir / f'{stem}_clips_index.csv'
    index_df.to_csv(index_path, index=False)
    _LOGGER.info(f"Wrote {len(index_df)} clip(s); index -> {index_path}")
    return index_df
