# Interpreting Results

This pipeline detects subtle body movement from a video using YOLO-Pose
(OpenPose-compatible 17-keypoint format), builds a PoseC3D-style
subject-centered heatmap representation, and makes an unsupervised
**move / no-move** decision per tracked person over time.  Multi-person runs
write one set of files per tracked person, e.g.
`output/statistics/recording_person2_motion_log.csv`.

Outputs are split across two folders inside `--output-dir`:

- `statistics/` — CSV / JSON tables and plots
- `videos/` — the poses JSON, the CNN-ready annotations, the annotated video, and split clips

---

## Output files

### `statistics/`

| File | What it contains |
|------|-----------------|
| `*_person<ID>_motion_log.csv` | Per-window move/no-move log for one tracked person |
| `*_person<ID>_motion_log.txt` | Human-readable version of the motion log |
| `*_person<ID>_motion_summary.json` | Aggregate motion statistics for one person |
| `*_person<ID>_motion_plot.png` | 2-panel motion plot (energy timeline + move/still band) |
| `*_all_person_motion_log.csv` | Combined motion windows for every tracked person |
| `*_all_person_motions.png` | Energy timelines, move/still bands, and aggregate activity for all people |

### `videos/`

| File | What it contains |
|------|-----------------|
| `*_poses.json` | Raw keypoint detections per frame (pixel coordinates + confidence) for all people |
| `*_pyskl.pkl` | CNN-ready PYSKL annotations for all people (see below) |
| `*_annotated.mp4` | Original video with boxes, tracker IDs, and pose skeletons overlaid (`--annotate-video`) |
| `*_person<ID>_bout<N>_<state>_<start>-<end>s.mp4` | One clip per movement bout per person (`--split-clips`) |
| `*_clips_index.csv` | Index of every split clip: person, state, times, frames, energy, file |

---

## Crop space & heatmap grid

Keypoints are detected in **pixel space**, then re-expressed for analysis:

1. **Subject-centered crop** (`pose_compact`): a single square box is fitted
   around all of a person's keypoints across all frames, expanded by padding.
2. **Resize**: the crop is scaled to a fixed **56×56 grid**.

This crop + resize removes camera-distance and framing confounds, so
coordinates change *only* when body parts move relative to the body — which is
what makes subtle movement detectable.  (This replaces the older
shoulder-width torso normalization.)

Two Gaussian heatmap modalities can be rendered on the grid:

- **Joint modality** — one channel per keypoint → `(17, T, 56, 56)`
- **Limb modality** — one channel per skeleton edge → `(16, T, 56, 56)`

The joint volume is used for Signal B; both are the representation a PoseC3D
CNN would consume after training data is labeled.

---

## Per-window motion log (`*_motion_log.csv`)

One row per sliding window (default 1.0 s window, 0.5 s step) per person.

| Column | Meaning |
|--------|---------|
| `person_id` | Tracker ID |
| `start_time` / `end_time` / `mid_time` | Window bounds and midpoint (seconds) |
| `start_frame` / `end_frame` | Source frame-index bounds |
| `energy_a` | **Decision signal.** Σ over keypoints of crop-normalized temporal variance (Signal A). Comparable across people. |
| `motion_energy` | Fused display magnitude (keypoint+joint+limb, per-person normalized). Not the decision variable. |
| `active_threshold` | The absolute move/still floor used — the **same constant for every person** (`--move-threshold`, default 1e-4). |
| `is_moving` | `True` when `energy_a > active_threshold` **and** `mean_keypoint_conf >= 0.8` (a low-confidence/occluded window is forced `still`, since its variance is detector jitter, not movement) |
| `state` | `moving` / `still` |
| `mean_keypoint_conf` | Mean confidence of valid keypoints in the window |
| `n_valid_kp` | Number of keypoints that contributed |
| `energy_head` / `energy_arms` / `energy_torso` / `energy_legs` / `dominant_region` | Per-body-part diagnostics (do not affect the decision) |

With `--signal-b`, two more columns are added (constant per person):

| Column | Meaning |
|--------|---------|
| `heatmap_l1_mean` | Mean L1 difference between consecutive rendered joint heatmaps (Signal B) |
| `ab_agreement` | `False` when Signal A reports motion but Signal B ≈ 0 (hint of a degenerate crop) |

> The move/still decision is **absolute**: `energy_a > move_threshold`, the same
> threshold for everyone. It answers "did this person move", not "did they move
> more than their own normal". `energy_a` is comparable across people; the fused
> `motion_energy` is a per-person-normalized display magnitude only.

---

## Summary statistics (`*_motion_summary.json`)

```
moving_fraction      Fraction of windows labelled "moving".
mean_motion_energy   Average motion_energy across all windows.
peak_motion_energy   Largest single-window motion_energy.
active_threshold     The absolute move/still threshold used (same for everyone).
n_windows            Number of sliding windows evaluated.
total_duration_s     Time span covered by the windows.
```

---

## CNN-ready annotations (`*_pyskl.pkl`)

A pickled list of per-person dicts in the PYSKL/PoseC3D annotation format:

| Key | Shape / type | Meaning |
|-----|--------------|---------|
| `frame_dir` | str | Unique clip id (`<stem>_person<ID>`) |
| `label` | int | Class id — placeholder `0` in the unsupervised setting |
| `img_shape` / `original_shape` | `(H, W)` | Original frame size (best-effort) |
| `total_frames` | int | Number of frames the person appears in |
| `keypoint` | `(1, T, 17, 2)` float32 | Raw **pixel** keypoints (M = 1 person) |
| `keypoint_score` | `(1, T, 17)` float32 | Per-joint confidence |

To train a PoseC3D classifier later: assign real `label` values and feed the
pickle to PYSKL/mmaction2.  The pipeline re-renders the crop/heatmap volume at
train time, so no re-extraction is needed.

---

## Reading the motion plot

**Per-person (`*_person<ID>_motion_plot.png`)**

| Panel | What to look for |
|-------|-----------------|
| **Motion Energy** | Filled area = motion level over time; the dashed red line is the person's move/still threshold; green dots mark windows labelled *moving*. |
| **Move / Still** | Green band = moving, grey = still, across the timeline. |

**All-person (`*_all_person_motions.png`)**

| Panel | What to look for |
|-------|-----------------|
| **Motion Energy Timeline** | One line per person; compare who moved when. |
| **Move / Still** | One row per person; green segments are their moving windows. |
| **Aggregate Activity** | Bars = mean motion energy per person; the black diamond line = moving fraction. |

---

## Interpreting `recording.mov` results

The clip is 628 frames (≈ 19.7 s); with the default `--skip-frames 10` it is
sampled to ~58 frames at ~2.9 fps, giving 39 windows per person.

| Person | moving_fraction | mean_motion_energy | peak_motion_energy | Interpretation |
|--------|-----------------|--------------------|--------------------|----------------|
| person4 | 0.128 | 4.1e-4 | 3.0e-3 | Most active of the stable tracks — a clip is cut around 12.5–15.5 s |
| person2 | 0.154 | 2.2e-4 | 1.2e-3 | Notable early movement — clip cut around 0–2.5 s |
| person3 | 0.077 | 2.1e-4 | 1.6e-3 | Occasional movement |
| person1 / person5 / person6 | ~0.05 | ~1e-4 | <1e-3 | Mostly still — calm seated listening |
| person7 | 0.25 | 6.2e-3 | 2.3e-2 | Short 4-window track (≈2 s); high energy but low sample count — likely a transient/pass-through detection |

**Notes:**
- Most stable tracks read as mostly *still*, consistent with calm seated
  listening; person2 and person4 have the clearest movement bouts and are the
  ones `--split-clips` cuts.
- Short tracks (person7) can show high `motion_energy` from just a few frames —
  weight them by `n_windows` / track duration before trusting them.

---

## Tips for comparing across participants or clips

- Use `moving_fraction` as the headline "how much did this person move" score —
  it is normalized and comparable across people.
- Compare `motion_energy` **within** a person over time, not across people
  (the threshold is per person).
- Run with `--signal-b` to corroborate the decision with the heatmap volume;
  check `ab_agreement`.
- Run with `--load-poses output/videos/recording_poses.json` to re-compute
  motion without re-running the pose detector.
