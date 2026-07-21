# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pose-Extraction** is a pipeline for detecting subtle body movement in seated subjects (e.g., listening to music) using YOLO-Pose, an OpenPose-compatible pose detection model. The system detects 17 keypoints per person, builds a PoseC3D-style subject-centered heatmap representation, and makes an unsupervised **move / no-move** decision per person over time.

**Core insight (from PoseC3D / the MiGA micro-gesture paper):** a subject-centered crop + resize removes camera-distance and framing confounds; motion is then measured as the temporal variance of keypoints in that normalized crop, making tiny relative movements detectable. The same keypoints are exported in PYSKL format so a PoseC3D CNN can later be trained on the identical data (gesture classification) without re-extraction.

---

## Rules
Read `.claude/rules/core-behavior.md` before any action.
For rule placement guidance see `.claude/rules/rules-location.md`.

---

## Architecture

The pipeline flows through four stages, each handled by a dedicated module:

### 1. **Pose Extraction** (`pose_extractor.py`)
- Wraps the ultralytics YOLO-Pose model (default: `yolov8x-pose.pt`, larger models are more accurate)
- Processes video frames and outputs 17-keypoint skeleton in COCO format
- Tracks each person with lightweight frame-to-frame box matching and writes
  stable `tracker_id` values plus bounding boxes
- Saves raw per-frame keypoint data (pixel coordinates + confidence) to JSON
- Model variants:
  - `yolov8n-pose` → fastest, lowest accuracy
  - `yolov8x-pose` → best for subtle movement (default)
  - `yolo11x-pose` → latest architecture, highest accuracy

### 2. **PoseC3D Representation** (`heatmap_volume.py`)
Pure representation layer following the PYSKL PoseC3D data-prep pipeline
(`build → pose_compact → uniform_sample → resize → generate`):
- **Subject-centered crop (`pose_compact`):** union bbox over all keypoints across all frames, expanded and forced square — removes camera-distance/framing confounds (the modern replacement for the old torso normalization).
- **Uniform sampling:** deterministic `T`-frame sampling (default 48) from variable-length clips.
- **Gaussian heatmap volumes:** joint modality `(K, T, H, W)` and limb modality `(E, T, H, W)`, σ=0.6 on a 56×56 grid, confidence-scaled. Used for Signal B and future CNN input.
- **PYSKL export (`to_pyskl_annotation`):** raw-pixel keypoints + `img_shape` in the exact dict format a PoseC3D CNN consumes (`keypoint (M,T,K,2)`, `keypoint_score (M,T,K)`, `label` placeholder 0).

### 3. **Unsupervised Move / No-Move** (`motion_detector.py`)
- **Signal A (primary):** slides a time window over crop-normalized keypoints; per window `motion_energy = Σ_k [Var(x_k) + Var(y_k)]`, confidence-weighted. Scale/framing invariant by construction.
- **Decision:** **absolute** noise-calibrated threshold on `energy_a` (crop-normalized keypoint variance) — `is_moving = energy_a > move_threshold`, the same for every person. Answers "did this person move", not "did they move more than their own normal".
- **Signal B (optional, `--signal-b`):** L1 temporal difference of consecutive rendered joint heatmaps, mean over T — corroborates Signal A per person.
- Emits a per-window DataFrame (the `_motion_log.csv` schema) that drives clip splitting and plots.

### 4. **Visualization** (`visualizer.py`)
- **Annotated video / per-person clips:** bounding boxes, tracker IDs, pose skeletons on the original video (unchanged).
- **Motion plots:** per-person 2-panel (motion-energy timeline + threshold; move/still band) and all-person 3-panel overview (energy lines, move/still bands per person, aggregate bar).

### 5. **Pipeline Orchestration** (`run_detection.py`)
- CLI entry point; chains the above stages; per-person analysis fanned out across worker processes.
- Optionally loads pre-computed poses (`--load-poses`) to skip re-detection.
- Outputs per input video:
  - `*_poses.json` → raw keypoints per frame
  - `*_pyskl.pkl` → CNN-ready PYSKL annotations (all persons)
  - `*_person<N>_motion_log.csv` / `.txt` → per-window move/no-move log
  - `*_person<N>_motion_summary.json` → per-person aggregate stats
  - `*_person<N>_motion_plot.png` → per-person motion plot
  - `*_all_person_motion_log.csv` / `*_all_person_motions.png` → combined
  - `*_annotated.mp4` → video with overlay (optional, `--annotate-video`)
  - `*_clips_index.csv` + bout clips → optional (`--split-clips`)

---

## Common Commands

### Extract poses from a video (full pipeline)
```bash
python -m src.run_detection recording.mov -o output
```

### Use a faster model
```bash
python -m src.run_detection recording.mov --checkpoint yolov8n-pose.pt
```

### Use a more accurate model
```bash
python -m src.run_detection recording.mov --checkpoint yolo11x-pose.pt
```

### Lower confidence threshold (catch subtler poses, add noise)
```bash
python -m src.run_detection recording.mov --confidence 0.10
```

### Skip every other frame (2× faster processing, half the detail)
```bash
python -m src.run_detection recording.mov --skip-frames 1
```

### Extract with annotated video overlay
```bash
python -m src.run_detection recording.mov --annotate-video
```

### Recompute motion without re-running detection
```bash
python -m src.run_detection recording.mov --load-poses output/videos/recording_poses.json
```

### Add heatmap corroboration (Signal B) and per-bout clips
```bash
python -m src.run_detection recording.mov --signal-b --split-clips
```

### Analyse a specific person in a multi-person video
```bash
python -m src.run_detection recording.mov --person-ids 1
```

---

## Data Flow & Key Concepts

### Crop space & heatmap grid
- Keypoints are detected in **pixel space**, then `pose_compact` shifts them into a **subject-centered square crop** and `resize_keypoints` scales that crop to a **56×56 grid**. This crop/resize is what makes motion scale- and framing-invariant (replaces the old shoulder-width normalization).
- **Joint modality:** one Gaussian channel per keypoint → `(17, T, 56, 56)`.
- **Limb modality:** one segment-Gaussian channel per skeleton edge → `(16, T, 56, 56)`.

### Per-window motion log (`*_motion_log.csv`)
One row per sliding window per person:
- **motion_energy:** Σ over keypoints of crop-normalized temporal variance (Signal A).
- **active_threshold:** the absolute move/still floor used (constant across all people; `--move-threshold`, default 1e-4).
- **is_moving / state:** the move/no-move decision (`is_moving == energy_a > active_threshold`). Note the decision is on `energy_a`, not the fused `motion_energy` (which is a per-person-normalized display magnitude).
- **mean_keypoint_conf / n_valid_kp:** detection quality inside the window.
- With `--signal-b`: **heatmap_l1_mean** (per-person) + **ab_agreement** flag.

### Summary statistics (`*_motion_summary.json`)
- **moving_fraction:** fraction of windows labelled moving.
- **mean_motion_energy / peak_motion_energy:** scalar motion level.
- **active_threshold, n_windows, total_duration_s.**

### CNN-ready annotations (`*_pyskl.pkl`)
List of per-person PYSKL dicts (`frame_dir`, `label`=0 placeholder, `img_shape`, `total_frames`, `keypoint (1,T,17,2)`, `keypoint_score (1,T,17)`) — the input a PoseC3D CNN trains on directly, unchanged.

---

## Debugging & Troubleshooting

### No keypoints detected
- Check video format and codec (MP4, MOV, AVI typically work)
- Lower `--confidence` threshold (default 0.15)
- Use a larger model (`yolo11x-pose` or `yolov8x-pose`)
- Ensure good lighting and clear upper-body visibility

### Poor pose tracking (jittery output)
- Increase `--confidence` to filter weak detections
- Annotated video (`--annotate-video`) reveals skeleton quality
- Check `*_motion_plot.png` for spikes; sharp peaks often indicate false detections

### Motion looks noisy or everything reads as "moving"
- The decision is an absolute threshold on `energy_a`. If a still person reads as moving, raise `--move-threshold`; if real subtle motion is missed, lower it.
- Run with `--signal-b`: if `ab_agreement` is False (Signal A says moving but heatmap L1 ≈ 0), the subject-centered crop is likely degenerate (too few valid keypoints).
- For very low fps videos, lower `--skip-frames` so windows contain ≥2 samples.

### Memory issues
- Use `--skip-frames` to process fewer frames
- Use a smaller model (`yolov8n-pose`)
- Process video in chunks by splitting input file

---

## Key Files & Modules

| File | Purpose |
|------|---------|
| `utils/pose_extractor.py` | YOLO-Pose + ByteTrack; keypoint detection from video |
| `heatmap_volume.py` | PoseC3D representation: crop, sampling, Gaussian volumes, PYSKL export |
| `motion_detector.py` | Windowed move/no-move (Signal A/B), per-window log, summaries |
| `video_splitter.py` | Per-person movement-bout clip splitting |
| `visualizer.py` | Annotated video rendering and motion plots |
| `run_detection.py` | CLI orchestrator; chains all stages |
| `RESULTS.md` | Format guide for all outputs (output file schemas) |
| `pyproject.toml` | Dependencies: ultralytics, opencv, numpy, pandas, scipy, matplotlib |

---

## Dependencies

Managed via `uv` (see `pyproject.toml`). Key packages:
- **ultralytics** — YOLO-Pose model
- **opencv-contrib-python** — video I/O and frame annotation
- **numpy, pandas** — data manipulation, Gaussian rendering, windowed stats
- **supervision** — ByteTrack multi-person tracking
- **matplotlib** — plotting
- **pillow** — image utilities

---

## Development Notes

### Confidence threshold tuning
- `--confidence 0.15` (default) balances detection and noise
- Lower (~0.05–0.10) catches subtle poses but adds false detections
- Higher (~0.25+) misses genuine subtle movements

### Model selection
- **For research:** `yolo11x-pose` (best accuracy, slow)
- **For production:** `yolov8x-pose` (balanced)
- **For speed:** `yolov8n-pose` (fastest, lowest accuracy)

### Move/no-move decision
- The decision is **absolute** (`energy_a > --move-threshold`), the same scale for every person — it answers whether a person moved, not whether they moved more than their own baseline.
- `motion_energy` is unitless (variance of normalized crop coordinates); compare it only within a person, or via `moving_fraction` across persons.
- Signal A needs only cropped keypoints (cheap, per window); Signal B renders the heatmap volume (heavier, per person) and is opt-in via `--signal-b`.

### Future CNN training
- `*_pyskl.pkl` is written in the PYSKL annotation format. To train a PoseC3D classifier later, add real `label` values and feed the pickle to PYSKL/mmaction2 — no re-extraction needed.

### Output size
- `*_poses.json` is compact (raw keypoints only)
- `*_motion_log.csv` is one row per sliding window per person
- `*_pyskl.pkl` stores full-length keypoint arrays (compact)
- `*_annotated.mp4` is largest; same size as input video

---
