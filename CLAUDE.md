# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pose-Extraction** is a pipeline for analyzing subtle body movements in seated subjects (e.g., listening to music) using YOLO-Pose, an OpenPose-compatible pose detection model. The system detects 17 keypoints per person, normalizes coordinates relative to the torso, and extracts kinematic features to quantify movement engagement.

**Core insight:** Torso normalization removes confounds from camera distance and body position, making tiny relative movements detectable — enabling analysis of micro-expressions, rhythmic hand movements, and postural dynamics.

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
- Saves raw per-frame keypoint data (pixel coordinates + confidence) to JSON
- Model variants:
  - `yolov8n-pose` → fastest, lowest accuracy
  - `yolov8x-pose` → best for subtle movement (default)
  - `yolo11x-pose` → latest architecture, highest accuracy

### 2. **Data Normalization & Kinematics** (`feature_extractor.py`)
- **Torso normalization:** Expresses every keypoint relative to shoulder midpoint, scaled by shoulder width. This removes global translation so relative movements become visible.
  - Reference point: midpoint between left and right shoulders = (0, 0)
  - Scale: shoulder width in pixels
  - Result: coordinates in "shoulder-width units", enabling cross-person comparison
- **Keypoint grouping:** HEAD, SHOULDERS, ARMS, TORSO subsets used in different computations
- **Feature computation:**
  - Speed: √(vx² + vy²) in shoulder-widths/second
  - Acceleration: rate of change of speed
  - Range of motion (ROM): min–max over 1-second window
  - Joint angles: elbow angles in degrees (via 3-point geometry)
- **Smoothing:** Savitzky–Golay filter on linearly interpolated signals to denoise
- **Frequency analysis:** FFT-based dominant frequency detection for each keypoint (Hz)

### 3. **Visualization** (`visualizer.py`)
- **Annotated video:** Overlays pose skeleton (green lines) and keypoint markers on original video
- **Movement plot:** 6-panel figure showing:
  1. Head movement speed (nose, ears)
  2. Arm movement speed (wrists, elbows)
  3. Vertical range of motion
  4. Elbow angles
  5. Overall body movement energy (area fill)
  6. Per-keypoint mean speed bar chart

### 4. **Pipeline Orchestration** (`main.py`)
- CLI entry point; chains the above stages
- Optionally loads pre-computed poses (`--load-poses`) to skip re-detection
- Outputs 7 files per input video:
  - `*_poses.json` → raw keypoints per frame
  - `*_keypoints.csv` → flat CSV of raw coordinates
  - `*_keypoints_norm.csv` → torso-normalized coordinates (use these for analysis)
  - `*_features.csv` → frame-level kinematics
  - `*_summary.json` → aggregate statistics
  - `*_movement_plot.png` → 6-panel visualization
  - `*_annotated.mp4` → video with overlay (optional)

---

## Common Commands

### Extract poses from a video (full pipeline)
```bash
python main.py recording.mov -o output
```

### Use a faster model
```bash
python main.py recording.mov --checkpoint yolov8n-pose.pt
```

### Use a more accurate model
```bash
python main.py recording.mov --checkpoint yolo11x-pose.pt
```

### Lower confidence threshold (catch subtler poses, add noise)
```bash
python main.py recording.mov --confidence 0.10
```

### Skip every other frame (2× faster processing, half the detail)
```bash
python main.py recording.mov --skip-frames 1
```

### Extract with annotated video overlay
```bash
python main.py recording.mov --annotate-video
```

### Recompute features without re-running detection
```bash
python main.py recording.mov --load-poses output/recording_poses.json
```

### Extract from a different person in multi-person video
```bash
python main.py recording.mov --person 1
```

---

## Data Flow & Key Concepts

### Coordinate systems

**Raw pixel space (`*_keypoints.csv`):**
- (x, y) in pixels, as detected by the model
- Dependent on camera distance, framing, person size

**Torso-normalized space (`*_keypoints_norm.csv`):**
- (x, y) in shoulder-width units relative to shoulder midpoint
- Invariant to camera distance and body size
- **This is the input for feature extraction** — it's what makes subtle movement analysis possible
- Example: nose at (0.04, -0.50) = 4% of a shoulder-width to the right, 0.5 shoulder-widths above the shoulders

### Frame-level features (`*_features.csv`)

For each keypoint:
- **speed:** magnitude of velocity (shoulder-widths/second)
- **accel:** magnitude of acceleration (shoulder-widths/second²)
- **rom_x / rom_y:** range of motion over a 1-second rolling window

Joint angles:
- **left_elbow_angle / right_elbow_angle:** degrees (180° = fully extended, 90° = right angle)

### Summary statistics (`*_summary.json`)

Per-keypoint aggregates:
- **mean_speed:** average velocity
- **p95_speed:** 95th percentile (robust to outliers)
- **active_fraction:** fraction of frames with speed > 1.5 units/s
- **dominant_freq_hz:** strongest periodic component (Hz)

Head & shoulder composites:
- **head_lateral_freq_hz / head_vertical_freq_hz:** Fourier frequency of nose oscillation
- **head_lateral_std / head_vertical_std:** postural stability (std of normalized position)
- **shoulder_sway_freq_hz / shoulder_sway_std:** shoulder midpoint dynamics
- **total_movement_energy:** mean speed across all upper-body keypoints (scalar engagement score)

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
- Check `*_movement_plot.png` for spikes; sharp peaks often indicate false detections

### Features look noisy
- The Savitzky–Goyal smoothing window is hardcoded at 7 frames
- For very low fps videos, consider `--skip-frames` or higher fps source
- Torso normalization requires stable shoulder detection — poor shoulder tracking degrades all downstream features

### Memory issues
- Use `--skip-frames` to process fewer frames
- Use a smaller model (`yolov8n-pose`)
- Process video in chunks by splitting input file

---

## Key Files & Modules

| File | Purpose |
|------|---------|
| `pose_extractor.py` | YOLO-Pose wrapper; keypoint detection from video |
| `feature_extractor.py` | Torso normalization, kinematics, FFT, aggregate stats |
| `visualizer.py` | Annotated video rendering and 6-panel plot |
| `main.py` | CLI orchestrator; chains all stages |
| `RESULTS.md` | Format guide for all outputs (output file schemas) |
| `pyproject.toml` | Dependencies: ultralytics, opencv, numpy, pandas, scipy, matplotlib |

---

## Dependencies

Managed via `uv` (see `pyproject.toml`). Key packages:
- **ultralytics** — YOLO-Pose model
- **opencv-contrib-python** — video I/O and frame annotation
- **numpy, pandas** — data manipulation
- **scipy** — signal processing (Savitzky–Golay, FFT)
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

### Torso normalization invariants
- Shoulder midpoint is always (0, 0) in normalized space — `shoulder_sway_std` always 0.0
- Use `left_shoulder_speed` / `right_shoulder_speed` from `*_features.csv` to measure shoulder sway instead
- Missing shoulder detections degrade all downstream coordinates (interpolation is best-effort)

### Output size
- `*_poses.json` is compact (raw keypoints only)
- `*_keypoints.csv` / `*_keypoints_norm.csv` are row-per-frame tables
- `*_features.csv` is same row count, wider (4–5 columns per keypoint)
- `*_annotated.mp4` is largest; same size as input video

---