# Pose Extraction

OpenPose-compatible pose extraction using YOLO-Pose (ultralytics) with
**ByteTrack** multi-person tracking for stable person IDs across frames.
For each tracked person it writes pose, feature, and movement-log files, and
can split a long recording into per-person movement clips.

## Run

```bash
uv run python -m src.run_detection assets/recording.mov -o output --annotate-video
```

`uv run` auto-creates and syncs `.venv` from `pyproject.toml` + `uv.lock`, then
runs inside it. No manual venv, activation, or `MPLCONFIGDIR` needed — the same
command works on Windows and macOS/Linux.

Re-use saved detections (skip the YOLO model) with:

```bash
uv run python -m src.run_detection assets/recording.mov \
  --load-poses output/videos/recording_poses.json
```

## Options

| Flag | Purpose |
|------|---------|
| `--annotate-video` | Write a full annotated copy of the video (all skeletons). |
| `--split-clips` | Write per-person, per-movement-bout clips to `videos/`. |
| `--skip-frames N` | Process every (N+1)th frame. **Primary speed/detail lever** — raise it to go faster on long videos, lower it for finer temporal detail (default 10). |
| `--workers N` | Worker processes for per-person analysis. `0` = auto, `1` = no multiprocessing. |
| `--checkpoint` | YOLO-Pose model (see below). |
| `--confidence` | Keypoint detection threshold (default 0.15). |
| `--person-ids 1,3` | Analyse only these tracker IDs. |
| `--move-threshold` | Absolute move/still floor on `energy_a` (default 1e-4). Lower = more sensitive. |
| `--min-track-seconds` / `--clip-bridge-seconds` / `--clip-min-seconds` | Tune clip splitting (see below). |

### Model selection

```
yolov8n-pose  – fastest, least accurate
yolov8s-pose  – small
yolov8m-pose  – medium
yolov8l-pose  – large
yolov8x-pose  – extra-large, strong accuracy
yolo11x-pose  – latest architecture, highest accuracy (default)
```

## Performance on long videos

The bottleneck is YOLO inference per frame. The pipeline already:

- **batches** frames into single inference calls, and
- **parallelises** per-person analysis across processes (`--workers`).

For very long inputs, the biggest lever you control is `--skip-frames`.

## Clip splitting

With `--split-clips`, each clip captures one person performing one continuous
movement, with the box + skeleton overlaid on that person only (full frame kept).
Guards against noise:

- **Min track duration** (`--min-track-seconds`) — ignores people who only pass
  through or transient false detections.
- **Absolute move threshold** (`--move-threshold`) — a window counts as active
  when its `energy_a` exceeds the fixed move/still floor (the same rule as the
  motion log), so clips align with the move/no-move decision.
- **Gap bridging** (`--clip-bridge-seconds`) — brief pauses inside one
  performance stay a single clip.

## Outputs

Files land under `output/statistics/` (CSV/JSON/plots) and `output/videos/`
(poses JSON, annotated video, and split clips).

See [RESULTS.md](RESULTS.md) for the full output schema and how to interpret it.


<!-- shoulder nghieng, thang. Normalize 0 hop li -->