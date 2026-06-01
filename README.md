# Pose Extraction

YOLO-Pose pipeline for detecting people in `assets/recording.mov`, tracking
each person across frames, and writing separate pose, feature, and movement log
files for every tracked individual.

## Run

```bash
MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python -m src.run_detection \
  assets/recording.mov \
  -o output/recording_individuals \
  --annotate-video
```

Use `--load-poses output/recording_individuals/recording_poses.json` to
regenerate features, logs, plots, and the annotated video without rerunning the
YOLO model.

## Outputs

For each tracked person:

- `recording_person<ID>_keypoints.csv`
- `recording_person<ID>_keypoints_norm.csv`
- `recording_person<ID>_features.csv`
- `recording_person<ID>_summary.json`
- `recording_person<ID>_movement_log.csv`
- `recording_person<ID>_movement_log.txt`
- `recording_person<ID>_movement_summary.json`
- `recording_person<ID>_movement_plot.png`

Shared files:

- `recording_poses.json` contains all detections, tracker IDs, boxes, and
  keypoints.
- `recording_all_person_movement_log.csv` combines every person's movement
  windows into one table.
- `recording_all_person_movements.png` graphs movement energy, movement labels,
  and aggregate activity for all tracked people.
- `recording_annotated.mp4` overlays per-person boxes, tracker IDs, and pose
  skeletons on the source video.
