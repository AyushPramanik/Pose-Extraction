# Interpreting Results

This pipeline extracts subtle body movements from a video using YOLO-Pose
(OpenPose-compatible 17-keypoint format).  Everything below applies to the
files produced in `output/`.

---

## Output files

| File | What it contains |
|------|-----------------|
| `*_poses.json` | Raw keypoint detections per frame (pixel coordinates + confidence) |
| `*_keypoints.csv` | Same data as the JSON, in a flat table |
| `*_keypoints_norm.csv` | **Torso-normalised** coordinates — best file for analysis |
| `*_features.csv` | Per-frame kinematics: speed, acceleration, range-of-motion, joint angles |
| `*_summary.json` | Aggregate statistics across the whole clip |
| `*_movement_plot.png` | 6-panel visual summary |
| `*_annotated.mp4` | Original video with pose skeleton overlaid |

---

## Coordinate system (`*_keypoints_norm.csv`)

All `_x` and `_y` columns are expressed in **shoulder-width units**, relative
to the midpoint between the two shoulders.

```
         ← negative x         positive x →
                    [shoulder midpoint = (0, 0)]
         ↑ negative y (up on screen)
         ↓ positive  y (down on screen)
```

A nose value of `(0.04, -0.50)` means the nose is 4% of a shoulder-width to
the right of centre and half a shoulder-width above the shoulders.

**Why normalise?**  A person who leans forward slightly or sits at a different
distance will produce different raw pixel values even if they made no relative
movement.  Normalised coordinates remove that confound, so values change *only*
when body parts move relative to the torso.  This is what makes subtle
movements detectable.

> **Note:** Because the shoulder midpoint is defined as `(0, 0)`,
> `shoulder_sway_std` in the summary will always be `0.0` — that is expected,
> not a bug.  To measure shoulder sway use `left_shoulder_speed` or
> `right_shoulder_speed` from `*_features.csv` instead.

---

## Feature columns (`*_features.csv`)

Each keypoint produces four columns.  Example for `nose`:

| Column | Meaning | Unit |
|--------|---------|------|
| `nose_speed` | Magnitude of velocity (√vx²+vy²) | shoulder-widths / second |
| `nose_accel` | Rate of change of speed | shoulder-widths / second² |
| `nose_rom_x` | Horizontal range of motion over the last 1 s | shoulder-widths |
| `nose_rom_y` | Vertical range of motion over the last 1 s | shoulder-widths |

`left_elbow_angle` / `right_elbow_angle` give the angle at the elbow joint in
degrees (180° = arm fully extended, 90° = right angle).

The first row will be NaN for speed/acceleration — that is normal; a
first-difference needs at least two frames.

---

## Summary statistics (`*_summary.json`)

### Per-keypoint fields

```
{keypoint}_mean_speed      Average speed across the clip.
{keypoint}_p95_speed       95th-percentile speed — the magnitude of the
                           largest bursts without being thrown off by
                           individual outlier frames.
{keypoint}_active_fraction Fraction of frames where speed > 1.5 units/s.
                           With normalised coordinates typical subtle
                           movements sit well below this threshold, so this
                           will often read 0.0.  It is most useful for
                           comparing one person against another or one clip
                           against another rather than as an absolute measure.
{keypoint}_dominant_freq_hz Frequency (Hz) of the strongest periodic pattern
                           in that keypoint's lateral position signal.
```

### Head and shoulder summary fields

```
head_lateral_freq_hz   Dominant horizontal oscillation frequency of the nose.
head_vertical_freq_hz  Dominant vertical oscillation frequency of the nose.
head_lateral_std       Standard deviation of horizontal head position.
                       Larger = more lateral head movement across the clip.
head_vertical_std      Standard deviation of vertical head position.
                       Larger = more nodding / bobbing.
shoulder_sway_freq_hz  Dominant frequency in the shoulder midpoint lateral
                       position (note: std is always 0 — see above).
total_movement_energy  Mean speed averaged over all upper-body keypoints and
                       all frames.  Single summary number for how much the
                       person moved overall.
```

---

## Reading the movement plot (`*_movement_plot.png`)

| Panel | What to look for |
|-------|-----------------|
| **Head Movement Speed** | Spikes = moments of clear head movement (nods, turns). A flat line near zero means the head was very still. |
| **Arm Movement Speed** | Wrist spikes correlate with hand gestures or rhythmic tapping. Compare left vs right to see asymmetric engagement. |
| **Vertical Range of Motion** | Sustained elevation = the body part held a different position for ≥1 s.  Sharp rise then fall = a brief gesture. |
| **Elbow Angle** | Drift upward = arms raising.  Oscillation = repeated arm movement.  A flat line means arms stayed in one position. |
| **Overall Body Movement Energy** | Filled area shows the total activity level over time.  Peaks are moments of higher engagement; valleys are stillness. |
| **Mean Speed per Keypoint** | Bar chart comparing which body part moved the most on average.  Taller bars = more active region. |

---

## Interpreting `recording.mov` results

The clip is 628 frames at ~31.8 fps (≈ 19.7 seconds).

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `total_movement_energy` | 0.019 | Low overall movement — consistent with calm, seated listening |
| `head_lateral_std` | 0.012 | Nose moved ±1.2% of shoulder-width laterally — very subtle |
| `head_vertical_std` | 0.006 | Less vertical than lateral — slight side-to-side presence |
| `head_lateral_freq_hz` | 0.051 Hz | One full sway cycle every ~20 s — very slow postural drift, not rhythmic tapping |
| `left_wrist_dominant_freq_hz` | 0.354 Hz | One cycle every ~2.8 s — the most rhythmically active point; worth inspecting in the annotated video |
| `right_wrist_dominant_freq_hz` | 0.202 Hz | One cycle every ~5 s — slower right-hand rhythm |
| `left_elbow_dominant_freq_hz` | 0.304 Hz | Matches left wrist, suggesting coupled arm movement |
| `shoulder_sway_std` | 0.000 | Expected (see note above) |

**What to investigate next:**
- Open `recording_annotated.mp4` and watch around the moments where the arm
  movement plot shows peaks — these are the highest-engagement instants.
- The left wrist (0.35 Hz) moves more rhythmically than the right (0.20 Hz);
  this asymmetry may reflect which hand is resting vs. active.
- The 0.051 Hz head frequency is too slow to be beat-tracking (music is
  typically 1–3 Hz).  A longer clip will reveal whether faster rhythmic
  components emerge.

---

## Tips for comparing across participants or clips

- Use `total_movement_energy` as the headline engagement score.
- Use `head_vertical_std` and `head_lateral_std` for postural stability.
- Use `left_wrist_dominant_freq_hz` / `right_wrist_dominant_freq_hz` for
  rhythmic hand activity.
- All values are normalised so they are comparable across people of different
  sizes and at different camera distances.
- Run with `--load-poses output/recording_poses.json` to re-compute features
  without re-running the pose detector.
