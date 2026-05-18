"""
Behavioral state classifier from OpenPose kinematic features.

Infers discrete actions and emotional states from pre-computed kinematics
via rule-based heuristics grounded in body language research. No labeled
training data is required; thresholds adapt to each clip's own statistics.

IMPORTANT: These are heuristic inferences, not ground-truth labels.
Treat outputs as hypotheses for further investigation.

Actions detected (per window):
  still         – negligible movement across all joints
  nodding       – predominantly vertical head oscillation
  head_shaking  – predominantly lateral head oscillation
  gesturing     – elevated wrist / arm speed
  arm_raised    – wrist position above shoulder midpoint (normalised coords)
  active        – high overall body movement (unspecific)

Emotional states inferred:
  excited       – high energy, rhythmic or expressive movement
  engaged       – moderate energy with purposeful movement
  calm          – low, smooth movement; stable posture
  disengaged    – persistently very low movement; passive stillness
  agitated      – high, irregular (non-rhythmic) movement
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ── window parameters ─────────────────────────────────────────────────────────
WINDOW_SECS  = 2.0   # analysis window length
OVERLAP_SECS = 0.5   # step = window - overlap

# ── directional dominance thresholds ─────────────────────────────────────────
VERT_DOM   = 0.58   # rom_y / (rom_x + rom_y) > this → vertical head motion
LAT_DOM    = 0.58   # rom_x / (rom_x + rom_y) > this → lateral head motion

# ── spatial thresholds (normalised torso units) ───────────────────────────────
ARM_RAISE_Y  = -0.30  # wrist_y_norm < this → arm raised above shoulder line
                      # (norm y is negative when wrist is above shoulder midpoint)

# ── adaptive threshold percentiles ───────────────────────────────────────────
# Thresholds are derived from the clip's own statistics so the classifier
# works whether the subject is very active or very still.
P_STILL   = 25   # below 25th-pctile energy → still
P_LOW     = 45
P_MID     = 60
P_HIGH    = 78
P_GESTURE = 75   # wrist speed above 75th-pctile → gesturing
P_HEAD    = 45   # head speed above 45th-pctile qualifies for direction analysis

# Irregularity: coefficient of variation (std/mean) of per-frame energy.
IRREG_THRESH = 1.6


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    start_time: float
    end_time:   float
    mid_time:   float
    actions:    list[str]
    emotion:    str
    confidence: float
    stats:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'start_time': round(self.start_time, 3),
            'end_time':   round(self.end_time,   3),
            'mid_time':   round(self.mid_time,   3),
            'actions':    ','.join(self.actions) if self.actions else 'still',
            'emotion':    self.emotion,
            'confidence': round(self.confidence, 3),
            **{k: round(v, 5) if isinstance(v, float) else v
               for k, v in self.stats.items()},
        }


# ── main classifier ───────────────────────────────────────────────────────────

class BehavioralClassifier:
    """
    Sliding-window behavioral state classifier.

    Parameters
    ----------
    fps : float
        Effective frames-per-second of the feature data.
    window_secs : float
        Length of each analysis window in seconds.
    overlap_secs : float
        Overlap between consecutive windows in seconds.
    """

    def __init__(
        self,
        fps: float,
        window_secs: float = WINDOW_SECS,
        overlap_secs: float = OVERLAP_SECS,
    ) -> None:
        self.fps = fps
        self.win   = max(4, int(fps * window_secs))
        self.step  = max(1, int(fps * (window_secs - overlap_secs)))

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        feat_df: pd.DataFrame,
        norm_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Classify behavioral states over the full clip.

        Parameters
        ----------
        feat_df : DataFrame
            Output of ``compute_features()`` — kinematic features.
        norm_df : DataFrame
            Output of ``normalize_to_torso()`` — torso-normalised positions.

        Returns
        -------
        results_df : DataFrame  (one row per window)
        summary    : dict
        """
        thr = self._adaptive_thresholds(feat_df)
        results: list[WindowResult] = []

        n = len(feat_df)
        start = 0
        while start < n:
            end = min(start + self.win, n)
            if end - start < 3:
                break
            wf = feat_df.iloc[start:end]
            wn = norm_df.iloc[start:end]
            results.append(self._classify_window(wf, wn, thr))
            start += self.step

        results_df = pd.DataFrame([r.to_dict() for r in results])
        summary = self._summarize(results, feat_df)
        return results_df, summary

    # ── threshold computation ─────────────────────────────────────────────────

    def _adaptive_thresholds(self, feat_df: pd.DataFrame) -> dict:
        thr: dict = {}

        speed_cols = [c for c in feat_df.columns if c.endswith('_speed')]
        if speed_cols:
            energy = feat_df[speed_cols].mean(axis=1).dropna()
            for name, p in (('still', P_STILL), ('low', P_LOW),
                            ('mid',   P_MID),   ('high', P_HIGH)):
                thr[f'energy_{name}'] = float(energy.quantile(p / 100)) if len(energy) else 0.0
            thr['energy_mean'] = float(energy.mean()) if len(energy) else 0.0

        # Head speed threshold
        if 'nose_speed' in feat_df.columns:
            hs = feat_df['nose_speed'].dropna()
            thr['head_speed_active'] = float(hs.quantile(P_HEAD / 100)) if len(hs) else 0.0

        # Wrist speed threshold for gesturing
        wrist_cols = [c for c in feat_df.columns if 'wrist_speed' in c]
        if wrist_cols:
            ws = pd.concat([feat_df[c].dropna() for c in wrist_cols])
            thr['wrist_gesture'] = float(ws.quantile(P_GESTURE / 100)) if len(ws) else 0.0

        return thr

    # ── per-window classification ─────────────────────────────────────────────

    def _classify_window(
        self,
        wf: pd.DataFrame,
        wn: pd.DataFrame,
        thr: dict,
    ) -> WindowResult:
        t0 = float(wf['time'].iloc[0])
        t1 = float(wf['time'].iloc[-1])
        stats = self._window_stats(wf, wn)
        actions = self._detect_actions(stats, thr)
        emotion, conf = self._infer_emotion(actions, stats, thr)
        return WindowResult(t0, t1, (t0 + t1) / 2, actions, emotion, conf, stats)

    def _window_stats(self, wf: pd.DataFrame, wn: pd.DataFrame) -> dict:
        s: dict = {}

        # ── overall energy ────────────────────────────────────────────────────
        speed_cols = [c for c in wf.columns if c.endswith('_speed')]
        if speed_cols:
            e = wf[speed_cols].mean(axis=1).dropna()
            s['movement_energy']       = float(e.mean())   if len(e) else 0.0
            s['movement_irregularity'] = (float(e.std() / e.mean())
                                          if len(e) > 1 and e.mean() > 1e-9 else 0.0)
        else:
            s['movement_energy'] = 0.0
            s['movement_irregularity'] = 0.0

        # ── head directionality ───────────────────────────────────────────────
        rx = float(wf['nose_rom_x'].mean()) if 'nose_rom_x' in wf.columns else 0.0
        ry = float(wf['nose_rom_y'].mean()) if 'nose_rom_y' in wf.columns else 0.0
        total = rx + ry
        s['head_rom_x']   = rx
        s['head_rom_y']   = ry
        s['head_vert_dom'] = ry / total if total > 1e-9 else 0.5
        s['head_lat_dom']  = rx / total if total > 1e-9 else 0.5

        # ── head speed ────────────────────────────────────────────────────────
        if 'nose_speed' in wf.columns:
            hs = wf['nose_speed'].dropna()
            s['head_speed_mean'] = float(hs.mean()) if len(hs) else 0.0
        else:
            s['head_speed_mean'] = 0.0

        # ── wrist speeds ──────────────────────────────────────────────────────
        wrist_speeds = []
        for side in ('left', 'right'):
            col = f'{side}_wrist_speed'
            if col in wf.columns:
                v = float(wf[col].dropna().mean())
                s[f'{side}_wrist_speed_mean'] = v
                wrist_speeds.append(v)
        s['wrist_speed_max'] = max(wrist_speeds) if wrist_speeds else 0.0

        # ── wrist height relative to shoulder midpoint (normalised) ──────────
        # In torso-normalised coords, y < 0 means ABOVE the shoulder midpoint.
        for side in ('left', 'right'):
            wyc = f'{side}_wrist_y'
            if wyc in wn.columns:
                wy = wn[wyc].dropna()
                s[f'{side}_wrist_y_norm'] = float(wy.mean()) if len(wy) else 0.0
            else:
                s[f'{side}_wrist_y_norm'] = 0.0

        # ── elbow angles ──────────────────────────────────────────────────────
        for side in ('left', 'right'):
            col = f'{side}_elbow_angle'
            if col in wf.columns:
                ea = wf[col].dropna()
                if len(ea):
                    s[f'{side}_elbow_angle_mean'] = float(ea.mean())
                    s[f'{side}_elbow_angle_std']  = float(ea.std()) if len(ea) > 1 else 0.0

        return s

    # ── action detection ──────────────────────────────────────────────────────

    def _detect_actions(self, s: dict, thr: dict) -> list[str]:
        actions: list[str] = []
        energy = s.get('movement_energy', 0.0)

        if energy <= thr.get('energy_still', 0.0):
            return ['still']

        head_speed = s.get('head_speed_mean', 0.0)
        head_active = head_speed > thr.get('head_speed_active', 0.0)

        if head_active and s.get('head_vert_dom', 0.5) > VERT_DOM:
            actions.append('nodding')

        if head_active and s.get('head_lat_dom', 0.5) > LAT_DOM:
            actions.append('head_shaking')

        if s.get('wrist_speed_max', 0.0) > thr.get('wrist_gesture', float('inf')):
            actions.append('gesturing')

        # Arm raised: wrist y well above shoulder midpoint (y negative in norm coords)
        for side in ('left', 'right'):
            if s.get(f'{side}_wrist_y_norm', 0.0) < ARM_RAISE_Y:
                if 'arm_raised' not in actions:
                    actions.append('arm_raised')

        if energy > thr.get('energy_high', float('inf')) and not actions:
            actions.append('active')

        return actions if actions else ['moving']

    # ── emotion inference ─────────────────────────────────────────────────────

    def _infer_emotion(
        self, actions: list[str], s: dict, thr: dict
    ) -> tuple[str, float]:
        energy     = s.get('movement_energy', 0.0)
        irreg      = s.get('movement_irregularity', 0.0)

        e_still = thr.get('energy_still', 0.0)
        e_low   = thr.get('energy_low',   0.0)
        e_mid   = thr.get('energy_mid',   0.0)
        e_high  = thr.get('energy_high',  float('inf'))

        if 'still' in actions:
            return 'disengaged', 0.55

        # High-energy states
        if energy >= e_high:
            if irreg > IRREG_THRESH:
                return 'agitated', 0.65
            if 'nodding' in actions or 'gesturing' in actions:
                return 'excited', 0.75
            if 'arm_raised' in actions:
                return 'excited', 0.65
            return 'excited', 0.55

        # Moderate energy
        if energy >= e_mid:
            if 'nodding' in actions:
                return 'engaged', 0.75
            if 'gesturing' in actions:
                return 'engaged', 0.70
            if 'arm_raised' in actions:
                return 'engaged', 0.65
            if irreg > IRREG_THRESH:
                return 'agitated', 0.55
            return 'engaged', 0.50

        # Low energy
        if energy >= e_low:
            if 'nodding' in actions:
                return 'engaged', 0.60
            if 'gesturing' in actions:
                return 'engaged', 0.55
            return 'calm', 0.60

        # Very low energy (below low threshold but not still)
        return 'calm', 0.55

    # ── summary ───────────────────────────────────────────────────────────────

    def _summarize(self, results: list[WindowResult], feat_df: pd.DataFrame) -> dict:
        if not results:
            return {}

        emotions = [r.emotion for r in results]
        all_actions = [a for r in results for a in r.actions]

        emotion_fracs: dict[str, float] = {}
        for e in set(emotions):
            emotion_fracs[e] = round(emotions.count(e) / len(emotions), 3)

        action_fracs: dict[str, float] = {}
        for a in set(all_actions):
            action_fracs[a] = round(all_actions.count(a) / len(results), 3)

        dominant = max(emotion_fracs, key=emotion_fracs.get)

        return {
            'dominant_emotion':   dominant,
            'emotion_fractions':  emotion_fracs,
            'action_fractions':   action_fracs,
            'n_windows':          len(results),
            'total_duration_s':   round(float(feat_df['time'].max() - feat_df['time'].min()), 2),
            'avg_confidence':     round(float(np.mean([r.confidence for r in results])), 3),
        }


# ── convenience: run directly on saved CSV files ──────────────────────────────

def classify_from_csvs(
    features_csv: str,
    norm_csv: str,
    fps: float,
    window_secs: float = WINDOW_SECS,
    overlap_secs: float = OVERLAP_SECS,
) -> tuple[pd.DataFrame, dict]:
    """
    Load pre-computed CSV files and run the behavioral classifier.

    Parameters
    ----------
    features_csv : str  Path to ``*_features.csv``
    norm_csv     : str  Path to ``*_keypoints_norm.csv``
    fps          : float  Effective fps (stored in ``*_summary.json``)

    Returns
    -------
    results_df, summary_dict
    """
    feat_df = pd.read_csv(features_csv)
    norm_df = pd.read_csv(norm_csv)
    clf = BehavioralClassifier(fps, window_secs, overlap_secs)
    return clf.analyze(feat_df, norm_df)
