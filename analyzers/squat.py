"""
Squat form analyzer.

Tracks, per frame:
  - knee angle (hip-knee-ankle), left & right
  - hip angle (shoulder-hip-knee), left & right
  - trunk lean angle from vertical
  - knee valgus proxy (knee-to-knee width vs ankle-to-ankle width)

Segments the video into reps using local minima of the average knee angle
(the bottom of each squat), then evaluates each rep against coaching
heuristics. These thresholds are common coaching rules of thumb, not a
substitute for a coach or physical therapist -- flagged in the report.
"""

import numpy as np

from analyzers.base import SportAnalyzer, FormReport, FormFlag
from pose_estimation.estimator import VideoLandmarks
from utils.angles import joint_angle, segment_angle_to_vertical, smooth_series, find_local_minima, midpoint, safe_nanmean

# Coaching heuristic thresholds
DEPTH_KNEE_ANGLE_MAX = 110.0       # knee angle at bottom must drop below this to count as "at depth"
EXCESSIVE_FORWARD_LEAN_DEG = 45.0  # trunk angle from vertical beyond this = too much forward lean
VALGUS_RATIO_THRESHOLD = 0.80      # knee-width / ankle-width below this = knees caving in
ASYMMETRY_DEG_THRESHOLD = 15.0     # left/right knee angle difference beyond this = imbalance


class SquatAnalyzer(SportAnalyzer):
    sport_name = "squat"

    def analyze(self, vl: VideoLandmarks) -> FormReport:
        n = len(vl.frames)
        knee_l = np.full(n, np.nan)
        knee_r = np.full(n, np.nan)
        hip_l = np.full(n, np.nan)
        hip_r = np.full(n, np.nan)
        trunk_lean = np.full(n, np.nan)
        valgus_ratio = np.full(n, np.nan)

        for i, f in enumerate(vl.frames):
            if not f.detected:
                continue
            try:
                hip_pt_l, knee_pt_l, ankle_pt_l = f.get_2d("left_hip"), f.get_2d("left_knee"), f.get_2d("left_ankle")
                hip_pt_r, knee_pt_r, ankle_pt_r = f.get_2d("right_hip"), f.get_2d("right_knee"), f.get_2d("right_ankle")
                sh_pt_l, sh_pt_r = f.get_2d("left_shoulder"), f.get_2d("right_shoulder")

                knee_l[i] = joint_angle(hip_pt_l, knee_pt_l, ankle_pt_l)
                knee_r[i] = joint_angle(hip_pt_r, knee_pt_r, ankle_pt_r)
                hip_l[i] = joint_angle(sh_pt_l, hip_pt_l, knee_pt_l)
                hip_r[i] = joint_angle(sh_pt_r, hip_pt_r, knee_pt_r)

                shoulder_mid = midpoint(sh_pt_l, sh_pt_r)
                hip_mid = midpoint(hip_pt_l, hip_pt_r)
                trunk_lean[i] = segment_angle_to_vertical(shoulder_mid, hip_mid)

                knee_width = np.linalg.norm(knee_pt_l - knee_pt_r)
                ankle_width = np.linalg.norm(ankle_pt_l - ankle_pt_r)
                valgus_ratio[i] = knee_width / ankle_width if ankle_width > 1e-6 else np.nan
            except (TypeError, ValueError):
                continue  # a landmark was missing/None this frame

        knee_avg = (knee_l + knee_r) / 2.0
        knee_avg_smooth = smooth_series(list(knee_avg), window=7)

        rep_bottoms = find_local_minima(knee_avg_smooth, min_prominence=15.0, min_distance=int(vl.fps * 0.6))

        report = FormReport(sport=self.sport_name, detection_rate=vl.detection_rate)
        report.metrics = {
            "knee_angle_left": knee_l,
            "knee_angle_right": knee_r,
            "hip_angle_left": hip_l,
            "hip_angle_right": hip_r,
            "trunk_lean_from_vertical": trunk_lean,
            "knee_valgus_ratio": valgus_ratio,
        }
        report.rep_boundaries = rep_bottoms

        depths, leans, valgus_events, asymmetries = [], [], 0, []

        for rep_idx, bottom_frame in enumerate(rep_bottoms):
            ts = vl.frames[bottom_frame].timestamp_sec
            min_knee = knee_avg_smooth[bottom_frame]
            depths.append(min_knee)

            if min_knee > DEPTH_KNEE_ANGLE_MAX:
                report.flags.append(FormFlag(
                    bottom_frame, ts, "warning",
                    f"Rep {rep_idx + 1}: didn't reach depth (knee angle {min_knee:.0f}\u00b0, "
                    f"target below {DEPTH_KNEE_ANGLE_MAX:.0f}\u00b0)."
                ))

            lean_at_bottom = trunk_lean[bottom_frame]
            leans.append(lean_at_bottom)
            if not np.isnan(lean_at_bottom) and lean_at_bottom > EXCESSIVE_FORWARD_LEAN_DEG:
                report.flags.append(FormFlag(
                    bottom_frame, ts, "warning",
                    f"Rep {rep_idx + 1}: excessive forward trunk lean ({lean_at_bottom:.0f}\u00b0 from vertical)."
                ))

            valgus_at_bottom = valgus_ratio[bottom_frame]
            if not np.isnan(valgus_at_bottom) and valgus_at_bottom < VALGUS_RATIO_THRESHOLD:
                valgus_events += 1
                report.flags.append(FormFlag(
                    bottom_frame, ts, "critical",
                    f"Rep {rep_idx + 1}: possible knee valgus (knees caving in) detected."
                ))

            asym = abs(knee_l[bottom_frame] - knee_r[bottom_frame])
            asymmetries.append(asym)
            if not np.isnan(asym) and asym > ASYMMETRY_DEG_THRESHOLD:
                report.flags.append(FormFlag(
                    bottom_frame, ts, "warning",
                    f"Rep {rep_idx + 1}: left/right knee angle asymmetry of {asym:.0f}\u00b0."
                ))

        report.summary = {
            "rep_count": len(rep_bottoms),
            "avg_bottom_knee_angle": safe_nanmean(depths) if depths else float("nan"),
            "avg_trunk_lean_at_bottom": safe_nanmean(leans) if leans else float("nan"),
            "avg_left_right_asymmetry": safe_nanmean(asymmetries) if asymmetries else float("nan"),
            "valgus_events": valgus_events,
        }
        return report
