"""
Football (soccer) injury-prevention analyzer.

Scope: this analyzes jump-landing and cutting/change-of-direction mechanics,
not general match play or shooting technique. That's a deliberate choice --
landing and deceleration mechanics are the movements with the strongest,
best-established links to non-contact injury (particularly ACL injury) in
field sports, and they're what real screening protocols like the Landing
Error Scoring System (LESS) and FIFA 11+ injury-prevention work actually
measure. Film a player performing a jump-and-land or a cutting maneuver.

See docs/REFERENCES.md for the specific studies behind each metric below
(Hewett et al. 2005 for valgus; Leppänen et al. for stiff landings) and the
gap between their 3D marker-based/force-plate methods and this analyzer's
single-camera 2D proxies.

Tracks, at the landing event(s):
  - knee flexion angle (hip-knee-ankle) per leg -- a "stiff" landing with
    little knee bend increases impact loading through the joint
  - knee valgus ratio (knee width / ankle width) -- knees caving inward on
    landing is one of the most consistently cited ACL injury risk factors
  - trunk flexion from vertical -- an upright trunk at landing reduces the
    body's ability to absorb force through the hips
  - left/right landing asymmetry, both in timing and in knee flexion

Landing frames are located by finding the apex of a jump (hip midpoint at
its highest point in frame) and then the first ground contact for each foot
afterward.
"""

import numpy as np

from analyzers.base import SportAnalyzer, FormReport, FormFlag
from pose_estimation.estimator import VideoLandmarks
from utils.angles import joint_angle, segment_angle_to_vertical, smooth_series, find_local_minima, midpoint, safe_nanmean

STIFF_LANDING_KNEE_ANGLE_MIN = 160.0   # knee angle at landing above this = minimal flexion, "stiff" landing
VALGUS_RATIO_THRESHOLD = 0.80          # same knee-width/ankle-width proxy used in the squat analyzer
UPRIGHT_TRUNK_MAX_DEG = 12.0           # trunk lean below this at landing = too little forward flexion
ASYMMETRY_KNEE_DEG_THRESHOLD = 15.0
ASYMMETRY_TIMING_SEC_THRESHOLD = 0.10


class FootballInjuryAnalyzer(SportAnalyzer):
    sport_name = "football"

    def analyze(self, vl: VideoLandmarks) -> FormReport:
        n = len(vl.frames)
        knee_l = np.full(n, np.nan)
        knee_r = np.full(n, np.nan)
        trunk_lean = np.full(n, np.nan)
        valgus_ratio = np.full(n, np.nan)
        hip_y = np.full(n, np.nan)
        ankle_y_l = np.full(n, np.nan)
        ankle_y_r = np.full(n, np.nan)

        for i, f in enumerate(vl.frames):
            if not f.detected:
                continue
            try:
                hip_pt_l, knee_pt_l, ankle_pt_l = f.get_2d("left_hip"), f.get_2d("left_knee"), f.get_2d("left_ankle")
                hip_pt_r, knee_pt_r, ankle_pt_r = f.get_2d("right_hip"), f.get_2d("right_knee"), f.get_2d("right_ankle")
                sh_l, sh_r = f.get_2d("left_shoulder"), f.get_2d("right_shoulder")

                knee_l[i] = joint_angle(hip_pt_l, knee_pt_l, ankle_pt_l)
                knee_r[i] = joint_angle(hip_pt_r, knee_pt_r, ankle_pt_r)

                shoulder_mid = midpoint(sh_l, sh_r)
                hip_mid = midpoint(hip_pt_l, hip_pt_r)
                trunk_lean[i] = segment_angle_to_vertical(shoulder_mid, hip_mid)
                hip_y[i] = hip_mid[1]

                knee_width = np.linalg.norm(knee_pt_l - knee_pt_r)
                ankle_width = np.linalg.norm(ankle_pt_l - ankle_pt_r)
                valgus_ratio[i] = knee_width / ankle_width if ankle_width > 1e-6 else np.nan

                ankle_y_l[i] = ankle_pt_l[1]
                ankle_y_r[i] = ankle_pt_r[1]
            except (TypeError, ValueError):
                continue

        report = FormReport(sport=self.sport_name, detection_rate=vl.detection_rate)
        report.metrics = {
            "knee_angle_left": knee_l,
            "knee_angle_right": knee_r,
            "trunk_lean_from_vertical": trunk_lean,
            "knee_valgus_ratio": valgus_ratio,
        }

        min_gap = max(3, int(vl.fps * 0.3))
        hip_y_smooth = smooth_series(list(hip_y), window=5)
        # Jump apex = hip midpoint at its highest point in frame = local MIN of y (image y grows downward)
        apex_frames = find_local_minima(hip_y_smooth, min_prominence=0.02, min_distance=min_gap)

        landing_events = []  # (apex_frame, left_landing_frame_or_None, right_landing_frame_or_None)
        search_window = int(vl.fps * 0.8)
        ankle_y_l_smooth = smooth_series(list(ankle_y_l), window=3)
        ankle_y_r_smooth = smooth_series(list(ankle_y_r), window=3)

        for apex in apex_frames:
            end = min(n - 1, apex + search_window)
            seg_l = ankle_y_l_smooth[apex:end + 1]
            seg_r = ankle_y_r_smooth[apex:end + 1]
            land_l = apex + int(np.nanargmax(seg_l)) if not np.all(np.isnan(seg_l)) else None
            land_r = apex + int(np.nanargmax(seg_r)) if not np.all(np.isnan(seg_r)) else None
            if land_l is not None or land_r is not None:
                landing_events.append((apex, land_l, land_r))

        if not landing_events:
            report.flags.append(FormFlag(0, 0.0, "warning", "Could not detect a jump/landing event in this clip."))
            return report

        landing_trunk_vals = []
        landing_valgus_vals = []

        for apex, land_l, land_r in landing_events:
            frame_ref = land_l if land_l is not None else land_r
            ts = vl.frames[frame_ref].timestamp_sec
            report.rep_boundaries.append(frame_ref)

            kl = knee_l[land_l] if land_l is not None else np.nan
            kr = knee_r[land_r] if land_r is not None else np.nan
            trunk_at_land = trunk_lean[frame_ref]
            valgus_at_land = valgus_ratio[frame_ref]
            landing_trunk_vals.append(trunk_at_land)
            landing_valgus_vals.append(valgus_at_land)

            for side, k in [("left", kl), ("right", kr)]:
                if not np.isnan(k) and k > STIFF_LANDING_KNEE_ANGLE_MIN:
                    report.flags.append(FormFlag(
                        frame_ref, ts, "warning",
                        f"Stiff {side} knee at landing ({k:.0f}\u00b0) \u2014 limited shock absorption."
                    ))

            if not np.isnan(valgus_at_land) and valgus_at_land < VALGUS_RATIO_THRESHOLD:
                report.flags.append(FormFlag(
                    frame_ref, ts, "critical",
                    f"Possible knee valgus at landing (ratio {valgus_at_land:.2f}) \u2014 established ACL injury risk factor."
                ))

            if not np.isnan(trunk_at_land) and trunk_at_land < UPRIGHT_TRUNK_MAX_DEG:
                report.flags.append(FormFlag(
                    frame_ref, ts, "warning",
                    f"Upright trunk at landing ({trunk_at_land:.0f}\u00b0 from vertical) \u2014 limited hip/trunk contribution to force absorption."
                ))

            if not (np.isnan(kl) or np.isnan(kr)) and abs(kl - kr) > ASYMMETRY_KNEE_DEG_THRESHOLD:
                report.flags.append(FormFlag(
                    frame_ref, ts, "warning",
                    f"Left/right knee flexion asymmetry of {abs(kl - kr):.0f}\u00b0 at landing."
                ))

            if land_l is not None and land_r is not None:
                timing_gap = abs(land_l - land_r) / vl.fps
                if timing_gap > ASYMMETRY_TIMING_SEC_THRESHOLD:
                    report.flags.append(FormFlag(
                        frame_ref, ts, "info",
                        f"Feet landed {timing_gap * 1000:.0f}ms apart \u2014 asymmetric (non-simultaneous) landing."
                    ))

        report.summary = {
            "landings_detected": len(landing_events),
            "avg_trunk_lean_at_landing": safe_nanmean(landing_trunk_vals) if landing_trunk_vals else float("nan"),
            "avg_knee_valgus_ratio_at_landing": safe_nanmean(landing_valgus_vals) if landing_valgus_vals else float("nan"),
        }
        return report
