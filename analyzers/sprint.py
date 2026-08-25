"""
Sprint form analyzer.

Tracks, per frame:
  - trunk lean from vertical
  - hip flexion angle per leg (shoulder-hip-knee) -- low values indicate strong
    knee drive during the swing phase, a key front-side mechanics marker
  - foot-strike events per leg (approximated as local maxima of ankle
    vertical pixel position, i.e. the foot's lowest point in frame)
  - overstride ratio at each foot strike: horizontal distance between the
    landing ankle and the hip midpoint, normalized by torso length

Assumes a roughly side-on (sagittal plane) camera view, which is standard
for sprint mechanics filming.
"""

import numpy as np

from analyzers.base import SportAnalyzer, FormReport, FormFlag
from pose_estimation.estimator import VideoLandmarks
from utils.angles import joint_angle, segment_angle_to_vertical, smooth_series, find_local_minima, midpoint, euclidean_distance, safe_nanmean

KNEE_DRIVE_HIP_ANGLE_MAX = 100.0     # hip angle should dip below this at peak knee drive
EXCESSIVE_TRUNK_LEAN_DEG = 25.0      # average trunk lean beyond this suggests over-leaning
OVERSTRIDE_RATIO_THRESHOLD = 0.45    # ankle-to-hip horizontal distance / torso length


class SprintAnalyzer(SportAnalyzer):
    sport_name = "sprint"

    def analyze(self, vl: VideoLandmarks) -> FormReport:
        n = len(vl.frames)
        trunk_lean = np.full(n, np.nan)
        hip_flex_l = np.full(n, np.nan)
        hip_flex_r = np.full(n, np.nan)
        ankle_y_l = np.full(n, np.nan)
        ankle_y_r = np.full(n, np.nan)
        overstride_l = np.full(n, np.nan)
        overstride_r = np.full(n, np.nan)

        for i, f in enumerate(vl.frames):
            if not f.detected:
                continue
            try:
                sh_l, sh_r = f.get_2d("left_shoulder"), f.get_2d("right_shoulder")
                hip_l, hip_r = f.get_2d("left_hip"), f.get_2d("right_hip")
                knee_l, knee_r = f.get_2d("left_knee"), f.get_2d("right_knee")
                ankle_l, ankle_r = f.get_2d("left_ankle"), f.get_2d("right_ankle")

                shoulder_mid = midpoint(sh_l, sh_r)
                hip_mid = midpoint(hip_l, hip_r)
                trunk_lean[i] = segment_angle_to_vertical(shoulder_mid, hip_mid)
                torso_len = euclidean_distance(shoulder_mid, hip_mid)

                hip_flex_l[i] = joint_angle(sh_l, hip_l, knee_l)
                hip_flex_r[i] = joint_angle(sh_r, hip_r, knee_r)

                ankle_y_l[i] = ankle_l[1]
                ankle_y_r[i] = ankle_r[1]

                if torso_len > 1e-6:
                    overstride_l[i] = abs(ankle_l[0] - hip_mid[0]) / torso_len
                    overstride_r[i] = abs(ankle_r[0] - hip_mid[0]) / torso_len
            except (TypeError, ValueError):
                continue

        min_stride_frames = max(3, int(vl.fps * 0.25))  # sprint strides are fast; ~4+ steps/sec typical

        # Footstrike = ankle at its lowest point in frame => local MAX of y (image y grows downward)
        contacts_l = find_local_minima(-smooth_series(list(ankle_y_l), window=3), min_prominence=0.005, min_distance=min_stride_frames)
        contacts_r = find_local_minima(-smooth_series(list(ankle_y_r), window=3), min_prominence=0.005, min_distance=min_stride_frames)

        # Knee drive peaks = local MIN of hip flexion angle (thigh most raised)
        drive_l = find_local_minima(smooth_series(list(hip_flex_l), window=3), min_prominence=10.0, min_distance=min_stride_frames)
        drive_r = find_local_minima(smooth_series(list(hip_flex_r), window=3), min_prominence=10.0, min_distance=min_stride_frames)

        report = FormReport(sport=self.sport_name, detection_rate=vl.detection_rate)
        report.metrics = {
            "trunk_lean_from_vertical": trunk_lean,
            "hip_flexion_left": hip_flex_l,
            "hip_flexion_right": hip_flex_r,
            "overstride_ratio_left": overstride_l,
            "overstride_ratio_right": overstride_r,
        }
        report.rep_boundaries = sorted(contacts_l + contacts_r)

        for side, contacts, overstride in [("left", contacts_l, overstride_l), ("right", contacts_r, overstride_r)]:
            for c in contacts:
                ts = vl.frames[c].timestamp_sec
                ratio = overstride[c]
                if not np.isnan(ratio) and ratio > OVERSTRIDE_RATIO_THRESHOLD:
                    report.flags.append(FormFlag(
                        c, ts, "warning",
                        f"Possible overstride on {side} foot strike (ratio {ratio:.2f})."
                    ))

        for side, drives, hip_flex in [("left", drive_l, hip_flex_l), ("right", drive_r, hip_flex_r)]:
            for d in drives:
                ts = vl.frames[d].timestamp_sec
                angle = hip_flex[d]
                if not np.isnan(angle) and angle > KNEE_DRIVE_HIP_ANGLE_MAX:
                    report.flags.append(FormFlag(
                        d, ts, "info",
                        f"Limited {side} knee drive at this stride (hip flexion {angle:.0f}\u00b0)."
                    ))

        avg_lean = safe_nanmean(trunk_lean) if n else float("nan")
        if not np.isnan(avg_lean) and avg_lean > EXCESSIVE_TRUNK_LEAN_DEG:
            report.flags.append(FormFlag(
                0, 0.0, "warning",
                f"Average trunk lean of {avg_lean:.0f}\u00b0 is higher than typical for max-velocity sprinting."
            ))

        contact_count = len(contacts_l) + len(contacts_r)
        duration = vl.frames[-1].timestamp_sec if vl.frames else 0.0
        cadence_spm = (contact_count / duration * 60.0) if duration > 0 else float("nan")

        report.summary = {
            "estimated_cadence_steps_per_min": cadence_spm,
            "avg_trunk_lean": avg_lean,
            "avg_overstride_ratio_left": safe_nanmean(overstride_l) if n else float("nan"),
            "avg_overstride_ratio_right": safe_nanmean(overstride_r) if n else float("nan"),
            "detected_foot_strikes": contact_count,
        }
        return report
