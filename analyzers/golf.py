"""
Golf swing form analyzer.

True swing-plane and shoulder-rotation analysis really needs multiple
synced camera angles (face-on + down-the-line); from a single 2D video we
approximate the two faults that are most visible from one camera:

  - "early extension": spine angle straightens up (stands up out of posture)
    between the top of the backswing and impact
  - "hip sway": lateral hip translation during the backswing instead of
    rotation around a stable center -- visible as horizontal hip drift
    in a face-on view

Swing phases are located automatically from wrist trajectory: address
(first stable frame), top of backswing (wrist at its highest point),
and impact (peak wrist speed after the top).
"""

import numpy as np

from analyzers.base import SportAnalyzer, FormReport, FormFlag
from pose_estimation.estimator import VideoLandmarks
from utils.angles import segment_angle_to_vertical, smooth_series, midpoint, euclidean_distance

EARLY_EXTENSION_THRESHOLD_DEG = 12.0   # spine angle change from top-of-swing to impact
HIP_SWAY_RATIO_THRESHOLD = 0.15        # lateral hip drift / shoulder width


class GolfSwingAnalyzer(SportAnalyzer):
    sport_name = "golf"

    def analyze(self, vl: VideoLandmarks) -> FormReport:
        n = len(vl.frames)
        spine_angle = np.full(n, np.nan)
        wrist_y = np.full(n, np.nan)
        wrist_x = np.full(n, np.nan)
        hip_x = np.full(n, np.nan)
        shoulder_width = np.full(n, np.nan)

        for i, f in enumerate(vl.frames):
            if not f.detected:
                continue
            try:
                sh_l, sh_r = f.get_2d("left_shoulder"), f.get_2d("right_shoulder")
                hip_l, hip_r = f.get_2d("left_hip"), f.get_2d("right_hip")
                wr_l, wr_r = f.get_2d("left_wrist"), f.get_2d("right_wrist")

                shoulder_mid = midpoint(sh_l, sh_r)
                hip_mid = midpoint(hip_l, hip_r)
                spine_angle[i] = segment_angle_to_vertical(shoulder_mid, hip_mid)
                shoulder_width[i] = euclidean_distance(sh_l, sh_r)

                wrist_mid = midpoint(wr_l, wr_r)
                wrist_y[i] = wrist_mid[1]
                wrist_x[i] = wrist_mid[0]
                hip_x[i] = hip_mid[0]
            except (TypeError, ValueError):
                continue

        report = FormReport(sport=self.sport_name, detection_rate=vl.detection_rate)
        report.metrics = {
            "spine_angle_from_vertical": spine_angle,
            "wrist_height": wrist_y,
            "hip_lateral_position": hip_x,
        }

        valid_idx = np.where(~np.isnan(wrist_y))[0]
        if len(valid_idx) < 5:
            report.flags.append(FormFlag(0, 0.0, "warning", "Not enough detected frames to segment the swing."))
            return report

        address_frame = int(valid_idx[0])
        wrist_y_smooth = smooth_series(list(wrist_y), window=5)

        # Top of backswing = highest wrist point (min y) in the first ~65% of the clip
        search_end = int(valid_idx[0] + 0.65 * (valid_idx[-1] - valid_idx[0]))
        search_slice = wrist_y_smooth[address_frame:search_end + 1]
        if np.all(np.isnan(search_slice)):
            report.flags.append(FormFlag(0, 0.0, "warning", "Could not locate top of backswing."))
            return report
        top_frame = address_frame + int(np.nanargmin(search_slice))

        # Impact = peak wrist speed after the top of backswing
        wrist_speed = np.full(n, np.nan)
        for i in range(top_frame + 1, min(n - 1, valid_idx[-1])):
            if not (np.isnan(wrist_x[i + 1]) or np.isnan(wrist_x[i - 1])):
                dx = wrist_x[i + 1] - wrist_x[i - 1]
                dy = wrist_y[i + 1] - wrist_y[i - 1]
                wrist_speed[i] = np.hypot(dx, dy)
        post_top_speed = wrist_speed[top_frame + 1:]
        impact_frame = top_frame + 1 + int(np.nanargmax(post_top_speed)) if not np.all(np.isnan(post_top_speed)) else top_frame

        report.rep_boundaries = [address_frame, top_frame, impact_frame]

        # Early extension check
        spine_top = spine_angle[top_frame]
        spine_impact = spine_angle[impact_frame]
        if not (np.isnan(spine_top) or np.isnan(spine_impact)):
            delta = spine_top - spine_impact  # positive = straightened up (lost posture)
            if delta > EARLY_EXTENSION_THRESHOLD_DEG:
                report.flags.append(FormFlag(
                    impact_frame, vl.frames[impact_frame].timestamp_sec, "warning",
                    f"Possible early extension: spine angle changed {delta:.0f}\u00b0 from top of swing to impact."
                ))

        # Hip sway check (comparing address to top-of-backswing hip position)
        if not (np.isnan(hip_x[address_frame]) or np.isnan(hip_x[top_frame]) or np.isnan(shoulder_width[address_frame])):
            sway = abs(hip_x[top_frame] - hip_x[address_frame])
            ratio = sway / shoulder_width[address_frame] if shoulder_width[address_frame] > 1e-6 else np.nan
            if not np.isnan(ratio) and ratio > HIP_SWAY_RATIO_THRESHOLD:
                report.flags.append(FormFlag(
                    top_frame, vl.frames[top_frame].timestamp_sec, "warning",
                    f"Possible hip sway during backswing (lateral drift ratio {ratio:.2f})."
                ))
        else:
            ratio = float("nan")

        report.summary = {
            "address_frame": address_frame,
            "top_of_backswing_frame": top_frame,
            "impact_frame": impact_frame,
            "spine_angle_change_top_to_impact": float(spine_top - spine_impact) if not (np.isnan(spine_top) or np.isnan(spine_impact)) else float("nan"),
            "hip_sway_ratio": float(ratio) if not np.isnan(ratio) else float("nan"),
        }
        return report
