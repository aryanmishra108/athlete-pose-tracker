"""
Cricket batting positioning analyzer.

Batting coaching is dominated by two positioning cues: "head over the ball"
(minimal lateral head movement away from the stance base as the shot is
played -- the classic marker of a stable base) and balanced weight transfer
through the shot rather than falling away. This analyzer approximates both
from a single face-on or slightly-angled video of one shot.

Tracks:
  - head lateral deviation: horizontal distance of the nose from the stance
    base (ankle midpoint), normalized by shoulder width -- large deviation
    at the shot moment means the head has moved away from the ball
  - hip lateral position: weight transfer/sway across the shot
  - trunk lean from vertical: balance, especially at shot completion

The "shot moment" is located as the peak combined wrist speed (bat swing),
the same technique used for impact detection in the golf analyzer.
"""

import numpy as np

from analyzers.base import SportAnalyzer, FormReport, FormFlag
from pose_estimation.estimator import VideoLandmarks
from utils.angles import segment_angle_to_vertical, smooth_series, midpoint, euclidean_distance

HEAD_DEVIATION_RATIO_THRESHOLD = 0.35   # |nose_x - ankle_mid_x| / shoulder_width at the shot moment
HIP_SWAY_RATIO_THRESHOLD = 0.25         # hip lateral drift address->shot / shoulder width
BALANCE_LEAN_THRESHOLD_DEG = 30.0       # trunk lean at shot completion beyond this = falling off balance


class CricketBattingAnalyzer(SportAnalyzer):
    sport_name = "cricket_batting"

    def analyze(self, vl: VideoLandmarks) -> FormReport:
        n = len(vl.frames)
        head_dev = np.full(n, np.nan)
        hip_x = np.full(n, np.nan)
        trunk_lean = np.full(n, np.nan)
        wrist_x = np.full(n, np.nan)
        wrist_y = np.full(n, np.nan)

        for i, f in enumerate(vl.frames):
            if not f.detected:
                continue
            try:
                nose = f.get_2d("nose")
                ankle_l, ankle_r = f.get_2d("left_ankle"), f.get_2d("right_ankle")
                hip_l, hip_r = f.get_2d("left_hip"), f.get_2d("right_hip")
                sh_l, sh_r = f.get_2d("left_shoulder"), f.get_2d("right_shoulder")
                wr_l, wr_r = f.get_2d("left_wrist"), f.get_2d("right_wrist")

                ankle_mid = midpoint(ankle_l, ankle_r)
                hip_mid = midpoint(hip_l, hip_r)
                shoulder_mid = midpoint(sh_l, sh_r)
                shoulder_width = euclidean_distance(sh_l, sh_r)

                if shoulder_width > 1e-6:
                    head_dev[i] = abs(nose[0] - ankle_mid[0]) / shoulder_width

                hip_x[i] = hip_mid[0]
                trunk_lean[i] = segment_angle_to_vertical(shoulder_mid, hip_mid)

                wrist_mid = midpoint(wr_l, wr_r)
                wrist_x[i], wrist_y[i] = wrist_mid[0], wrist_mid[1]
            except (TypeError, ValueError):
                continue

        report = FormReport(sport=self.sport_name, detection_rate=vl.detection_rate)
        report.metrics = {
            "head_lateral_deviation": head_dev,
            "hip_lateral_position": hip_x,
            "trunk_lean_from_vertical": trunk_lean,
        }

        valid_idx = np.where(~np.isnan(wrist_x))[0]
        if len(valid_idx) < 5:
            report.flags.append(FormFlag(0, 0.0, "warning", "Not enough detected frames to locate the shot."))
            return report

        address_frame = int(valid_idx[0])

        wrist_speed = np.full(n, np.nan)
        for i in range(1, n - 1):
            if not (np.isnan(wrist_x[i - 1]) or np.isnan(wrist_x[i + 1])):
                wrist_speed[i] = np.hypot(wrist_x[i + 1] - wrist_x[i - 1], wrist_y[i + 1] - wrist_y[i - 1])

        if np.all(np.isnan(wrist_speed)):
            report.flags.append(FormFlag(0, 0.0, "warning", "Could not detect bat-swing motion."))
            return report

        shot_frame = int(np.nanargmax(wrist_speed))
        completion_frame = int(valid_idx[-1])
        report.rep_boundaries = [address_frame, shot_frame, completion_frame]

        ts_shot = vl.frames[shot_frame].timestamp_sec
        dev_at_shot = head_dev[shot_frame]
        if not np.isnan(dev_at_shot) and dev_at_shot > HEAD_DEVIATION_RATIO_THRESHOLD:
            report.flags.append(FormFlag(
                shot_frame, ts_shot, "warning",
                f"Head moved noticeably off the base at the shot (deviation ratio {dev_at_shot:.2f}) -- "
                f"'head over the ball' is compromised."
            ))

        if not (np.isnan(hip_x[address_frame]) or np.isnan(hip_x[shot_frame])):
            shoulder_width_ref = euclidean_distance(
                vl.frames[address_frame].get_2d("left_shoulder"), vl.frames[address_frame].get_2d("right_shoulder")
            )
            if shoulder_width_ref > 1e-6:
                sway_ratio = abs(hip_x[shot_frame] - hip_x[address_frame]) / shoulder_width_ref
                if sway_ratio > HIP_SWAY_RATIO_THRESHOLD:
                    report.flags.append(FormFlag(
                        shot_frame, ts_shot, "info",
                        f"Notable hip lateral shift from stance to shot (ratio {sway_ratio:.2f}) -- "
                        f"check whether this is intentional weight transfer or a loss of base."
                    ))
            else:
                sway_ratio = float("nan")
        else:
            sway_ratio = float("nan")

        lean_at_completion = trunk_lean[completion_frame]
        if not np.isnan(lean_at_completion) and lean_at_completion > BALANCE_LEAN_THRESHOLD_DEG:
            report.flags.append(FormFlag(
                completion_frame, vl.frames[completion_frame].timestamp_sec, "warning",
                f"Trunk lean of {lean_at_completion:.0f}\u00b0 at shot completion -- possible loss of balance through the shot."
            ))

        report.summary = {
            "address_frame": address_frame,
            "shot_frame": shot_frame,
            "head_deviation_at_shot": float(dev_at_shot) if not np.isnan(dev_at_shot) else float("nan"),
            "hip_sway_ratio": float(sway_ratio) if not np.isnan(sway_ratio) else float("nan"),
            "trunk_lean_at_completion": float(lean_at_completion) if not np.isnan(lean_at_completion) else float("nan"),
        }
        return report
