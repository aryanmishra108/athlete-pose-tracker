"""
Cricket fast-bowling positioning analyzer.

Focuses on the two events sports-science research consistently ties to both
performance and lower-back injury risk in fast bowlers (stress fractures of
the lumbar vertebrae are the single most common serious injury in young fast
bowlers). See docs/REFERENCES.md for the specific studies (Elliott 2000;
Bayne et al. 2016) and, importantly, the gap between what they measured (3D
marker-based motion capture) and what this analyzer measures (2D proxies
from one camera):

  - front foot contact (FFC): the front leg should brace with the knee
    fairly extended to convert run-up momentum into ball speed. A collapsing
    front knee wastes that momentum.
  - trunk lateral flexion at FFC: bowling actions with high side-on trunk
    lean at this instant (a "mixed" action) carry the strongest documented
    association with lumbar stress injury.
  - shoulder-hip separation (counter-rotation) at FFC: a rough 2D proxy for
    how "front-on" vs "side-on" the action is -- large separation combined
    with high lateral flexion is the highest-risk combination.

Film side-on or slightly front-of-side-on, full delivery stride in frame,
one delivery per clip. Front foot and bowling arm are auto-detected rather
than assumed, since the analyzer doesn't know bowler handedness in advance.
"""

import numpy as np

from analyzers.base import SportAnalyzer, FormReport, FormFlag
from pose_estimation.estimator import VideoLandmarks
from utils.angles import (
    joint_angle, segment_angle_to_vertical, segment_angle_to_horizontal,
    smooth_series, find_local_minima, midpoint,
)

FRONT_KNEE_BRACE_MIN_DEG = 150.0        # front knee angle at FFC should be at/above this (extended, braced)
LATERAL_FLEXION_RISK_DEG = 30.0         # trunk lean at FFC beyond this = high lumbar-stress-injury marker
SEPARATION_MIXED_ACTION_DEG = 30.0      # shoulder-hip separation beyond this = more front-on/mixed action


class CricketBowlingAnalyzer(SportAnalyzer):
    sport_name = "cricket_bowling"

    def analyze(self, vl: VideoLandmarks) -> FormReport:
        n = len(vl.frames)
        knee_l = np.full(n, np.nan)
        knee_r = np.full(n, np.nan)
        ankle_y_l = np.full(n, np.nan)
        ankle_y_r = np.full(n, np.nan)
        trunk_lean = np.full(n, np.nan)
        separation = np.full(n, np.nan)
        wrist_l_x = np.full(n, np.nan)
        wrist_l_y = np.full(n, np.nan)
        wrist_r_x = np.full(n, np.nan)
        wrist_r_y = np.full(n, np.nan)

        for i, f in enumerate(vl.frames):
            if not f.detected:
                continue
            try:
                hip_l, knee_pt_l, ankle_pt_l = f.get_2d("left_hip"), f.get_2d("left_knee"), f.get_2d("left_ankle")
                hip_r, knee_pt_r, ankle_pt_r = f.get_2d("right_hip"), f.get_2d("right_knee"), f.get_2d("right_ankle")
                sh_l, sh_r = f.get_2d("left_shoulder"), f.get_2d("right_shoulder")
                wr_l, wr_r = f.get_2d("left_wrist"), f.get_2d("right_wrist")

                knee_l[i] = joint_angle(hip_l, knee_pt_l, ankle_pt_l)
                knee_r[i] = joint_angle(hip_r, knee_pt_r, ankle_pt_r)
                ankle_y_l[i] = ankle_pt_l[1]
                ankle_y_r[i] = ankle_pt_r[1]

                shoulder_mid = midpoint(sh_l, sh_r)
                hip_mid = midpoint(hip_l, hip_r)
                trunk_lean[i] = segment_angle_to_vertical(shoulder_mid, hip_mid)

                shoulder_angle = segment_angle_to_horizontal(sh_l, sh_r)
                hip_angle = segment_angle_to_horizontal(hip_l, hip_r)
                separation[i] = abs(shoulder_angle - hip_angle)

                wrist_l_x[i], wrist_l_y[i] = wr_l[0], wr_l[1]
                wrist_r_x[i], wrist_r_y[i] = wr_r[0], wr_r[1]
            except (TypeError, ValueError):
                continue

        report = FormReport(sport=self.sport_name, detection_rate=vl.detection_rate)
        report.metrics = {
            "knee_angle_left": knee_l,
            "knee_angle_right": knee_r,
            "trunk_lean_from_vertical": trunk_lean,
            "shoulder_hip_separation": separation,
        }

        # Bowling arm = whichever wrist reaches the higher peak speed overall
        def wrist_speed(xs, ys):
            speed = np.full(n, np.nan)
            for i in range(1, n - 1):
                if not (np.isnan(xs[i - 1]) or np.isnan(xs[i + 1])):
                    speed[i] = np.hypot(xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1])
            return speed

        speed_l = wrist_speed(wrist_l_x, wrist_l_y)
        speed_r = wrist_speed(wrist_r_x, wrist_r_y)
        max_l = np.nanmax(speed_l) if not np.all(np.isnan(speed_l)) else -1
        max_r = np.nanmax(speed_r) if not np.all(np.isnan(speed_r)) else -1

        if max_l < 0 and max_r < 0:
            report.flags.append(FormFlag(0, 0.0, "warning", "Could not detect wrist motion to locate release."))
            return report

        bowling_speed = speed_l if max_l >= max_r else speed_r
        release_frame = int(np.nanargmax(bowling_speed))

        # Front foot contact = the footstrike (ankle at its lowest point) closest before release
        min_gap = max(3, int(vl.fps * 0.15))
        ankle_y_l_smooth = smooth_series(list(ankle_y_l), window=3)
        ankle_y_r_smooth = smooth_series(list(ankle_y_r), window=3)
        contacts_l = find_local_minima(-ankle_y_l_smooth, min_prominence=0.01, min_distance=min_gap)
        contacts_r = find_local_minima(-ankle_y_r_smooth, min_prominence=0.01, min_distance=min_gap)

        candidates = [(c, "left", knee_l) for c in contacts_l if c < release_frame] + \
                     [(c, "right", knee_r) for c in contacts_r if c < release_frame]

        if not candidates:
            report.flags.append(FormFlag(0, 0.0, "warning", "Could not detect front foot contact before release."))
            report.summary = {"release_frame": release_frame}
            return report

        ffc_frame, front_leg, front_knee_series = max(candidates, key=lambda c: c[0])
        report.rep_boundaries = [ffc_frame, release_frame]

        ts = vl.frames[ffc_frame].timestamp_sec
        front_knee_angle = front_knee_series[ffc_frame]
        lean_at_ffc = trunk_lean[ffc_frame]
        separation_at_ffc = separation[ffc_frame]

        if not np.isnan(front_knee_angle) and front_knee_angle < FRONT_KNEE_BRACE_MIN_DEG:
            report.flags.append(FormFlag(
                ffc_frame, ts, "warning",
                f"Front ({front_leg}) knee flexed to {front_knee_angle:.0f}\u00b0 at front foot contact -- "
                f"limited leg bracing, less efficient momentum transfer."
            ))

        if not np.isnan(lean_at_ffc) and lean_at_ffc > LATERAL_FLEXION_RISK_DEG:
            report.flags.append(FormFlag(
                ffc_frame, ts, "critical",
                f"Trunk lean of {lean_at_ffc:.0f}\u00b0 at front foot contact -- high lateral flexion is a "
                f"documented lumbar stress-fracture risk marker in fast bowlers (see docs/REFERENCES.md)."
            ))

        if not np.isnan(separation_at_ffc) and separation_at_ffc > SEPARATION_MIXED_ACTION_DEG:
            report.flags.append(FormFlag(
                ffc_frame, ts, "info",
                f"Shoulder-hip separation of {separation_at_ffc:.0f}\u00b0 at front foot contact suggests a "
                f"more front-on/mixed action -- worth combining with the lateral-flexion reading above."
            ))

        report.summary = {
            "front_foot_contact_frame": ffc_frame,
            "release_frame": release_frame,
            "front_leg": front_leg,
            "front_knee_angle_at_ffc": float(front_knee_angle) if not np.isnan(front_knee_angle) else float("nan"),
            "trunk_lateral_flexion_at_ffc": float(lean_at_ffc) if not np.isnan(lean_at_ffc) else float("nan"),
            "shoulder_hip_separation_at_ffc": float(separation_at_ffc) if not np.isnan(separation_at_ffc) else float("nan"),
        }
        return report
