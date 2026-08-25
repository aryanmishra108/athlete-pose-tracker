"""Shared pytest fixtures. Builds synthetic FrameLandmarks/VideoLandmarks
without needing MediaPipe installed or a real video file, so the analyzer
logic can be tested in isolation from the pose-estimation backend."""

import numpy as np
import pytest

from pose_estimation.estimator import FrameLandmarks, VideoLandmarks, LANDMARK_NAMES

LM_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


def _blank_pose():
    return np.zeros((33, 3))


def make_squat_cycle_video(n_frames=90, fps=30.0, cycles=2, min_knee_deg=80, max_knee_deg=170):
    """
    A synthetic video where hip/knee/ankle positions trace out `cycles` full
    squat reps. Angles are placed geometrically (not just made up numbers)
    so joint_angle() computed from these points actually lands near the
    target knee angle at each rep's bottom -- this tests the real geometry
    pipeline, not a mocked-out shortcut.
    """
    frames = []
    for i in range(n_frames):
        t = i / n_frames
        # Oscillate knee bend between two angles across `cycles` reps
        phase = (np.sin(2 * np.pi * cycles * t) + 1) / 2  # 0..1
        knee_deg = max_knee_deg - phase * (max_knee_deg - min_knee_deg)
        knee_rad = np.radians(knee_deg)

        pts = _blank_pose()
        for side, x_off in [("left", -0.06), ("right", 0.06)]:
            hip = np.array([0.5 + x_off, 0.45])
            ankle = np.array([0.5 + x_off, 0.90])
            # place knee so the hip-knee-ankle angle equals knee_deg, bending forward in x
            shank_len = 0.22
            bend = (np.pi - knee_rad) / 2
            knee = ankle + shank_len * np.array([np.sin(bend), -np.cos(bend)])
            shoulder = np.array([0.5 + x_off, 0.15])
            wrist = np.array([0.5 + x_off, 0.30])

            pts[LM_INDEX[f"{side}_hip"]] = [hip[0], hip[1], 0.95]
            pts[LM_INDEX[f"{side}_knee"]] = [knee[0], knee[1], 0.95]
            pts[LM_INDEX[f"{side}_ankle"]] = [ankle[0], ankle[1], 0.95]
            pts[LM_INDEX[f"{side}_shoulder"]] = [shoulder[0], shoulder[1], 0.95]
            pts[LM_INDEX[f"{side}_wrist"]] = [wrist[0], wrist[1], 0.95]
        pts[LM_INDEX["nose"]] = [0.5, 0.10, 0.95]

        fl = FrameLandmarks(
            frame_index=i, timestamp_sec=i / fps,
            landmarks_2d=pts.copy(), landmarks_3d=pts[:, :3].copy(), detected=True,
        )
        frames.append(fl)

    return VideoLandmarks(fps=fps, width=640, height=480, frames=frames)


@pytest.fixture
def squat_video():
    return make_squat_cycle_video(n_frames=90, cycles=2, min_knee_deg=80, max_knee_deg=170)


@pytest.fixture
def shallow_squat_video():
    """A 'squat' that never gets below 150 degrees knee angle -- should trigger a depth flag."""
    return make_squat_cycle_video(n_frames=90, cycles=2, min_knee_deg=150, max_knee_deg=175)


@pytest.fixture
def empty_video():
    """A video with no pose ever detected, to test graceful degradation."""
    frames = [
        FrameLandmarks(frame_index=i, timestamp_sec=i / 30.0, detected=False)
        for i in range(30)
    ]
    return VideoLandmarks(fps=30.0, width=640, height=480, frames=frames)
