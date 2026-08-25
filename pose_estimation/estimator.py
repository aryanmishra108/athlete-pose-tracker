"""
Wraps MediaPipe's PoseLandmarker (Tasks API) to extract per-frame 2D
(pixel-space) and 3D (world-space, metric, hip-relative) landmarks from a
video. This is the single point of contact with MediaPipe -- every analyzer
downstream consumes the plain FrameLandmarks objects produced here, not
MediaPipe types directly, so the rest of the codebase doesn't care which
pose backend is used.

NOTE ON API VERSION: MediaPipe's older "Solutions" API (mp.solutions.pose)
has been removed from current PyPI releases (see
https://github.com/google-ai-edge/mediapipe/issues/6200 and similar --
this affects everyone on a recent mediapipe version, not just this project).
This module uses the actively-maintained Tasks API
(mediapipe.tasks.vision.PoseLandmarker) instead, which requires a small
model file downloaded once and cached locally (see download_pose_model()).

Note on 3D: MediaPipe Pose is monocular (single RGB camera), so the 3D world
landmarks it produces are a *learned estimate*, not a triangulated
measurement -- accurate enough for relative joint angles and symmetry checks,
but depth values (Z) are noisier than X/Y and shouldn't be trusted for
absolute measurements like true stride length in meters.
"""

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np

# The 33 MediaPipe Pose landmarks, in the fixed order the model always
# outputs them. This topology is shared by both the old Solutions API and
# the current Tasks API, but since Solutions may not be importable at all,
# we hardcode the names here rather than deriving them from mp.solutions.
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Skeleton edges for drawing, matching the standard MediaPipe Pose topology.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
MODEL_FILENAMES = {0: "pose_landmarker_lite.task", 1: "pose_landmarker_full.task", 2: "pose_landmarker_heavy.task"}
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "athlete_pose_tracker")


def download_pose_model(model_complexity: int = 1) -> str:
    """
    Ensure the PoseLandmarker model file for the given complexity is present
    locally, downloading it from Google's model repository on first use.
    Returns the local file path. Requires internet access on first call only
    -- the model is cached under ~/.cache/athlete_pose_tracker after that.
    """
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    filename = MODEL_FILENAMES[model_complexity]
    local_path = os.path.join(MODEL_CACHE_DIR, filename)
    if not os.path.exists(local_path):
        url = MODEL_URLS[model_complexity]
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not download the pose model from {url}. "
                f"Check your internet connection, or manually download the file and "
                f"place it at {local_path}. Original error: {e}"
            )
    return local_path


@dataclass
class FrameLandmarks:
    """Pose landmarks for a single video frame."""
    frame_index: int
    timestamp_sec: float
    # (33, 3) array of (x, y, visibility) in normalized pixel space [0,1]
    landmarks_2d: Optional[np.ndarray] = None
    # (33, 3) array of (x, y, z) in meters, hip-center-relative
    landmarks_3d: Optional[np.ndarray] = None
    detected: bool = False

    def get_2d(self, name: str) -> Optional[np.ndarray]:
        """Return the (x, y) pixel-normalized position of a named landmark."""
        if not self.detected:
            return None
        idx = LANDMARK_NAMES.index(name)
        return self.landmarks_2d[idx, :2]

    def get_3d(self, name: str) -> Optional[np.ndarray]:
        """Return the (x, y, z) world position (meters) of a named landmark."""
        if not self.detected:
            return None
        idx = LANDMARK_NAMES.index(name)
        return self.landmarks_3d[idx, :3]

    def visibility(self, name: str) -> float:
        if not self.detected:
            return 0.0
        idx = LANDMARK_NAMES.index(name)
        return float(self.landmarks_2d[idx, 2])


@dataclass
class VideoLandmarks:
    """All per-frame landmarks for a processed video, plus video metadata."""
    fps: float
    width: int
    height: int
    frames: list = field(default_factory=list)  # list[FrameLandmarks]

    @property
    def detection_rate(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f.detected for f in self.frames) / len(self.frames)


class PoseEstimator:
    """Runs MediaPipe's PoseLandmarker over a video file and collects landmarks per frame."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        if model_complexity not in (0, 1, 2):
            raise ValueError("model_complexity must be 0 (lite), 1 (full), or 2 (heavy)")
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def process_video(self, video_path: str, progress_callback=None) -> VideoLandmarks:
        """
        Run pose estimation on every frame of a video file.

        Args:
            video_path: path to a video file readable by OpenCV.
            progress_callback: optional fn(frame_idx, total_frames) called per frame,
                useful for driving a UI progress bar.

        Returns:
            VideoLandmarks with one FrameLandmarks entry per video frame.
        """
        model_path = download_pose_model(self.model_complexity)

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            num_poses=1,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        result = VideoLandmarks(fps=fps, width=width, height=height)

        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)
                pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                fl = FrameLandmarks(
                    frame_index=frame_idx,
                    timestamp_sec=frame_idx / fps,
                )

                if pose_result.pose_landmarks and pose_result.pose_world_landmarks:
                    # num_poses=1, so take the first (only) detected person
                    landmarks_2d = pose_result.pose_landmarks[0]
                    landmarks_3d = pose_result.pose_world_landmarks[0]
                    fl.detected = True
                    fl.landmarks_2d = np.array(
                        [[lm.x, lm.y, lm.visibility] for lm in landmarks_2d]
                    )
                    fl.landmarks_3d = np.array(
                        [[lm.x, lm.y, lm.z] for lm in landmarks_3d]
                    )

                result.frames.append(fl)
                frame_idx += 1

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        cap.release()
        return result
