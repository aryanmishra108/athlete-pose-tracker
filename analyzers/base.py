"""
Abstract base for sport-specific form analyzers. Each analyzer takes the
full VideoLandmarks output from PoseEstimator and produces a FormReport:
a set of per-frame metrics (angle time series), rep/phase segmentation,
and rule-based flags describing form issues.

Adding a new sport = subclassing SportAnalyzer and implementing analyze().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from pose_estimation.estimator import VideoLandmarks


@dataclass
class FormFlag:
    """A single rule-based form issue detected during the movement."""
    frame_index: int
    timestamp_sec: float
    severity: str  # "info" | "warning" | "critical"
    message: str


@dataclass
class FormReport:
    sport: str
    # name -> per-frame array (NaN where undetected), aligned to video frames
    metrics: Dict[str, np.ndarray] = field(default_factory=dict)
    flags: List[FormFlag] = field(default_factory=list)
    rep_boundaries: List[int] = field(default_factory=list)  # frame indices marking reps
    summary: Dict[str, float] = field(default_factory=dict)  # headline numbers for the report
    detection_rate: float = 0.0


class SportAnalyzer(ABC):
    sport_name: str = "generic"

    @abstractmethod
    def analyze(self, video_landmarks: VideoLandmarks) -> FormReport:
        """Compute metrics, rep segmentation, and form flags for this sport."""
        raise NotImplementedError
