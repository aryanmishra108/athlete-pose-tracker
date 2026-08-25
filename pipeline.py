"""
Top-level orchestration: video file -> pose landmarks -> sport analyzer -> FormReport.
This is the single function the Streamlit app (or a CLI/notebook) calls.
"""

from analyzers.base import FormReport
from analyzers.squat import SquatAnalyzer
from analyzers.sprint import SprintAnalyzer
from analyzers.golf import GolfSwingAnalyzer
from analyzers.football import FootballInjuryAnalyzer
from analyzers.cricket_bowling import CricketBowlingAnalyzer
from analyzers.cricket_batting import CricketBattingAnalyzer
from pose_estimation.estimator import PoseEstimator, VideoLandmarks

ANALYZERS = {
    "squat": SquatAnalyzer,
    "sprint": SprintAnalyzer,
    "golf": GolfSwingAnalyzer,
    "football": FootballInjuryAnalyzer,
    "cricket_bowling": CricketBowlingAnalyzer,
    "cricket_batting": CricketBattingAnalyzer,
}


def run_analysis(video_path: str, sport: str, progress_callback=None,
                  model_complexity: int = 1) -> tuple[VideoLandmarks, FormReport]:
    """
    Run the full pipeline on a video file for the given sport.

    Args:
        video_path: path to the uploaded video.
        sport: one of "squat", "sprint", "golf", "football", "cricket_bowling", "cricket_batting".
        progress_callback: optional fn(frame_idx, total_frames) for UI progress.
        model_complexity: MediaPipe Pose model complexity (0=fastest, 2=most accurate).

    Returns:
        (video_landmarks, form_report)
    """
    if sport not in ANALYZERS:
        raise ValueError(f"Unknown sport '{sport}'. Choose from: {list(ANALYZERS)}")

    estimator = PoseEstimator(model_complexity=model_complexity)
    video_landmarks = estimator.process_video(video_path, progress_callback=progress_callback)

    analyzer = ANALYZERS[sport]()
    report = analyzer.analyze(video_landmarks)

    return video_landmarks, report
