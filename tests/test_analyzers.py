"""
Tests for the sport analyzers. The squat analyzer gets real behavioral
tests (rep counting, depth flagging) since its geometry is the simplest to
construct synthetically and verify by hand. The rest get a smoke test:
every analyzer must run without exceptions on both a normal synthetic video
and a video with zero pose detections, since analyzers run unattended in
the Streamlit app and a crash there is a bad user experience, not just a
failed assertion.
"""

import pytest

from analyzers.squat import SquatAnalyzer
from analyzers.sprint import SprintAnalyzer
from analyzers.golf import GolfSwingAnalyzer
from analyzers.football import FootballInjuryAnalyzer
from analyzers.cricket_bowling import CricketBowlingAnalyzer
from analyzers.cricket_batting import CricketBattingAnalyzer

ALL_ANALYZERS = [SquatAnalyzer, SprintAnalyzer, GolfSwingAnalyzer,
                 FootballInjuryAnalyzer, CricketBowlingAnalyzer, CricketBattingAnalyzer]


class TestSquatAnalyzer:
    def test_counts_expected_number_of_reps(self, squat_video):
        report = SquatAnalyzer().analyze(squat_video)
        # the fixture was built with 2 full squat cycles
        assert len(report.rep_boundaries) == 2

    def test_deep_squat_does_not_flag_depth(self, squat_video):
        report = SquatAnalyzer().analyze(squat_video)
        depth_flags = [f for f in report.flags if "depth" in f.message]
        assert depth_flags == []

    def test_shallow_squat_flags_depth(self, shallow_squat_video):
        report = SquatAnalyzer().analyze(shallow_squat_video)
        depth_flags = [f for f in report.flags if "depth" in f.message]
        assert len(depth_flags) >= 1

    def test_summary_rep_count_matches_boundaries(self, squat_video):
        report = SquatAnalyzer().analyze(squat_video)
        assert report.summary["rep_count"] == len(report.rep_boundaries)

    def test_symmetric_synthetic_squat_has_no_asymmetry_flag(self, squat_video):
        # the fixture places left/right joints as mirror images, so L/R
        # knee angles should match and no asymmetry flag should fire
        report = SquatAnalyzer().analyze(squat_video)
        asym_flags = [f for f in report.flags if "asymmetry" in f.message]
        assert asym_flags == []


@pytest.mark.parametrize("analyzer_cls", ALL_ANALYZERS)
def test_analyzer_runs_without_exception_on_normal_video(analyzer_cls, squat_video):
    # squat_video is used as a generic "some pose data exists" fixture here --
    # the point of this test is crash-safety across all analyzers, not that a
    # squat clip makes biomechanical sense as football/cricket footage.
    report = analyzer_cls().analyze(squat_video)
    assert report.sport == analyzer_cls.sport_name
    assert 0.0 <= report.detection_rate <= 1.0


@pytest.mark.parametrize("analyzer_cls", ALL_ANALYZERS)
def test_analyzer_handles_zero_detections_gracefully(analyzer_cls, empty_video):
    # must not raise, even though every frame has detected=False
    report = analyzer_cls().analyze(empty_video)
    assert report.sport == analyzer_cls.sport_name
