"""Unit tests for utils/angles.py -- the pure-math layer everything else builds on."""

import numpy as np
import pytest

from utils.angles import (
    joint_angle, segment_angle_to_vertical, segment_angle_to_horizontal,
    midpoint, euclidean_distance, smooth_series, find_local_minima,
)


class TestJointAngle:
    def test_straight_line_is_180_degrees(self):
        a, b, c = np.array([0, 0]), np.array([1, 0]), np.array([2, 0])
        assert joint_angle(a, b, c) == pytest.approx(180.0)

    def test_right_angle(self):
        a, b, c = np.array([1, 0]), np.array([0, 0]), np.array([0, 1])
        assert joint_angle(a, b, c) == pytest.approx(90.0)

    def test_degenerate_points_return_nan(self):
        a = b = c = np.array([0, 0])
        assert np.isnan(joint_angle(a, b, c))

    def test_symmetric_under_swap(self):
        a, b, c = np.array([1, 1]), np.array([0, 0]), np.array([2, -1])
        assert joint_angle(a, b, c) == pytest.approx(joint_angle(c, b, a))


class TestSegmentAngleToVertical:
    def test_perfectly_vertical_is_zero(self):
        top, bottom = np.array([0.5, 0.0]), np.array([0.5, 1.0])
        assert segment_angle_to_vertical(top, bottom) == pytest.approx(0.0, abs=1e-6)

    def test_perfectly_horizontal_is_90(self):
        top, bottom = np.array([0.0, 0.5]), np.array([1.0, 0.5])
        assert segment_angle_to_vertical(top, bottom) == pytest.approx(90.0)

    def test_45_degree_lean(self):
        top, bottom = np.array([0.0, 0.0]), np.array([1.0, 1.0])
        assert segment_angle_to_vertical(top, bottom) == pytest.approx(45.0)


class TestSegmentAngleToHorizontal:
    def test_perfectly_horizontal_is_zero(self):
        a, b = np.array([0, 0.5]), np.array([1, 0.5])
        assert segment_angle_to_horizontal(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_perfectly_vertical_is_90(self):
        a, b = np.array([0.5, 0]), np.array([0.5, 1])
        assert abs(segment_angle_to_horizontal(a, b)) == pytest.approx(90.0)


def test_midpoint():
    a, b = np.array([0, 0]), np.array([2, 4])
    np.testing.assert_allclose(midpoint(a, b), [1, 2])


def test_euclidean_distance():
    a, b = np.array([0, 0]), np.array([3, 4])
    assert euclidean_distance(a, b) == pytest.approx(5.0)


class TestSmoothSeries:
    def test_constant_series_unchanged(self):
        values = [5.0] * 10
        out = smooth_series(values, window=3)
        np.testing.assert_allclose(out, values)

    def test_smooths_out_single_spike(self):
        values = [10.0] * 5 + [1000.0] + [10.0] * 5
        out = smooth_series(values, window=5)
        # the spike frame itself should be pulled well below the raw spike value
        assert out[5] < 500.0

    def test_ignores_nans_in_window(self):
        values = [1.0, np.nan, 3.0, np.nan, 5.0]
        out = smooth_series(values, window=3)
        assert not np.isnan(out[0])


class TestFindLocalMinima:
    def test_finds_single_dip(self):
        values = np.array([10, 8, 5, 2, 5, 8, 10], dtype=float)
        minima = find_local_minima(values, min_prominence=3.0, min_distance=1)
        assert 3 in minima

    def test_flat_series_has_no_minima(self):
        values = np.full(20, 5.0)
        minima = find_local_minima(values, min_prominence=1.0, min_distance=2)
        assert minima == []

    def test_respects_min_distance(self):
        # two dips very close together -- min_distance should collapse them to one
        values = np.array([10, 2, 10, 2, 10], dtype=float)
        minima = find_local_minima(values, min_prominence=3.0, min_distance=4)
        assert len(minima) <= 1
