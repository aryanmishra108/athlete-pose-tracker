"""
Geometry helpers for computing joint angles from landmark positions.
Works with both 2D (x, y) and 3D (x, y, z) points -- just pass whichever
you have; numpy handles both transparently.
"""

import numpy as np


def safe_nanmean(values) -> float:
    """
    np.nanmean, but returns NaN silently instead of raising a RuntimeWarning
    when every element is NaN (which happens whenever a metric couldn't be
    computed for any frame, e.g. a body part was never detected). Analyzers
    use this instead of calling np.nanmean directly on anything that might
    be all-NaN.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at vertex `b`, formed by rays b->a and b->c, in degrees.
    E.g. for knee angle: a=hip, b=knee, c=ankle.
    """
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return float("nan")
    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def segment_angle_to_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    """
    Angle of the segment top->bottom relative to true vertical, in degrees.
    Useful for trunk/spine lean or shin angle. Works in 2D (x, y) where
    y increases downward (image coordinates).
    """
    top, bottom = np.asarray(top, dtype=float), np.asarray(bottom, dtype=float)
    vec = bottom - top
    vertical = np.array([0.0, 1.0])
    denom = np.linalg.norm(vec) * np.linalg.norm(vertical)
    if denom < 1e-8:
        return float("nan")
    cosine = np.clip(np.dot(vec, vertical) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def segment_angle_to_horizontal(a: np.ndarray, b: np.ndarray) -> float:
    """
    Signed angle of the segment a->b relative to horizontal, in degrees,
    range (-90, 90]. Useful for measuring the rotation of a left-right body
    line (e.g. the shoulder line or hip line) rather than its lean -- the
    shoulder-line and hip-line angles can be compared to approximate
    shoulder-hip separation (trunk counter-rotation).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    vec = b - a
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def smooth_series(values: list, window: int = 5) -> np.ndarray:
    """Simple moving average to smooth a noisy angle/metric time series.
    NaNs (missed detections) are ignored in the average rather than propagated."""
    arr = np.array(values, dtype=float)
    out = np.full_like(arr, np.nan)
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        window_vals = arr[lo:hi]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            out[i] = np.mean(valid)
    return out


def signed_angle_between_vectors_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Signed angle in degrees from v1 to v2, in a 2D plane, range (-180, 180].
    Useful for rotational separation between two body segments projected
    onto a plane (e.g. shoulder line vs hip line, viewed from above, to
    measure trunk counter-rotation).
    """
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return float(np.degrees(np.arctan2(cross, dot)))


def find_local_maxima(values: np.ndarray, min_prominence: float = 5.0, min_distance: int = 10):
    """Same as find_local_minima but for peaks -- implemented by negating the signal."""
    arr = np.asarray(values, dtype=float)
    return find_local_minima(-arr, min_prominence=min_prominence, min_distance=min_distance)


def find_local_minima(values: np.ndarray, min_prominence: float = 5.0, min_distance: int = 10):
    """
    Lightweight local-minima finder (no scipy dependency) -- used for rep
    counting, e.g. finding the bottom of each squat from the knee-angle
    time series. Returns indices of detected minima.
    """
    minima = []
    n = len(values)
    i = 1
    while i < n - 1:
        if np.isnan(values[i]):
            i += 1
            continue
        # walking window to find a local dip
        window_lo = max(0, i - min_distance)
        window_hi = min(n, i + min_distance + 1)
        local_window = values[window_lo:window_hi]
        local_window = local_window[~np.isnan(local_window)]
        if len(local_window) == 0:
            i += 1
            continue
        if values[i] == np.nanmin(local_window):
            neighborhood_max = np.nanmax(local_window)
            if neighborhood_max - values[i] >= min_prominence:
                if not minima or (i - minima[-1]) >= min_distance:
                    minima.append(i)
        i += 1
    return minima
