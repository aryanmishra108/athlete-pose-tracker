"""
Visualization helpers:
  - draw_skeleton_overlay: annotate a video with the pose skeleton + a live
    metric readout, writing a new video file
  - plot_metric_timeseries: matplotlib chart of one or more metrics over time,
    with rep boundaries and flags marked
  - plot_3d_skeleton_frame: plotly 3D scatter of a single frame's world
    landmarks, for inspecting the monocular 3D estimate
"""

from typing import Dict, List, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pose_estimation.estimator import VideoLandmarks, LANDMARK_NAMES, POSE_CONNECTIONS
from analyzers.base import FormReport

# Shared palette (see theme.py for the full rationale) -- kept here too since
# these are drawn with cv2/matplotlib/plotly rather than CSS.
BG_VOID = "#0F1316"
BG_PANEL = "#171D22"
BORDER_HAIR = "#2B343B"
TEXT_PRIMARY = "#ECEFF2"
TEXT_SECONDARY = "#8A97A3"
AMBER = "#FFB627"
CYAN = "#47D6E0"
RED = "#FF5C5C"

# cv2 uses BGR, not RGB
_CYAN_BGR = (224, 214, 71)
_AMBER_BGR = (39, 182, 255)
_PANEL_BGR = (34, 29, 23)


def draw_skeleton_overlay(video_path: str, vl: VideoLandmarks, output_path: str,
                           live_metric: Optional[np.ndarray] = None,
                           live_metric_label: str = "") -> str:
    """
    Re-render the input video with the pose skeleton drawn on every frame,
    plus an optional live-updating metric value (e.g. current knee angle)
    in the corner. Returns the output path.
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, vl.fps, (vl.width, vl.height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(vl.frames):
            break

        fl = vl.frames[frame_idx]
        if fl.detected:
            pts = fl.landmarks_2d
            for (start_idx, end_idx) in POSE_CONNECTIONS:
                x1, y1 = int(pts[start_idx, 0] * vl.width), int(pts[start_idx, 1] * vl.height)
                x2, y2 = int(pts[end_idx, 0] * vl.width), int(pts[end_idx, 1] * vl.height)
                cv2.line(frame, (x1, y1), (x2, y2), _CYAN_BGR, 2)
            for i in range(pts.shape[0]):
                x, y = int(pts[i, 0] * vl.width), int(pts[i, 1] * vl.height)
                cv2.circle(frame, (x, y), 3, _AMBER_BGR, -1)

        if live_metric is not None and frame_idx < len(live_metric) and not np.isnan(live_metric[frame_idx]):
            text = f"{live_metric_label.upper()}: {live_metric[frame_idx]:.0f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (14, 14), (14 + tw + 16, 14 + th + 20), _PANEL_BGR, -1)
            cv2.putText(frame, text, (22, 14 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _AMBER_BGR, 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path


def plot_metric_timeseries(report: FormReport, metric_names: List[str], fps: float, title: str = ""):
    """Return a matplotlib Figure plotting the requested metrics over time, styled to match the app theme."""
    line_colors = [CYAN, AMBER, "#B39DDB", "#7CFFB2"]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)

    for i, name in enumerate(metric_names):
        if name not in report.metrics:
            continue
        values = report.metrics[name]
        t = np.arange(len(values)) / fps
        ax.plot(t, values, label=name.replace("_", " "), color=line_colors[i % len(line_colors)], linewidth=1.6)

    for b in report.rep_boundaries:
        ax.axvline(b / fps, color=TEXT_SECONDARY, linestyle="--", alpha=0.4, linewidth=1)

    severity_color = {"info": CYAN, "warning": AMBER, "critical": RED}
    for flag in report.flags:
        ax.axvline(flag.timestamp_sec, color=severity_color.get(flag.severity, TEXT_SECONDARY), alpha=0.7, linewidth=1)

    for spine in ax.spines.values():
        spine.set_color(BORDER_HAIR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xlabel("Time (s)", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("Angle (degrees)", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(title or "Metrics over time", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left")
    legend = ax.legend(loc="upper right", fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER_HAIR)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    ax.grid(alpha=0.15, color=TEXT_SECONDARY)
    fig.tight_layout()
    return fig


def plot_3d_skeleton_frame(vl: VideoLandmarks, frame_index: int):
    """Return a plotly Figure showing the 3D world-landmark skeleton for one frame, styled to match the app theme."""
    fl = vl.frames[frame_index]
    if not fl.detected:
        return go.Figure(layout=dict(paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL))

    pts = fl.landmarks_3d  # (33, 3): x, y, z in meters, hip-relative
    xs, ys, zs = pts[:, 0], pts[:, 2], -pts[:, 1]  # reorient so "up" looks up in the plot

    edge_x, edge_y, edge_z = [], [], []
    for (i, j) in POSE_CONNECTIONS:
        edge_x += [xs[i], xs[j], None]
        edge_y += [ys[i], ys[j], None]
        edge_z += [zs[i], zs[j], None]

    axis_style = dict(
        backgroundcolor=BG_PANEL, gridcolor=BORDER_HAIR, showbackground=True,
        zerolinecolor=BORDER_HAIR, color=TEXT_SECONDARY,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                                line=dict(color=CYAN, width=4), name="skeleton"))
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                                marker=dict(size=4, color=AMBER), name="joints",
                                text=LANDMARK_NAMES, hoverinfo="text"))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m)", **axis_style),
            yaxis=dict(title="depth (m)", **axis_style),
            zaxis=dict(title="height (m)", **axis_style),
            aspectmode="data",
        ),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_SECONDARY, family="IBM Plex Mono, monospace"),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        title=dict(text=f"FRAME {frame_index} \u00b7 T={fl.timestamp_sec:.2f}S", font=dict(color=TEXT_PRIMARY, size=13)),
    )
    return fig
