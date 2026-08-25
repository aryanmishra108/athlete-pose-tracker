"""
Streamlit UI for the athlete pose/form analyzer.

Run with:
    streamlit run app.py

Design note: analysis results are cached in st.session_state, keyed to the
uploaded file + sport + model settings. Streamlit re-runs this whole script
top-to-bottom on *every* widget interaction (moving the 3D-frame slider,
switching tabs, etc.) -- without this cache, each of those interactions
would silently skip re-displaying results (since they'd have been gated
behind `if st.button("Analyze")`, which is only True on the run where you
actually clicked it) or, worse, force a full re-analysis. Caching means
pose estimation runs exactly once per Analyze click, and everything else
(browsing frames, re-reading the report) is instant.

Visual identity lives in theme.py -- see that file for the full rationale.
"""

import hashlib
import os
import tempfile

import streamlit as st

from pipeline import run_analysis, ANALYZERS
from utils.visualization import draw_skeleton_overlay, plot_metric_timeseries, plot_3d_skeleton_frame
from theme import inject_theme, render_header, render_eyebrow, render_flag_card

st.set_page_config(page_title="Athlete Form Analyzer", layout="wide")

SPORT_METRIC_PRESETS = {
    "squat": ["knee_angle_left", "knee_angle_right", "trunk_lean_from_vertical"],
    "sprint": ["hip_flexion_left", "hip_flexion_right", "trunk_lean_from_vertical"],
    "golf": ["spine_angle_from_vertical", "hip_lateral_position"],
    "football": ["knee_angle_left", "knee_angle_right", "trunk_lean_from_vertical"],
    "cricket_bowling": ["knee_angle_left", "knee_angle_right", "trunk_lean_from_vertical"],
    "cricket_batting": ["head_lateral_deviation", "trunk_lean_from_vertical"],
}

SPORT_FILMING_TIPS = {
    "squat": "Film from the side or front, full body in frame.",
    "sprint": "Film from the side (sagittal view).",
    "golf": "Film face-on, full swing in frame.",
    "football": "Film front-on or 3/4 view, capturing a jump-landing or cutting movement.",
    "cricket_bowling": "Film side-on, full delivery stride from run-up through release in frame.",
    "cricket_batting": "Film face-on or slightly side-angled, one shot per clip.",
}

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def sport_label(s: str) -> str:
    return s.replace("_", " ").title()


def file_signature(uploaded_file) -> str:
    """Cheap fingerprint of an uploaded file's identity, used to detect a new upload."""
    return hashlib.md5(f"{uploaded_file.name}:{uploaded_file.size}".encode()).hexdigest()


def run_and_cache_analysis(uploaded_file, sport, model_complexity):
    """Run the full pipeline once and stash everything the UI needs in session_state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress_bar = st.progress(0.0, text="Running pose estimation...")

        def on_progress(idx, total):
            if total > 0:
                progress_bar.progress(min(idx / total, 1.0), text=f"Processing frame {idx}/{total}")

        video_landmarks, report = run_analysis(
            input_path, sport, progress_callback=on_progress, model_complexity=model_complexity
        )
        progress_bar.empty()

        output_path = os.path.join(tmpdir, "annotated.mp4")
        first_metric_key = SPORT_METRIC_PRESETS[sport][0]
        draw_skeleton_overlay(
            input_path, video_landmarks, output_path,
            live_metric=report.metrics.get(first_metric_key),
            live_metric_label=first_metric_key.replace("_", " "),
        )
        with open(output_path, "rb") as f:
            annotated_video_bytes = f.read()  # read into memory -- tmpdir is deleted when this block exits

    st.session_state["analysis"] = {
        "signature": file_signature(uploaded_file),
        "sport": sport,
        "model_complexity": model_complexity,
        "video_landmarks": video_landmarks,
        "report": report,
        "annotated_video_bytes": annotated_video_bytes,
    }


def render_results():
    data = st.session_state["analysis"]
    video_landmarks = data["video_landmarks"]
    report = data["report"]
    sport = data["sport"]

    if video_landmarks.detection_rate < 0.5:
        st.warning(
            f"Pose was only detected in {video_landmarks.detection_rate:.0%} of frames. "
            "Results may be unreliable \u2014 try a clearer video with the full body visible."
        )

    tab_video, tab_feedback, tab_charts, tab_3d = st.tabs(["Video", "Feedback", "Metrics", "3D viewer"])

    with tab_video:
        render_eyebrow("Tracking feed")
        with st.container(border=True):
            st.video(data["annotated_video_bytes"])
        st.caption(f"Pose detected in {video_landmarks.detection_rate:.0%} of frames.")

        render_eyebrow("Summary")
        summary_items = list(report.summary.items())
        if summary_items:
            cols = st.columns(min(len(summary_items), 4))
            for i, (key, value) in enumerate(summary_items):
                display_val = f"{value:.1f}" if isinstance(value, float) else str(value)
                cols[i % len(cols)].metric(key.replace("_", " ").title(), display_val)

    with tab_feedback:
        render_eyebrow("Form flags")
        if not report.flags:
            st.success("No form issues flagged by the current rule set.")
        else:
            for flag in sorted(report.flags, key=lambda f: SEVERITY_ORDER.get(f.severity, 3)):
                render_flag_card(flag.severity, flag.timestamp_sec, flag.message)

    with tab_charts:
        render_eyebrow(f"{sport_label(sport)} \u2014 angle readout")
        fig = plot_metric_timeseries(report, SPORT_METRIC_PRESETS[sport], video_landmarks.fps, title=sport_label(sport))
        st.pyplot(fig)
        st.caption("Dashed gray lines mark detected reps/events; colored lines mark flagged moments.")

    with tab_3d:
        render_eyebrow("Reconstructed pose")
        with st.container(border=True):
            frame_idx = st.slider(
                "Frame", 0, len(video_landmarks.frames) - 1, len(video_landmarks.frames) // 2, key="frame_3d_slider"
            )
            fig3d = plot_3d_skeleton_frame(video_landmarks, frame_idx)
            st.plotly_chart(fig3d, use_container_width=True)
        st.caption(
            "3D landmarks are MediaPipe's monocular world-coordinate estimate, not a triangulated "
            "measurement \u2014 reliable for relative joint angles, less so for absolute distances."
        )


def main():
    inject_theme()
    render_header(
        kicker="Computer vision \u00b7 Biomechanics",
        title="Athlete Form Analyzer",
        subtitle="Upload a video for automatic pose tracking, joint-angle measurement, and rule-based form feedback, built on MediaPipe's monocular 2D/3D pose model.",
    )

    with st.sidebar:
        st.header("Settings")
        sport = st.selectbox("Sport / movement", list(ANALYZERS.keys()), format_func=sport_label)
        model_complexity = st.select_slider(
            "Model accuracy", options=[0, 1, 2], value=1,
            format_func=lambda x: {0: "Fastest", 1: "Balanced", 2: "Most accurate"}[x],
        )
        st.markdown("---")
        render_eyebrow(f"Filming \u2014 {sport_label(sport)}")
        st.caption(f"{SPORT_FILMING_TIPS[sport]} Good lighting and minimal motion blur help detection either way.")

    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "m4v"])

    if uploaded is None:
        st.info("Upload a video to get started.")
        return

    current_sig = file_signature(uploaded)

    def compute_is_stale():
        cached = st.session_state.get("analysis")
        return (
            cached is None
            or cached["signature"] != current_sig
            or cached["sport"] != sport
            or cached["model_complexity"] != model_complexity
        )

    is_stale = compute_is_stale()
    has_cached = st.session_state.get("analysis") is not None

    col1, col2 = st.columns([1, 3])
    with col1:
        clicked = st.button("Analyze" if is_stale else "Re-analyze", type="primary")
    with col2:
        if has_cached and not is_stale:
            st.caption("Showing cached results for this video/sport/setting combo \u2014 browsing frames or tabs won't re-run analysis.")
        elif has_cached and is_stale:
            st.caption("Video, sport, or accuracy setting changed since the last analysis \u2014 click to (re)analyze.")

    if clicked:
        run_and_cache_analysis(uploaded, sport, model_complexity)
        is_stale = compute_is_stale()  # recompute now that session_state was just updated

    if st.session_state.get("analysis") is not None and not is_stale:
        render_results()
    elif st.session_state.get("analysis") is not None and is_stale:
        st.info("Click Analyze to process this video with the current settings.")


if __name__ == "__main__":
    main()
