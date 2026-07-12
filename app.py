"""
AI Sports Performance Analyzer
--------------------------------
A Streamlit application that uses OpenCV + MediaPipe Pose to analyze
sports videos: running posture, jump performance, and throwing mechanics.

Features:
- Upload sports video
- Detect human pose (MediaPipe Pose)
- Draw skeleton overlay on every frame
- Running posture analysis (trunk lean, cadence, knee lift, symmetry)
- Jump analysis (flight time -> estimated jump height, takeoff knee flexion)
- Throw angle analysis (release angle, elbow extension, arm speed)
- Composite 0-100 performance score
- Download processed (annotated) video
- Download a text performance report
"""

import os
import math
import tempfile
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
import mediapipe_compat as mp

# ---------------------------------------------------------------------------
# Page config & MediaPipe setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Sports Performance Analyzer",
    page_icon="🏃",
    layout="wide",
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LM = mp_pose.PoseLandmark

KEYPOINT_NAMES = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]


# ---------------------------------------------------------------------------
# Geometry / signal-processing helpers
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c):
    """Angle in degrees at vertex b, formed by points a-b-c."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosine_angle = np.dot(ba, bc) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def get_landmark_xy(landmarks, idx, width, height):
    lm = landmarks[idx]
    return [lm.x * width, lm.y * height]


def extract_key_points(landmarks, width, height):
    pts = {}
    for name in KEYPOINT_NAMES:
        idx = getattr(LM, name).value
        pts[name] = get_landmark_xy(landmarks, idx, width, height)
    return pts


def smooth_signal(signal, window=5):
    """Simple moving-average smoothing, dependency-free."""
    signal = np.array(signal, dtype=float)
    if len(signal) == 0:
        return signal
    window = max(1, min(window, len(signal)))
    if window == 1:
        return signal
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(signal)]


def find_peaks_simple(signal, min_distance=5):
    """Lightweight local-maxima peak finder (no scipy dependency)."""
    peaks = []
    n = len(signal)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    return peaks


# ---------------------------------------------------------------------------
# Video processing: pose detection + skeleton drawing + metric extraction
# ---------------------------------------------------------------------------

def process_video(input_path, dominant_arm="Right", progress_callback=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or math.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    metrics = {
        "time": [],
        "left_knee_angle": [],
        "right_knee_angle": [],
        "trunk_angle": [],
        "hip_y": [],
        "elbow_angle": [],
        "forearm_angle": [],
        "wrist_speed": [],
    }

    prev_wrist = None
    frame_idx = 0
    detected_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            detected_frames += 1
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_pose_connections_style(),
            )

            lm = results.pose_landmarks.landmark
            pts = extract_key_points(lm, width, height)
            t = frame_idx / fps

            l_knee = calculate_angle(pts["LEFT_HIP"], pts["LEFT_KNEE"], pts["LEFT_ANKLE"])
            r_knee = calculate_angle(pts["RIGHT_HIP"], pts["RIGHT_KNEE"], pts["RIGHT_ANKLE"])

            mid_shoulder = [
                (pts["LEFT_SHOULDER"][0] + pts["RIGHT_SHOULDER"][0]) / 2,
                (pts["LEFT_SHOULDER"][1] + pts["RIGHT_SHOULDER"][1]) / 2,
            ]
            mid_hip = [
                (pts["LEFT_HIP"][0] + pts["RIGHT_HIP"][0]) / 2,
                (pts["LEFT_HIP"][1] + pts["RIGHT_HIP"][1]) / 2,
            ]
            vertical_ref = [mid_hip[0], mid_hip[1] - 100]
            trunk_angle = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

            if dominant_arm == "Right":
                shoulder, elbow, wrist = pts["RIGHT_SHOULDER"], pts["RIGHT_ELBOW"], pts["RIGHT_WRIST"]
            else:
                shoulder, elbow, wrist = pts["LEFT_SHOULDER"], pts["LEFT_ELBOW"], pts["LEFT_WRIST"]

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            forearm_angle = math.degrees(
                math.atan2(-(wrist[1] - elbow[1]), (wrist[0] - elbow[0]))
            )

            if prev_wrist is not None:
                dist = math.hypot(wrist[0] - prev_wrist[0], wrist[1] - prev_wrist[1])
                wrist_speed = dist * fps
            else:
                wrist_speed = 0.0
            prev_wrist = wrist

            metrics["time"].append(t)
            metrics["left_knee_angle"].append(l_knee)
            metrics["right_knee_angle"].append(r_knee)
            metrics["trunk_angle"].append(trunk_angle)
            metrics["hip_y"].append(mid_hip[1])
            metrics["elbow_angle"].append(elbow_angle)
            metrics["forearm_angle"].append(forearm_angle)
            metrics["wrist_speed"].append(wrist_speed)
        else:
            prev_wrist = None

        writer.write(frame)
        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress_callback(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    pose.close()

    duration = frame_idx / fps if fps else 0.0
    detection_rate = (detected_frames / frame_idx * 100) if frame_idx else 0.0

    return output_path, metrics, fps, duration, detection_rate


# ---------------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------------

def analyze_running(metrics, duration):
    left = np.array(metrics["left_knee_angle"])
    right = np.array(metrics["right_knee_angle"])
    trunk = np.array(metrics["trunk_angle"])

    if len(left) < 10:
        return None

    left_s = smooth_signal(left, 5)
    right_s = smooth_signal(right, 5)

    min_dist = max(3, int(len(left_s) / 20))
    left_peaks = find_peaks_simple(left_s, min_distance=min_dist)
    right_peaks = find_peaks_simple(right_s, min_distance=min_dist)

    steps = len(left_peaks) + len(right_peaks)
    duration_min = max(duration / 60.0, 1e-6)
    cadence = steps / duration_min

    avg_trunk_lean = float(np.mean(trunk))
    left_rom = float(np.max(left_s) - np.min(left_s))
    right_rom = float(np.max(right_s) - np.min(right_s))
    symmetry_index = abs(left_rom - right_rom) / max(left_rom, right_rom, 1e-6) * 100
    min_knee_angle = float(min(np.min(left_s), np.min(right_s)))

    trunk_score = max(0.0, 100 - abs(avg_trunk_lean - 10) * 4)
    cadence_score = max(0.0, 100 - abs(cadence - 175) * 1.5)
    symmetry_score = max(0.0, 100 - symmetry_index)
    knee_lift_score = max(0.0, 100 - abs(min_knee_angle - 75) * 1.2)

    overall = float(np.mean([trunk_score, cadence_score, symmetry_score, knee_lift_score]))

    return {
        "Average Trunk Lean (deg)": round(avg_trunk_lean, 1),
        "Cadence (steps/min)": round(cadence, 1),
        "Steps Detected": steps,
        "Symmetry Index (%)": round(symmetry_index, 1),
        "Min Knee Angle During Swing (deg)": round(min_knee_angle, 1),
        "scores": {
            "Trunk Lean": round(trunk_score, 1),
            "Cadence": round(cadence_score, 1),
            "Symmetry": round(symmetry_score, 1),
            "Knee Lift": round(knee_lift_score, 1),
        },
        "overall_score": round(overall, 1),
    }


def analyze_jump(metrics, fps):
    hip_y = np.array(metrics["hip_y"])
    left = np.array(metrics["left_knee_angle"])
    right = np.array(metrics["right_knee_angle"])

    if len(hip_y) < 10:
        return None

    hip_y_s = smooth_signal(hip_y, 5)

    baseline_window = max(5, int(len(hip_y_s) * 0.15))
    baseline = float(np.median(hip_y_s[:baseline_window]))
    signal_range = float(np.max(hip_y_s) - np.min(hip_y_s))
    threshold = signal_range * 0.15 + 3

    airborne = hip_y_s < (baseline - threshold)
    flight_frames = int(np.sum(airborne))
    flight_time = flight_frames / fps if fps else 0.0

    g = 9.81
    jump_height_m = g * (flight_time ** 2) / 8
    jump_height_cm = jump_height_m * 100

    if np.any(airborne):
        first_air_idx = int(np.argmax(airborne))
        window_start = max(0, first_air_idx - 8)
        if first_air_idx > window_start:
            takeoff_knee = float(
                min(np.min(left[window_start:first_air_idx + 1]),
                    np.min(right[window_start:first_air_idx + 1]))
            )
        else:
            takeoff_knee = float(min(left[first_air_idx], right[first_air_idx]))
    else:
        takeoff_knee = float(min(np.min(left), np.min(right)))

    height_score = max(0.0, min(100.0, (jump_height_cm / 60.0) * 100))
    knee_score = max(0.0, 100 - abs(takeoff_knee - 100) * 1.5)
    overall = float(np.mean([height_score, knee_score]))

    return {
        "Flight Time (s)": round(flight_time, 3),
        "Estimated Jump Height (cm)": round(jump_height_cm, 1),
        "Takeoff Knee Flexion (deg)": round(takeoff_knee, 1),
        "scores": {
            "Jump Height": round(height_score, 1),
            "Takeoff Knee Flexion": round(knee_score, 1),
        },
        "overall_score": round(overall, 1),
    }


def analyze_throw(metrics, fps):
    speed = np.array(metrics["wrist_speed"])
    elbow = np.array(metrics["elbow_angle"])
    forearm = np.array(metrics["forearm_angle"])
    times = metrics["time"]

    if len(speed) < 10:
        return None

    speed_s = smooth_signal(speed, 3)
    release_idx = int(np.argmax(speed_s))

    max_speed = float(speed_s[release_idx])
    release_elbow_angle = float(elbow[release_idx])
    release_angle_raw = float(forearm[release_idx])

    norm_release_angle = abs(release_angle_raw)
    if norm_release_angle > 90:
        norm_release_angle = 180 - norm_release_angle

    angle_score = max(0.0, 100 - abs(norm_release_angle - 40) * 3)
    elbow_score = max(0.0, 100 - abs(release_elbow_angle - 170) * 2)
    speed_score = max(0.0, min(100.0, (max_speed / 2000.0) * 100))

    overall = float(np.mean([angle_score, elbow_score, speed_score]))

    return {
        "Release Moment (s)": round(times[release_idx], 2),
        "Release Angle (deg from horizontal)": round(norm_release_angle, 1),
        "Elbow Angle at Release (deg)": round(release_elbow_angle, 1),
        "Peak Arm Speed (px/s, relative)": round(max_speed, 1),
        "scores": {
            "Release Angle": round(angle_score, 1),
            "Elbow Extension": round(elbow_score, 1),
            "Arm Speed": round(speed_score, 1),
        },
        "overall_score": round(overall, 1),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(analysis_type, summary, video_name, detection_rate, duration):
    lines = []
    lines.append("=" * 62)
    lines.append("AI SPORTS PERFORMANCE ANALYZER - PERFORMANCE REPORT")
    lines.append("=" * 62)
    lines.append(f"Video analyzed     : {video_name}")
    lines.append(f"Analysis type      : {analysis_type}")
    lines.append(f"Video duration     : {duration:.2f} s")
    lines.append(f"Pose detection rate: {detection_rate:.1f}% of frames")
    lines.append(f"Generated on       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-" * 62)

    if summary is None:
        lines.append("No reliable pose data could be extracted from this video.")
        lines.append("Try a clip with better lighting and the full body clearly visible.")
        lines.append("=" * 62)
        return "\n".join(lines)

    lines.append("DETAILED METRICS")
    for key, value in summary.items():
        if key in ("scores", "overall_score"):
            continue
        lines.append(f"  {key}: {value}")

    lines.append("-" * 62)
    lines.append("COMPONENT SCORES (out of 100)")
    for key, value in summary["scores"].items():
        lines.append(f"  {key}: {value}")

    lines.append("-" * 62)
    lines.append(f"OVERALL PERFORMANCE SCORE: {summary['overall_score']} / 100")
    lines.append("=" * 62)
    lines.append("Note: Speed and jump-height figures are estimates derived from")
    lines.append("pose landmark motion and frame timing, not calibrated sensors.")
    lines.append("=" * 62)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "processed": False,
        "output_path": None,
        "summary": None,
        "analysis_type": None,
        "video_name": None,
        "detection_rate": 0.0,
        "duration": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main():
    init_session_state()

    st.title("🏃 AI Sports Performance Analyzer")
    st.caption(
        "Upload a sports video to detect human pose, draw a skeleton overlay, "
        "and generate an automated performance analysis and score."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Running Posture", "Jump Analysis", "Throw Angle Analysis"],
        )

        dominant_arm = "Right"
        if analysis_type == "Throw Angle Analysis":
            dominant_arm = st.radio("Throwing Arm", ["Right", "Left"], horizontal=True)

        st.markdown("---")
        st.markdown("**Tips for best results**")
        st.markdown(
            "- Keep the full body visible in frame\n"
            "- Use good, even lighting\n"
            "- A side-on camera angle works best for running and throwing\n"
            "- A front-on angle works best for vertical jumps"
        )

    uploaded_file = st.file_uploader(
        "Upload a sports video", type=["mp4", "mov", "avi", "mkv", "m4v"]
    )

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        tfile.close()
        input_path = tfile.name

        st.video(input_path)

        run_clicked = st.button("🚀 Run Analysis", type="primary")

        if run_clicked:
            progress_bar = st.progress(0, text="Processing video...")

            def update_progress(p):
                progress_bar.progress(p, text=f"Processing video... {int(p * 100)}%")

            try:
                with st.spinner("Detecting pose and analyzing performance..."):
                    output_path, metrics, fps, duration, detection_rate = process_video(
                        input_path, dominant_arm, update_progress
                    )
                progress_bar.progress(1.0, text="Done!")
            except Exception as exc:
                progress_bar.empty()
                st.error(f"Something went wrong while processing the video: {exc}")
                return

            if analysis_type == "Running Posture":
                summary = analyze_running(metrics, duration)
            elif analysis_type == "Jump Analysis":
                summary = analyze_jump(metrics, fps)
            else:
                summary = analyze_throw(metrics, fps)

            st.session_state["output_path"] = output_path
            st.session_state["summary"] = summary
            st.session_state["analysis_type"] = analysis_type
            st.session_state["video_name"] = uploaded_file.name
            st.session_state["detection_rate"] = detection_rate
            st.session_state["duration"] = duration
            st.session_state["processed"] = True

    if st.session_state["processed"]:
        st.markdown("## 📊 Results")

        summary = st.session_state["summary"]
        analysis_type = st.session_state["analysis_type"]
        detection_rate = st.session_state["detection_rate"]

        st.caption(f"Pose detected in {detection_rate:.1f}% of video frames.")

        if summary is None:
            st.warning(
                "No reliable pose could be detected in this video. "
                "Try a clip with better lighting and the full body clearly visible."
            )
        else:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Overall Performance Score", f"{summary['overall_score']} / 100")
                st.markdown("**Component Scores**")
                for name, value in summary["scores"].items():
                    st.progress(min(max(value, 0) / 100.0, 1.0), text=f"{name}: {value}/100")

            with col2:
                st.markdown("**Detailed Metrics**")
                display_items = {
                    k: v for k, v in summary.items() if k not in ("scores", "overall_score")
                }
                st.table(display_items)

        st.markdown("## 🎬 Processed Video (with skeleton overlay)")
        if st.session_state["output_path"] and os.path.exists(st.session_state["output_path"]):
            st.video(st.session_state["output_path"])

        col_a, col_b = st.columns(2)
        with col_a:
            if st.session_state["output_path"] and os.path.exists(st.session_state["output_path"]):
                with open(st.session_state["output_path"], "rb") as f:
                    st.download_button(
                        "⬇️ Download Processed Video",
                        data=f.read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )
        with col_b:
            report_text = generate_report(
                analysis_type,
                summary,
                st.session_state["video_name"],
                detection_rate,
                st.session_state["duration"],
            )
            st.download_button(
                "⬇️ Download Report",
                data=report_text,
                file_name="performance_report.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
