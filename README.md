# AI Sports Performance Analyzer

A Streamlit application that analyzes sports videos with OpenCV and MediaPipe pose detection. Upload a clip to generate an annotated video, sport-specific performance metrics, a composite score, and a downloadable text report.

> Metrics are camera-based estimates for feedback and exploration. They are not a substitute for calibrated motion-capture equipment or coaching/medical advice.

## Features

- Upload MP4, MOV, AVI, MKV, or M4V videos.
- Detect full-body pose landmarks frame by frame.
- Generate a downloadable video with a skeleton overlay.
- Analyze three movement types:
  - **Running posture:** trunk lean, cadence, knee motion, and left/right symmetry.
  - **Vertical jump:** estimated flight time, jump height, and takeoff knee flexion.
  - **Throwing mechanics:** release angle, elbow extension, and relative arm speed.
- Display component scores and an overall score out of 100.
- Download a plain-text performance report.

## Tech stack

- [Streamlit](https://streamlit.io/)
- OpenCV
- NumPy
- MediaPipe Tasks Pose Landmarker

## Getting started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd "HackZen Original"
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`) to open in your browser.

## How to use

1. Select **Running Posture**, **Jump Analysis**, or **Throw Angle Analysis** in the sidebar.
2. For throws, choose the throwing arm.
3. Upload a sports video.
4. Click **Run Analysis**.
5. Review the metrics and score, then download the annotated video and report.

For the most reliable results, use even lighting, keep the full body in frame, and minimize camera movement. Side views work best for running and throwing; front views work well for vertical jumps.

## Project structure

```text
.
|-- app.py                 # Streamlit UI, video processing, and analysis logic
|-- mediapipe_compat.py    # Compatibility layer for the MediaPipe Tasks API
|-- requirements.txt       # Python dependencies
`-- README.md
```

## MediaPipe compatibility note

`mediapipe_compat.py` adapts the MediaPipe Tasks Pose Landmarker to the legacy-style interface used by the app. It also includes a Windows workaround for a missing `free()` export in the MediaPipe Tasks DLL distributed with the pinned package version. Keep this file alongside `app.py`.

The pose model is downloaded automatically on first use and cached in `.mp_models/`. Ensure the machine can access the internet the first time an analysis runs, or place a compatible `pose_landmarker_full.task`, `pose_landmarker_lite.task`, or `pose_landmarker.task` file in that folder beforehand.

## Limitations

- Results depend on video resolution, framing, lighting, clothing contrast, and occlusion.
- Speed is reported in pixels/second and is relative, not real-world velocity.
- Jump height is inferred from estimated flight time and is not calibrated.
- The current analysis focuses on one primary pose per frame.
