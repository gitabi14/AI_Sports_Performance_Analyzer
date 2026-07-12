import os
import cv2
import numpy as np

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import pose_landmarker as pl
    from mediapipe.tasks.python.vision.core import image as image_lib
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.core import mediapipe_c_bindings as c_bindings
except Exception:
    vision = None
    pl = None
    image_lib = None
    base_options_lib = None
    c_bindings = None


def _patch_windows_tasks_loader():
    """Work around MediaPipe 0.10.30's Windows DLL missing ``free``.

    The Tasks Python loader unconditionally registers ``libmediapipe.dll.free``
    even though the distributed Windows DLL does not export it.  ``free`` is
    only used to dispose of native error strings; successful pose inference
    does not call it.  The no-op fallback lets the otherwise functional DLL
    initialize and avoids masking the actual task API.
    """
    if os.name != "nt" or c_bindings is None:
        return

    original_loader = c_bindings.load_raw_library
    if getattr(original_loader, "_hackzen_free_workaround", False):
        return

    def load_raw_library(signatures=()):
        try:
            return original_loader(signatures)
        except AttributeError as exc:
            if "function 'free' not found" not in str(exc):
                raise

            def missing_free(_pointer):
                # The wheel lacks the matching exported deallocator. This
                # branch is only used when MediaPipe returns a native error.
                return None

            c_bindings._shared_lib.free = missing_free
            return original_loader(signatures)

    load_raw_library._hackzen_free_workaround = True
    c_bindings.load_raw_library = load_raw_library


_patch_windows_tasks_loader()


class _Landmark:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _LandmarksWrapper:
    def __init__(self, lm_list):
        self.landmark = lm_list


class _Results:
    def __init__(self, pose_landmarks=None):
        self.pose_landmarks = pose_landmarks


class _Pose:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if pl is None:
            raise ImportError("mediapipe.tasks not available in this environment")

        self._landmarker = None
        # Determine model cache path
        cache_dir = os.path.join(os.getcwd(), ".mp_models")
        os.makedirs(cache_dir, exist_ok=True)
        # Candidate model filenames
        candidates = [
            "pose_landmarker_full.task",
            "pose_landmarker_lite.task",
            "pose_landmarker.task",
        ]
        model_path = None
        for name in candidates:
            p = os.path.join(cache_dir, name)
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            # Try to download a commonly-hosted task file (best-effort).
            urls = [
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            ]
            for url in urls:
                try:
                    import urllib.request
                    dest = os.path.join(cache_dir, os.path.basename(url))
                    if not os.path.exists(dest):
                        urllib.request.urlretrieve(url, dest)
                    model_path = dest
                    break
                except Exception:
                    model_path = None

        if model_path is None:
            raise FileNotFoundError(
                "Could not find or download a MediaPipe pose landmarker model."
                " Please provide a .task model file in the project .mp_models folder"
            )

        base_options = base_options_lib.BaseOptions(model_asset_path=model_path)
        options = pl.PoseLandmarkerOptions(
            base_options=base_options,
            # The application calls ``process`` once per frame, without a
            # monotonically increasing timestamp.  IMAGE mode is therefore
            # the compatible Tasks API mode (VIDEO mode requires
            # ``detect_for_video(..., timestamp_ms)``).
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # create landmarker
        try:
            self._landmarker = pl.PoseLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"Failed to create PoseLandmarker: {e}")

    def process(self, frame_rgb):
        # frame_rgb expected as numpy RGB image
        if image_lib is None:
            raise ImportError("mediapipe.tasks image helper not available")
        # MediaPipe Tasks 0.10.30 constructs images directly from a NumPy
        # array; older examples used the now-removed ``create_from_array``.
        mp_image = image_lib.Image(
            image_lib.ImageFormat.SRGB, np.ascontiguousarray(frame_rgb)
        )
        res = self._landmarker.detect(mp_image)
        if res and res.pose_landmarks:
            # take first pose
            p = res.pose_landmarks[0]
            lm_list = []
            for lm in p:
                lm_list.append(_Landmark(x=lm.x, y=lm.y, z=getattr(lm, 'z', 0.0)))
            return _Results(_LandmarksWrapper(lm_list))
        return _Results(None)

    def close(self):
        if self._landmarker:
            self._landmarker.close()


class _DrawingUtils:
    @staticmethod
    def draw_landmarks(image, landmarks, connections, landmark_drawing_spec=None, connection_drawing_spec=None):
        if landmarks is None or not hasattr(landmarks, 'landmark'):
            return
        h, w = image.shape[:2]
        pts = []
        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        if connections:
            for c in connections:
                a, b = c
                if a < len(pts) and b < len(pts):
                    cv2.line(image, pts[a], pts[b], (0, 128, 255), 2)


class _DrawingStyles:
    @staticmethod
    def get_default_pose_landmarks_style():
        return None

    @staticmethod
    def get_default_pose_connections_style():
        return None


class _Solutions:
    def __init__(self):
        self.pose = type('pose_mod', (), {})()
        self.pose.Pose = _Pose
        # provide Landmark enum-like mapping
        class LM:
            NOSE = type('v', (), {'value': 0})
            LEFT_SHOULDER = type('v', (), {'value': 11})
            RIGHT_SHOULDER = type('v', (), {'value': 12})
            LEFT_ELBOW = type('v', (), {'value': 13})
            RIGHT_ELBOW = type('v', (), {'value': 14})
            LEFT_WRIST = type('v', (), {'value': 15})
            RIGHT_WRIST = type('v', (), {'value': 16})
            LEFT_HIP = type('v', (), {'value': 23})
            RIGHT_HIP = type('v', (), {'value': 24})
            LEFT_KNEE = type('v', (), {'value': 25})
            RIGHT_KNEE = type('v', (), {'value': 26})
            LEFT_ANKLE = type('v', (), {'value': 27})
            RIGHT_ANKLE = type('v', (), {'value': 28})
            LEFT_HEEL = type('v', (), {'value': 29})
            RIGHT_HEEL = type('v', (), {'value': 30})
            LEFT_FOOT_INDEX = type('v', (), {'value': 31})
            RIGHT_FOOT_INDEX = type('v', (), {'value': 32})
        self.pose.PoseLandmark = LM


mp = type('mp', (), {})()
mp.solutions = _Solutions()
mp.drawing_utils = _DrawingUtils()
mp.drawing_styles = _DrawingStyles()

# ``app.py`` imports this file as ``mp``.  Export the same attributes at the
# module level as the legacy ``mediapipe`` package so calls such as
# ``mp.solutions.pose`` continue to work.
solutions = mp.solutions
solutions.drawing_utils = mp.drawing_utils
solutions.drawing_styles = mp.drawing_styles
drawing_utils = mp.drawing_utils
drawing_styles = mp.drawing_styles

try:
    # expose POSE_CONNECTIONS similar to legacy API
    if pl is not None:
        mp.solutions.pose_connections = pl.PoseLandmarksConnections.POSE_LANDMARKS
        mp.solutions.pose_connections = [(c.start, c.end) for c in pl.PoseLandmarksConnections.POSE_LANDMARKS]
        mp.solutions.pose.POSE_CONNECTIONS = [(c.start, c.end) for c in pl.PoseLandmarksConnections.POSE_LANDMARKS]
except Exception:
    pass
