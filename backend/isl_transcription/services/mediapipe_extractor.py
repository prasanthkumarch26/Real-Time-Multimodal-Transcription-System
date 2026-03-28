"""
mediapipe_extractor.py

Extracts hand, pose, and face landmarks from a JPEG frame using MediaPipe Tasks
HolisticLandmarker (compatible with mediapipe >= 0.10.13 on Python 3.12).

The landmark vector is a flat 1D numpy array combining:
  - 33 pose landmarks × 4 coordinates (x, y, z, visibility) = 132 values
  - 478 face landmarks × 3 coordinates = 1434 values
  - 21 left hand landmarks × 3 coordinates = 63 values
  - 21 right hand landmarks × 3 coordinates = 63 values
Total: 1692 features per frame

Note: Feature count changed from 1629 to 1662 due to Tasks API using visibility on pose.
If a landmark type is not detected in a frame, zeros are used.
"""

import numpy as np
import cv2
import mediapipe as mp
import os
import logging

logger = logging.getLogger(__name__)

# Model path
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "holistic_landmarker.task"
)

# Feature sizes (Tasks API)
N_POSE = 33 * 4        # x, y, z, visibility
N_FACE = 478 * 3       # x, y, z
N_LEFT_HAND = 21 * 3   # x, y, z
N_RIGHT_HAND = 21 * 3  # x, y, z
TOTAL_FEATURES = N_POSE + N_FACE + N_LEFT_HAND + N_RIGHT_HAND  # 1692


def _build_landmarker():
    """Build and return a HolisticLandmarker instance."""
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"HolisticLandmarker task model not found at {_MODEL_PATH}. "
            "Run: python -c \"import urllib.request; urllib.request.urlretrieve("
            "'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
            "holistic_landmarker/float16/latest/holistic_landmarker.task', "
            "'backend/isl_transcription/models/holistic_landmarker.task')\""
        )
    
    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return HolisticLandmarker.create_from_options(options)


# Singleton landmarker — created once and reused
try:
    _landmarker = _build_landmarker()
    logger.info("HolisticLandmarker loaded successfully.")
except Exception as e:
    _landmarker = None
    logger.warning(f"HolisticLandmarker unavailable: {e}")


def _extract_keypoints(result) -> np.ndarray:
    """Flatten all HolisticLandmarker results into a single feature vector."""
    pose = (
        np.array([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks]).flatten()
        if result.pose_landmarks else np.zeros(N_POSE)
    )
    face = (
        np.array([[l.x, l.y, l.z] for l in result.face_landmarks]).flatten()
        if result.face_landmarks else np.zeros(N_FACE)
    )
    lh = (
        np.array([[l.x, l.y, l.z] for l in result.left_hand_landmarks]).flatten()
        if result.left_hand_landmarks else np.zeros(N_LEFT_HAND)
    )
    rh = (
        np.array([[l.x, l.y, l.z] for l in result.right_hand_landmarks]).flatten()
        if result.right_hand_landmarks else np.zeros(N_RIGHT_HAND)
    )
    return np.concatenate([pose, face, lh, rh])


def extract_landmarks(jpeg_bytes: bytes) -> np.ndarray | None:
    """
    Decode JPEG bytes and run MediaPipe HolisticLandmarker on the frame.

    Returns:
        1D float32 numpy array of TOTAL_FEATURES features, or None on failure.
    """
    if _landmarker is None:
        return None

    try:
        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            logger.warning("Failed to decode JPEG frame")
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = _landmarker.detect(mp_image)

        return _extract_keypoints(result).astype(np.float32)

    except Exception as e:
        logger.error(f"Error in extract_landmarks: {e}", exc_info=True)
        return None
