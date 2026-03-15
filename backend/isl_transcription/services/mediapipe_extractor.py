"""
mediapipe_extractor.py

Extracts hand, pose, and face landmarks from a JPEG frame using MediaPipe Holistic.

The landmark vector is a flat 1D numpy array combining:
  - 21 hand landmarks × 2 hands × 3 coordinates = 126 values
  - 33 pose landmarks × 3 coordinates = 99 values
  - 468 face landmarks × 3 coordinates = 1404 values
Total: 1629 features per frame

If a landmark type is not detected in a frame, zeros are used.
"""

import numpy as np
import cv2
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

mp_holistic = mp.solutions.holistic  # type: ignore

# Reuse a single Holistic instance for efficiency
_holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Feature sizes
N_POSE = 33 * 4       # x, y, z, visibility
N_FACE = 468 * 3      # x, y, z
N_LEFT_HAND = 21 * 3  # x, y, z
N_RIGHT_HAND = 21 * 3


def _extract_keypoints(results) -> np.ndarray:
    """Flatten all MediaPipe landmark results into a single feature vector."""
    pose = (
        np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(N_POSE)
    )
    face = (
        np.array([[l.x, l.y, l.z] for l in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(N_FACE)
    )
    lh = (
        np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(N_LEFT_HAND)
    )
    rh = (
        np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(N_RIGHT_HAND)
    )
    return np.concatenate([pose, face, lh, rh])


def extract_landmarks(jpeg_bytes: bytes) -> np.ndarray | None:
    """
    Decode JPEG bytes and run MediaPipe Holistic on the frame.

    Returns:
        1D float32 numpy array of features, or None on failure.
    """
    try:
        # Decode JPEG → BGR numpy array
        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            logger.warning("Failed to decode JPEG frame")
            return None

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        results = _holistic.process(frame_rgb)

        return _extract_keypoints(results).astype(np.float32)

    except Exception as e:
        logger.error(f"Error in extract_landmarks: {e}", exc_info=True)
        return None
