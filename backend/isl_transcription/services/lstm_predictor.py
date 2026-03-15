"""
lstm_predictor.py

LSTM-based ISL word classifier.

Architecture:
  - Input: (30 frames × 1629 landmark features)
  - 3 LSTM layers (stacked)
  - Dense softmax output over vocabulary

Supported words (7-class prototype):
  hello, thanks, iloveyou, please, yes, no, good

If the trained model file (action.h5) is not found, a stub predictor is used
that responds with random words for development / UI testing.
"""

import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# ─── Vocabulary (must match training label order) ──────────────────────────
ACTIONS = ["hello", "thanks", "iloveyou", "please", "yes", "no", "good"]
SEQUENCE_LENGTH = 30      # frames per prediction window
PREDICTION_THRESHOLD = 0.85  # confidence threshold — discard low-confidence preds

# ─── Model path ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "action.h5")


class LSTMPredictor:
    def __init__(self):
        self._model = None
        self._stub = True  # Stub mode until real model is loaded

        try:
            from tensorflow.keras.models import load_model  # type: ignore
            if os.path.exists(MODEL_PATH):
                logger.info(f"Loading LSTM model from {MODEL_PATH}")
                self._model = load_model(MODEL_PATH)
                self._stub = False
                logger.info("LSTM model loaded successfully.")
            else:
                logger.warning(
                    f"LSTM model not found at {MODEL_PATH}. "
                    "Running in STUB mode — returning dummy predictions."
                )
        except ImportError:
            logger.warning("TensorFlow not available. Running in STUB mode.")

    def predict(self, sequence: np.ndarray) -> str | None:
        """
        Run prediction on a (30, 1629) float32 sequence.

        Returns:
            Predicted word string if confidence > threshold, else None.
        """
        if sequence.shape != (SEQUENCE_LENGTH, sequence.shape[-1]):
            logger.warning(f"Unexpected sequence shape: {sequence.shape}")
            return None

        if self._stub or self._model is None:
            # STUB: return a random word occasionally for UI development
            if np.random.random() > 0.85:
                return np.random.choice(ACTIONS)
            return None

        input_data = np.expand_dims(sequence, axis=0)  # (1, 30, features)
        predictions = self._model.predict(input_data, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])

        logger.debug(f"ISL prediction: {ACTIONS[predicted_idx]} ({confidence:.2%})")

        if confidence >= PREDICTION_THRESHOLD:
            return ACTIONS[predicted_idx]

        return None


# Global singleton
predictor = LSTMPredictor()
