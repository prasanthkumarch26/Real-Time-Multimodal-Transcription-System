import numpy as np
from collections import deque
from threading import Lock

SAMPLE_RATE = 16000

# Sliding window config
WINDOW_SECONDS = 3.0       # Keep last 3 seconds of audio
STEP_SECONDS = 0.5         # Slide every 0.5 seconds
MIN_SPEECH_SECONDS = 0.5   # Minimum audio length to run inference

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SAMPLES = int(SAMPLE_RATE * STEP_SECONDS)
MIN_SPEECH_SAMPLES = int(SAMPLE_RATE * MIN_SPEECH_SECONDS)


class SlidingAudioBuffer:
    """
    A per-session audio ring buffer that accumulates PCM float32 audio
    and yields fixed-size overlapping windows for transcription.

    The buffer holds up to WINDOW_SAMPLES worth of audio.
    A new window is yielded each time STEP_SAMPLES of new audio has arrived.
    """

    def __init__(self):
        # Use a deque to efficiently pop from the left
        self._buffer: deque[float] = deque(maxlen=WINDOW_SAMPLES)
        self._samples_since_last_step: int = 0
        self._lock = Lock()

    def add_samples(self, samples: np.ndarray) -> np.ndarray | None:
        """
        Add new float32 audio samples to the buffer.
        Returns a numpy window if enough new audio has accumulated, else None.
        """
        with self._lock:
            for sample in samples:
                self._buffer.append(sample)

            self._samples_since_last_step += len(samples)

            if self._samples_since_last_step >= STEP_SAMPLES:
                self._samples_since_last_step = 0
                window = np.array(self._buffer, dtype=np.float32)
                if len(window) >= MIN_SPEECH_SAMPLES:
                    return window
        return None

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._samples_since_last_step = 0
