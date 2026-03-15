import numpy as np
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)


class AudioTranscriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initializes the faster-whisper model.
        - model_size: whisper model size (tiny, base, small, medium, large-v2)
        - device: 'cpu' or 'cuda'
        - compute_type: 'int8' for fast CPU, 'float16' for GPU
        """
        logger.info(f"Loading faster-whisper model: {model_size} on {device} ({compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded.")

    def transcribe(self, audio_data: np.ndarray, initial_prompt: str = "") -> str:
        """
        Transcribes a numpy float32 audio window.

        Args:
            audio_data: 1D float32 numpy array at 16kHz.
            initial_prompt: Previous transcription text for context.
                            Whisper uses this to avoid repeated or broken sentences.

        Returns:
            Transcribed text as a string.
        """
        if len(audio_data) == 0:
            return ""

        kwargs = {
            "beam_size": 5,
            "language": "en",
            "condition_on_previous_text": True,
            "vad_filter": False,  # We do our own VAD upstream
        }

        # Pass previous text as context prompt if available
        if initial_prompt.strip():
            kwargs["initial_prompt"] = initial_prompt.strip()

        segments, _info = self.model.transcribe(audio_data, **kwargs)
        text = " ".join(segment.text for segment in segments)
        return text.strip()


# Global singleton — model is loaded once at startup
transcriber = AudioTranscriber()
