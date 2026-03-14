import numpy as np
from faster_whisper import WhisperModel

class AudioTranscriber:
    def __init__(self, model_size="distil-large-v3", device="cpu", compute_type="int8"):
        """
        Initializes the faster-whisper model.
        Using 'distil-large-v3' model and 'int8' for faster CPU execution by default.
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribes continuous audio data (numpy array).
        faster-whisper expects a 1D numpy array of float32 containing the audio samples at 16kHz.
        """
        segments, info = self.model.transcribe(audio_data, beam_size=5)
        
        text = " ".join([segment.text for segment in segments])
        return text.strip()

# Global transcriber instance
transcriber = AudioTranscriber()
