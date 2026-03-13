import numpy as np
import noisereduce as nr

def reduce_noise(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Applies noise reduction to the audio data using spectral gating.
    Expected input is a 1D float32 or int16 numpy array.
    """
    # Prevent crashing if audio chunk is too small
    if len(audio_data) < sample_rate // 2:
        return audio_data
        
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.8)
    return reduced_noise
