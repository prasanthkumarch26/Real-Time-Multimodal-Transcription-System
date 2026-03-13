import webrtcvad
import numpy as np

# Initialize VAD with aggressiveness level from 0 to 3
vad = webrtcvad.Vad(2)

def is_speech(audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
    """
    Check if a given numpy array containing audio data contains speech.
    Since webrtcvad expects precise 10, 20, or 30 ms frames of 16-bit PCM,
    we'll frame the data and check if any frame contains speech.
    
    audio_data is expected to be a 1D numpy array of float32 or int16.
    """
    # Convert float32 to int16 PCM if necessary
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        pcm_data = (audio_data * 32767).astype(np.int16)
    else:
        pcm_data = audio_data.astype(np.int16)

    raw_bytes = pcm_data.tobytes()
    frame_duration_ms = 30
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2) # 2 bytes per sample

    speech_frames = 0
    total_frames = 0

    for i in range(0, len(raw_bytes) - frame_size + 1, frame_size):
        frame = raw_bytes[i:i + frame_size]
        try:
            if vad.is_speech(frame, sample_rate):
                speech_frames += 1
        except Exception as e:
            pass
        total_frames += 1

    # If any frame has speech, consider the chunk as containing speech
    # or require a ratio (e.g., > 10% frames are speech)
    if total_frames == 0:
        return False
        
    return (speech_frames / total_frames) > 0.05
