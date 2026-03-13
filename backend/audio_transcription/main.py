from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import io
from pydub import AudioSegment

from services.noise_reduction import reduce_noise
from services.vad import is_speech
from services.transcriber import transcriber

app = FastAPI(title="Audio Transcription Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/attranscribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to /attranscribe")
    try:
        while True:
            # Receive audio chunk (bytes) from frontend
            audio_bytes = await websocket.receive_bytes()
            
            try:
                # Convert bytes to pydub AudioSegment
                # pydub can automatically detect format from file-like object using ffprobe/ffmpeg
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                
                # Convert to 16kHz mono (required by faster-whisper and VAD)
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                
                # Get raw audio data as numpy array (int16)
                samples = np.array(audio_segment.get_array_of_samples())
                
                # Normalize to float32 between -1.0 and 1.0
                samples_float32 = samples.astype(np.float32) / 32768.0
                
                # 1. Voice Activity Detection
                if not is_speech(samples_float32, sample_rate=16000):
                    # No speech detected, skip transcription
                    print("No speech detected softly skipping...")
                    continue
                    
                # 2. Noise reduction
                cleaned_audio = reduce_noise(samples_float32, sample_rate=16000)
                
                # 3. Transcribe using faster-whisper
                text = transcriber.transcribe(cleaned_audio)
                
                if text.strip():
                    response = {
                        "id": "AT",
                        "Transcription": text
                    }
                    await websocket.send_text(json.dumps(response))
                    print(f"Sent: {text}")
                
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
