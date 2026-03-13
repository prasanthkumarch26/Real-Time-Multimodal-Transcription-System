from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import io
# pydub removed as we handle raw PCM directly

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
                # Frontend now sends chunked raw 16kHz mono Int16 PCM bytes directly.
                samples = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Normalize to float32 between -1.0 and 1.0 (Required by VAD & Whisper)
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
