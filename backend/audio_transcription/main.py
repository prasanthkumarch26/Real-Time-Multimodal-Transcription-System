from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import logging

from services.audio_buffer import SlidingAudioBuffer
from services.noise_reduction import reduce_noise
from services.vad import is_speech
from services.transcriber import transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Transcription Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "audio_transcription"}


@app.websocket("/attranscribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to /attranscribe")

    # Each WebSocket session gets its own buffer and context
    session_buffer = SlidingAudioBuffer()
    context_prompt = ""  # Rolling context for Whisper initial_prompt

    try:
        while True:
            # Receive raw PCM bytes from the AudioWorklet
            # Frontend sends Float32Array binary (4 bytes per sample at 16kHz, mono)
            audio_bytes = await websocket.receive_bytes()

            try:
                # Decode Float32 PCM sent by AudioWorklet
                samples = np.frombuffer(audio_bytes, dtype=np.float32).copy()

                if len(samples) == 0:
                    continue

                # 1. Voice Activity Detection on this incoming chunk
                if not is_speech(samples, sample_rate=16000):
                    logger.debug("Silence detected — skipping chunk")
                    continue

                # 2. Accumulate into the sliding window buffer
                window = session_buffer.add_samples(samples)

                if window is None:
                    # Not enough audio in the window yet
                    continue

                # 3. Noise reduction on the window
                cleaned_audio = reduce_noise(window, sample_rate=16000)

                # 4. Transcribe with context prompting
                text = transcriber.transcribe(cleaned_audio, initial_prompt=context_prompt)

                if text.strip():
                    # Update rolling context with last ~100 words to keep prompt short
                    all_words = (context_prompt + " " + text).split()
                    context_prompt = " ".join(all_words[-100:])

                    response = {
                        "id": "AT",
                        "Transcription": text,
                    }
                    await websocket.send_text(json.dumps(response))
                    logger.info(f"Transcribed: {text}")

            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info("Client disconnected from /attranscribe")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
