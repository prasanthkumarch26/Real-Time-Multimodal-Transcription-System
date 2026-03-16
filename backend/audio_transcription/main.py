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

    # Per-session state
    session_buffer = SlidingAudioBuffer()
    context_prompt = ""  # Rolling context for Whisper initial_prompt

    # Speech / silence state machine
    # After SILENCE_CHUNKS_THRESHOLD consecutive silent 30ms chunks (~300ms),
    # we consider the utterance finished and clear the buffer.
    SILENCE_CHUNKS_THRESHOLD = 10
    silence_streak = 0
    is_active = False   # True while the user is speaking

    try:
        while True:
            # Receive raw Float32 PCM bytes from the AudioWorklet (16kHz, mono)
            audio_bytes = await websocket.receive_bytes()

            try:
                samples = np.frombuffer(audio_bytes, dtype=np.float32).copy()
                if len(samples) == 0:
                    continue

                # ── 1. Voice Activity Detection ──────────────────────────────
                speech_detected = is_speech(samples, sample_rate=16000)

                if not speech_detected:
                    silence_streak += 1

                    if silence_streak >= SILENCE_CHUNKS_THRESHOLD and is_active:
                        # Utterance just ended — wipe stale audio from buffer
                        # so Whisper cannot re-transcribe old speech during silence
                        session_buffer.clear()
                        is_active = False
                        context_prompt = ""   # fresh context for next utterance
                        logger.info("Speech ended — buffer cleared, now idle")

                    # Always skip transcription while silent
                    continue

                # ── 2. Speech active ──────────────────────────────────────────
                silence_streak = 0
                if not is_active:
                    is_active = True
                    logger.info("Speech started")

                # ── 3. Accumulate into sliding window ─────────────────────────
                window = session_buffer.add_samples(samples)
                if window is None:
                    continue   # window not full yet

                # ── 4. Noise reduction ────────────────────────────────────────
                cleaned_audio = reduce_noise(window, sample_rate=16000)

                # ── 5. Transcribe with context prompting ──────────────────────
                text = transcriber.transcribe(cleaned_audio, initial_prompt=context_prompt)

                if text.strip():
                    # Cap context at last 100 words to avoid prompt bloat
                    all_words = (context_prompt + " " + text).split()
                    context_prompt = " ".join(all_words[-100:])

                    response = {"id": "AT", "Transcription": text}
                    await websocket.send_text(json.dumps(response))
                    logger.info(f"Transcribed: {text}")

            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info("Client disconnected from /attranscribe")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
