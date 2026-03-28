import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import logging
import time
import functools

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

MAX_CONCURRENT_CONNECTIONS = 100
active_connections = 0
queue_drops = 0
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "audio_transcription", "active_connections": active_connections, "queue_drops": queue_drops}


@app.websocket("/attranscribe")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections
    if active_connections >= MAX_CONCURRENT_CONNECTIONS:
        logger.warning("Connection rejected: Max concurrent connections reached (Backpressure)")
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    await websocket.accept()
    active_connections += 1
    logger.info(f"Client connected to /attranscribe. Active connections: {active_connections}")

    # Per-session state
    session_buffer = SlidingAudioBuffer()
    context_prompt = ""  # Rolling context for Whisper initial_prompt

    SILENCE_CHUNKS_THRESHOLD = 10
    silence_streak = 0
    is_active = False   # True while the user is speaking

    loop = asyncio.get_event_loop()
    
    # Bounded queue for tail-drop policy (backpressure)
    chunk_queue = asyncio.Queue(maxsize=15)

    async def process_audio():
        nonlocal silence_streak, is_active, context_prompt
        ttft_logged = False
        start_time = None
        
        while True:
            try:
                item = await chunk_queue.get()
                if item is None:  # Shutdown signal
                    break
                samples, queued_at = item
                queue_wait_ms = (time.time() - queued_at) * 1000
                    
                # ── 1. Voice Activity Detection ──────────────────────────────
                speech_detected = is_speech(samples, sample_rate=16000)

                if not speech_detected:
                    silence_streak += 1

                    if silence_streak >= SILENCE_CHUNKS_THRESHOLD and is_active:
                        session_buffer.clear()
                        is_active = False
                        context_prompt = ""   
                        logger.info("Speech ended — buffer cleared, now idle")
                    continue

                # ── 2. Speech active ──────────────────────────────────────────
                silence_streak = 0
                if not is_active:
                    is_active = True
                    start_time = time.time()
                    logger.info("Speech started")

                # ── 3. Accumulate into sliding window ─────────────────────────
                window = session_buffer.add_samples(samples)
                if window is None:
                    continue   

                # ── 4. Noise reduction (offloaded) ───────────────────────────
                nr_func = functools.partial(reduce_noise, window, sample_rate=16000)
                cleaned_audio = await loop.run_in_executor(executor, nr_func)

                # ── 5. Transcribe with context prompting (offloaded) ─────────
                tr_func = functools.partial(transcriber.transcribe, cleaned_audio, initial_prompt=context_prompt)
                
                t0_inf = time.time()
                text = await loop.run_in_executor(executor, tr_func)
                inference_ms = (time.time() - t0_inf) * 1000

                if text.strip():
                    if not ttft_logged and start_time:
                        ttft = (time.time() - start_time) * 1000
                        logger.info(f"Time to First Token (TTFT): {ttft:.1f}ms")
                        ttft_logged = True

                    # Cap context at last 100 words to avoid prompt bloat
                    all_words = (context_prompt + " " + text).split()
                    context_prompt = " ".join(all_words[-100:])

                    response = {
                        "id": "AT", 
                        "Transcription": text,
                        "metrics": {
                            "queue_wait_ms": round(queue_wait_ms, 2),
                            "inference_ms": round(inference_ms, 2)
                        }
                    }
                    await websocket.send_text(json.dumps(response))
                    logger.info(f"Transcribed: {text} | QWait: {queue_wait_ms:.1f}ms | Inf: {inference_ms:.1f}ms")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)

    processor_task = asyncio.create_task(process_audio())

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            try:
                samples = np.frombuffer(audio_bytes, dtype=np.float32).copy()
                if len(samples) == 0:
                    continue
                
                # Implement Tail-drop policy: Drop chunk if queue is full
                if chunk_queue.full():
                    global queue_drops
                    queue_drops += 1
                    logger.warning("Queue full: Tail-dropping audio chunk to maintain real-time SLA")
                    continue
                
                await chunk_queue.put((samples, time.time()))
            except Exception as e:
                logger.error(f"Error receiving audio chunk: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info("Client disconnected from /attranscribe")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        active_connections -= 1
        processor_task.cancel()
        logger.info(f"Cleaned up connection. Active connections: {active_connections}")
