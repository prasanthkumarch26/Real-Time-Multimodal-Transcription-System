from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import numpy as np
import logging

from services.mediapipe_extractor import extract_landmarks
from services.lstm_predictor import predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ISL Transcription Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "isl_transcription"}


@app.websocket("/isltranscribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ISL Client connected to /isltranscribe")

    # Sequence buffer: collect N frames before predicting
    sequence: list[np.ndarray] = []
    SEQUENCE_LENGTH = 30  # LSTM expects 30 frames

    try:
        while True:
            # Receive JSON message with base64-encoded JPEG frame
            text_data = await websocket.receive_text()

            try:
                payload = json.loads(text_data)
                frame_data_url = payload.get("frame", "")

                if not frame_data_url:
                    continue

                # Decode base64 JPEG to numpy image
                header, encoded = frame_data_url.split(",", 1)
                img_bytes = base64.b64decode(encoded)

                # Extract MediaPipe landmarks from frame
                landmarks = extract_landmarks(img_bytes)

                if landmarks is not None:
                    sequence.append(landmarks)

                    # Keep a rolling window of SEQUENCE_LENGTH frames
                    if len(sequence) > SEQUENCE_LENGTH:
                        sequence.pop(0)

                    # Only predict when we have a full sequence
                    if len(sequence) == SEQUENCE_LENGTH:
                        input_array = np.array(sequence, dtype=np.float32)
                        word = predictor.predict(input_array)

                        if word:
                            response = {
                                "id": "ISL",
                                "Transcription": word,
                            }
                            await websocket.send_text(json.dumps(response))
                            logger.info(f"ISL Prediction: {word}")

            except Exception as e:
                logger.error(f"Error processing ISL frame: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info("ISL Client disconnected")
    except Exception as e:
        logger.error(f"ISL WebSocket error: {e}", exc_info=True)
