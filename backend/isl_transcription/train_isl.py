import os
import cv2
import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from services.mediapipe_extractor import extract_landmarks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "action.h5")
SEQUENCE_LENGTH = 30
FEATURES_LENGTH = 1692  # Tasks API: 33×4 pose + 478×3 face + 21×3 lh + 21×3 rh

def load_data():
    cache_x = os.path.join(os.path.dirname(__file__), "X.npy")
    cache_y = os.path.join(os.path.dirname(__file__), "y.npy")
    cache_a = os.path.join(os.path.dirname(__file__), "actions.npy")
    
    if os.path.exists(cache_x) and os.path.exists(cache_y) and os.path.exists(cache_a):
        logger.info("Found cached numpy arrays, skipping extraction...")
        return np.load(cache_x), np.load(cache_y), np.load(cache_a).tolist()

    sequences, labels = [], []
    
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset path {DATASET_PATH} does not exist.")
        return [], [], []

    # Get valid class directories
    actions = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    label_map = {label: num for num, label in enumerate(actions)}
    
    logger.info(f"Discovered {len(actions)} classes: {actions}")

    for action in actions:
        action_path = os.path.join(DATASET_PATH, action)
        videos = [v for v in os.listdir(action_path) if v.endswith(".MOV")]
        
        for video_filename in videos:
            video_path = os.path.join(action_path, video_filename)
            cap = cv2.VideoCapture(video_path)
            
            sequence_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # mediapipe_extractor expects JPEG bytes
                _, buffer = cv2.imencode('.jpg', frame)
                jpeg_bytes = buffer.tobytes()
                
                landmarks = extract_landmarks(jpeg_bytes)
                if landmarks is not None:
                    sequence_frames.append(landmarks)
                    
            cap.release()
            
            seq_len = len(sequence_frames)
            if seq_len == 0:
                logger.warning(f"No landmarks extracted from {video_filename}")
                continue
                
            # Sampling: Take evenly spaced frames up to SEQUENCE_LENGTH
            if seq_len > SEQUENCE_LENGTH:
                indices = np.linspace(0, seq_len - 1, SEQUENCE_LENGTH, dtype=int)
                sequence_frames = [sequence_frames[i] for i in indices]
            elif seq_len < SEQUENCE_LENGTH:
                # Pad with zeros
                padding = [np.zeros(FEATURES_LENGTH)] * (SEQUENCE_LENGTH - seq_len)
                sequence_frames.extend(padding)
                
            sequences.append(sequence_frames)
            labels.append(label_map[action])
            logger.info(f"Processed video {video_filename} for class '{action}'")

    X_arr = np.array(sequences)
    y_arr = np.array(labels)
    
    np.save(cache_x, X_arr)
    np.save(cache_y, y_arr)
    np.save(cache_a, np.array(actions))
    
    return X_arr, y_arr, actions

def build_model(num_classes):
    model = Sequential()
    # Using default tanh activation which cuDNN optimizes for GPU
    model.add(LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES_LENGTH)))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=False))
    model.add(BatchNormalization())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    logger.info("Starting Data Loading from Dataset...")
    X, y, actions = load_data()
    
    if len(X) == 0:
        logger.error("No data collected")
        return
        
    logger.info(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    
    y_cat = to_categorical(y).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=42)
    
    logger.info("Building LSTM model...")
    model = build_model(len(actions))
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_categorical_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]
    
    logger.info("Training model...")
    model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_test, y_test), callbacks=callbacks)
    
    logger.info(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
    
    # Save vocabulary
    actions_path = os.path.join(os.path.dirname(__file__), "models", "actions.txt")
    with open(actions_path, "w") as f:
        f.write(",".join(actions))
    logger.info(f"Saved vocabulary actions to {actions_path}")

if __name__ == "__main__":
    main()
