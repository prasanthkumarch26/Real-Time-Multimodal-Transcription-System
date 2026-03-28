import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset_features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SEQ_LEN = 30
FEATURES = 1692
EPOCHS = 200
BATCH_SIZE = 16  # smaller batch size since dataset is small

# Augmentation params
AUG_MULTIPLIER = 5
SCALE_MIN = 0.85
SCALE_MAX = 1.15
SHIFT_MAX = 0.05
NOISE_STD = 0.005

def augment_sequence(seq: np.ndarray) -> np.ndarray:
    frames = seq.shape[0]
    
    # Temporal Crop/Pad
    if frames > SEQ_LEN:
        start = np.random.randint(0, frames - SEQ_LEN)
        aug_seq = seq[start:start + SEQ_LEN].copy()
    elif frames < SEQ_LEN:
        pad_len = SEQ_LEN - frames
        aug_seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
    else:
        aug_seq = seq.copy()
        
    scale = np.random.uniform(SCALE_MIN, SCALE_MAX)
    shift_x = np.random.uniform(-SHIFT_MAX, SHIFT_MAX)
    shift_y = np.random.uniform(-SHIFT_MAX, SHIFT_MAX)
    
    # Apply to pose
    for i in range(0, 132, 4):
        mask = aug_seq[:, i] != 0
        if np.any(mask):
            aug_seq[mask, i] = aug_seq[mask, i] * scale + shift_x
            aug_seq[mask, i+1] = aug_seq[mask, i+1] * scale + shift_y
            
    # Apply to face, hands
    for i in range(132, 1692, 3):
        mask = aug_seq[:, i] != 0
        if np.any(mask):
            aug_seq[mask, i] = aug_seq[mask, i] * scale + shift_x
            aug_seq[mask, i+1] = aug_seq[mask, i+1] * scale + shift_y
            
    # Noise
    noise = np.random.normal(0, NOISE_STD, aug_seq.shape)
    mask_nonzero = aug_seq != 0
    aug_seq = aug_seq + (noise * mask_nonzero)
    
    return np.float32(aug_seq)

def extract_base_sequence(seq: np.ndarray) -> np.ndarray:
    frames = seq.shape[0]
    if frames > SEQ_LEN:
        start = (frames - SEQ_LEN) // 2
        res = seq[start:start + SEQ_LEN].copy()
    elif frames < SEQ_LEN:
        pad_len = SEQ_LEN - frames
        res = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
    else:
        res = seq.copy()
    return np.float32(res)

def main():
    if not os.path.exists(DATA_DIR):
        logger.error(f"Features directory not found: {DATA_DIR}")
        return
        
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(classes)
    logger.info(f"Found {num_classes} classes: {classes}")
    
    X, y = [], []
    
    # Load and augment data
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, class_name)
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        for npy_file in npy_files:
            seq = np.load(os.path.join(class_dir, npy_file))
            
            # Base exact sequence
            X.append(extract_base_sequence(seq))
            y.append(label)
            
            # Augmented sequences
            for _ in range(AUG_MULTIPLIER):
                X.append(augment_sequence(seq))
                y.append(label)
                
    X = np.array(X)
    y = to_categorical(y, num_classes)
    
    logger.info(f"Total sequences (including augmentations): {X.shape[0]}")
    logger.info(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQ_LEN, FEATURES)),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "action.h5")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    ]
    
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Save actions mapping
    actions_path = os.path.join(MODEL_DIR, "actions.txt")
    with open(actions_path, "w") as f:
        f.write(",".join(classes))
    logger.info(f"Model saved to {model_path}. Actions saved to {actions_path}.")

if __name__ == "__main__":
    main()
