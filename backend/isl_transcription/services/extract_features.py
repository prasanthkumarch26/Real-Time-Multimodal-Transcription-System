import os
import cv2
import numpy as np
import logging
from mediapipe_extractor import extract_landmarks, TOTAL_FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_features")

def process_video(video_path: str) -> np.ndarray:
    """Read a video and extract landmarks frame-by-frame."""
    cap = cv2.VideoCapture(video_path)
    frames_features = []
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return np.empty((0, TOTAL_FEATURES))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Encode as JPEG bytes for mediapipe_extractor
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            logger.warning(f"Failed to encode frame {frame_count} of {video_path}")
            continue
            
        jpeg_bytes = encoded_image.tobytes()
        features = extract_landmarks(jpeg_bytes)
        
        if features is not None:
            frames_features.append(features)
        else:
            # If nothing detected, append zeros to maintain temporal structure
            frames_features.append(np.zeros(TOTAL_FEATURES, dtype=np.float32))
            
        frame_count += 1
        
    cap.release()
    return np.array(frames_features, dtype=np.float32)

def main():
    if not os.path.exists(DATASET_DIR):
        logger.error(f"Dataset directory not found: {DATASET_DIR}")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    logger.info(f"Found {len(classes)} classes: {classes}")
    
    total_processed = 0
    total_skipped = 0
    
    for class_name in classes:
        class_in_dir = os.path.join(DATASET_DIR, class_name)
        class_out_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_out_dir, exist_ok=True)
        
        videos = [v for v in os.listdir(class_in_dir) if v.lower().endswith(('.mov', '.mp4', '.avi'))]
        logger.info(f"Processing class '{class_name}' with {len(videos)} videos...")
        
        for video_name in videos:
            video_path = os.path.join(class_in_dir, video_name)
            out_name = os.path.splitext(video_name)[0] + ".npy"
            out_path = os.path.join(class_out_dir, out_name)
            
            # Skip if already processed
            if os.path.exists(out_path):
                logger.info(f"Skipping {video_name} - already extracted.")
                total_skipped += 1
                continue
                
            logger.info(f"Extracting features from {video_name}...")
            features = process_video(video_path)
            
            if features.shape[0] > 0:
                np.save(out_path, features)
                total_processed += 1
            else:
                logger.warning(f"No frames extracted from {video_name}")
                
    logger.info(f"Extraction complete. Processed {total_processed} new videos, skipped {total_skipped}.")

if __name__ == "__main__":
    main()
