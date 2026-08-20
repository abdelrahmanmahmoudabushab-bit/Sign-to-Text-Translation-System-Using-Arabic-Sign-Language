#!/usr/bin/env python3
"""
JSL Keypoint Extractor (Resumable Version)

Processes all downloaded JSL sign videos through MediaPipe Holistic,
extracts normalized 225-D keypoint sequences, synchronizes indices with
jsl_manifest.json (removing any failed downloads), and saves coordinate data
to datasets/jsl_keypoints.npy.

Implements file-level caching in datasets/jsl_keypoints_cache/ to allow
immediate recovery and resumption in case of external hangs or system restarts.
"""

import os
import json
import logging
import numpy as np

# Reconfigure console output encoding
import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.DataLoader import DataLoader, N_FRAMES, N_KEYPOINTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(DB_DIR, "datasets", "jsl_manifest.json")
VIDEOS_DIR = os.path.join(DB_DIR, "datasets", "jsl", "videos")
CACHE_DIR = os.path.join(DB_DIR, "datasets", "jsl_keypoints_cache")
OUTPUT_KEYPOINTS_PATH = os.path.join(DB_DIR, "datasets", "jsl_keypoints.npy")

def extract_all():
    if not os.path.exists(MANIFEST_PATH):
        logger.error("Manifest not found at %s. Run crawl_soosl.py first.", MANIFEST_PATH)
        return

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    logger.info("Verifying downloaded videos...")
    valid_manifest = []
    video_paths = []
    
    for item in manifest:
        filename = item.get("videoFilename")
        if not filename:
            continue
        full_path = os.path.join(VIDEOS_DIR, filename)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 1024:
            valid_manifest.append(item)
            video_paths.append(full_path)

    total_files = len(video_paths)
    logger.info("Found %d downloaded videos ready for feature extraction.", total_files)
    
    if total_files == 0:
        logger.warning("No valid videos found. Run download_jsl_videos.py first.")
        return

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    x_data = []
    skipped_count = 0

    logger.info("Starting MediaPipe Holistic keypoint extraction (resumable)...")
    for idx, path in enumerate(video_paths):
        cache_file = os.path.join(CACHE_DIR, f"{idx}.npy")
        
        # Try loading from cache
        if os.path.exists(cache_file):
            try:
                x_extracted = np.load(cache_file)
                x_data.append(x_extracted)
                skipped_count += 1
                if skipped_count % 200 == 0:
                    logger.info("Loaded %d cached video keypoints...", skipped_count)
                continue
            except Exception as e:
                logger.warning("Cache file %s corrupted, re-extracting: %s", cache_file, e)

        # Log target filename before processing (so we instantly know the culprit if it hangs)
        filename = os.path.basename(path)
        logger.info("[%d/%d] Processing video: %s", idx + 1, total_files, filename)

        try:
            # load_inference_data returns shape (1, 60, 225)
            x_extracted = DataLoader.load_inference_data(path)
            
            if x_extracted is not None:
                # Remove batch dimension to store as (60, 225)
                features = x_extracted[0]
            else:
                features = np.zeros((N_FRAMES, N_KEYPOINTS))
                
        except Exception as e:
            logger.warning("Failed to extract keypoints from %s: %s", filename, e)
            features = np.zeros((N_FRAMES, N_KEYPOINTS))

        # Save to cache
        np.save(cache_file, features)
        x_data.append(features)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
            logger.info("Processed %d/%d video frames...", idx + 1, total_files)

    # Convert to NumPy array
    x_array = np.array(x_data, dtype=np.float32)
    logger.info("Extracted coordinates dataset shape: %s", str(x_array.shape))

    # Save final unified coordinate array
    np.save(OUTPUT_KEYPOINTS_PATH, x_array)
    logger.info("Saved final unified keypoint dataset to: %s", OUTPUT_KEYPOINTS_PATH)

    # Save the synchronized manifest back to align indices with the embedding database
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(valid_manifest, f, ensure_ascii=False, indent=2)
    logger.info("Synchronized manifest written back to: %s", MANIFEST_PATH)

if __name__ == "__main__":
    extract_all()
