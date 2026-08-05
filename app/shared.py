"""
Shared utilities for the Arabic Sign Language Translation System.

Contains functions that were previously duplicated across multiple scripts:
- discover_sequences(): KArSL-502 dataset directory walker
- build_model(): CNN-LSTM model architecture definition
"""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def discover_sequences(
    data_dir: str,
    target_sign_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Walk the KArSL-502 directory structure to find all frame sequences.
    Optimized with os.scandir and JSON caching.
    """
    import json
    if target_sign_ids is None:
        from app.DataLoader import actions
        target_sign_ids = set(actions.keys())

    logger.info("Scanning dataset directory: %s", data_dir)

    if not os.path.exists(data_dir):
        logger.warning("Dataset directory does not exist: %s", data_dir)
        return []

    cache_path = os.path.join(data_dir, "discovered_sequences_cache.json")
    all_sequences = []

    # Try loading cache first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # Ensure the cache structure is what we expect
            if isinstance(cached_data, list):
                all_sequences = [(os.path.join(data_dir, item[0]), item[1]) for item in cached_data]
                logger.info("Loaded %d sequences from cache: %s", len(all_sequences), cache_path)
        except Exception as e:
            logger.warning("Failed to load sequences cache: %s. Re-scanning...", e)
            all_sequences = []

    if not all_sequences:
        logger.info("Cache not found, invalid, or empty. Scanning directory structure...")
        try:
            raw_discovered = []
            # Use os.scandir for high performance traversal
            with os.scandir(data_dir) as signers:
                for signer in signers:
                    if signer.is_dir():
                        with os.scandir(signer.path) as sessions:
                            for session in sessions:
                                if session.is_dir():
                                    with os.scandir(session.path) as splits:
                                        for split in splits:
                                            if split.is_dir():
                                                with os.scandir(split.path) as sign_ids:
                                                    for sign_id in sign_ids:
                                                        if sign_id.is_dir():
                                                            with os.scandir(sign_id.path) as folders:
                                                                for folder in folders:
                                                                    if folder.is_dir():
                                                                        # Verify folder has frames before caching
                                                                        try:
                                                                            if any(f.name.lower().endswith(('.jpg', '.png')) for f in os.scandir(folder.path)):
                                                                                rel_path = os.path.relpath(folder.path, data_dir)
                                                                                raw_discovered.append((rel_path, sign_id.name))
                                                                                all_sequences.append((folder.path, sign_id.name))
                                                                        except OSError:
                                                                            pass
            
            # Save raw relative paths to cache
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(raw_discovered, f, indent=2)
                logger.info("Saved %d valid sequences to cache: %s", len(raw_discovered), cache_path)
            except Exception as e:
                logger.warning("Failed to save sequences cache: %s", e)
        except Exception as e:
            logger.exception("Error during dataset sequence discovery: %s", e)

    # Filter by target sign ids
    sequences = [(path, sign_id) for path, sign_id in all_sequences if sign_id in target_sign_ids]
    n_classes = len(set(s[1] for s in sequences))
    logger.info("Found %d matching sequences across %d sign classes", len(sequences), n_classes)
    return sequences


def build_model(
    n_classes: int,
    n_frames: int = 60,
    n_keypoints: int = 225,
):
    """
    Build CNN-LSTM model for sign language classification.

    Architecture: Conv1D blocks → Bidirectional LSTM → Dense → Softmax

    This is the single source of truth for the model architecture.
    Both train.py and train_from_landmarks.py import from here.

    Args:
        n_classes: Number of output classes.
        n_frames: Temporal sequence length (default 60).
        n_keypoints: Feature vector size per frame (default 225).

    Returns:
        Uncompiled Keras Sequential model.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # Input
        layers.Input(shape=(n_frames, n_keypoints)),

        # Conv1D block 1 - extract local temporal features
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Conv1D block 2 - deeper features
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Bidirectional LSTM - capture temporal dependencies
        layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3)),
        layers.Bidirectional(layers.LSTM(128, dropout=0.3)),

        # Dense classification head
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Output
        layers.Dense(n_classes, activation='softmax')
    ])

    return model
