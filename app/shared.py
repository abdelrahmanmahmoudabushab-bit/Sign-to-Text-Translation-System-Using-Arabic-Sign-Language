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

    Structure: data_dir / signer / session / split / signID / session_folder / frames.jpg

    Args:
        data_dir: Root path of the KArSL-502 dataset.
        target_sign_ids: Set of sign ID strings to filter (e.g. {'0001', '0002', ...}).
                         If None, imports `actions` from DataLoader and uses its keys.

    Returns:
        List of (frames_dir_path, sign_id_str) tuples for the target signs.
    """
    if target_sign_ids is None:
        from app.DataLoader import actions
        target_sign_ids = set(actions.keys())

    sequences: List[Tuple[str, str]] = []

    logger.info("Scanning dataset directory: %s", data_dir)
    logger.info("Looking for %d target sign IDs...", len(target_sign_ids))

    if not os.path.exists(data_dir):
        logger.warning("Dataset directory does not exist: %s", data_dir)
        return sequences

    try:
        signers = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for signer in signers:
            signer_path = os.path.join(data_dir, signer)
            sessions = [d for d in os.listdir(signer_path) if os.path.isdir(os.path.join(signer_path, d))]
            for session in sessions:
                session_path = os.path.join(signer_path, session)
                splits = [d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))]
                for split in splits:
                    split_path = os.path.join(session_path, split)
                    sign_ids = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
                    for sign_id in sign_ids:
                        if sign_id not in target_sign_ids:
                            continue
                        sign_path = os.path.join(split_path, sign_id)
                        try:
                            folders = os.listdir(sign_path)
                            for folder in folders:
                                folder_path = os.path.join(sign_path, folder)
                                sequences.append((folder_path, sign_id))
                        except Exception:
                            pass
    except Exception as e:
        logger.exception("Error during dataset sequence discovery: %s", e)

    n_classes = len(set(s[1] for s in sequences))
    logger.info("Found %d sequences across %d sign classes", len(sequences), n_classes)
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
