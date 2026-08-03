"""
Download KArSL-502 Landmarks dataset and train the model using all 3 signers.
This uses pre-extracted MediaPipe landmarks (parquet files) instead of raw video frames,
which is ~100x faster than extracting from video.

Usage:
    python train_from_landmarks.py
    python train_from_landmarks.py --landmarks_dir /path/to/landmarks --epochs 50
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.DataLoader import actions, arabic_labels, N_FRAMES, N_KEYPOINTS
from app.shared import build_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def download_landmarks(landmarks_dir: str, kaggle_dataset: str) -> bool:
    """Download the landmarks dataset from Kaggle if parquet files are missing."""
    parquet_dir = os.path.join(landmarks_dir, "train_landmark_files")
    if os.path.exists(parquet_dir):
        count = sum(1 for r, d, f in os.walk(parquet_dir) for fn in f if fn.endswith('.parquet'))
        if count > 0:
            logger.info("Landmarks already downloaded (%d parquet files)", count)
            return True

    print("=" * 60)
    print("Downloading KArSL-502 Landmarks from Kaggle...")
    print("(Pre-extracted MediaPipe landmarks — much smaller than video)")
    print("=" * 60)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            kaggle_dataset,
            path=landmarks_dir,
            unzip=True,
            quiet=False
        )
        print("Download complete!")
        return True
    except Exception as e:
        logger.error("Error downloading: %s", e)
        return False


def load_parquet_landmarks(parquet_path: str, sign_id: str):
    """
    Load a single parquet file and convert landmarks to the same 225-keypoint
    format used by DataLoader.extract_keypoints().

    Uses O(n) grouped lookup instead of the previous O(n²) per-frame scan.
    """
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(parquet_path)
        data = table.to_pydict()
    except Exception:
        return None

    frames_col = data.get('frame', [])
    types_col = data.get('type', [])
    landmark_idx_col = data.get('landmark_index', [])
    x_col = data.get('x', [])
    y_col = data.get('y', [])
    z_col = data.get('z', [])

    if not frames_col:
        return None

    # Group by frame first — O(n) instead of O(n * frames)
    frame_data = defaultdict(list)
    for i in range(len(frames_col)):
        frame_data[frames_col[i]].append(i)

    keypoints_sequence = []

    for frame_num in sorted(frame_data.keys()):
        pose = np.zeros(33 * 3)
        left_hand = np.zeros(21 * 3)
        right_hand = np.zeros(21 * 3)

        for i in frame_data[frame_num]:
            ltype = types_col[i]
            lidx = landmark_idx_col[i]
            x, y, z = float(x_col[i]), float(y_col[i]), float(z_col[i])

            if ltype == 'pose' and lidx < 33:
                pose[lidx * 3] = x
                pose[lidx * 3 + 1] = y
                pose[lidx * 3 + 2] = z
            elif ltype == 'left_hand' and lidx < 21:
                left_hand[lidx * 3] = x
                left_hand[lidx * 3 + 1] = y
                left_hand[lidx * 3 + 2] = z
            elif ltype == 'right_hand' and lidx < 21:
                right_hand[lidx * 3] = x
                right_hand[lidx * 3 + 1] = y
                right_hand[lidx * 3 + 2] = z

        # Apply same landmark adjustment as DataLoader.extract_keypoints()
        nose = pose[:3]
        lh_wrist = left_hand[:3]
        rh_wrist = right_hand[:3]

        pose_adj = adjust_landmarks(pose, nose)
        lh_adj = adjust_landmarks(left_hand, lh_wrist)
        rh_adj = adjust_landmarks(right_hand, rh_wrist)

        keypoints = np.concatenate((pose_adj, lh_adj, rh_adj))
        keypoints_sequence.append(keypoints)

    # Pad/truncate to N_FRAMES
    if len(keypoints_sequence) > N_FRAMES:
        keypoints_sequence = keypoints_sequence[:N_FRAMES]
    while len(keypoints_sequence) < N_FRAMES:
        keypoints_sequence.append(np.zeros(N_KEYPOINTS))

    return np.array(keypoints_sequence)


def adjust_landmarks(arr: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Same as DataLoader.adjust_landmarks — center landmarks relative to reference point."""
    arr_reshaped = arr.reshape(-1, 3)
    center_repeated = np.tile(center, (len(arr_reshaped), 1))
    arr_adjusted = arr_reshaped - center_repeated
    return arr_adjusted.reshape(-1)


def extract_all_landmarks(landmarks_dir: str, cache_dir: str):
    """Extract keypoints from all parquet files for the target signs."""
    cache_x = os.path.join(cache_dir, "X_keypoints.npy")
    cache_y = os.path.join(cache_dir, "y_labels.npy")

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        X = np.load(cache_x)
        y = np.load(cache_y)
        logger.info("Loaded from cache: %d samples, %d classes", X.shape[0], len(set(y)))
        return X, y

    os.makedirs(cache_dir, exist_ok=True)

    csv_path = os.path.join(landmarks_dir, "train.csv")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info("Processing %d sequences from all 3 signers...", len(rows))

    target_sign_ids = set(actions.keys())
    X_list = []
    y_list = []
    skipped = 0
    matched = 0

    for idx, row in enumerate(rows):
        sign_id = row['sign_id']
        if sign_id not in target_sign_ids:
            continue

        matched += 1
        parquet_path = os.path.join(landmarks_dir, row['path'])

        if not os.path.exists(parquet_path):
            skipped += 1
            if skipped <= 5:
                logger.warning("  Missing: %s", parquet_path)
            continue

        if matched % 50 == 0 or matched == 1:
            label = actions[sign_id]
            arabic = arabic_labels.get(label, '?')
            logger.info(
                "  [%d] Processing signer %s, sign %s (%s), %s frames",
                matched, row['signer'], sign_id, arabic, row['num_frames'],
            )

        keypoints = load_parquet_landmarks(parquet_path, sign_id)
        if keypoints is not None:
            X_list.append(keypoints)
            y_list.append(actions[sign_id])
        else:
            skipped += 1

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info("Extraction complete: %d samples, %d classes, %d skipped",
                X.shape[0], len(set(y)), skipped)

    np.save(cache_x, X)
    np.save(cache_y, y)
    logger.info("Cache saved to %s", cache_dir)

    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train ArSL model from pre-extracted landmarks")
    parser.add_argument("--landmarks_dir", type=str,
                       default=r"D:\signo v6\datasets\karsl-landmarks",
                       help="Path to downloaded landmarks dataset")
    parser.add_argument("--cache_dir", type=str,
                       default=r"D:\signo v6\datasets\keypoints_cache",
                       help="Path for keypoint cache files")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to save trained model")
    parser.add_argument("--kaggle_dataset", type=str,
                       default="abdalrhmantwfik/karsl-502-landmarks",
                       help="Kaggle dataset identifier")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split ratio")

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "app", "conv1_lstm.keras"
        )

    print("=" * 60)
    print("Arabic Sign Language — Full 3-Signer Training Pipeline")
    print("=" * 60)
    print(f"Landmarks dir: {args.landmarks_dir}")
    print(f"Cache dir:     {args.cache_dir}")
    print(f"Model output:  {args.model_path}")
    print(f"Epochs:        {args.epochs}")
    print()

    # Step 1: Download landmarks if needed
    if not download_landmarks(args.landmarks_dir, args.kaggle_dataset):
        print("Failed to download landmarks. Aborting.")
        sys.exit(1)

    # Step 2: Extract keypoints from parquet files
    X, y = extract_all_landmarks(args.landmarks_dir, args.cache_dir)

    if len(X) == 0:
        print("ERROR: No samples extracted!")
        sys.exit(1)

    # Step 3: Remap labels to contiguous range [0, n_classes)
    unique_labels = sorted(np.unique(y))
    n_classes = len(unique_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    reverse_label_map = {new: old for old, new in label_map.items()}
    y_mapped = np.array([label_map[label] for label in y])

    # Save label mapping
    mapping_path = os.path.join(args.cache_dir, "label_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "label_to_idx": {str(int(k)): int(v) for k, v in label_map.items()},
            "idx_to_label": {str(int(k)): int(v) for k, v in reverse_label_map.items()},
            "idx_to_arabic": {str(int(v)): arabic_labels.get(int(k), "?") for k, v in label_map.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"Label mapping saved ({n_classes} classes)")

    # Step 4: 3-way split — train / val / test (fixes data leakage)
    from sklearn.model_selection import train_test_split

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_mapped, test_size=args.test_size, random_state=42, stratify=y_mapped
    )
    val_ratio = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 5: Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    np.save(os.path.join(args.cache_dir, "norm_mean.npy"), mean)
    np.save(os.path.join(args.cache_dir, "norm_std.npy"), std)
    print("Normalization stats saved")

    # Step 6: Build and train model
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import callbacks
    from sklearn.utils.class_weight import compute_class_weight

    model = build_model(n_classes)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Class weights
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes.astype(int), class_weights))

    # Callbacks
    cb_list = [
        callbacks.ModelCheckpoint(
            args.model_path, monitor='val_accuracy',
            save_best_only=True, verbose=1, mode='max'
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1, mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger(
            os.path.join(os.path.dirname(args.model_path), 'training_log.csv')
        )
    ]

    print(f"\nStarting training: {len(X_train)} samples, {n_classes} classes")
    print(f"Input shape: {X_train.shape}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1
    )

    # Step 7: Evaluate on held-out test set
    from sklearn.metrics import classification_report
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    unique_test_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = []
    for label in unique_test_labels:
        orig = reverse_label_map.get(label, label)
        arabic = arabic_labels.get(int(orig), f"class_{label}")
        target_names.append(f"{label}:{arabic}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    print(f"\nModel saved to: {args.model_path}")
    print("Done! The model is ready for the web app.")


if __name__ == "__main__":
    main()
