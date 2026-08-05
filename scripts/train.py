"""
Training script for Arabic Sign Language CNN-LSTM model.
Extracts MediaPipe keypoints from KArSL-502 RGB frames and trains
a Conv1D + Bidirectional LSTM model for sign classification.

Usage:
    python train.py --data_dir "D:/signo v6/datasets/karsl-502" --epochs 100
"""

import os
import sys
import json
import argparse
import logging

# Reconfigure stdout/stderr to UTF-8 for printing Arabic text in Windows terminals
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path for app package imports (script is now in scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.DataLoader import DataLoader, actions, arabic_labels, N_FRAMES, N_KEYPOINTS
from app.shared import discover_sequences, build_model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def extract_keypoints_from_sequences(sequences, cache_dir, max_per_class=None):
    """
    Extract MediaPipe keypoints from image frame sequences.
    Caches results to disk for reuse.

    Args:
        sequences: list of (frames_dir, sign_id) tuples
        cache_dir: directory to save extracted keypoints
        max_per_class: limit samples per class (for debugging)

    Returns:
        X: numpy array of shape (n_samples, N_FRAMES, N_KEYPOINTS)
        y: numpy array of class labels
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_x_path = os.path.join(cache_dir, "X_keypoints.npy")
    cache_y_path = os.path.join(cache_dir, "y_labels.npy")

    # Load from cache if available
    if os.path.exists(cache_x_path) and os.path.exists(cache_y_path):
        logger.info("Loading cached keypoints...")
        X = np.load(cache_x_path, mmap_mode='r')
        y = np.load(cache_y_path)
        logger.info("Loaded %d samples from cache", X.shape[0])
        return X, y

    logger.info("Extracting keypoints from %d sequences...", len(sequences))
    logger.info("This will take a while (MediaPipe processes each frame)...")

    X_list = []
    y_list = []

    # Count per class
    class_counts = {}
    skipped = 0

    import mediapipe as mp
    mp_holistic = mp.solutions.holistic

    for idx, (frames_dir, sign_id) in enumerate(sequences):
        label = actions[sign_id]

        # Limit per class if specified
        if max_per_class and class_counts.get(label, 0) >= max_per_class:
            continue

        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info(
                "  Processing sequence %d/%d (sign: %s -> %s) [%d samples extracted]",
                idx + 1, len(sequences), sign_id, arabic_labels.get(label, '?'), len(X_list),
            )

        try:
            from PIL import Image

            results_list = []
            frame_files = sorted([
                f for f in os.listdir(frames_dir)
                if f.lower().endswith(('.jpg', '.png'))
            ])

            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as holistic:
                for fname in frame_files:
                    image_path = os.path.join(frames_dir, fname)
                    image = np.array(Image.open(image_path).convert('RGB'))

                    results = holistic.process(image)
                    keypoints = DataLoader.extract_keypoints(results)
                    results_list.append(keypoints)

            # Truncate if too many frames
            if len(results_list) > N_FRAMES:
                results_list = results_list[:N_FRAMES]

            # Pad if too few frames
            while len(results_list) < N_FRAMES:
                results_list.append(np.zeros(N_KEYPOINTS))

            X_list.append(np.array(results_list))
            y_list.append(label)
            class_counts[label] = class_counts.get(label, 0) + 1

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                logger.warning("  Skipped %s: %s", frames_dir, e)

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info("Extraction complete: %d samples, %d classes", X.shape[0], len(set(y)))
    logger.info("Skipped: %d sequences", skipped)

    # Save cache
    logger.info("Saving cache to %s...", cache_dir)
    np.save(cache_x_path, X)
    np.save(cache_y_path, y)

    return X, y


class KeypointSequence(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size=32, shuffle=True):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        return np.array(batch_x, dtype='float32'), np.array(batch_y, dtype='int32')

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_model(X_train, y_train, X_val, y_val, n_classes, model_save_path,
                class_weight_dict=None, epochs=100):
    """
    Train the model with best practices for accuracy:
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Class weights for imbalanced data
    """
    model = build_model(n_classes)

    # Compile with Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks for best performance
    cb_list = [
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.CSVLogger(
            os.path.join(os.path.dirname(model_save_path), 'training_log.csv')
        )
    ]

    # Initialize generators to stream batch data in-place
    train_gen = KeypointSequence(X_train, y_train, batch_size=32, shuffle=True)
    val_gen = KeypointSequence(X_val, y_val, batch_size=32, shuffle=False)

    logger.info("Training with %d samples, validating with %d samples", len(X_train), len(X_val))
    logger.info("Number of classes: %d", n_classes)
    logger.info("Input shape: %s", X_train.shape)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cb_list,
        class_weight=class_weight_dict,
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate model and print detailed metrics."""
    from sklearn.metrics import classification_report

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Detailed classification report
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Map labels to Arabic words for the report
    target_names = []
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    for label in unique_labels:
        arabic = arabic_labels.get(label, f"class_{label}")
        target_names.append(f"{label}:{arabic}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Arabic Sign Language CNN-LSTM Model")
    parser.add_argument("--data_dir", type=str,
                       default=r"D:\signo v6\datasets\karsl-502",
                       help="Path to KArSL-502 dataset")
    parser.add_argument("--cache_dir", type=str,
                       default=r"D:\signo v6\datasets\keypoints_cache",
                       help="Path to cache extracted keypoints")
    parser.add_argument("--model_path", type=str,
                       default=None,
                       help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Max training epochs")
    parser.add_argument("--max_per_class", type=int, default=None,
                       help="Max samples per class (for quick testing)")
    parser.add_argument("--test_size", type=float, default=0.15,
                       help="Test split ratio (held-out evaluation)")
    parser.add_argument("--val_size", type=float, default=0.15,
                       help="Validation split ratio (from training set)")

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app", "conv1_lstm.keras"
        )

    print("="*60)
    print("Arabic Sign Language - CNN-LSTM Training Pipeline")
    print("="*60)
    print(f"Dataset:    {args.data_dir}")
    print(f"Cache:      {args.cache_dir}")
    print(f"Model:      {args.model_path}")
    print(f"Epochs:     {args.epochs}")
    print(f"Test size:  {args.test_size}")
    print(f"Val size:   {args.val_size}")
    print()

    # Step 1: Discover sequences (using shared utility)
    sequences = discover_sequences(args.data_dir)

    if len(sequences) == 0:
        print("ERROR: No sequences found! Check your data_dir path.")
        print("Expected structure: signer/session/split/signID/session_folder/frames.jpg")
        sys.exit(1)

    # Step 2: Extract keypoints
    X, y = extract_keypoints_from_sequences(
        sequences,
        args.cache_dir,
        max_per_class=args.max_per_class
    )

    # Step 3: Remap labels to contiguous range [0, n_classes)
    unique_labels = sorted(np.unique(y))
    n_classes = len(unique_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    reverse_label_map = {new: old for old, new in label_map.items()}
    y_mapped = np.array([label_map[label] for label in y])

    # Save label mapping
    mapping_path = os.path.join(args.cache_dir, "label_mapping.json")
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "label_to_idx": {str(int(k)): int(v) for k, v in label_map.items()},
            "idx_to_label": {str(int(k)): int(v) for k, v in reverse_label_map.items()},
            "idx_to_arabic": {str(int(v)): arabic_labels.get(int(k), "?") for k, v in label_map.items()}
        }, f, ensure_ascii=False, indent=2)

    print(f"\nLabel mapping saved to {mapping_path}")
    print(f"Total samples: {len(X)}, Classes: {n_classes}")

    # Step 4: 3-way split — train / val / test (fixes data leakage from previous version)
    unique_vals, class_counts = np.unique(y_mapped, return_counts=True)
    min_count = class_counts.min() if len(class_counts) > 0 else 0
    stratify_option = y_mapped if min_count >= 2 else None
    if stratify_option is not None:
        print("Using stratified splits.")
    else:
        print("Some classes have only 1 sample. Falling back to non-stratified split.")

    # First split: separate held-out test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_mapped,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify_option
    )

    # Second split: separate validation from training
    stratify_tv = y_trainval if stratify_option is not None else None
    val_ratio = args.val_size / (1.0 - args.test_size)  # Adjust ratio relative to trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=42,
        stratify=stratify_tv
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    # Apply normalization (avoid in-place on mmap arrays)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Save normalization stats
    np.save(os.path.join(args.cache_dir, "norm_mean.npy"), mean)
    np.save(os.path.join(args.cache_dir, "norm_std.npy"), std)

    # Compute class weights for imbalanced data
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes.astype(int), class_weights))

    # Step 5: Train (validation set is truly separate from test set)
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        n_classes,
        args.model_path,
        class_weight_dict=class_weight_dict,
        epochs=args.epochs
    )

    # Step 6: Evaluate on held-out test set (no leakage)
    accuracy = evaluate_model(model, X_test, y_test)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best model saved to: {args.model_path}")
    print(f"Final test accuracy: {accuracy*100:.2f}%")
    print(f"\nThe model is ready for use with the Django web application!")


if __name__ == "__main__":
    main()
