"""
Evaluate the coordinate-based CNN-LSTM model on all cached test samples
to measure accuracy and per-class performance across all words.
"""

import os
import sys
import numpy as np
import json
import time

# Reconfigure console output encoding to UTF-8 for Arabic rendering
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path (script is now in scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.util as util
from app.DataLoader import arabic_labels, actions

def main():
    print("=" * 60)
    print("SIGNO VOCABULARY EVALUATION RUNNER")
    print("=" * 60)

    # 1. Locate cache directories
    base_dir = r"D:\signo v6\datasets"
    cache_dirs = ["keypoints_cache_test_opt2", "keypoints_cache_test", "keypoints_cache"]
    
    selected_cache = None
    for d in cache_dirs:
        path = os.path.join(base_dir, d)
        if os.path.exists(os.path.join(path, "X_keypoints.npy")):
            selected_cache = path
            break

    if not selected_cache:
        print("Error: Could not locate a keypoints cache containing X_keypoints.npy")
        return

    print(f"Loading cached test dataset from: {selected_cache}")
    
    # 2. Load dataset
    try:
        X_test = np.load(os.path.join(selected_cache, "X_keypoints.npy"))
        y_test = np.load(os.path.join(selected_cache, "y_labels.npy"))
        
        with open(os.path.join(selected_cache, "label_mapping.json"), "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        idx_to_arabic = {int(k): v for k, v in mapping_data.get("idx_to_arabic", {}).items()}
    except Exception as e:
        print(f"Error loading dataset files: {e}")
        return

    n_samples = X_test.shape[0]
    print(f"Successfully loaded {n_samples} test samples.")
    print(f"Input shape: {X_test.shape}")
    print(f"Unique classes in test set: {len(np.unique(y_test))}")
    print("-" * 60)

    # Initialize model
    util._init()

    if util._model is None:
        print("Error: Model not initialized or loaded.")
        return

    print("Running predictions on all samples...")
    start_time = time.time()
    
    correct_count = 0
    predictions = []
    
    # Per-class metrics
    class_correct = {}
    class_total = {}
    confusions = []

    label_to_idx = mapping_data.get("label_to_idx", {})

    for i in range(n_samples):
        # Extract a single sample and add batch dimension: (1, 60, 225)
        x_sample = np.expand_dims(X_test[i], axis=0)
        true_class_id = int(y_test[i])
        
        # Map original class ID to sequential index
        true_seq_idx = label_to_idx.get(str(true_class_id), true_class_id)
        true_label = idx_to_arabic.get(int(true_seq_idx), "?")
        
        # Predict (disabling video path skips VLM arbitration)
        pred_label = util.predict(x_sample, video_path=None)
        predictions.append(pred_label)

        # Update stats
        class_total[true_label] = class_total.get(true_label, 0) + 1
        if pred_label == true_label:
            correct_count += 1
            class_correct[true_label] = class_correct.get(true_label, 0) + 1
        else:
            confusions.append({
                "index": i,
                "true": true_label,
                "predicted": pred_label or "[Rest Position / Filtered]"
            })

        # Print progress every 50 samples
        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            print(f"  Processed {i + 1}/{n_samples} samples ({speed:.1f} samples/sec)...")

    total_time = time.time() - start_time
    accuracy = (correct_count / n_samples) * 100

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Test Samples:  {n_samples}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Overall Accuracy:    {accuracy:.2f}%")
    print(f"Total Time Elapsed:  {total_time:.2f} seconds")
    print(f"Average Latency:     {(total_time / n_samples) * 1000:.1f} ms/sample")
    print("=" * 60)

    # Output per-class summary
    print("\nClass Performance Breakdown (Top 20 classes by count):")
    print(f"{'Arabic Sign Word':<25} | {'Accuracy':<10} | {'Count (Correct/Total)':<10}")
    print("-" * 60)
    
    sorted_classes = sorted(class_total.items(), key=lambda item: item[1], reverse=True)
    for word, total in sorted_classes[:20]:
        correct = class_correct.get(word, 0)
        acc = (correct / total) * 100
        print(f"{word:<25} | {acc:>8.1f}% | {correct}/{total}")

    # Output top mistakes
    if confusions:
        print("\nCommon Confusions / Mistakes:")
        from collections import Counter
        mistake_counts = Counter([f"'{c['true']}' mispredicted as '{c['predicted']}'" for c in confusions])
        for mistake, count in mistake_counts.most_common(10):
            print(f"  - {mistake} ({count} times)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
