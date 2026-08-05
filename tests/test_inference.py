"""
Test script: pick a random sequence from the dataset and run inference.
Uses the shared discover_sequences() utility.
"""

import os
import sys
import numpy as np
import random
import tensorflow as tf

# Reconfigure console output encoding to UTF-8 for Arabic rendering
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path (script is now in tests/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.DataLoader import DataLoader, actions, arabic_labels, N_FRAMES, N_KEYPOINTS
from app.shared import discover_sequences
from app.util import predict


def test_random_sequence():
    dataset_dir = r"D:\signo v6\datasets\karsl-502"

    # 1. Discover all sequences using shared utility
    print("Scanning dataset directory to pick a test sample...")
    sequences = discover_sequences(dataset_dir)

    if not sequences:
        print("No test sequences found.")
        return

    # 2. Pick a random sample sequence (filtering out empty directories dynamically)
    while True:
        test_sample = random.choice(sequences)
        frames_dir, sign_id = test_sample
        # Check if directory exists and contains images
        if os.path.exists(frames_dir):
            try:
                images = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png'))]
                if len(images) >= 5:
                    break
            except OSError:
                pass
        # Remove empty folder from list so we don't pick it again
        sequences.remove(test_sample)
        if not sequences:
            print("No non-empty test sequences found.")
            return

    true_idx = actions[sign_id]
    true_label_arabic = arabic_labels.get(true_idx, "?")

    print("\n" + "="*60)
    print("INFERENCE TEST SAMPLE DETECTED")
    print("="*60)
    print(f"Directory:    {frames_dir}")
    print(f"Sign ID:      {sign_id}")
    print(f"True Label:   {true_label_arabic} (Class {true_idx})")
    print("="*60)

    # 3. Extract keypoints using MediaPipe Holistic
    print("\nProcessing frame images through MediaPipe Holistic...")
    import mediapipe as mp
    from PIL import Image

    mp_holistic = mp.solutions.holistic
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])[:N_FRAMES]

    results_list = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for fname in frame_files:
            image_path = os.path.join(frames_dir, fname)
            image = np.array(Image.open(image_path).convert('RGB'))
            results = holistic.process(image)
            keypoints = DataLoader.extract_keypoints(results)
            results_list.append(keypoints)

    while len(results_list) < N_FRAMES:
        results_list.append(np.zeros(N_KEYPOINTS))

    x = np.expand_dims(np.array(results_list), axis=0)  # Add batch dimension (1, 60, 225)

    # 4. Predict
    print("Running model prediction...")
    predicted_label = predict(x)

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"True Translation:      {true_label_arabic}")
    print(f"Model Prediction:      {predicted_label}")
    print(f"Match Status:          {'PASS' if true_label_arabic == predicted_label else 'FAIL'}")
    print("="*60)

if __name__ == "__main__":
    test_random_sequence()
