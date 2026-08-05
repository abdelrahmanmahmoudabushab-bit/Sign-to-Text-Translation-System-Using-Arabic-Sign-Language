"""
Test script: pick N random sequences and simulate a multi-word signed sentence.
Runs inference on each, then optionally smooths via local Ollama LLM.
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


def simulate_sentence_translation(num_words=4):
    dataset_dir = r"D:\signo v6\datasets\karsl-502"

    # 1. Discover all sequences using shared utility
    print("Scanning dataset directory to select multiple word signs...")
    sequences = discover_sequences(dataset_dir)

    if not sequences:
        print("No test sequences found.")
        return

    # 2. Pick N random sample sequences to construct a "signed sentence"
    signed_sentence = [random.choice(sequences) for _ in range(num_words)]

    print("\n" + "="*60)
    true_words = []
    for idx, (path, sign_id) in enumerate(signed_sentence):
        class_idx = actions[sign_id]
        arabic_word = arabic_labels.get(class_idx, "?")
        true_words.append(arabic_word)
        print(f"Word {idx+1}: {arabic_word} (Sign ID: {sign_id})")

    print("="*60)
    print(f"Target Gesture Sentence: {' '.join(true_words)}")
    print("="*60 + "\n")

    # 3. Process each word sequentially using MediaPipe + CNN-LSTM
    import mediapipe as mp
    from PIL import Image
    mp_holistic = mp.solutions.holistic

    predicted_words = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for idx, (frames_dir, sign_id) in enumerate(signed_sentence):
            print(f"Translating Gesture {idx+1}/{num_words}...")

            frame_files = sorted([
                f for f in os.listdir(frames_dir)
                if f.lower().endswith(('.jpg', '.png'))
            ])[:N_FRAMES]

            results_list = []
            for fname in frame_files:
                image_path = os.path.join(frames_dir, fname)
                image = np.array(Image.open(image_path).convert('RGB'))
                results = holistic.process(image)
                keypoints = DataLoader.extract_keypoints(results)
                results_list.append(keypoints)

            while len(results_list) < N_FRAMES:
                results_list.append(np.zeros(N_KEYPOINTS))

            x = np.expand_dims(np.array(results_list), axis=0)  # Shape: (1, 60, 225)

            # Predict individual sign word
            predicted_word = predict(x)
            predicted_words.append(predicted_word)
            print(f"  -> Predicted: {predicted_word}")

    # 4. Show final translation output
    raw_translation = " ".join(predicted_words)
    true_sentence = " ".join(true_words)

    print("\n" + "="*60)
    print("SENTENCE TRANSLATION OUTCOME")
    print("="*60)
    print(f"Target Sentence (True):  {true_sentence}")
    print(f"Model Output (Raw):      {raw_translation}")

    # Perform active LLM smoothing via local Ollama
    print("\n[Active Local LLM Translation & Smoothing]")
    from app.llm_util import smooth_sign_sentence
    translation = smooth_sign_sentence(predicted_words)
    print(f"Grammatical Arabic:      {translation['arabic']}")
    print(f"English Translation:     {translation['english']}")
    print("="*60)

if __name__ == "__main__":
    simulate_sentence_translation(num_words=3)  # Simulate a 3-word signed phrase
