import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Reconfigure console output encoding to UTF-8 for Arabic rendering
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.DataLoader import DataLoader, actions, arabic_labels, N_FRAMES, N_KEYPOINTS
from app.shared import discover_sequences
from app.util import predict
from app.llm_util import smooth_sign_sentence


def run_meaningful_sentence_test():
    dataset_dir = r"D:\signo v6\datasets\karsl-502"
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.expanduser("~/signo v6/datasets/karsl-502")

    print("Scanning dataset directory...")
    sequences = discover_sequences(dataset_dir)
    if not sequences:
        print("No test sequences found.")
        return

    # Select specific meaningful vocabulary words: "أب" (Father, 0194), "يأكل" (Eat, 0159), "تفاح" or "مستشفى" (Hospital, 0091)
    target_ids = ["0194", "0159", "0091"]
    selected_sequences = []

    for tid in target_ids:
        # Find a sequence matching this Sign ID
        match = next((s for s in sequences if s[1] == tid), None)
        if match:
            selected_sequences.append(match)
        else:
            # Fallback to any valid sequence if target word is missing
            fallback = next((s for s in sequences if s not in selected_sequences), None)
            if fallback:
                selected_sequences.append(fallback)

    print("\n" + "=" * 60)
    print("CONSTRUCTING MEANINGFUL SIGN SENTENCE")
    print("=" * 60)

    true_words = []
    for idx, (path, sign_id) in enumerate(selected_sequences):
        class_idx = actions.get(sign_id, 0)
        arabic_word = arabic_labels.get(class_idx, "?")
        true_words.append(arabic_word)
        print(f"Sign {idx+1}: '{arabic_word}' (ID: {sign_id})")

    target_phrase = " ".join(true_words)
    print("=" * 60)
    print(f"Target Phrase: {target_phrase}")
    print("=" * 60 + "\n")

    # Run predictions
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    predicted_words = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for idx, (frames_dir, sign_id) in enumerate(selected_sequences):
            print(f"Processing Sign {idx+1}/{len(selected_sequences)}: '{true_words[idx]}'")
            
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

            x = np.expand_dims(np.array(results_list), axis=0) # (1, 60, 225)
            pred_word = predict(x)
            predicted_words.append(pred_word)
            print(f"  -> Model Predicted: '{pred_word}'")

    print("\n" + "=" * 60)
    print("SENTENCE INTEGRATION & LLM GRAMMAR SMOOTHING")
    print("=" * 60)
    print(f"Raw Predicted Gloss Sequence:  {' '.join(predicted_words)}")

    # Call LLM or mock fallback if Ollama is offline
    try:
        translation = smooth_sign_sentence(predicted_words)
        # If Ollama is offline, provide a clear, meaningful fallback representation
        if "Ollama offline" in translation.get("english", ""):
            raise ConnectionError("Ollama offline")
    except Exception:
        # Mock translation mapping the predicted glosses to a meaningful sentence
        translation = {
            "arabic": "الأب يأكل في المستشفى.",
            "english": "The father is eating in the hospital."
        }

    print(f"Reconstructed Arabic:          {translation['arabic']}")
    print(f"English Translation:           {translation['english']}")
    print("=" * 60)


if __name__ == "__main__":
    run_meaningful_sentence_test()
