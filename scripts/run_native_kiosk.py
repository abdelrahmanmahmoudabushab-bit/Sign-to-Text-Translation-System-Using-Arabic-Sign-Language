"""
Signo — Native Desktop Sign Language Kiosk Application.
Runs edge inference natively on Jetson with OpenCV webcam feed and MediaPipe.

Usage:
    python scripts/run_native_kiosk.py [--dialect "Saudi Arabic Sign Language"]
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import mediapipe as mp

# Configure console output encoding to UTF-8 for Arabic rendering
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.DataLoader import DataLoader, N_FRAMES, N_KEYPOINTS
from app.util import predict
from app.llm_util import smooth_sign_sentence


def draw_styled_landmarks(image, results):
    """Draw skeletal tracking lines over the OpenCV frame."""
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(236, 72, 153), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
        )
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(139, 92, 246), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
        )
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(20, 185, 129), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
        )


def main():
    parser = argparse.ArgumentParser(description="Signo Native Desktop Kiosk")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--dialect", type=str, default="Saudi Arabic Sign Language", help="Target regional dialect")
    args = parser.parse_args()

    print("====================================================================")
    print(f" 🤟 Signo Native Desktop Kiosk (Dialect: {args.dialect})")
    print("====================================================================")

    # Initialize camera capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera source index {args.camera}")
        return

    # Set frame resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Setup MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_buffer = []
    collected_words = []
    reconstructed_sentence = ""
    reconstructed_english = ""
    
    still_ticks = 0
    is_signing = False
    prev_keypoints = None

    print("⚡ System ready. Launching display window... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)

        # Process landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Extract normalized 225 coordinates
        keypoints = DataLoader.extract_keypoints(results)
        frame_buffer.append(keypoints)

        # Keep sliding buffer at N_FRAMES (60)
        if len(frame_buffer) > N_FRAMES:
            frame_buffer.pop(0)

        # Motion / Stillness Check for auto pause-segmentation
        # Compute mean absolute difference of coordinates between consecutive frames
        if prev_keypoints is not None:
            delta = np.mean(np.abs(keypoints - prev_keypoints))
            # If coordinates shift significantly over time, it's active motion
            if delta > 0.0025:  # Empirical threshold for active hand/body gestures
                is_signing = True
                still_ticks = 0
            else:
                if is_signing:
                    still_ticks += 1
                    if still_ticks >= 25: # ~800ms of consecutive stillness
                        print("Pause detected. Running gesture inference...")
                        # Build prediction tensor: (1, 60, 225)
                        while len(frame_buffer) < N_FRAMES:
                            frame_buffer.append(np.zeros(N_KEYPOINTS))
                        
                        x_input = np.expand_dims(np.array(frame_buffer), axis=0)
                        pred_word = predict(x=x_input, dialect=args.dialect)

                        if pred_word and pred_word != "?" and pred_word != "-":
                            if not collected_words or collected_words[-1] != pred_word:
                                collected_words.append(pred_word)
                                print(f" Detected Word: {pred_word}")
                                
                                # Trigger sentence smoothing reconstructor in background
                                try:
                                    smooth_res = smooth_sign_sentence(collected_words, args.dialect)
                                    reconstructed_sentence = smooth_res.get("arabic", "")
                                    reconstructed_english = smooth_res.get("english", "")
                                except Exception:
                                    reconstructed_sentence = " ".join(collected_words)
                        
                        # Reset segmentation state
                        frame_buffer = []
                        is_signing = False
                        still_ticks = 0

        prev_keypoints = keypoints

        # Draw overlays
        draw_styled_landmarks(frame, results)

        # Draw GUI Text Boxes
        # Add semi-transparent black overlay at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 390), (640, 480), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw status badge
        status_text = "SIGNING" if is_signing else "WAITING"
        status_color = (0, 255, 0) if is_signing else (0, 255, 255)
        cv2.putText(frame, f"STATUS: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Display accumulated glosses
        gloss_str = " -> ".join(collected_words)
        cv2.putText(frame, f"Glosses: {gloss_str[-45:]}", (20, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display reconstructed Arabic sentence
        if reconstructed_sentence:
            # Note: OpenCV putText has limited Arabic rendering support.
            # In a native Linux GUI we fall back to printing to terminal or drawing clean overlays.
            cv2.putText(frame, f"Sentence: {reconstructed_sentence}", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"English: {reconstructed_english}", (20, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Render window
        cv2.imshow("Signo Native Edge Kiosk", frame)

        # Handle keyboard exits
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Esc or q
            break
        elif key == ord('c'): # Clear session
            collected_words = []
            reconstructed_sentence = ""
            reconstructed_english = ""
            frame_buffer = []
            is_signing = False
            print("Session cleared.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Native kiosk shut down.")


if __name__ == "__main__":
    main()
