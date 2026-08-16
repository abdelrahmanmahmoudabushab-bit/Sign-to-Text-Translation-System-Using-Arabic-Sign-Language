"""
Signo — Native Desktop Sign Language Kiosk Application.
Runs edge inference natively on Jetson with OpenCV and Pygame GUI.

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
import pygame
import arabic_reshaper
from bidi.algorithm import get_display

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

    # 1. Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Signo v6 — Native Edge Translation Kiosk")
    screen = pygame.display.set_mode((1024, 768))
    clock = pygame.time.Clock()

    # Helper function to match and load fonts
    def get_font(size):
        sys_fonts = ['dejavusans', 'liberationsans', 'arial', 'sans']
        for f in sys_fonts:
            path = pygame.font.match_font(f)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.Font(None, size)

    font_title = get_font(28)
    font_body = get_font(20)
    font_bold = get_font(22)
    font_large_arabic = get_font(36)

    def render_arabic(text, font, color):
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return font.render(bidi_text, True, color)

    # 2. Camera Setup
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera source index {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 3. MediaPipe Setup
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Kiosk State Variables
    frame_buffer = []
    collected_words = []
    reconstructed_sentence = ""
    reconstructed_english = ""
    
    still_ticks = 0
    is_signing = False
    prev_keypoints = None
    dialects = ["Saudi Arabic Sign Language", "Gulf Sign Language", "Levantine Sign Language", "Egyptian Sign Language"]
    current_dialect = args.dialect if args.dialect in dialects else dialects[0]

    # Button Rectangles
    btn_clear = pygame.Rect(700, 100, 280, 60)
    btn_dialect = pygame.Rect(700, 190, 280, 60)
    btn_reset_cam = pygame.Rect(700, 280, 280, 60)
    btn_exit = pygame.Rect(700, 370, 280, 60)

    running = True

    while running:
        # Handle Pygame Events
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_c:
                    collected_words = []
                    reconstructed_sentence = ""
                    reconstructed_english = ""
                    frame_buffer = []
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    if btn_clear.collidepoint(mouse_pos):
                        collected_words = []
                        reconstructed_sentence = ""
                        reconstructed_english = ""
                        frame_buffer = []
                        print("Session cleared.")
                    elif btn_dialect.collidepoint(mouse_pos):
                        idx = (dialects.index(current_dialect) + 1) % len(dialects)
                        current_dialect = dialects[idx]
                        print(f"Dialect switched to: {current_dialect}")
                    elif btn_reset_cam.collidepoint(mouse_pos):
                        cap.release()
                        cap = cv2.VideoCapture(args.camera)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        frame_buffer = []
                        print("Webcam device re-initialized.")
                    elif btn_exit.collidepoint(mouse_pos):
                        running = False

        # Read Video Frame
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)

        # Process landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Extract normalized coordinates
        keypoints = DataLoader.extract_keypoints(results)
        frame_buffer.append(keypoints)

        # Keep sliding buffer at N_FRAMES (60)
        if len(frame_buffer) > N_FRAMES:
            frame_buffer.pop(0)

        # Motion Check for pause-segmentation
        if prev_keypoints is not None:
            delta = np.mean(np.abs(keypoints - prev_keypoints))
            # Print diagnostic motion delta to terminal
            if delta > 0.0005:
                print(f"DEBUG - Delta: {delta:.6f} | Still Ticks: {still_ticks} | Signing: {is_signing}")
                
            if delta > 0.0025:  # Active movement threshold
                is_signing = True
                still_ticks = 0
            else:
                if is_signing:
                    still_ticks += 1
                    if still_ticks >= 25: # ~800ms of stillness
                        print("Pause detected. Running inference...")
                        while len(frame_buffer) < N_FRAMES:
                            frame_buffer.append(np.zeros(N_KEYPOINTS))
                        
                        x_input = np.expand_dims(np.array(frame_buffer), axis=0)
                        pred_word = predict(x=x_input, dialect=current_dialect)

                        if pred_word and pred_word != "?" and pred_word != "-":
                            if not collected_words or collected_words[-1] != pred_word:
                                collected_words.append(pred_word)
                                print(f"Detected: {pred_word}")
                                
                                try:
                                    smooth_res = smooth_sign_sentence(collected_words, current_dialect)
                                    reconstructed_sentence = smooth_res.get("arabic", "")
                                    reconstructed_english = smooth_res.get("english", "")
                                except Exception:
                                    reconstructed_sentence = " ".join(collected_words)
                        
                        # Reset segmentation state
                        frame_buffer = []
                        is_signing = False
                        still_ticks = 0

        prev_keypoints = keypoints

        # Draw MediaPipe overlays onto OpenCV frame before blitting to Pygame
        draw_styled_landmarks(frame, results)

        # Convert OpenCV frame (BGR) to Pygame surface (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        frame_surf = pygame.transform.scale(frame_surf, (640, 480))

        # ─── Render Pygame GUI Layout ───────────────────────────────────────
        screen.fill((15, 23, 42)) # slate-900 background

        # Draw Video Card container
        pygame.draw.rect(screen, (30, 41, 59), (18, 18, 644, 484), 2, border_radius=8) # border
        screen.blit(frame_surf, (20, 20))

        # Status badge overlay on top left of video
        status_text = "SIGNING" if is_signing else "WAITING"
        status_color = (16, 185, 129) if is_signing else (245, 158, 11) # emerald vs amber
        pygame.draw.rect(screen, (30, 41, 59), (30, 30, 160, 36), border_radius=6)
        pygame.draw.circle(screen, status_color, (45, 48), 6)
        lbl_status = font_bold.render(status_text, True, (255, 255, 255))
        screen.blit(lbl_status, (62, 37))

        # Draw Control Sidebar Panel
        pygame.draw.rect(screen, (30, 41, 59), (680, 20, 324, 480), border_radius=8)
        lbl_sidebar = font_title.render("Controls", True, (255, 255, 255))
        screen.blit(lbl_sidebar, (700, 40))

        # Render Interactive Buttons with hover transitions
        def draw_button(rect, label, color, hover_color):
            is_hovered = rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen, hover_color if is_hovered else color, rect, border_radius=8)
            lbl_btn = font_bold.render(label, True, (255, 255, 255))
            text_rect = lbl_btn.get_rect(center=rect.center)
            screen.blit(lbl_btn, text_rect)

        draw_button(btn_clear, "Clear Session", (51, 65, 85), (71, 85, 105))
        draw_button(btn_dialect, f"Dialect: {current_dialect.split()[0]}", (51, 65, 85), (71, 85, 105))
        draw_button(btn_reset_cam, "Reset Webcam", (51, 65, 85), (71, 85, 105))
        draw_button(btn_exit, "Exit App", (239, 68, 68), (220, 38, 38)) # Red button

        # Draw Bottom Translation Card
        pygame.draw.rect(screen, (30, 41, 59), (20, 520, 984, 220), border_radius=8)
        lbl_dashboard = font_title.render("Live Translation Board", True, (255, 255, 255))
        screen.blit(lbl_dashboard, (40, 540))

        # Render glosses sequence
        gloss_str = " -> ".join(collected_words) if collected_words else "No signs recorded yet..."
        lbl_gloss = font_body.render(f"Glosses: {gloss_str}", True, (148, 163, 184))
        screen.blit(lbl_gloss, (40, 580))

        # Render Reconstructed Arabic Sentence
        if reconstructed_sentence:
            lbl_sentence = render_arabic(reconstructed_sentence, font_large_arabic, (16, 185, 129))
            screen.blit(lbl_sentence, (40, 620))

            lbl_english = font_body.render(f"English: {reconstructed_english}", True, (255, 255, 255))
            screen.blit(lbl_english, (40, 680))
        else:
            lbl_placeholder = font_body.render("Waiting for gestures pause to translate...", True, (71, 85, 105))
            screen.blit(lbl_placeholder, (40, 620))

        # Update Display & tick clock
        pygame.display.flip()
        clock.tick(30) # Lock to 30 FPS to save CPU/GPU cycles

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    pygame.quit()
    print("Native kiosk shut down successfully.")


if __name__ == "__main__":
    main()
