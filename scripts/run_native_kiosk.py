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
import base64
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

# Bypass VLM arbitration for instant kiosk response (no 15s freeze)
os.environ["SIGNO_BYPASS_ARBITRATION"] = "1"

from app.DataLoader import DataLoader, N_FRAMES, N_KEYPOINTS
from app.util import predict_candidates
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


def resample_sequence(frames, target_len=60):
    """Linearly interpolates a list of frame keypoints to exactly target_len frames."""
    src_len = len(frames)
    if src_len == 0:
        return [np.zeros(225) for _ in range(target_len)]
    if src_len == target_len:
        return frames
    
    new_frames = []
    for i in range(target_len):
        idx = i * (src_len - 1) / (target_len - 1)
        idx_low = int(np.floor(idx))
        idx_high = int(np.ceil(idx))
        weight = idx - idx_low
        
        val_low = np.array(frames[idx_low])
        val_high = np.array(frames[idx_high])
        
        interpolated = (1 - weight) * val_low + weight * val_high
        new_frames.append(interpolated)
    return new_frames


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
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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
    
    is_active = False # Tracks if camera processing is active (starts stopped)
    record_state = "IDLE" # "IDLE", "COUNTDOWN", "RECORDING"
    countdown_start_time = 0.0
    recording_start_time = 0.0
    is_left_handed = False # Swaps hands for left-handed signers
    dialects = ["Saudi Arabic Sign Language", "Gulf Sign Language", "Levantine Sign Language", "Egyptian Sign Language"]
    current_dialect = args.dialect if args.dialect in dialects else dialects[0]

    # Button Rectangles (Manual Control Mode Layout)
    btn_start_stop = pygame.Rect(700, 80, 280, 50)
    btn_record_sign = pygame.Rect(700, 140, 280, 50)
    btn_translate = pygame.Rect(700, 200, 280, 50)
    btn_clear = pygame.Rect(700, 260, 280, 50)
    btn_handedness = pygame.Rect(700, 320, 280, 50)
    btn_dialect = pygame.Rect(700, 380, 280, 50)
    btn_exit = pygame.Rect(700, 440, 280, 50)

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
                    if btn_start_stop.collidepoint(mouse_pos):
                        is_active = not is_active
                        frame_buffer = []
                        record_state = "IDLE"
                        print(f"[STATUS] Kiosk processing {'STARTED' if is_active else 'STOPPED'}")
                    elif btn_record_sign.collidepoint(mouse_pos):
                        if is_active:
                            if record_state == "IDLE":
                                import time
                                record_state = "COUNTDOWN"
                                countdown_start_time = time.time()
                                frame_buffer = []
                                print("[ACTION] Get Ready countdown started...")
                            else:
                                print("[WARNING] Cannot start recording: system is not IDLE.")
                        else:
                            print("[WARNING] Cannot start recording: kiosk is stopped.")
                    elif btn_translate.collidepoint(mouse_pos):
                        if collected_words:
                            print(f"[ACTION] Running translation smoothing on: {collected_words}")
                            try:
                                smooth_res = smooth_sign_sentence(collected_words, current_dialect)
                                reconstructed_sentence = smooth_res.get("arabic", "")
                                reconstructed_english = smooth_res.get("english", "")
                                print(f"[TRANSLATE] Arabic: {reconstructed_sentence} | English: {reconstructed_english}")
                            except Exception as e:
                                print(f"[TRANSLATE] Smoothing failed: {e}")
                                reconstructed_sentence = " ".join([c[0] for c in collected_words])
                        else:
                            print("[WARNING] Cannot translate: no words recorded.")
                    elif btn_clear.collidepoint(mouse_pos):
                        collected_words = []
                        reconstructed_sentence = ""
                        reconstructed_english = ""
                        frame_buffer = []
                        record_state = "IDLE"
                        print("[ACTION] Session cleared.")
                    elif btn_handedness.collidepoint(mouse_pos):
                        is_left_handed = not is_left_handed
                        print(f"[ACTION] Dominant Hand swapped. Left-Handed Mode: {is_left_handed}")
                    elif btn_dialect.collidepoint(mouse_pos):
                        idx = (dialects.index(current_dialect) + 1) % len(dialects)
                        current_dialect = dialects[idx]
                        print(f"[ACTION] Dialect switched to: {current_dialect}")
                    elif btn_exit.collidepoint(mouse_pos):
                        running = False

        # Read Video Frame
        ret, frame = cap.read()
        if not ret:
            break

        results = None
        if is_active:
            # Process landmarks (processed on original, un-mirrored frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            if record_state == "COUNTDOWN":
                import time
                elapsed = time.time() - countdown_start_time
                remaining = 3.0 - elapsed
                if remaining <= 0:
                    record_state = "RECORDING"
                    recording_start_time = time.time()
                    frame_buffer = []
                    print("[ACTION] Recording started...")
            
            elif record_state == "RECORDING":
                import time
                # Extract normalized coordinates
                keypoints = DataLoader.extract_keypoints(results)
                frame_buffer.append(keypoints)

                elapsed_rec = time.time() - recording_start_time
                if elapsed_rec >= 2.0:
                    print(f"[ACTION] Recording complete. Captured {len(frame_buffer)} frames in {elapsed_rec:.2f}s. Resampling...")
                    # Resample captured frames linearly to exactly N_FRAMES (60)
                    resampled_buffer = resample_sequence(frame_buffer, N_FRAMES)
                    
                    # If left-handed mode is enabled, swap left & right hand channels and mirror X axis
                    if is_left_handed:
                        mirrored_buffer = []
                        for kps in resampled_buffer:
                            kp_reshaped = kps.reshape(-1, 3)
                            kp_reshaped[:, 0] = -kp_reshaped[:, 0] # Mirror horizontal coordinates
                            pose_part = kp_reshaped[:33]
                            lh_part = kp_reshaped[33:54]
                            rh_part = kp_reshaped[54:75]
                            kp_swapped = np.concatenate((pose_part, rh_part, lh_part))
                            mirrored_buffer.append(kp_swapped.flatten())
                        resampled_buffer = mirrored_buffer
                        print("[ACTION] Left-Handed coordinates mirrored and swapped successfully.")

                    x_input = np.expand_dims(np.array(resampled_buffer), axis=0)
                    try:
                        pred_candidates = predict_candidates(x=x_input, dialect=current_dialect)
                        print(f"[PREDICT] Candidates: {pred_candidates}")
                        if pred_candidates and pred_candidates[0] != "?" and pred_candidates[0] != "-":
                            if not collected_words or collected_words[-1][0] != pred_candidates[0]:
                                collected_words.append(pred_candidates)
                                print(f"Captured: {pred_candidates}")
                    except Exception as e:
                        print(f"[PREDICT] FAILED: {e}")
                        import traceback; traceback.print_exc()

                    # Reset to idle preview
                    frame_buffer = []
                    record_state = "IDLE"

        # Draw MediaPipe overlays onto OpenCV frame before flipping
        if is_active and results:
            draw_styled_landmarks(frame, results)

        # Flip horizontally for natural mirror effect (done AFTER landmark extraction & drawing to keep coordinates un-mirrored)
        frame = cv2.flip(frame, 1)

        # Draw text overlays on the flipped frame so they are not mirrored
        if is_active:
            if record_state == "COUNTDOWN":
                import time
                elapsed = time.time() - countdown_start_time
                remaining = 3.0 - elapsed
                if remaining > 0:
                    countdown_val = int(np.ceil(remaining))
                    cv2.putText(frame, f"STARTING IN: {countdown_val}", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (59, 130, 246), 4, cv2.LINE_AA)
            elif record_state == "RECORDING":
                import time
                elapsed_rec = time.time() - recording_start_time
                progress_pct = int(min(100, (elapsed_rec / 2.0) * 100))
                cv2.putText(frame, f"RECORDING ({progress_pct}%)", (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (239, 68, 68), 2, cv2.LINE_AA)
                cv2.circle(frame, (100, 50), 12, (239, 68, 68), -1)

        if not is_active:
            # Draw placeholder when paused
            cv2.rectangle(frame, (80, 200), (560, 280), (15, 23, 42), -1)
            cv2.rectangle(frame, (80, 200), (560, 280), (51, 65, 85), 2)
            cv2.putText(frame, "KIOSK INACTIVE", (150, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (239, 68, 68), 2)
            cv2.putText(frame, "Click Start Kiosk to translate signs", (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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
        if is_active:
            if record_state == "IDLE":
                status_text = "READY"
                status_color = (16, 185, 129) # Green
            elif record_state == "COUNTDOWN":
                status_text = "GET READY"
                status_color = (59, 130, 246) # Blue
            elif record_state == "RECORDING":
                import time
                elapsed_rec = time.time() - recording_start_time
                progress_pct = int(min(100, (elapsed_rec / 2.0) * 100))
                status_text = f"RECORDING ({progress_pct}%)"
                status_color = (239, 68, 68) # Red
        else:
            status_text = "INACTIVE"
            status_color = (100, 116, 139) # slate-500 grey

        pygame.draw.rect(screen, (30, 41, 59), (30, 30, 200, 36), border_radius=6)
        pygame.draw.circle(screen, status_color, (45, 48), 6)
        lbl_status = font_bold.render(status_text, True, (255, 255, 255))
        screen.blit(lbl_status, (62, 37))

        # Draw Control Sidebar Panel
        pygame.draw.rect(screen, (30, 41, 59), (680, 20, 324, 480), border_radius=8)
        lbl_sidebar = font_title.render("Controls", True, (255, 255, 255))
        screen.blit(lbl_sidebar, (700, 35))

        # Render Interactive Buttons with hover transitions
        def draw_button(rect, label, color, hover_color):
            is_hovered = rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen, hover_color if is_hovered else color, rect, border_radius=8)
            lbl_btn = font_bold.render(label, True, (255, 255, 255))
            text_rect = lbl_btn.get_rect(center=rect.center)
            screen.blit(lbl_btn, text_rect)

        # Dynamic Start/Stop button color and label
        if is_active:
            draw_button(btn_start_stop, "Stop Kiosk", (220, 38, 38), (185, 28, 28)) # Dark red stops
        else:
            draw_button(btn_start_stop, "Start Kiosk", (16, 185, 129), (5, 150, 105)) # Green starts

        # Capture / Record Sign button (Cyan/Teal theme)
        draw_button(btn_record_sign, "Capture Sign", (14, 116, 144), (8, 145, 178))

        # Translate Board button (Violet theme)
        draw_button(btn_translate, "Translate Board", (109, 40, 217), (124, 58, 237))

        draw_button(btn_clear, "Clear Session", (51, 65, 85), (71, 85, 105))
        # Handedness button
        hand_label = "Hand: Left-Handed" if is_left_handed else "Hand: Right-Handed"
        draw_button(btn_handedness, hand_label, (51, 65, 85), (71, 85, 105))
        draw_button(btn_dialect, f"Dialect: {current_dialect.split()[0]}", (51, 65, 85), (71, 85, 105))
        draw_button(btn_exit, "Exit App", (71, 85, 105), (100, 116, 139))

        # Draw Bottom Translation Card
        pygame.draw.rect(screen, (30, 41, 59), (20, 520, 984, 220), border_radius=8)
        lbl_dashboard = font_title.render("Live Translation Board", True, (255, 255, 255))
        screen.blit(lbl_dashboard, (40, 540))

        # Render glosses sequence using the top prediction for each position
        gloss_str = " -> ".join([c[0] for c in collected_words]) if collected_words else "No signs recorded yet..."
        lbl_gloss = font_body.render(f"Glosses: {gloss_str}", True, (148, 163, 184))
        screen.blit(lbl_gloss, (40, 580))

        # Render Reconstructed Arabic Sentence
        if reconstructed_sentence:
            lbl_sentence = render_arabic(reconstructed_sentence, font_large_arabic, (16, 185, 129))
            screen.blit(lbl_sentence, (40, 620))

            lbl_english = font_body.render(f"English: {reconstructed_english}", True, (255, 255, 255))
            screen.blit(lbl_english, (40, 680))
        else:
            lbl_placeholder = font_body.render("Click Capture Sign, then Translate Board...", True, (71, 85, 105))
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
