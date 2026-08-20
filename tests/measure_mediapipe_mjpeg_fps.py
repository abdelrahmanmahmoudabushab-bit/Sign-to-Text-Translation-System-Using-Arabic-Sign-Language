#!/usr/bin/env python3
import cv2
import mediapipe as mp
import time

print("Measuring MediaPipe Holistic FPS with MJPG format on Jetson...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Could not open camera.")
    sys.exit(1)

mp_holistic = mp.solutions.holistic
start_time = time.time()
frame_count = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    print("Running MediaPipe tracking loop for 5 seconds...")
    while time.time() - start_time < 5.0:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        frame_count += 1

elapsed = time.time() - start_time
fps = frame_count / elapsed
print(f"Elapsed: {elapsed:.2f}s, Frames: {frame_count}, Average FPS: {fps:.2f}")
cap.release()
