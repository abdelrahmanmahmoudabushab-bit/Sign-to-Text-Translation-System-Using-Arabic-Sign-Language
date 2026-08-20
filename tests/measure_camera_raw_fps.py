#!/usr/bin/env python3
import cv2
import time

print("Measuring RAW Camera Hardware FPS on Jetson (without MediaPipe)...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Could not open camera.")
    import sys; sys.exit(1)

# Warm up camera
for _ in range(5):
    cap.read()

start_time = time.time()
frame_count = 0

print("Reading camera frames for 5 seconds...")
while time.time() - start_time < 5.0:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

elapsed = time.time() - start_time
fps = frame_count / elapsed
print(f"Elapsed: {elapsed:.2f}s, Frames: {frame_count}, Raw Camera FPS: {fps:.2f}")
cap.release()
