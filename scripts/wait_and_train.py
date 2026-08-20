import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

keypoints_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "jsl_keypoints.npy")

print("Waiting for keypoint extraction to finish writing jsl_keypoints.npy...")

# Check if keypoints file was updated recently or check extraction script completion
import psutil

def is_extractor_running():
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            cmd = p.info.get('cmdline')
            if cmd and any('extract_jsl_keypoints.py' in arg for arg in cmd):
                return True
        except Exception:
            pass
    return False

while is_extractor_running():
    time.sleep(5)

print("\n" + "="*60)
print("STAGE 1 COMPLETE: Keypoint extraction finished!")
print("STAGE 2 STARTING: High-precision embedding training...")
print("="*60)

from scripts.train_jsl_embeddings import train_and_compile
train_and_compile()

print("\n" + "="*60)
print("STAGE 3 STARTING: End-to-End System Inference Test...")
print("="*60)

from tests.test_jsl_inference import test_jsl_inference
test_jsl_inference()

print("\n" + "="*60)
print("ALL STAGES COMPLETED SUCCESSFULLY!")
print("="*60)

