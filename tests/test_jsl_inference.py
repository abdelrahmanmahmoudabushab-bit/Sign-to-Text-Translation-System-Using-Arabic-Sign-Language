#!/usr/bin/env python3
"""
Test JSL Embedding Inference Pathway

Runs a mock test query against the JSL nearest-neighbor lookup database 
inside app/util.py to ensure that the metric embedding classification 
is fully functional and does not throw errors.
"""

import os
import sys
import numpy as np

# Reconfigure console output encoding to UTF-8
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.util import predict
from app.DataLoader import N_FRAMES, N_KEYPOINTS

def test_jsl_inference():
    print("="*60)
    print("RUNNING JSL EMBEDDING INTEGRATION TEST")
    print("="*60)
    
    # 1. Generate random mockup keypoint tensor (60 frames, 225 keypoints)
    # We add values to simulate non-rest hands movement (motion_std > 0.0001)
    dummy_x = np.random.normal(0, 0.5, size=(1, N_FRAMES, N_KEYPOINTS))
    
    # Ensure it's not detected as rest position
    dummy_x[0, :, 99:225] = np.random.normal(0.5, 0.2, size=(N_FRAMES, 126))
    
    # 2. Run prediction with Jordanian dialect
    dialect = "Jordanian Arabic Sign Language"
    print(f"Dialect under test: {dialect}")
    
    try:
        print("Running utility predict() route...")
        predicted_word = predict(x=dummy_x, dialect=dialect)
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"JSL Lookup Prediction: {predicted_word}")
        print("Status:                PASS (Execution succeeded without crashes)")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
        print(f"Error encountered:     {e}")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    test_jsl_inference()
