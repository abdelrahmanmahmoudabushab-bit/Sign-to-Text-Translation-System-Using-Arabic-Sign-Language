#!/usr/bin/env python3
"""Quick diagnostic test — run on Jetson to check full pipeline."""
import sys, os, numpy as np, traceback

os.environ['SIGNO_BYPASS_ARBITRATION'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.DataLoader import DataLoader, N_FRAMES, N_KEYPOINTS

print("=== TEST 1: Model Init ===")
try:
    from app import util
    util._init()
    print(f"Model loaded: {util._model is not None} (type: {util._model['type'] if util._model else None})")
    print(f"Label mapping loaded: {util._idx_to_arabic is not None} ({len(util._idx_to_arabic) if util._idx_to_arabic else 0} classes)")
    print(f"Norm stats loaded: mean={util._norm_mean is not None}, std={util._norm_std is not None}")
    if util._idx_to_arabic:
        items = list(util._idx_to_arabic.items())[:5]
        print(f"First 5 labels: {dict(items)}")
except Exception as e:
    print(f"INIT FAILED: {e}")
    traceback.print_exc()

print()
print("=== TEST 2: Inference with random data ===")
try:
    from app.util import predict
    x = np.random.randn(1, N_FRAMES, N_KEYPOINTS).astype(np.float32)
    result = predict(x=x, dialect='Saudi Arabic Sign Language', history=[])
    print(f"Prediction result: {repr(result)}")
except Exception as e:
    print(f"INFERENCE FAILED: {e}")
    traceback.print_exc()

print()
print("=== TEST 3: Inference with zeros (rest position) ===")
try:
    x_zeros = np.zeros((1, N_FRAMES, N_KEYPOINTS), dtype=np.float32)
    result = predict(x=x_zeros, dialect='Saudi Arabic Sign Language', history=[])
    print(f"Rest result: {repr(result)}")
except Exception as e:
    print(f"REST CHECK FAILED: {e}")
    traceback.print_exc()

print()
print("=== TEST 4: Check cache paths ===")
try:
    from app.util import _get_cache_candidates
    candidates = _get_cache_candidates()
    for c in candidates:
        exists = os.path.isdir(c)
        has_mapping = os.path.exists(os.path.join(c, "label_mapping.json"))
        has_mean = os.path.exists(os.path.join(c, "norm_mean.npy"))
        has_model = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "conv1_lstm.keras"))
        print(f"  {c}: exists={exists} mapping={has_mapping} norm={has_mean}")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
    for f in ["conv1_lstm.engine", "conv1_lstm.onnx", "conv1_lstm.keras"]:
        fp = os.path.join(model_dir, f)
        print(f"  Model {f}: exists={os.path.exists(fp)}")
except Exception as e:
    print(f"CACHE CHECK FAILED: {e}")
    traceback.print_exc()

print()
print("=== DONE ===")
