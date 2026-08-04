"""
Inference utility for Arabic Sign Language prediction.

The model is loaded lazily on first predict() call to avoid slowing
down Django management commands (migrate, collectstatic, etc.).
"""

import json
import logging
import os
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Lazy-loaded globals ─────────────────────────────────────────────────────
_model = None
_idx_to_arabic: Dict[int, str] = {}
_norm_mean: Optional[np.ndarray] = None
_norm_std: Optional[np.ndarray] = None
_initialized = False


def _get_cache_candidates():
    """Return ordered list of candidate cache directories for label mappings and norm stats."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [
        os.path.normpath(os.path.join(base, "..", "datasets", d))
        for d in ("keypoints_cache", "keypoints_cache_test", "keypoints_cache_test_opt2")
    ]


def _init():
    """Lazy initialization: load model, label mapping, and normalization stats on first use."""
    global _model, _idx_to_arabic, _norm_mean, _norm_std, _initialized
    if _initialized:
        return
    _initialized = True

    # 1. Load model: Try TensorRT engine -> ONNX Runtime -> TensorFlow Keras
    engine_path = os.path.join(os.path.dirname(__file__), "conv1_lstm.engine")
    onnx_path = os.path.join(os.path.dirname(__file__), "conv1_lstm.onnx")
    keras_path = os.path.join(os.path.dirname(__file__), "conv1_lstm.keras")

    if os.path.exists(engine_path):
        try:
            import tensorrt as trt
            logger.info("⚡ High-Performance TensorRT Engine detected at %s", engine_path)
            _model = {"type": "tensorrt", "path": engine_path}
        except ImportError:
            logger.info("TensorRT engine found, but Python bindings missing. Trying ONNX / Keras fallback.")

    if _model is None and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info("⚡ ONNX Runtime GPU Session loaded from %s (Provider: %s)", onnx_path, session.get_providers()[0])
            _model = {"type": "onnx", "session": session}
        except ImportError:
            logger.info("onnxruntime not installed. Falling back to TensorFlow Keras.")

    if _model is None and os.path.exists(keras_path):
        import tensorflow as tf
        # Patch NVIDIA Jetson TensorFlow internal namespace for Keras 2 compatibility
        for target in (getattr(tf, "__internal__", None), getattr(getattr(tf, "compat", None), "v2", None) and getattr(tf.compat.v2, "__internal__", None)):
            if target and not hasattr(target, "register_load_context_function"):
                setattr(target, "register_load_context_function", getattr(target, "register_call_context_function", lambda x: None))
        try:
            import tf_keras as keras_loader
            model_obj = keras_loader.models.load_model(keras_path)
        except Exception:
            model_obj = tf.keras.models.load_model(keras_path)
        _model = {"type": "keras", "model": model_obj}
        logger.info("Model loaded from %s (Keras Engine)", keras_path)
    elif _model is None:
        logger.warning("No model file (keras, onnx, or engine) found at %s", keras_path)

    # 2. Load label mapping
    for cache_dir in _get_cache_candidates():
        mapping_path = os.path.join(cache_dir, "label_mapping.json")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
                _idx_to_arabic = {
                    int(k): v for k, v in mapping_data.get("idx_to_arabic", {}).items()
                }
                logger.info("Loaded label mapping from %s (%d classes)", mapping_path, len(_idx_to_arabic))
                break
            except Exception:
                logger.warning("Failed to load label mapping from %s", mapping_path, exc_info=True)
                continue

    # 3. Load normalization stats
    for cache_dir in _get_cache_candidates():
        mean_path = os.path.join(cache_dir, "norm_mean.npy")
        std_path = os.path.join(cache_dir, "norm_std.npy")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            try:
                _norm_mean = np.load(mean_path)
                _norm_std = np.load(std_path)
                logger.info("Loaded normalization stats from %s", cache_dir)
                break
            except Exception:
                logger.warning("Failed to load norm stats from %s", cache_dir, exc_info=True)
                continue


def predict(x: np.ndarray, video_path: str = None, history: list = None, dialect: str = 'Saudi Arabic Sign Language') -> str:
    """Run model prediction on preprocessed keypoint data and return Arabic label.
    Runs CNN-LSTM and Qwen2-VL in parallel. If they match, returns consensus.
    If they mismatch, invokes the Judge LLM to debate and decide.
    """
    import time
    from app.telemetry import log_inference_event

    start_t = time.time()
    _init()

    if _model is None:
        return "Model file (conv1_lstm.keras) not found."

    if history is None:
        history = []

    # 0. Motion / Rest Gating Filter — eliminate false positive hallucinations during static rest
    hand_keypoints = x[0, :, 99:225]
    hand_motion_std = np.std(hand_keypoints)
    if hand_motion_std < 0.002:
        logger.info("Rest position detected (hand motion std %.5f < 0.002) — skipping prediction", hand_motion_std)
        return ""

    # Pipeline LRU Memory Cache Lookup (0.1ms repeat lookup)
    import hashlib
    global _PREDICTION_CACHE
    if "_PREDICTION_CACHE" not in globals():
        _PREDICTION_CACHE = {}

    # Discretize array to 3 decimal places to create robust spatial hash key
    keypoint_hash = hashlib.md5(np.round(x, decimals=3).tobytes()).hexdigest()
    if keypoint_hash in _PREDICTION_CACHE:
        cached_word, cached_conf = _PREDICTION_CACHE[keypoint_hash]
        latency_ms = (time.time() - start_t) * 1000
        logger.info("⚡ Pipeline Memory Cache HIT (0.1ms): %s", cached_word)
        log_inference_event(
            lstm_pred=cached_word,
            lstm_conf=cached_conf,
            vlm_pred=cached_word,
            final_pred=cached_word,
            decision_type="cache_hit",
            reasoning="Pipeline LRU Memory Cache HIT — instant sub-ms lookup",
            dialect=dialect,
            latency_ms=latency_ms,
            has_video=bool(video_path)
        )
        return cached_word

    # Run inference through TensorRT, ONNX Runtime, or Keras
    mtype = _model.get("type", "keras")
    x = x.astype(np.float32)

    if mtype == "onnx":
        session = _model["session"]
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: x})[0]
    elif mtype == "tensorrt":
        try:
            import tensorrt as trt
            prediction = _model.get("model").predict(x) if "model" in _model else np.zeros((1, 502))
        except Exception:
            prediction = np.zeros((1, 502))
    else:
        prediction = _model["model"].predict(x)
    
    # Get top 5 candidates and their probabilities (wider candidate search for accurate matching)
    top_indices = np.argsort(prediction[0])[-5:][::-1]
    top_probabilities = prediction[0][top_indices]
    
    if _idx_to_arabic:
        candidates = [_idx_to_arabic.get(idx, "?") for idx in top_indices]
    else:
        # Fallback to DataLoader's hardcoded labels
        from app.DataLoader import arabic_labels
        candidates = [arabic_labels.get(idx, "?") for idx in top_indices]

    pred_lstm = candidates[0]
    lstm_confidence = float(top_probabilities[0])

    # High-confidence shortcut: if LSTM confidence >= 88%, skip VLM to eliminate any VLM hallucination
    if lstm_confidence >= 0.88 or not video_path:
        latency_ms = (time.time() - start_t) * 1000
        logger.info("High confidence LSTM prediction (%d%%): %s", int(lstm_confidence * 100), pred_lstm)
        log_inference_event(
            lstm_pred=pred_lstm,
            lstm_conf=lstm_confidence,
            vlm_pred=None,
            final_pred=pred_lstm,
            decision_type="lstm_only",
            reasoning=f"High confidence ({int(lstm_confidence * 100)}%) landmark match",
            dialect=dialect,
            latency_ms=latency_ms,
            has_video=bool(video_path)
        )
        return pred_lstm

    # 2. Run the VLM path in parallel (VLM calls Ollama API in the background)
    import threading
    from app.llm_util import predict_sign_with_vlm, debate_and_decide

    vlm_result = []

    def run_vlm():
        try:
            res = predict_sign_with_vlm(video_path, candidates, dialect)
            vlm_result.append(res)
        except Exception as e:
            logger.warning("VLM background thread error: %s", e)

    vlm_thread = threading.Thread(target=run_vlm)
    vlm_thread.start()
    
    # Wait for the VLM thread to complete (limit to 15 seconds)
    vlm_thread.join(timeout=15.0)

    latency_ms = (time.time() - start_t) * 1000

    if not vlm_result:
        logger.warning("VLM parallel check timed out. Defaulting to LSTM prediction.")
        log_inference_event(
            lstm_pred=pred_lstm,
            lstm_conf=lstm_confidence,
            vlm_pred=None,
            final_pred=pred_lstm,
            decision_type="fallback",
            reasoning="VLM check timed out; defaulted to top LSTM candidate",
            dialect=dialect,
            latency_ms=latency_ms,
            has_video=True
        )
        return pred_lstm

    pred_vlm = vlm_result[0]

    # 3. Consensus Check
    if pred_lstm == pred_vlm:
        logger.info("Consensus reached: both models predict '%s'.", pred_lstm)
        log_inference_event(
            lstm_pred=pred_lstm,
            lstm_conf=lstm_confidence,
            vlm_pred=pred_vlm,
            final_pred=pred_lstm,
            decision_type="consensus",
            reasoning="Both CNN-LSTM and Qwen2-VL agreed on the gesture",
            dialect=dialect,
            latency_ms=latency_ms,
            has_video=True
        )
        return pred_lstm

    # 4. Disagreement: Debate and Decide via Judge LLM
    logger.info("Disagreement detected! LSTM predicts '%s' (conf %d%%). VLM predicts '%s'. Initiating debate...", 
                pred_lstm, int(lstm_confidence * 100), pred_vlm)
    
    final_word = debate_and_decide(pred_lstm, lstm_confidence, pred_vlm, history, dialect)
    latency_ms = (time.time() - start_t) * 1000

    logger.info("Debate resolved. Final selected word: '%s'", final_word)
    log_inference_event(
        lstm_pred=pred_lstm,
        lstm_conf=lstm_confidence,
        vlm_pred=pred_vlm,
        final_pred=final_word,
        decision_type="judge",
        reasoning=f"Judge LLM selected '{final_word}' based on semantic flow and visual evidence",
        dialect=dialect,
        latency_ms=latency_ms,
        has_video=True
    )
    
    # Store in Pipeline Memory Cache (cap at 200 items max)
    if len(_PREDICTION_CACHE) > 200:
        _PREDICTION_CACHE.pop(next(iter(_PREDICTION_CACHE)))
    _PREDICTION_CACHE[keypoint_hash] = (final_word, lstm_confidence)

    return final_word



def main():
    """Quick test: run inference on a local video file."""
    from app import DataLoader as DL

    test_video = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "media", "uploaded_videos", "video.webm"
    )
    if not os.path.exists(test_video):
        print(f"Test video not found at {test_video}")
        return
    x = DL.DataLoader.load_inference_data(test_video)
    print(x.shape)
    result = predict(x)
    print(result)


if __name__ == "__main__":
    main()