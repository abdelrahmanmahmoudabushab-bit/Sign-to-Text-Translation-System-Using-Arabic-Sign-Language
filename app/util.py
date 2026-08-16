"""
Inference utility for Arabic Sign Language prediction.

The model is loaded lazily on first predict() call to avoid slowing
down Django management commands (migrate, collectstatic, etc.).
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Lazy-loaded globals ─────────────────────────────────────────────────────
_model = None
_idx_to_arabic: Dict[int, str] = {}
_norm_mean: Optional[np.ndarray] = None
_norm_std: Optional[np.ndarray] = None
_initialized = False
_PREDICTION_CACHE: Dict[str, list] = {}
_INIT_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()


def _get_cache_candidates():
    """Return ordered list of candidate cache directories for label mappings and norm stats."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.normpath(os.path.join(base, "..", "datasets", d))
        for d in ("keypoints_cache", "keypoints_cache_test", "keypoints_cache_test_opt2")
    ]
    # Add absolute paths for Jetson Orin Nano target environments
    candidates.append("/home/jetson/signo v6/datasets/keypoints_cache")
    # Jetson Docker fallback: check app/ directory itself for bundled norm stats
    candidates.append(os.path.dirname(os.path.abspath(__file__)))
    return candidates


def _init():
    """Lazy initialization: load model, label mapping, and normalization stats on first use."""
    global _model, _idx_to_arabic, _norm_mean, _norm_std, _initialized
    if _initialized:
        return
    with _INIT_LOCK:
        if _initialized:  # Double-check after acquiring lock
            return

    # 1. Load model: Try TensorRT engine -> ONNX Runtime -> TensorFlow Keras
    engine_path = os.path.join(os.path.dirname(__file__), "conv1_lstm.engine")
    onnx_path = os.path.join(os.path.dirname(__file__), "conv1_lstm.onnx")
    keras_path = os.path.join(os.path.dirname(__file__), "conv1_lstm.keras")

    if os.path.exists(engine_path):
        try:
            from app.trt_runner import TensorRTRunner
            runner = TensorRTRunner(engine_path)
            _model = {"type": "tensorrt", "runner": runner}
            logger.info("⚡ High-Performance TensorRT Engine loaded & warmed up at %s", engine_path)
        except Exception as trt_err:
            logger.info("TensorRT engine build/load failed (%s). Trying ONNX / Keras fallback.", trt_err)

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
            model_obj = keras_loader.models.load_model(keras_path, compile=False)
        except Exception:
            model_obj = tf.keras.models.load_model(keras_path, compile=False)
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

    # 4. Load persistent prediction disk cache
    cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media", "prediction_cache.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                _PREDICTION_CACHE.update(json.load(f))
            logger.info("Loaded %d persistent predictions from disk cache", len(_PREDICTION_CACHE))
        except Exception as e:
            logger.warning("Failed to load prediction disk cache: %s", e)

    _initialized = True

def _save_prediction_cache():
    """Persist prediction cache to disk atomically (write to temp, then rename)."""
    cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media", "prediction_cache.json")
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with _CACHE_LOCK:
            cache_snapshot = dict(_PREDICTION_CACHE)
        # Write to temp file first, then atomically replace
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(cache_file), suffix='.tmp')
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cache_snapshot, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, cache_file)
        except Exception:
            # Clean up temp file on failure
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning("Failed to save prediction disk cache: %s", e)


def _predict_coordinates(
    x: np.ndarray, 
    video_path: str = None, 
    dialect: str = 'Saudi Arabic Sign Language', 
    start_t: float = None
) -> dict:
    """Core coordinate-based CNN-LSTM prediction logic."""
    from app.telemetry import log_inference_event
    
    if start_t is None:
        start_t = time.time()
        
    # 0. Motion / Rest Gating Filter — eliminate false positive hallucinations during static rest
    hand_keypoints = x[0, :, 99:225]
    hand_motion_std = np.std(hand_keypoints)
    if hand_motion_std < 0.002:
        logger.info("Rest position detected (hand motion std %.5f < 0.002) — skipping prediction", hand_motion_std)
        return {
            "pred_lstm": "",
            "lstm_confidence": 1.0,
            "candidates": [],
            "candidate_confidences": [],
            "is_cache_hit": False,
            "is_rest": True
        }

    # Normalize keypoints using training stats (with epsilon guard against near-zero std)
    if _norm_mean is not None and _norm_std is not None:
        safe_std = np.maximum(_norm_std, 1e-7)
        x = (x - _norm_mean) / safe_std

    # Pipeline LRU Memory Cache Lookup (0.1ms repeat lookup)
    global _PREDICTION_CACHE

    # Discretize array to 3 decimal places to create robust spatial hash key
    keypoint_hash = hashlib.sha256(np.round(x, decimals=3).tobytes()).hexdigest()
    with _CACHE_LOCK:
        cached = _PREDICTION_CACHE.get(keypoint_hash)
    if cached is not None:
        cached_word, cached_conf = cached
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
        return {
            "pred_lstm": cached_word,
            "lstm_confidence": cached_conf,
            "candidates": [cached_word],
            "candidate_confidences": [cached_conf],
            "is_cache_hit": True,
            "keypoint_hash": keypoint_hash,
            "is_rest": False
        }

    # Run inference through TensorRT, ONNX Runtime, or Keras
    mtype = _model.get("type", "keras")
    x = x.astype(np.float32)

    if mtype == "onnx":
        session = _model["session"]
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: x})[0]
    elif mtype == "tensorrt":
        runner = _model["runner"]
        prediction = runner.predict(x)
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

    return {
        "pred_lstm": pred_lstm,
        "lstm_confidence": lstm_confidence,
        "candidates": candidates,
        "candidate_confidences": [float(p) for p in top_probabilities],
        "is_cache_hit": False,
        "keypoint_hash": keypoint_hash,
        "is_rest": False
    }


def predict(x: np.ndarray = None, video_path: str = None, history: list = None, dialect: str = 'Saudi Arabic Sign Language') -> str:
    """Run model prediction on preprocessed keypoint data and return Arabic label.
    Supports parallel pipeline (when x is None) or sequential fallback.
    """
    global _PREDICTION_CACHE
    from app.telemetry import log_inference_event

    start_t = time.time()
    _init()

    if _model is None:
        return "Model file (conv1_lstm.keras) not found."

    if history is None:
        history = []

def _arbitrate_prediction(
    pred_lstm: str,
    lstm_confidence: float,
    candidates: list,
    candidate_confidences: list,
    keypoint_hash: str,
    video_path: Optional[str],
    base64_frames: Optional[list],
    history: list,
    dialect: str,
    start_t: float
) -> str:
    """Unified arbitration logic across CNN-LSTM, Qwen2-VL, and Judge LLM."""
    from app.telemetry import log_inference_event
    from app.llm_util import debate_and_decide

    # 1. High-confidence shortcut OR bypass flag set OR no video available
    if lstm_confidence >= 0.95 or not video_path or os.environ.get("SIGNO_BYPASS_ARBITRATION") == "1":
        latency_ms = (time.time() - start_t) * 1000
        logger.info("Bypassing VLM arbitration. Predict: %s", pred_lstm)
        log_inference_event(
            lstm_pred=pred_lstm,
            lstm_conf=lstm_confidence,
            vlm_pred=None,
            final_pred=pred_lstm,
            decision_type="lstm_only",
            reasoning="Arbitration bypassed (high confidence or bypass flag)",
            dialect=dialect,
            latency_ms=latency_ms,
            has_video=bool(video_path)
        )
        return pred_lstm

    # 2. VLM Vision query
    vlm_result = []

    def run_vlm_query():
        try:
            if base64_frames:
                from app.llm_util import query_vlm_with_frames
                matched, ann = query_vlm_with_frames(base64_frames, candidates, dialect)
                vlm_result.append((matched, ann))
            elif video_path:
                from app.llm_util import predict_sign_with_vlm
                res_val = predict_sign_with_vlm(video_path, candidates, dialect)
                pred_vlm = res_val
                ann = ""
                for tag in [" (negation)", " (question)", " (emphasis)"]:
                    if tag in res_val:
                        pred_vlm = res_val.replace(tag, "")
                        ann = tag
                        break
                vlm_result.append((pred_vlm, ann))
        except Exception as e:
            logger.warning("VLM prediction query failed: %s", e)

    t_vlm = threading.Thread(target=run_vlm_query, daemon=True)
    t_vlm.start()
    t_vlm.join(timeout=15.0)

    latency_ms = (time.time() - start_t) * 1000

    if not vlm_result:
        logger.warning("VLM query failed or timed out. Defaulting to LSTM prediction.")
        log_inference_event(
            lstm_pred=pred_lstm,
            lstm_conf=lstm_confidence,
            vlm_pred=None,
            final_pred=pred_lstm,
            decision_type="fallback",
            reasoning="VLM check timed out or failed; defaulted to top LSTM candidate",
            dialect=dialect,
            latency_ms=latency_ms,
            has_video=True
        )
        return pred_lstm

    pred_vlm, grammar_annotation = vlm_result[0]
    pred_vlm_annotated = f"{pred_vlm}{grammar_annotation}"

    # 3. Consensus Check
    if pred_lstm == pred_vlm:
        logger.info("Consensus reached: both models predict '%s'.", pred_lstm)
        final_pred = f"{pred_lstm}{grammar_annotation}"
        log_inference_event(
            lstm_pred=pred_lstm,
            lstm_conf=lstm_confidence,
            vlm_pred=pred_vlm_annotated,
            final_pred=final_pred,
            decision_type="consensus",
            reasoning=f"Both CNN-LSTM and Qwen2-VL agreed on '{pred_lstm}' with marker '{grammar_annotation.strip() or 'none'}'",
            dialect=dialect,
            latency_ms=latency_ms,
            has_video=True
        )
        return final_pred

    # 4. Disagreement: Debate and Decide
    logger.info("Disagreement detected! LSTM predicts '%s' (conf %d%%). VLM predicts '%s'. Initiating debate...",
                pred_lstm, int(lstm_confidence * 100), pred_vlm)

    final_word = debate_and_decide(candidates, candidate_confidences, pred_vlm, history, dialect)
    latency_ms = (time.time() - start_t) * 1000

    if final_word == "?":
        logger.info("Debate resulted in ABSTENTION. Returning '?'")
        log_inference_event(
            lstm_pred=pred_lstm,
            lstm_conf=lstm_confidence,
            vlm_pred=pred_vlm_annotated,
            final_pred="?",
            decision_type="judge",
            reasoning="Judge LLM abstained: unable to establish a confident consensus",
            dialect=dialect,
            latency_ms=latency_ms,
            has_video=True
        )
        return "?"

    final_pred = f"{final_word}{grammar_annotation}"
    logger.info("Debate resolved. Final selected word: '%s'", final_pred)
    log_inference_event(
        lstm_pred=pred_lstm,
        lstm_conf=lstm_confidence,
        vlm_pred=pred_vlm_annotated,
        final_pred=final_pred,
        decision_type="judge",
        reasoning=f"Judge LLM selected '{final_word}' based on semantic flow and visual evidence.",
        dialect=dialect,
        latency_ms=latency_ms,
        has_video=True
    )

    with _CACHE_LOCK:
        if len(_PREDICTION_CACHE) > 2000:
            _PREDICTION_CACHE.pop(next(iter(_PREDICTION_CACHE)))
        _PREDICTION_CACHE[keypoint_hash] = [final_pred, lstm_confidence]
    _save_prediction_cache()

    return final_pred


def predict(x: np.ndarray = None, video_path: str = None, history: list = None, dialect: str = 'Saudi Arabic Sign Language') -> str:
    """Run model prediction on preprocessed keypoint data or raw video, and return Arabic label."""
    start_t = time.time()
    _init()

    if _model is None:
        return "Model file (conv1_lstm.keras) not found."

    if history is None:
        history = []

    # ─── CASE A: Pre-extracted coordinates (Offline test / evaluation) ────────
    if x is not None:
        res = _predict_coordinates(x, video_path, dialect, start_t)
        if res.get("is_rest", False) or res.get("is_cache_hit", False):
            return res["pred_lstm"]

        return _arbitrate_prediction(
            pred_lstm=res["pred_lstm"],
            lstm_confidence=res["lstm_confidence"],
            candidates=res["candidates"],
            candidate_confidences=res.get("candidate_confidences", []),
            keypoint_hash=res["keypoint_hash"],
            video_path=video_path,
            base64_frames=None,
            history=history,
            dialect=dialect,
            start_t=start_t
        )

    # ─── CASE B: Live video request (Parallel MediaPipe & VLM extraction) ──────
    from app.DataLoader import DataLoader
    from app.vlm_util import extract_video_frames

    lstm_result = {}
    vlm_prep = {}

    def run_lstm_pipeline():
        try:
            x_extracted = DataLoader.load_inference_data(video_path)
            if x_extracted is not None:
                res = _predict_coordinates(x_extracted, video_path, dialect, start_t)
                lstm_result.update(res)
        except Exception as e:
            logger.error("LSTM parallel thread error: %s", e)

    def run_vlm_prep():
        try:
            base64_frames = extract_video_frames(video_path, count=6)
            if base64_frames:
                vlm_prep["base64_frames"] = base64_frames
        except Exception as e:
            logger.error("VLM prep parallel thread error: %s", e)

    t_lstm = threading.Thread(target=run_lstm_pipeline, daemon=True)
    t_vlm = threading.Thread(target=run_vlm_prep, daemon=True)

    t_lstm.start()
    t_vlm.start()

    t_lstm.join(timeout=45.0)
    t_vlm.join(timeout=45.0)

    if not lstm_result:
        logger.error("LSTM parallel pipeline failed or timed out.")
        return "Could not process video frames."

    if lstm_result.get("is_rest", False) or lstm_result.get("is_cache_hit", False):
        return lstm_result["pred_lstm"]

    return _arbitrate_prediction(
        pred_lstm=lstm_result["pred_lstm"],
        lstm_confidence=lstm_result["lstm_confidence"],
        candidates=lstm_result["candidates"],
        candidate_confidences=lstm_result.get("candidate_confidences", []),
        keypoint_hash=lstm_result["keypoint_hash"],
        video_path=video_path,
        base64_frames=vlm_prep.get("base64_frames"),
        history=history,
        dialect=dialect,
        start_t=start_t
    )


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