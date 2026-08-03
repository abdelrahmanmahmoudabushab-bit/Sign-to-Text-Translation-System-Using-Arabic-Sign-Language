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

    # 1. Load model
    model_path = os.path.join(os.path.dirname(__file__), "conv1_lstm.keras")
    if os.path.exists(model_path):
        import tensorflow as tf
        _model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded from %s", model_path)
    else:
        logger.warning("Model file not found at %s", model_path)

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

    Args:
        x: numpy array of shape (1, N_FRAMES, N_KEYPOINTS).
        video_path: Path to the uploaded video file.
        history: List of previously translated words in the current session.
        dialect: Selected sign language dialect name.

    Returns:
        Predicted Arabic word string, or an error message if model is unavailable.
    """
    _init()

    if _model is None:
        return "Model file (conv1_lstm.keras) not found."

    if history is None:
        history = []

    # 1. Apply normalization if stats are available (must match training preprocessing)
    if _norm_mean is not None and _norm_std is not None:
        x = (x - _norm_mean) / (_norm_std + 1e-8)

    prediction = _model.predict(x)
    
    # Get top 3 candidates and their probabilities
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_probabilities = prediction[0][top_indices]
    
    if _idx_to_arabic:
        candidates = [_idx_to_arabic.get(idx, "?") for idx in top_indices]
    else:
        # Fallback to DataLoader's hardcoded labels
        from app.DataLoader import arabic_labels
        candidates = [arabic_labels.get(idx, "?") for idx in top_indices]

    pred_lstm = candidates[0]
    lstm_confidence = top_probabilities[0]

    # If no video path is provided, run only the local LSTM path
    if not video_path:
        logger.info("Direct LSTM prediction (no video path): %s (confidence %d%%)", pred_lstm, int(lstm_confidence * 100))
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

    if not vlm_result:
        logger.warning("VLM parallel check timed out. Defaulting to LSTM prediction.")
        return pred_lstm

    pred_vlm = vlm_result[0]

    # 3. Consensus Check
    if pred_lstm == pred_vlm:
        logger.info("Consensus reached: both models predict '%s'.", pred_lstm)
        return pred_lstm

    # 4. Disagreement: Debate and Decide via Judge LLM
    logger.info("Disagreement detected! LSTM predicts '%s' (conf %d%%). VLM predicts '%s'. Initiating debate...", 
                pred_lstm, int(lstm_confidence * 100), pred_vlm)
    
    final_word = debate_and_decide(pred_lstm, lstm_confidence, pred_vlm, history, dialect)
    logger.info("Debate resolved. Final selected word: '%s'", final_word)
    
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