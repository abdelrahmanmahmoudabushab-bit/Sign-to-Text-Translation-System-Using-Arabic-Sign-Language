import json
import logging
import os
import queue
import threading
import warnings
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

logger = logging.getLogger(__name__)

# ─── 20x SWE MediaPipe Holistic Graph Instance Pool ──────────────────────────
class HolisticPool:
    """Thread-safe pool of stateful MediaPipe Holistic instances to eliminate setup latency."""
    def __init__(self, max_size=3):
        self.pool = queue.Queue()
        self.max_size = max_size
        self.current_size = 0
        self.lock = threading.Lock()

    def acquire(self) -> mp.solutions.holistic.Holistic:
        if self.pool.empty() and self.current_size < self.max_size:
            with self.lock:
                if self.pool.empty() and self.current_size < self.max_size:
                    logger.info("Initializing new MediaPipe Holistic instance for pool (Instance #%d)...", self.current_size + 1)
                    instance = mp.solutions.holistic.Holistic(
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.current_size += 1
                    return instance

        logger.debug("Acquiring MediaPipe Holistic instance from pool...")
        return self.pool.get()

    def release(self, instance: mp.solutions.holistic.Holistic):
        self.pool.put(instance)

_HOLISTIC_POOL = None
_POOL_LOCK = threading.Lock()

def get_holistic_pool() -> HolisticPool:
    global _HOLISTIC_POOL
    if _HOLISTIC_POOL is None:
        with _POOL_LOCK:
            if _HOLISTIC_POOL is None:
                _HOLISTIC_POOL = HolisticPool(max_size=3)
    return _HOLISTIC_POOL

N_FRAMES = 60   # N Frames Per Prediction
N_KEYPOINTS = 225  # N Keypoints captured by mediapipe (33*3 + 21*3 + 21*3)

# ─── Lazy load label configuration from JSON ──────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "label_config.json")
_config_loaded = False

arabic_labels = {}
actions = {}


def _ensure_config_loaded():
    global _config_loaded
    if _config_loaded:
        return
    _config_loaded = True
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
                _config = json.load(_f)
            arabic_labels.update({int(k): v for k, v in _config.get("arabic_labels", {}).items()})
            actions.update(_config.get("actions", {}))
        except Exception as e:
            logger.warning("Failed to parse label_config.json: %s", e)
    else:
        logger.warning("label_config.json not found at %s — using empty mappings", _CONFIG_PATH)


# Initialize on module load, but wrap in function for clean retry/lazy access if needed
_ensure_config_loaded()


class DataLoader:
    """MediaPipe keypoint extraction and data loading utilities for ArSL videos."""

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    @staticmethod
    def extract_keypoints(results) -> np.ndarray:
        """Extract pose + hand keypoints from MediaPipe Holistic results.

        Returns a 1D array of shape (N_KEYPOINTS,) = (225,):
            - 33*3 adjusted pose landmarks
            - 21*3 adjusted left-hand landmarks
            - 21*3 adjusted right-hand landmarks
        """
        pose = (
            np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 3)
        )
        lh = (
            np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3)
        )
        rh = (
            np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3)
        )

        nose = pose[:3]
        lh_wrist = lh[:3]
        rh_wrist = rh[:3]

        pose_adjusted = DataLoader.adjust_landmarks(pose, nose)
        lh_adjusted = DataLoader.adjust_landmarks(lh, lh_wrist)
        rh_adjusted = DataLoader.adjust_landmarks(rh, rh_wrist)

        return np.concatenate((pose_adjusted, lh_adjusted, rh_adjusted))

    @staticmethod
    def draw_styled_landmarks(image: np.ndarray, results) -> None:
        """Draw MediaPipe landmarks on an image for visualization."""
        DataLoader.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, DataLoader.mp_holistic.POSE_CONNECTIONS,
            DataLoader.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            DataLoader.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
        DataLoader.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, DataLoader.mp_holistic.HAND_CONNECTIONS,
            DataLoader.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            DataLoader.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
        DataLoader.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, DataLoader.mp_holistic.HAND_CONNECTIONS,
            DataLoader.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            DataLoader.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

    @staticmethod
    def adjust_landmarks(arr: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Center landmarks relative to a reference point (nose or wrist)."""
        arr_reshaped = arr.reshape(-1, 3)
        center_repeated = np.tile(center, (len(arr_reshaped), 1))
        arr_adjusted = arr_reshaped - center_repeated
        return arr_adjusted.reshape(-1)

    @staticmethod
    def load_data(frames_dir: str) -> Optional[np.ndarray]:
        """Load keypoints from a directory of frame images.

        Returns:
            numpy array of shape (N_FRAMES, N_KEYPOINTS), or None on failure.
        """
        results_list = []

        pool = get_holistic_pool()
        holistic = pool.acquire()
        try:
            for filename in sorted(os.listdir(frames_dir)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(frames_dir, filename)
                    image = np.array(Image.open(image_path).convert('RGB'))

                    if image.shape[-1] == 1:
                        image = np.repeat(image, 3, axis=-1)

                    results = holistic.process(image)
                    results_extracted = DataLoader.extract_keypoints(results)
                    results_list.append(results_extracted)
        finally:
            pool.release(holistic)

        # Truncate if over N_FRAMES (previously silently dropped the entire sequence)
        if len(results_list) > N_FRAMES:
            logger.info(
                "Sequence in %s has %d frames (> %d). Truncating to %d.",
                frames_dir, len(results_list), N_FRAMES, N_FRAMES,
            )
            results_list = results_list[:N_FRAMES]

        # Pad if under N_FRAMES
        while len(results_list) < N_FRAMES:
            results_list.append(np.zeros(N_KEYPOINTS))

        return np.array(results_list)

    @staticmethod
    def load_inference_data(path: str, flip_horizontal: bool = False) -> Optional[np.ndarray]:
        """Load keypoints from a video file for inference.

        Args:
            path: Path to the video file (e.g., .webm).
            flip_horizontal: Whether to mirror flip frames (default False).

        Returns:
            numpy array of shape (1, N_FRAMES, N_KEYPOINTS), or None on failure.
        """
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            logger.error("Could not open video file %s", path)
            return None

        results_list = []

        pool = get_holistic_pool()
        holistic = pool.acquire()
        try:
            while len(results_list) < N_FRAMES:
                ret, image = cap.read()
                if not ret:
                    logger.debug("End of video file reached or no frames to read")
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if flip_horizontal:
                    image = cv2.flip(image, 1)

                if image.shape[-1] == 1:
                    image = np.repeat(image, 3, axis=-1)

                results = holistic.process(image)
                results_extracted = DataLoader.extract_keypoints(results)
                results_list.append(results_extracted)
        finally:
            pool.release(holistic)

        cap.release()

        if len(results_list) > N_FRAMES:
            results_list = results_list[:N_FRAMES]

        while len(results_list) < N_FRAMES:
            results_list.append(np.zeros(N_KEYPOINTS))

        results_array = np.array(results_list)
        results_array = np.expand_dims(results_array, axis=0)

        return results_array if results_array is not None else np.zeros((1, N_FRAMES, N_KEYPOINTS))
