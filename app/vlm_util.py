import os
import cv2
import base64
import logging

logger = logging.getLogger(__name__)

def extract_video_frames(video_path: str, count: int = 6) -> list:
    """
    Extracts count (default 6) evenly spaced frames from a video
    and returns them as a list of base64-encoded JPEG strings.

    Uses frame seeking instead of loading all frames into memory
    to minimize RAM usage on memory-constrained devices (Jetson).

    Args:
        video_path: Path to the input video file.
        count: Number of frames to extract.

    Returns:
        A list of base64-encoded JPEG image strings, or empty list if extraction failed.
    """
    if not os.path.exists(video_path):
        logger.error("Input video file does not exist at %s", video_path)
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < count:
        logger.warning("Video has only %d frames; minimum %d required.", total_frames, count)
        cap.release()
        return []

    # Calculate frame indices evenly spaced across the timeline
    indices = [int(total_frames * (i + 0.5) / count) for i in range(count)]
    base64_frames = []

    for idx in indices:
        clamped_idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize frame to 320x240 to keep it lightweight for network transit
        resized = cv2.resize(frame, (320, 240))
        
        # Encode as JPG
        ret, buffer = cv2.imencode('.jpg', resized)
        if not ret:
            continue
            
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(img_base64)

    cap.release()
    logger.info("Successfully extracted %d sequential frames from %s", len(base64_frames), video_path)
    return base64_frames
