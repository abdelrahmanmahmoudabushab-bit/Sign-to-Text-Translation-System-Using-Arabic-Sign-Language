import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_storyboard(video_path: str, output_path: str = "media/temp_grid.jpg") -> str:
    """
    Extracts 6 evenly spaced frames from a video and stitches them
    into a 2x3 storyboard grid image for VLM analysis.

    Args:
        video_path: Path to the input video file.
        output_path: Path where the stitched grid image should be saved.

    Returns:
        The path to the saved grid image, or None if extraction failed.
    """
    if not os.path.exists(video_path):
        logger.error("Input video file does not exist at %s", video_path)
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 6:
        logger.warning("Video has only %d frames; minimum 6 required.", total_frames)
        cap.release()
        return None

    # Calculate frame indices at 10%, 26%, 42%, 58%, 74%, 90% of the timeline
    frame_percentages = [0.10, 0.26, 0.42, 0.58, 0.74, 0.90]
    frame_indices = [int(total_frames * p) for p in frame_percentages]
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize frame to 320x240 to keep the stitched image lightweight
            resized = cv2.resize(frame, (320, 240))
            frames.append(resized)
        else:
            logger.warning("Could not read frame at index %d", idx)

    cap.release()

    if len(frames) < 6:
        logger.error("Failed to extract 6 frames (only got %d)", len(frames))
        return None

    # Stitch top row (frames 0, 1, 2) and bottom row (frames 3, 4, 5)
    try:
        row1 = np.hstack(frames[0:3])
        row2 = np.hstack(frames[3:6])
        grid = np.vstack([row1, row2])

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, grid)
        logger.info("Successfully generated storyboard grid at %s", output_path)
        return output_path
    except Exception as e:
        logger.exception("Error stitching and saving grid image: %s", e)
        return None
