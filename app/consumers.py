import logging
import numpy as np
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from app.DataLoader import N_FRAMES, N_KEYPOINTS
from app.util import predict

logger = logging.getLogger(__name__)


class SignLanguageConsumer(AsyncJsonWebsocketConsumer):
    """
    WebSocket Consumer for processing real-time coordinate streams frame-by-frame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_buffer = []

    async def connect(self):
        await self.accept()
        self.frame_buffer = []
        logger.info("⚡ Live WebSocket Translation connection accepted.")

    async def disconnect(self, close_code):
        self.frame_buffer = []
        logger.info("⚡ Live WebSocket Translation disconnected.")

    async def receive_json(self, content):
        """
        Receives frame payload:
        {
          "keypoints": [225 floats...],
          "dialect": "Saudi Arabic Sign Language",
          "reset": false
        }
        """
        if content.get("reset", False):
            self.frame_buffer = []
            return

        keypoints = content.get("keypoints")
        if not keypoints or len(keypoints) != N_KEYPOINTS:
            return

        # Add keypoints to sequence frame buffer
        self.frame_buffer.append(keypoints)

        # Keep buffer to sliding window of N_FRAMES (60)
        if len(self.frame_buffer) > N_FRAMES:
            self.frame_buffer.pop(0)

        # Trigger inference when buffer is fully populated (60 frames)
        if len(self.frame_buffer) == N_FRAMES:
            # Construct prediction input array: shape (1, 60, 225)
            x_input = np.expand_dims(np.array(self.frame_buffer), axis=0) # (1, 60, 225)

            # Perform prediction on background threadpool to avoid blocking event loop
            dialect = content.get("dialect", "Saudi Arabic Sign Language")
            pred_word = await database_sync_to_async(self._run_inference)(x_input, dialect)

            if pred_word and pred_word != "?":
                await self.send_json({
                    "status": "success",
                    "word": pred_word
                })

    def _run_inference(self, x, dialect):
        try:
            # Sequence predict
            return predict(x=x, dialect=dialect)
        except Exception as e:
            logger.error("Error during real-time WebSocket inference: %s", e)
            return None
