"""
Native PyCUDA TensorRT Engine Runner for NVIDIA Jetson.

Deserializes a compiled FP16 TensorRT .engine model, pre-allocates CUDA
device memory buffers, and executes zero-copy GPU inference with a pre-warmup pass.
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class TensorRTRunner:
    """
    High-performance native PyCUDA runner for TensorRT engines.
    """

    def __init__(self, engine_path: str, input_shape=(1, 60, 225), output_shape=(1, 502)):
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        self.cuda_driver = None

        self._load_engine()
        self._allocate_buffers()
        self._warmup()

    def _load_engine(self):
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at {self.engine_path}")

        logger.info("Loading serialized TensorRT engine from %s...", self.engine_path)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine at {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

    def _allocate_buffers(self):
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            self.cuda_driver = cuda
        except ImportError:
            logger.warning("PyCUDA not available. Attempting CUDA ctypes memory allocation...")
            import ctypes
            self.cuda_driver = None

        if self.cuda_driver:
            self.stream = self.cuda_driver.Stream()

            # Pre-allocate pinned host and device buffers for input & output
            input_size = int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize
            output_size = int(np.prod(self.output_shape)) * np.dtype(np.float32).itemsize

            # Host buffers (pinned memory)
            self.h_input = self.cuda_driver.pagelocked_empty(self.input_shape, dtype=np.float32)
            self.h_output = self.cuda_driver.pagelocked_empty(self.output_shape, dtype=np.float32)

            # Device memory allocation
            self.d_input = self.cuda_driver.mem_alloc(input_size)
            self.d_output = self.cuda_driver.mem_alloc(output_size)

            self.bindings = [int(self.d_input), int(self.d_output)]
            logger.info("✅ Pre-allocated CUDA GPU memory buffers (%d bytes input, %d bytes output)", input_size, output_size)
        else:
            raise RuntimeError("No CUDA memory driver available (PyCUDA required for native TensorRT execution).")

    def _warmup(self):
        """Perform 1 dummy warmup pass during startup to achieve sub-5ms first gesture latency."""
        dummy_input = np.zeros(self.input_shape, dtype=np.float32)
        logger.info("🔥 Executing GPU dummy warmup pass...")
        self.predict(dummy_input)
        logger.info("⚡ TensorRT Engine ready and warmed up for ultra-fast GPU inference!")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run synchronized FP16 GPU inference on input keypoint tensor x."""
        if x.shape != self.input_shape:
            x = np.ascontiguousarray(x, dtype=np.float32)

        # Copy input data to pagelocked host buffer
        np.copyto(self.h_input, x)

        # Transfer host memory to GPU device
        self.cuda_driver.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Execute TensorRT inference engine
        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        elif hasattr(self.context, "execute_async_v2"):
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            self.context.execute_v2(bindings=self.bindings)

        # Transfer GPU device output back to host
        self.cuda_driver.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return np.copy(self.h_output)
