"""
TensorRT Export & Optimization Script for Signo (Jetson Orin Nano)

Converts the trained Keras model (conv1_lstm.keras) to ONNX format,
and compiles an optimized FP16 TensorRT engine (conv1_lstm.engine)
for ultra-fast GPU inference on NVIDIA Jetson.

Usage:
    python export_tensorrt.py
"""

import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("export_tensorrt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH = os.path.join(BASE_DIR, "app", "conv1_lstm.keras")
ONNX_PATH = os.path.join(BASE_DIR, "app", "conv1_lstm.onnx")
ENGINE_PATH = os.path.join(BASE_DIR, "app", "conv1_lstm.engine")


def keras_to_onnx():
    """Convert Keras model to ONNX format."""
    print("=" * 60)
    print("Step 1: Converting Keras model to ONNX...")
    print("=" * 60)

    if not os.path.exists(KERAS_PATH):
        logger.error("Keras model not found at %s", KERAS_PATH)
        return False

    try:
        import tensorflow as tf
        import tf2onnx

        logger.info("Loading Keras model from %s...", KERAS_PATH)
        model = tf.keras.models.load_model(KERAS_PATH)
        
        input_signature = [
            tf.TensorSpec(shape=(1, 60, 225), dtype=tf.float32, name="input_1")
        ]

        logger.info("Converting to ONNX (opset 13)...")
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13,
            output_path=ONNX_PATH
        )
        logger.info("✅ Successfully exported ONNX model to: %s", ONNX_PATH)
        return True
    except ImportError:
        logger.warning("tf2onnx not installed. Installing tf2onnx...")
        os.system(f"{sys.executable} -m pip install tf2onnx onnx")
        try:
            import tensorflow as tf
            import tf2onnx

            model = tf.keras.models.load_model(KERAS_PATH)
            input_signature = [tf.TensorSpec(shape=(1, 60, 225), dtype=tf.float32, name="input_1")]
            tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13, output_path=ONNX_PATH)
            logger.info("✅ Successfully exported ONNX model to: %s", ONNX_PATH)
            return True
        except Exception as e:
            logger.error("Failed to convert Keras to ONNX: %s", e)
            return False
    except Exception as e:
        logger.error("Error during ONNX conversion: %s", e)
        return False


def onnx_to_tensorrt():
    """Build FP16 TensorRT engine from ONNX model."""
    print()
    print("=" * 60)
    print("Step 2: Compiling TensorRT FP16 Engine for Jetson...")
    print("=" * 60)

    if not os.path.exists(ONNX_PATH):
        logger.error("ONNX model not found at %s", ONNX_PATH)
        return False

    # Attempt to build via trtexec (available on JetPack / Jetson)
    import subprocess
    trtexec_cmd = [
        "trtexec",
        f"--onnx={ONNX_PATH}",
        f"--saveEngine={ENGINE_PATH}",
        "--fp16",
        "--workspace=1024",
        "--verbose=False"
    ]

    try:
        logger.info("Running trtexec with command: %s", " ".join(trtexec_cmd))
        result = subprocess.run(trtexec_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ Successfully compiled TensorRT FP16 engine to: %s", ENGINE_PATH)
            return True
        else:
            logger.warning("trtexec stderr: %s", result.stderr)
    except FileNotFoundError:
        logger.warning("trtexec tool not found in PATH (typical on non-Jetson host).")

    # Fallback to Python tensorrt library if installed
    try:
        import tensorrt as trt

        logger.info("Building engine using TensorRT Python API...")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(ONNX_PATH, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error("TensorRT parser error: %s", parser.get_error(error))
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
        if builder.platform_has_tf32:
            config.set_flag(trt.BuilderFlag.FP16)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to build serialized engine")
            return False

        with open(ENGINE_PATH, 'wb') as f:
            f.write(serialized_engine)

        logger.info("✅ Successfully compiled TensorRT FP16 engine to: %s", ENGINE_PATH)
        return True
    except Exception as e:
        logger.warning("TensorRT Python API build failed: %s", e)
        logger.info("ℹ️ Note: TensorRT engines are built on the target Jetson Orin Nano hardware during deployment.")
        return False


def main():
    print("=" * 55)
    print("  Signo — TensorRT Export Pipeline")
    print("=" * 55)

    success_onnx = keras_to_onnx()
    if success_onnx:
        onnx_to_tensorrt()

    print()
    print("Summary:")
    print(f"  - ONNX Model:     {'Ready' if os.path.exists(ONNX_PATH) else 'Missing'}")
    print(f"  - TensorRT Engine: {'Ready' if os.path.exists(ENGINE_PATH) else 'Will compile on Jetson at first boot'}")
    print("=" * 55)


if __name__ == "__main__":
    main()
