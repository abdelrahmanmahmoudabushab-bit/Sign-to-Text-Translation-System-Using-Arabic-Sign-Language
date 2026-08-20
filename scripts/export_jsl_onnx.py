import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_jsl_embeddings import build_embedder

def export_onnx():
    embedder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "jsl_embedder.keras")
    onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "jsl_embedder.onnx")
    
    embedder = tf.keras.models.load_model(embedder_path, compile=False, safe_mode=False)
    
    import tf2onnx
    input_signature = [tf.TensorSpec([None, 60, 225], tf.float32, name="input")]
    model_proto, _ = tf2onnx.convert.from_keras(embedder, input_signature=input_signature, opset=13)
    
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"Successfully exported JSL Embedder to ONNX format at {onnx_path}")

if __name__ == "__main__":
    export_onnx()
