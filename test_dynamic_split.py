import onnxruntime as ort
import numpy as np
import time

def test_dynamic():
    print("Testing dynamic ONNX models with SPLIT providers (Encoder=DML, Others=CPU)...")
    dml_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    cpu_providers = ['CPUExecutionProvider']
    
    # We load the dynamic version created earlier
    model_dir = "experiments/GAME-1.0-small-onnx-opset18"
    
    try:
        print("\nLoading Encoder on DML...")
        encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=dml_providers)
        print("Encoder loaded successfully.")
        
        # Test encoder
        B, L = 1, 44100 * 5 # 5 seconds
        inputs = {
            "waveform": np.random.randn(B, L).astype(np.float32),
            "duration": np.array([5.0], dtype=np.float32)
        }
        
        print(f"Running Encoder inference with shape {inputs['waveform'].shape}...")
        start = time.time()
        enc_out = encoder.run(None, inputs)
        print(f"Encoder inference done in {time.time() - start:.4f}s")
        print(f"Output x_seg shape: {enc_out[0].shape}")
        
    except Exception as e:
        print(f"Encoder Failed: {e}")

    try:
        print("\nLoading Segmenter on CPU...")
        segmenter = ort.InferenceSession(f"{model_dir}/segmenter.onnx", providers=cpu_providers)
        print("Segmenter loaded successfully.")
        
        # Test segmenter with a dynamic shape (B=1, T=100)
        B, T, C = 1, 100, 128
        inputs = {
            "x_seg": np.random.randn(B, T, C).astype(np.float32),
            "language": np.zeros((B,), dtype=np.int64),
            "known_boundaries": np.ones((B, T), dtype=bool),
            "prev_boundaries": np.ones((B, T), dtype=bool),
            "t": np.array(0.5, dtype=np.float32),
            "maskT": np.ones((B, T), dtype=bool),
            "threshold": np.array(0.5, dtype=np.float32),
            "radius": np.array(2, dtype=np.int64)
        }
        
        # Ensure only required inputs are passed
        seg_inputs = [i.name for i in segmenter.get_inputs()]
        inputs = {k: v for k, v in inputs.items() if k in seg_inputs}
        
        print(f"Running Segmenter inference with shape {inputs['x_seg'].shape}...")
        start = time.time()
        out = segmenter.run(None, inputs)
        print(f"Segmenter inference done in {time.time() - start:.4f}s")
        print(f"Output shape: {out[0].shape}")
        
    except Exception as e:
        print(f"Segmenter Failed: {e}")

if __name__ == "__main__":
    test_dynamic()
