import onnxruntime as ort
import numpy as np
import time

def test_dynamic():
    print("Testing dynamic ONNX models with CUDA provider...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # We load the dynamic version created earlier
    model_dir = "experiments/GAME-1.0-small-onnx-opset18"
    
    try:
        print("\nLoading Segmenter...")
        segmenter = ort.InferenceSession(f"{model_dir}/segmenter.onnx", providers=providers)
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

    try:
        print("\nLoading Estimator...")
        estimator = ort.InferenceSession(f"{model_dir}/estimator.onnx", providers=providers)
        print("Estimator loaded successfully.")
        
        # Test estimator with a dynamic shape (B=1, T=100, N=10)
        B, T, C, N = 1, 100, 128, 10
        inputs = {
            "x_est": np.random.randn(B, T, C).astype(np.float32),
            "boundaries": np.zeros((B, T), dtype=bool),
            "maskT": np.ones((B, T), dtype=bool),
            "maskN": np.ones((B, N), dtype=bool),
            "threshold": np.array(0.5, dtype=np.float32)
        }
        # mock some boundaries
        inputs["boundaries"][0, ::10] = True
        
        est_inputs = [i.name for i in estimator.get_inputs()]
        inputs = {k: v for k, v in inputs.items() if k in est_inputs}
        
        print(f"Running Estimator inference with T={T}, N={N}...")
        start = time.time()
        out = estimator.run(None, inputs)
        print(f"Estimator inference done in {time.time() - start:.4f}s")
        print(f"Presence shape: {out[0].shape}, Scores shape: {out[1].shape}")
        
    except Exception as e:
        print(f"Estimator Failed: {e}")

if __name__ == "__main__":
    test_dynamic()
