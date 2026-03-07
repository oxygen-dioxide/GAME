import onnxruntime as ort
import numpy as np
import time
import os

def benchmark_model(model_path, inputs, provider_list, iterations=10):
    try:
        # Create session with specific providers
        sess = ort.InferenceSession(model_path, providers=provider_list)
        
        # Warmup
        for _ in range(3):
            sess.run(None, inputs)
            
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            sess.run(None, inputs)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        return avg_time
    except Exception as e:
        print(f"Error testing {model_path} with {provider_list}: {e}")
        return float('inf')

def main():
    model_dir = "experiments/GAME-1.0-small-onnx-static"
    
    print("="*50)
    print("BENCHMARKING PURE CPU vs DML (Partial Fallback)")
    print("="*50)
    
    # Define inputs for STATIC models
    L_STATIC = 441000
    T_STATIC = 1000
    N_STATIC = 200
    C_STATIC = 128
    
    encoder_inputs = {
        "waveform": np.random.randn(1, L_STATIC).astype(np.float32),
        "duration": np.array([10.0], dtype=np.float32)
    }
    
    segmenter_inputs = {
        "x_seg": np.random.randn(1, T_STATIC, C_STATIC).astype(np.float32),
        "language": np.zeros((1,), dtype=np.int64),
        "known_boundaries": np.ones((1, T_STATIC), dtype=bool),
        "prev_boundaries": np.ones((1, T_STATIC), dtype=bool),
        "t": np.array(0.5, dtype=np.float32),
        "maskT": np.ones((1, T_STATIC), dtype=bool),
        "threshold": np.array(0.5, dtype=np.float32),
        "radius": np.array(2, dtype=np.int64)
    }
    
    estimator_inputs = {
        "x_est": np.random.randn(1, T_STATIC, C_STATIC).astype(np.float32),
        "boundaries": np.zeros((1, T_STATIC), dtype=bool),
        "maskT": np.ones((1, T_STATIC), dtype=bool),
        "maskN": np.ones((1, N_STATIC), dtype=bool),
        "threshold": np.array(0.5, dtype=np.float32)
    }
    estimator_inputs["boundaries"][0, ::5] = True # mock 200 boundaries
    
    models_to_test = [
        ("encoder.onnx", encoder_inputs),
        ("segmenter.onnx", segmenter_inputs),
        ("estimator.onnx", estimator_inputs)
    ]
    
    total_cpu_time = 0
    total_dml_time = 0
    
    for model_name, inputs in models_to_test:
        model_path = os.path.join(model_dir, model_name)
        print(f"\nTesting {model_name}...")
        
        # Test Pure CPU
        cpu_time = benchmark_model(model_path, inputs, ['CPUExecutionProvider'], iterations=10)
        print(f"  -> Pure CPU Avg Time: {cpu_time:.4f}s")
        total_cpu_time += cpu_time
        
        # Test DML (with CPU fallback)
        dml_time = benchmark_model(model_path, inputs, ['DmlExecutionProvider', 'CPUExecutionProvider'], iterations=10)
        print(f"  -> DML (w/ Fallback) Avg Time: {dml_time:.4f}s")
        total_dml_time += dml_time
        
        if dml_time < cpu_time:
            print(f"  [RESULT] DML is {cpu_time/dml_time:.2f}x FASTER")
        else:
            print(f"  [RESULT] CPU is {dml_time/cpu_time:.2f}x FASTER")

    print("\n" + "="*50)
    print("TOTAL ESTIMATED PIPELINE TIME (1 Iteration of each model)")
    print("="*50)
    print(f"Pure CPU Total: {total_cpu_time:.4f}s")
    print(f"DML Total:      {total_dml_time:.4f}s")
    if total_dml_time < total_cpu_time:
        print(f"Overall DML is {total_cpu_time/total_dml_time:.2f}x FASTER")
    else:
        print(f"Overall Pure CPU is {total_dml_time/total_cpu_time:.2f}x FASTER")

if __name__ == "__main__":
    main()
