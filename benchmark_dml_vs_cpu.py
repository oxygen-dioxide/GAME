#!/usr/bin/env python3
"""Benchmark DML vs CPU to verify GPU acceleration"""

import time
import pathlib
import numpy as np
import onnxruntime as ort

def benchmark_provider(model_path: pathlib.Path, provider: str, num_runs: int = 10):
    """Benchmark a specific provider"""
    
    print(f"\n{'='*60}")
    print(f"Testing with {provider}")
    print(f"{'='*60}")
    
    # Create session
    providers = [provider] if provider != 'DmlExecutionProvider' else ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(
        model_path.as_posix(),
        providers=providers
    )
    
    print(f"Active providers: {session.get_providers()}")
    
    # Prepare input
    waveform = np.random.randn(1, 160000).astype(np.float32)  # 10 seconds at 16kHz
    duration = np.array([10.0], dtype=np.float32)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        session.run(None, {"waveform": waveform, "duration": duration})
    
    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        output = session.run(None, {"waveform": waveform, "duration": duration})
        end = time.perf_counter()
        elapsed = (end - start) * 1000  # Convert to ms
        times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.2f} ms")
    
    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    
    return avg_time

def main():
    model_path = pathlib.Path("experiments/GAME-1.0-small-onnx-opset17/encoder.onnx")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("="*60)
    print("DirectML vs CPU Benchmark")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Input: 10 seconds of audio (160000 samples)")
    
    # Test CPU
    cpu_time = benchmark_provider(model_path, 'CPUExecutionProvider', num_runs=10)
    
    # Test DML
    dml_time = benchmark_provider(model_path, 'DmlExecutionProvider', num_runs=10)
    
    # Compare
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")
    print(f"CPU Average: {cpu_time:.2f} ms")
    print(f"DML Average: {dml_time:.2f} ms")
    speedup = cpu_time / dml_time
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1.5:
        print(f"\n✓ GPU acceleration is working! ({speedup:.2f}x faster)")
        print("DirectML is successfully using your GPU.")
    elif speedup > 1.0:
        print(f"\n⚠ Minor speedup detected ({speedup:.2f}x)")
        print("GPU may be used but performance gain is limited.")
    else:
        print(f"\n❌ No speedup detected (CPU is faster)")
        print("DirectML may not be using the GPU properly.")
    
    print("\nTo verify GPU usage:")
    print("1. Open Task Manager (Ctrl+Shift+Esc)")
    print("2. Go to Performance tab")
    print("3. Check GPU 0 (integrated) and GPU 1 (discrete)")
    print("4. Run this script again and watch for activity")
    print("5. The GPU with activity spikes is being used")

if __name__ == "__main__":
    main()
