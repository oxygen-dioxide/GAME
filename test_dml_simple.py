import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
import os

def test():
    # 1. Create a dummy model
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return torch.matmul(x, x) + x

    m = SimpleModel()
    torch.onnx.export(
        m, torch.randn(100, 100), "dummy.onnx",
        opset_version=17,
        dynamic_axes={"x": {0: "dim0", 1: "dim1"}}
    )
    
    # 2. Test execution providers
    print("Available Providers:", ort.get_available_providers())
    
    sess_cpu = ort.InferenceSession("dummy.onnx", providers=["CPUExecutionProvider"])
    print("CPU Session Active Providers:", sess_cpu.get_providers())
    
    sess_dml = ort.InferenceSession("dummy.onnx", providers=["DmlExecutionProvider"])
    print("DML Session Active Providers:", sess_dml.get_providers())
    
    # 3. Benchmark
    x = np.random.randn(2000, 2000).astype(np.float32)
    
    start = time.time()
    for _ in range(50):
        _ = sess_cpu.run(None, {"x": x})
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.4f}s")
    
    start = time.time()
    for _ in range(50):
        _ = sess_dml.run(None, {"x": x})
    dml_time = time.time() - start
    print(f"DML Time: {dml_time:.4f}s")

if __name__ == "__main__":
    test()
