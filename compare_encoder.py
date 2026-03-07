import onnxruntime as ort
import numpy as np

def compare_encoders():
    print("Loading models with CPUExecutionProvider...")
    providers = ['CPUExecutionProvider']
    
    # Load opset 17 model
    sess_17 = ort.InferenceSession("experiments/GAME-1.0-small-onnx-opset17/encoder.onnx", providers=providers)
    # Load opset 20 model
    sess_20 = ort.InferenceSession("experiments/GAME-1.0-small-onnx-opset20/encoder.onnx", providers=providers)
    
    # Create dummy inputs
    # The pure infer script uses:
    # waveform: shape (1, L_STATIC) float32
    # duration: shape (1,) float32
    L_STATIC = 441000  # 10 seconds
    
    # Use random noise as dummy waveform to avoid trivial constant folding effects
    np.random.seed(42)
    dummy_waveform = np.random.randn(1, L_STATIC).astype(np.float32)
    dummy_duration = np.array([10.0], dtype=np.float32)
    
    inputs = {
        "waveform": dummy_waveform,
        "duration": dummy_duration
    }
    
    print("\nRunning inference for opset 17...")
    out_17 = sess_17.run(None, inputs)
    print("Opset 17 outputs shapes:")
    for i, out in enumerate(out_17):
        print(f"  Output {i}: {out.shape}, dtype: {out.dtype}")
        
    print("\nRunning inference for opset 20...")
    out_20 = sess_20.run(None, inputs)
    print("Opset 20 outputs shapes:")
    for i, out in enumerate(out_20):
        print(f"  Output {i}: {out.shape}, dtype: {out.dtype}")
        
    print("\nComparing numerical differences:")
    names = ["x_seg", "x_est", "maskT"]
    all_match = True
    
    for i in range(len(out_17)):
        v17 = out_17[i]
        v20 = out_20[i]
        
        if v17.shape != v20.shape:
            print(f"❌ Output {i} ({names[i]}) shape mismatch: {v17.shape} vs {v20.shape}")
            all_match = False
            continue
            
        if v17.dtype != v20.dtype:
            print(f"❌ Output {i} ({names[i]}) dtype mismatch: {v17.dtype} vs {v20.dtype}")
            all_match = False
            continue
            
        # For boolean masks (maskT)
        if v17.dtype == bool:
            match = np.array_equal(v17, v20)
            if not match:
                mismatch_count = np.sum(v17 != v20)
                print(f"❌ Output {i} ({names[i]}) bool values mismatch! {mismatch_count} different elements.")
                all_match = False
            else:
                print(f"✅ Output {i} ({names[i]}) exactly matches.")
        else:
            # For float outputs
            # Check for NaN or Inf
            if np.isnan(v17).any() or np.isnan(v20).any():
                print(f"⚠️ Output {i} ({names[i]}) contains NaN values!")
            
            diff = np.abs(v17 - v20)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            if max_diff > 1e-4:
                print(f"❌ Output {i} ({names[i]}) numerical diff too large. Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
                all_match = False
            elif max_diff > 0:
                print(f"⚠️ Output {i} ({names[i]}) matches closely but not exactly. Max diff: {max_diff:.6f}")
            else:
                print(f"✅ Output {i} ({names[i]}) exactly matches.")

    print("\nOverall result:")
    if all_match:
        print("🟢 The outputs of opset 17 and opset 20 are functionally identical.")
    else:
        print("🔴 There are differences in the outputs between the two opsets.")

if __name__ == "__main__":
    compare_encoders()
