import numpy as np
import onnxruntime as ort
import librosa
from inference.slicer2 import Slicer
import sys

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=5)

def compare_tensors(cpu_tensor, dml_tensor, name, tolerance=1e-5):
    """Compares two numpy tensors and prints the difference."""
    if cpu_tensor.shape != dml_tensor.shape:
        print(f"❌ {name}: Shape mismatch! CPU: {cpu_tensor.shape}, DML: {dml_tensor.shape}")
        return False
    
    if not np.allclose(cpu_tensor, dml_tensor, atol=tolerance):
        diff = np.abs(cpu_tensor - dml_tensor)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        print(f"❌ {name}: Tensors differ significantly!")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Average difference: {avg_diff:.6f}")
        # print("CPU:\n", cpu_tensor)
        # print("DML:\n", dml_tensor)
        return False
    
    print(f"✓ {name}: Outputs are consistent.")
    return True

def run_and_compare():
    model_dir = "experiments/GAME-1.0-small-onnx-opset17"
    
    # --- Load providers ---
    cpu_providers = ['CPUExecutionProvider']
    dml_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    print(f"Loading models from: {model_dir}")
    
    # --- Load models for both providers ---
    cpu_encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=cpu_providers)
    dml_encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=dml_providers)
    
    cpu_segmenter = ort.InferenceSession(f"{model_dir}/segmenter.onnx", providers=cpu_providers)
    dml_segmenter = ort.InferenceSession(f"{model_dir}/segmenter.onnx", providers=dml_providers)
    
    cpu_estimator = ort.InferenceSession(f"{model_dir}/estimator.onnx", providers=cpu_providers)
    dml_estimator = ort.InferenceSession(f"{model_dir}/estimator.onnx", providers=dml_providers)
    
    cpu_dur2bd = ort.InferenceSession(f"{model_dir}/dur2bd.onnx", providers=cpu_providers)
    dml_dur2bd = ort.InferenceSession(f"{model_dir}/dur2bd.onnx", providers=dml_providers)
    
    cpu_bd2dur = ort.InferenceSession(f"{model_dir}/bd2dur.onnx", providers=cpu_providers)
    dml_bd2dur = ort.InferenceSession(f"{model_dir}/bd2dur.onnx", providers=dml_providers)
    
    # --- Prepare input data ---
    waveform, _ = librosa.load("未命名1.wav", sr=44100, mono=True)
    slicer = Slicer(sr=44100, threshold=-40., min_length=1000, min_interval=200, max_sil_kept=100)
    chunk = slicer.slice(waveform)[0] # Use first chunk
    chunk_wav = chunk["waveform"][np.newaxis, :].astype(np.float32)
    chunk_duration = np.array([len(chunk["waveform"]) / 44100], dtype=np.float32)
    
    # --- Step 1: Encoder ---
    print("\n--- 1. Testing Encoder ---")
    cpu_enc_input = {"waveform": chunk_wav, "duration": chunk_duration}
    cpu_x_seg, cpu_x_est, cpu_maskT = cpu_encoder.run(None, cpu_enc_input)
    dml_x_seg, dml_x_est, dml_maskT = dml_encoder.run(None, cpu_enc_input)

    if not compare_tensors(cpu_x_seg, dml_x_seg, "Encoder:x_seg"): return
    if not compare_tensors(cpu_x_est, dml_x_est, "Encoder:x_est"): return
    if not compare_tensors(cpu_maskT, dml_maskT, "Encoder:maskT"): return

    # --- Step 2: dur2bd ---
    print("\n--- 2. Testing dur2bd ---")
    known_durations = np.array([[chunk_duration[0]]], dtype=np.float32)
    cpu_dur2bd_input = {"durations": known_durations, "maskT": cpu_maskT}
    cpu_known_boundaries = cpu_dur2bd.run(None, cpu_dur2bd_input)[0]
    dml_known_boundaries = dml_dur2bd.run(None, cpu_dur2bd_input)[0]

    if not compare_tensors(cpu_known_boundaries, dml_known_boundaries, "dur2bd:known_boundaries"): return
    
    # --- Step 3: Segmenter (D3PM Loop) ---
    print("\n--- 3. Testing Segmenter (D3PM Loop) ---")
    t0, nsteps = 0.0, 8
    ts = [t0 + i * ((1 - t0) / nsteps) for i in range(nsteps)]
    
    cpu_boundaries = cpu_known_boundaries
    dml_boundaries = dml_known_boundaries
    
    for i, t in enumerate(ts):
        print(f"  --- D3PM Step {i} (t={t:.3f}) ---")
        seg_input = {
            "language": np.array([0], dtype=np.int64),
            "t": np.array(t, dtype=np.float32),
            "threshold": np.array(0.2, dtype=np.float32),
            "radius": np.array(2, dtype=np.int64),
        }
        
        cpu_seg_input = {**seg_input, "x_seg": cpu_x_seg, "maskT": cpu_maskT, "known_boundaries": cpu_known_boundaries, "prev_boundaries": cpu_boundaries}
        dml_seg_input = {**seg_input, "x_seg": dml_x_seg, "maskT": dml_maskT, "known_boundaries": dml_known_boundaries, "prev_boundaries": dml_boundaries}
        
        cpu_boundaries = cpu_segmenter.run(None, cpu_seg_input)[0]
        dml_boundaries = dml_segmenter.run(None, dml_seg_input)[0]
        
        if not compare_tensors(cpu_boundaries, dml_boundaries, f"Segmenter:boundaries (t={t:.3f})"): 
            print("🛑 Discrepancy found in Segmenter!")
            return

    # --- Step 4: bd2dur ---
    print("\n--- 4. Testing bd2dur ---")
    cpu_bd2dur_input = {"boundaries": cpu_boundaries, "maskT": cpu_maskT}
    dml_bd2dur_input = {"boundaries": dml_boundaries, "maskT": dml_maskT}
    
    cpu_durations, cpu_maskN = cpu_bd2dur.run(None, cpu_bd2dur_input)
    dml_durations, dml_maskN = dml_bd2dur.run(None, dml_bd2dur_input)

    if not compare_tensors(cpu_durations, dml_durations, "bd2dur:durations"): return
    if not compare_tensors(cpu_maskN, dml_maskN, "bd2dur:maskN"): return
    
    # --- Step 5: Estimator ---
    print("\n--- 5. Testing Estimator ---")
    cpu_est_input = {"x_est": cpu_x_est, "boundaries": cpu_boundaries, "maskT": cpu_maskT, "maskN": cpu_maskN, "threshold": np.array(0.2, dtype=np.float32)}
    dml_est_input = {"x_est": dml_x_est, "boundaries": dml_boundaries, "maskT": dml_maskT, "maskN": dml_maskN, "threshold": np.array(0.2, dtype=np.float32)}

    cpu_presence, cpu_scores = cpu_estimator.run(None, cpu_est_input)
    dml_presence, dml_scores = dml_estimator.run(None, dml_est_input)
    
    if not compare_tensors(cpu_presence, dml_presence, "Estimator:presence"): return
    if not compare_tensors(cpu_scores, dml_scores, "Estimator:scores"): return
    
    print("\n🎉 All model outputs are consistent between CPU and DML for the first chunk.")

if __name__ == "__main__":
    run_and_compare()
