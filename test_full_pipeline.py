import onnxruntime as ort
import numpy as np

def run_pipeline(opset_ver):
    print(f"--- Testing Opset {opset_ver} ---")
    base_path = f"experiments/GAME-1.0-small-onnx-opset{opset_ver}/"
    providers = ['CPUExecutionProvider']
    
    # Load all models
    sess_encoder = ort.InferenceSession(base_path + "encoder.onnx", providers=providers)
    sess_dur2bd = ort.InferenceSession(base_path + "dur2bd.onnx", providers=providers)
    sess_seg = ort.InferenceSession(base_path + "segmenter.onnx", providers=providers)
    sess_bd2dur = ort.InferenceSession(base_path + "bd2dur.onnx", providers=providers)
    sess_est = ort.InferenceSession(base_path + "estimator.onnx", providers=providers)
    
    # 1. Encoder
    import librosa
    L_STATIC = 441000
    N_STATIC = 200
    
    # Load actual audio file instead of random noise
    try:
        wav, sr = librosa.load("未命名1.wav", sr=44100, mono=True)
        # Pad or truncate to L_STATIC (10 seconds)
        if len(wav) < L_STATIC:
            wav = np.pad(wav, (0, L_STATIC - len(wav)))
        else:
            wav = wav[:L_STATIC]
        dummy_waveform = np.expand_dims(wav, axis=0).astype(np.float32)
        print("Loaded 未命名1.wav successfully.")
    except Exception as e:
        print(f"Failed to load wav: {e}. Falling back to noise.")
        np.random.seed(42)
        dummy_waveform = (np.random.randn(1, L_STATIC) * 0.1).astype(np.float32)
        
    dummy_duration = np.array([10.0], dtype=np.float32)
    
    enc_out = sess_encoder.run(None, {"waveform": dummy_waveform, "duration": dummy_duration})
    x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
    print(f"[Encoder] x_seg: {x_seg.shape}, x_est: {x_est.shape}, maskT: {maskT.shape}")
    
    # 2. dur2bd
    padded_known_durs = np.zeros((1, N_STATIC), dtype=np.float32)
    padded_known_durs[0, 0] = 10.0
    dur2bd_inputs = [i.name for i in sess_dur2bd.get_inputs()]
    dur2bd_args = {"durations": padded_known_durs}
    if "maskT" in dur2bd_inputs:
        dur2bd_args["maskT"] = maskT
        
    known_boundaries = sess_dur2bd.run(None, dur2bd_args)[0]
    print(f"[dur2bd] known_boundaries: {known_boundaries.shape}, values head: {known_boundaries[0, :5]}")
    
    # 3. segmenter
    seg_inputs = [i.name for i in sess_seg.get_inputs()]
    seg_args = {
        "x_seg": x_seg,
        "maskT": maskT,
        "threshold": np.array(0.2, dtype=np.float32),
        "radius": np.array(2, dtype=np.int64) # ~0.02s
    }
    if "language" in seg_inputs:
        seg_args["language"] = np.array([0], dtype=np.int64)
    if "known_boundaries" in seg_inputs:
        seg_args["known_boundaries"] = known_boundaries
    if "prev_boundaries" in seg_inputs:
        seg_args["prev_boundaries"] = known_boundaries
    if "t" in seg_inputs:
        seg_args["t"] = np.array(0.0, dtype=np.float32)
        
    boundaries = sess_seg.run(None, seg_args)[0]
    print(f"[Segmenter] boundaries: {boundaries.shape}, non-zero count: {np.count_nonzero(boundaries)}")
    
    # 4. bd2dur
    bd2dur_out = sess_bd2dur.run(None, {
        "boundaries": boundaries,
        "maskT": maskT
    })
    durations, maskN = bd2dur_out[0], bd2dur_out[1]
    print(f"[bd2dur] durations: {durations.shape}, maskN: {maskN.shape}")
    print(f"         valid notes count (maskN=True): {np.sum(maskN)}")
    
    # Pad to N_STATIC if needed (mimicking the inference code)
    actual_n = maskN.shape[1]
    if actual_n < N_STATIC:
        pad_maskN = np.zeros((1, N_STATIC), dtype=bool)
        pad_maskN[0, :actual_n] = maskN[0]
        padded_maskN = pad_maskN
        pad_durs = np.zeros((1, N_STATIC), dtype=np.float32)
        pad_durs[0, :actual_n] = durations[0]
        padded_durations = pad_durs
    else:
        padded_maskN = maskN[:, :N_STATIC]
        padded_durations = durations[:, :N_STATIC]
        
    # 5. estimator
    est_inputs = [i.name for i in sess_est.get_inputs()]
    est_args = {
        "x_est": x_est,
        "boundaries": boundaries,
        "maskT": maskT,
        "maskN": padded_maskN,
        "threshold": np.array(0.2, dtype=np.float32)
    }
    est_args = {k: v for k, v in est_args.items() if k in est_inputs}
    
    est_out = sess_est.run(None, est_args)
    presence, scores = est_out[0], est_out[1]
    
    # Combine final note info
    valid_notes = padded_maskN[0].astype(bool)
    final_pres = presence[0][valid_notes]
    final_scores = scores[0][valid_notes]
    final_durs = padded_durations[0][valid_notes]
    
    print(f"[Estimator] presence: {presence.shape}, scores: {scores.shape}")
    print(f"            Notes present: {np.sum(final_pres)} / {len(final_pres)}")
    
    return {
        "enc_seg": x_seg, "enc_est": x_est, "enc_maskT": maskT,
        "known_bounds": known_boundaries,
        "bounds": boundaries,
        "durs": durations, "maskN": maskN,
        "pres": presence, "scores": scores,
        "final_notes": np.sum(final_pres)
    }

if __name__ == "__main__":
    out17 = run_pipeline("17")
    out20 = run_pipeline("20")
    
    print("\n=== SUMMARY COMPARISON ===")
    print(f"Final valid notes generated -> Opset17: {out17['final_notes']} | Opset20: {out20['final_notes']}")
    
    if out17['final_notes'] == 0 and out20['final_notes'] > 0:
        print("💡 The bug is confirmed: Opset17 produces 0 notes while Opset20 produces notes.")
        
    # Find the first step that diverges significantly
    diff_known_b = np.max(out17['known_bounds'] ^ out20['known_bounds']) # boolean xor
    diff_b = np.max(out17['bounds'] ^ out20['bounds']) # boolean xor
    diff_durs = np.max(np.abs(out17['durs'] - out20['durs']))
    diff_scores = np.max(np.abs(out17['scores'] - out20['scores']))
    
    print("\nMaximum absolute differences between Opset17 and Opset20:")
    print(f"1. dur2bd (known_boundaries) : {diff_known_b}")
    print(f"2. segmenter (boundaries)    : {diff_b}")
    print(f"3. bd2dur (durations)        : {diff_durs}")
    print(f"4. estimator (scores)        : {diff_scores}")

    print("\n=== Tensor Content Dump for Opset 17 ===")
    pres17 = out17['pres']
    scores17 = out17['scores']
    maskN17 = out17['maskN']
    
    valid_notes17 = maskN17[0].astype(bool)
    
    print("\n[Opset 17] First 35 elements of maskN (Valid Flags):")
    print(valid_notes17[:35])
    
    print("\n[Opset 17] Presence (Boolean True/False based on threshold) for the first 35 elements:")
    print(pres17[0, :35])
    
    print("\n[Opset 17] Scores (Raw float values) for the first 35 elements:")
    # Using np.set_printoptions to show floats clearly
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    print(scores17[0, :35])
    
    print("\n=== Tensor Content Dump for Opset 20 ===")
    pres20 = out20['pres']
    scores20 = out20['scores']
    maskN20 = out20['maskN']
    
    valid_notes20 = maskN20[0].astype(bool)
    
    print("\n[Opset 20] First 35 elements of maskN (Valid Flags):")
    print(valid_notes20[:35])
    
    print("\n[Opset 20] Presence (Boolean True/False based on threshold) for the first 35 elements:")
    print(pres20[0, :35])
    
    print("\n[Opset 20] Scores (Raw float values) for the first 35 elements:")
    print(scores20[0, :35])
