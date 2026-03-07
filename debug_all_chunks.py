import numpy as np
import onnxruntime as ort
import librosa
from inference.slicer2 import Slicer

def test_model(model_dir, model_name):
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    providers = ['CPUExecutionProvider']
    
    encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=providers)
    segmenter = ort.InferenceSession(f"{model_dir}/segmenter.onnx", providers=providers)
    estimator = ort.InferenceSession(f"{model_dir}/estimator.onnx", providers=providers)
    dur2bd = ort.InferenceSession(f"{model_dir}/dur2bd.onnx", providers=providers)
    bd2dur = ort.InferenceSession(f"{model_dir}/bd2dur.onnx", providers=providers)
    
    # Load audio
    waveform, _ = librosa.load("未命名1.wav", sr=44100, mono=True)
    
    # Slice
    slicer = Slicer(sr=44100, threshold=-40., min_length=1000, min_interval=200, max_sil_kept=100)
    chunks = slicer.slice(waveform)
    print(f"Total chunks: {len(chunks)}")
    
    # D3PM parameters
    t0 = 0.0
    nsteps = 8
    step = (1 - t0) / nsteps
    ts = [t0 + i * step for i in range(nsteps)]
    
    total_notes = 0
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_wav = chunk["waveform"]
        chunk_duration = len(chunk_wav) / 44100
        
        if chunk_wav.ndim == 1:
            chunk_wav_2d = chunk_wav[np.newaxis, :]
        else:
            chunk_wav_2d = chunk_wav
        
        # Encoder
        enc_out = encoder.run(None, {
            "waveform": chunk_wav_2d.astype(np.float32),
            "duration": np.array([chunk_duration], dtype=np.float32),
        })
        x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
        
        # dur2bd
        known_durations = np.array([[chunk_duration]], dtype=np.float32)
        dur2bd_out = dur2bd.run(None, {
            "durations": known_durations,
            "maskT": maskT,
        })
        known_boundaries = dur2bd_out[0]
        
        # D3PM loop
        boundaries = known_boundaries
        for t in ts:
            seg_out = segmenter.run(None, {
                "x_seg": x_seg,
                "language": np.array([0], dtype=np.int64),
                "known_boundaries": known_boundaries,
                "prev_boundaries": boundaries,
                "t": np.array(t, dtype=np.float32),
                "maskT": maskT,
                "threshold": np.array(0.2, dtype=np.float32),
                "radius": np.array(2, dtype=np.int64),
            })
            boundaries = seg_out[0]
        
        # bd2dur
        bd2dur_out = bd2dur.run(None, {
            "boundaries": boundaries,
            "maskT": maskT,
        })
        durations, maskN = bd2dur_out[0], bd2dur_out[1]
        
        # Estimator
        est_out = estimator.run(None, {
            "x_est": x_est,
            "boundaries": boundaries,
            "maskT": maskT,
            "maskN": maskN,
            "threshold": np.array(0.2, dtype=np.float32),
        })
        presence, scores = est_out[0], est_out[1]
        
        # Extract valid notes
        valid = maskN[0].astype(bool)
        durations_valid = durations[0][valid]
        presence_valid = presence[0][valid]
        
        # Count valid present notes
        valid_present = (durations_valid > 0) & presence_valid
        chunk_notes = valid_present.sum()
        total_notes += chunk_notes
        
        print(f"  Chunk {chunk_idx}: duration={chunk_duration:.3f}s, valid_notes={valid.sum()}, present_notes={chunk_notes}")
    
    print(f"\nTotal notes: {total_notes}")
    return total_notes

# Test both models
count17 = test_model("experiments/GAME-1.0-small-onnx-opset17", "opset17")
count20 = test_model("experiments/GAME-1.0-small-onnx-opset20", "opset20")

print(f"\n{'='*60}")
print(f"Summary: opset17={count17} notes, opset20={count20} notes")
print(f"Difference: {count20 - count17} notes")
print(f"{'='*60}")
