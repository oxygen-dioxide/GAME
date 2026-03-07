import numpy as np
import onnxruntime as ort
import librosa
from inference.slicer2 import Slicer

# Load models
model_dir = "experiments/GAME-1.0-small-onnx-opset20"
providers = ['CPUExecutionProvider']

encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=providers)
segmenter = ort.InferenceSession(f"{model_dir}/segmenter.onnx", providers=providers)
estimator = ort.InferenceSession(f"{model_dir}/estimator.onnx", providers=providers)
dur2bd = ort.InferenceSession(f"{model_dir}/dur2bd.onnx", providers=providers)
bd2dur = ort.InferenceSession(f"{model_dir}/bd2dur.onnx", providers=providers)

# Load audio
waveform, _ = librosa.load("未命名1.wav", sr=44100, mono=True)
print(f"Waveform shape: {waveform.shape}, duration: {len(waveform)/44100:.2f}s")

# Slice
slicer = Slicer(sr=44100, threshold=-40., min_length=1000, min_interval=200, max_sil_kept=100)
chunks = slicer.slice(waveform)
print(f"Sliced into {len(chunks)} chunks")

# Process first chunk
chunk = chunks[0]
chunk_wav = chunk["waveform"]
chunk_offset = chunk["offset"]
chunk_duration = len(chunk_wav) / 44100

print(f"\nChunk 0: offset={chunk_offset:.3f}s, duration={chunk_duration:.3f}s, samples={len(chunk_wav)}")

# Encoder
if chunk_wav.ndim == 1:
    chunk_wav_2d = chunk_wav[np.newaxis, :]
else:
    chunk_wav_2d = chunk_wav

enc_out = encoder.run(None, {
    "waveform": chunk_wav_2d.astype(np.float32),
    "duration": np.array([chunk_duration], dtype=np.float32),
})
x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
print(f"Encoder: x_seg={x_seg.shape}, x_est={x_est.shape}, maskT={maskT.shape}")
print(f"maskT sum: {maskT.sum()}, mean: {maskT.mean():.3f}")

# dur2bd
known_durations = np.array([[chunk_duration]], dtype=np.float32)
dur2bd_out = dur2bd.run(None, {
    "durations": known_durations,
    "maskT": maskT,
})
known_boundaries = dur2bd_out[0]
print(f"dur2bd: known_boundaries={known_boundaries.shape}, sum={known_boundaries.sum():.3f}")

# Segmenter (single step, t=0)
seg_out = segmenter.run(None, {
    "x_seg": x_seg,
    "language": np.array([0], dtype=np.int64),
    "known_boundaries": known_boundaries,
    "prev_boundaries": known_boundaries,
    "t": np.array(0.0, dtype=np.float32),
    "maskT": maskT,
    "threshold": np.array(0.2, dtype=np.float32),
    "radius": np.array(2, dtype=np.int64),
})
boundaries = seg_out[0]
print(f"Segmenter: boundaries={boundaries.shape}, sum={boundaries.sum():.3f}, max={boundaries.max():.3f}")

# bd2dur
bd2dur_out = bd2dur.run(None, {
    "boundaries": boundaries,
    "maskT": maskT,
})
durations, maskN = bd2dur_out[0], bd2dur_out[1]
print(f"bd2dur: durations={durations.shape}, maskN={maskN.shape}")
print(f"maskN sum: {maskN.sum()}, durations[maskN>0]: {durations[maskN>0]}")

# Estimator
est_out = estimator.run(None, {
    "x_est": x_est,
    "boundaries": boundaries,
    "maskT": maskT,
    "maskN": maskN,
    "threshold": np.array(0.2, dtype=np.float32),
})
presence, scores = est_out[0], est_out[1]
print(f"Estimator: presence={presence.shape}, scores={scores.shape}")

# Extract valid notes
valid = maskN[0].astype(bool)
durations_valid = durations[0][valid]
presence_valid = presence[0][valid]
scores_valid = scores[0][valid]

print(f"\nValid notes: {valid.sum()}")
print(f"Durations: {durations_valid}")
print(f"Presence: {presence_valid}")
print(f"Scores: {scores_valid}")
