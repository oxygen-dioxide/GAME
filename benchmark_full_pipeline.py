#!/usr/bin/env python3
"""
Full pipeline benchmark script for GAME ONNX inference.
Measures the speed of each component in the pipeline on a given audio file.
"""

import time
import pathlib
import numpy as np
import onnxruntime as ort
import librosa
from inference.slicer2 import Slicer

class Timing:
    """A simple helper class to store and print timing info."""
    def __init__(self):
        self.stages = {}
        self.start_time = None
        self.last_time = None

    def start(self):
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        print("Benchmark started...")

    def a_lap(self, name):
        now = time.perf_counter()
        elapsed = (now - self.last_time) * 1000  # ms
        self.stages[name] = elapsed
        self.last_time = now
        print(f"  - Stage '{name}' took: {elapsed:.2f} ms")

    def stop(self):
        total_elapsed = (time.perf_counter() - self.start_time) * 1000 # ms
        print(f"\nBenchmark finished.")
        print(f"Total time: {total_elapsed:.2f} ms")
        return total_elapsed

def run_benchmark():
    # --- Configuration ---
    # Use the specified opset 17 model for the benchmark
    model_dir = "experiments/GAME-1.0-small-onnx-opset17"
    audio_path = "E:\\创作文件夹\\God Knows\\God knows... - 平野綾_vocals.wav"
    providers = ['CPUExecutionProvider']
    
    timer = Timing()
    timer.start()

    # --- Load Models ---
    try:
        encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=providers)
        segmenter = ort.InferenceSession(f"{model_dir}/segmenter.onnx", providers=providers)
        estimator = ort.InferenceSession(f"{model_dir}/estimator.onnx", providers=providers)
        dur2bd = ort.InferenceSession(f"{model_dir}/dur2bd.onnx", providers=providers)
        bd2dur = ort.InferenceSession(f"{model_dir}/bd2dur.onnx", providers=providers)
        with open(pathlib.Path(model_dir) / "config.json", "r") as f:
            config = json.load(f)
        samplerate = config['samplerate']
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    timer.a_lap("Model Loading")

    # --- Load Audio ---
    try:
        waveform, _ = librosa.load(audio_path, sr=samplerate, mono=True)
    except Exception as e:
        print(f"❌ Error loading audio file '{audio_path}': {e}")
        return
    timer.a_lap(f"Audio Loading ({len(waveform)/samplerate:.2f}s)")
    
    # --- Slicing ---
    slicer = Slicer(sr=samplerate, threshold=-40., min_length=1000, min_interval=200, max_sil_kept=100)
    chunks = slicer.slice(waveform)
    timer.a_lap(f"Slicing ({len(chunks)} chunks)")

    # --- Inference Loop ---
    total_inference_time = 0
    total_notes = 0
    
    # D3PM parameters from infer.py defaults
    t0, nsteps = 0.0, 8
    ts = [t0 + i * ((1 - t0) / nsteps) for i in range(nsteps)]
    seg_threshold, seg_radius, est_threshold = 0.2, 0.02, 0.2
    boundary_radius_frames = round(seg_radius / config['timestep'])

    for i, chunk in enumerate(chunks):
        loop_start = time.perf_counter()
        
        chunk_wav = chunk["waveform"][np.newaxis, :].astype(np.float32)
        chunk_duration = np.array([len(chunk["waveform"]) / samplerate], dtype=np.float32)

        # Encoder
        enc_out = encoder.run(None, {"waveform": chunk_wav, "duration": chunk_duration})
        x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
        
        # Start with empty boundaries for `extract` mode
        known_boundaries = np.zeros_like(maskT, dtype=bool)
        
        # Segmenter (D3PM loop)
        boundaries = known_boundaries
        for t in ts:
            seg_out = segmenter.run(None, {
                "x_seg": x_seg, "language": np.array([0], dtype=np.int64),
                "known_boundaries": known_boundaries, "prev_boundaries": boundaries,
                "t": np.array(t, dtype=np.float32), "maskT": maskT,
                "threshold": np.array(seg_threshold, dtype=np.float32),
                "radius": np.array(boundary_radius_frames, dtype=np.int64),
            })
            boundaries = seg_out[0]
        
        # bd2dur
        durations, maskN = bd2dur.run(None, {"boundaries": boundaries, "maskT": maskT})
        
        # Estimator
        presence, scores = estimator.run(None, {
            "x_est": x_est, "boundaries": boundaries, "maskT": maskT, "maskN": maskN,
            "threshold": np.array(est_threshold, dtype=np.float32)
        })
        
        loop_end = time.perf_counter()
        loop_elapsed = (loop_end - loop_start) * 1000
        total_inference_time += loop_elapsed
        
        valid = (maskN[0].astype(bool)) & (durations[0] > 0) & (presence[0])
        total_notes += valid.sum()
        
        print(f"  - Chunk {i+1}/{len(chunks)} processed in {loop_elapsed:.2f} ms, found {valid.sum()} notes.")

    timer.stages['Inference (Total)'] = total_inference_time
    
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"Audio File: {audio_path}")
    print(f"Model: {model_dir}")
    print(f"Execution Provider: {providers[0]}")
    print(f"Total Notes Extracted: {total_notes}")
    print("\n--- STAGE TIMINGS ---")
    for name, elapsed in timer.stages.items():
        print(f"  - {name:<25}: {elapsed:.2f} ms")
        
    timer.stop()


if __name__ == "__main__":
    import json
    run_benchmark()
