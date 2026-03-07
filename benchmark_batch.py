#!/usr/bin/env python3
"""
Batch inference benchmark script for GAME ONNX.
Tests the performance of processing audio chunks in batches.
"""

import time
import pathlib
import numpy as np
import onnxruntime as ort
import librosa
import json
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

def batch_generator(chunks, batch_size):
    """Yields batches of padded audio chunks."""
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        if not batch_chunks:
            continue
            
        max_len = max(len(c['waveform']) for c in batch_chunks)
        
        padded_wavs = []
        for chunk in batch_chunks:
            wav = chunk['waveform']
            pad_len = max_len - len(wav)
            padded_wav = np.pad(wav, (0, pad_len), 'constant')
            padded_wavs.append(padded_wav)
            
        yield np.stack(padded_wavs, axis=0).astype(np.float32)

def run_batch_benchmark():
    # --- Configuration ---
    model_dir = "experiments/GAME-1.0-small-onnx-opset17"
    audio_path = "E:\\创作文件夹\\God Knows\\God knows... - 平野綾_vocals.wav"
    batch_size = 4  # Set batch size to 4
    
    # --- Create Optimized Session Options ---
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    
    print(f"✓ Graph Optimization Level: ORT_ENABLE_ALL")
    print(f"✓ Memory Pattern Optimization: Enabled")
    print(f"✓ Batch Size: {batch_size}")
    
    timer = Timing()
    timer.start()

    # --- Load Models ---
    try:
        encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", sess_options=sess_options, providers=providers)
        segmenter = ort.InferenceSession(f"{model_dir}/segmenter.onnx", sess_options=sess_options, providers=providers)
        estimator = ort.InferenceSession(f"{model_dir}/estimator.onnx", sess_options=sess_options, providers=providers)
        dur2bd = ort.InferenceSession(f"{model_dir}/dur2bd.onnx", sess_options=sess_options, providers=providers)
        bd2dur = ort.InferenceSession(f"{model_dir}/bd2dur.onnx", sess_options=sess_options, providers=providers)
        with open(pathlib.Path(model_dir) / "config.json", "r") as f:
            config = json.load(f)
        samplerate = config['samplerate']
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    timer.a_lap("Model Loading (Optimized)")

    # --- Load Audio ---
    try:
        waveform, _ = librosa.load(audio_path, sr=samplerate, mono=True)
    except Exception as e:
        print(f"❌ Error loading audio file '{audio_path}': {e}")
        return
    timer.a_lap(f"Audio Loading ({len(waveform)/samplerate:.2f}s)")
    
    # --- Slicing ---
    slicer = Slicer(sr=samplerate, threshold=-32., min_length=5000, min_interval=500, max_sil_kept=500)
    chunks = slicer.slice(waveform)
    timer.a_lap(f"Slicing (Relaxed) ({len(chunks)} chunks)")

    # --- Inference Loop ---
    total_inference_time = 0
    total_notes = 0
    
    t0, nsteps = 0.0, 8
    ts = [t0 + i * ((1 - t0) / nsteps) for i in range(nsteps)]
    seg_threshold, seg_radius, est_threshold = 0.2, 0.02, 0.2
    boundary_radius_frames = round(seg_radius / config['timestep'])
    
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch_idx, batch_wavs in enumerate(batch_generator(chunks, batch_size)):
        loop_start = time.perf_counter()
        
        current_batch_size = batch_wavs.shape[0]
        batch_durations = np.array([batch_wavs.shape[1] / samplerate] * current_batch_size, dtype=np.float32)

        # Encoder
        enc_out = encoder.run(None, {"waveform": batch_wavs, "duration": batch_durations})
        x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
        
        known_boundaries = np.zeros_like(maskT, dtype=bool)
        
        boundaries = known_boundaries
        for t in ts:
            seg_out = segmenter.run(None, {
                "x_seg": x_seg, "language": np.array([0] * current_batch_size, dtype=np.int64),
                "known_boundaries": known_boundaries, "prev_boundaries": boundaries,
                "t": np.array(t, dtype=np.float32), "maskT": maskT,
                "threshold": np.array(seg_threshold, dtype=np.float32),
                "radius": np.array(boundary_radius_frames, dtype=np.int64),
            })
            boundaries = seg_out[0]
        
        durations, maskN = bd2dur.run(None, {"boundaries": boundaries, "maskT": maskT})
        
        presence, scores = estimator.run(None, {
            "x_est": x_est, "boundaries": boundaries, "maskT": maskT, "maskN": maskN,
            "threshold": np.array(est_threshold, dtype=np.float32)
        })
        
        loop_end = time.perf_counter()
        loop_elapsed = (loop_end - loop_start) * 1000
        total_inference_time += loop_elapsed
        
        # Process results for each item in the batch
        batch_notes = 0
        for i in range(current_batch_size):
            valid = (maskN[i].astype(bool)) & (durations[i] > 0) & (presence[i])
            batch_notes += valid.sum()
        total_notes += batch_notes
        
        print(f"  - Batch {batch_idx+1}/{num_batches} (size={current_batch_size}) processed in {loop_elapsed:.2f} ms, found {batch_notes} notes.")

    timer.stages['Inference (Total)'] = total_inference_time
    
    print("\n" + "="*60)
    print("Batch Benchmark Summary")
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
    run_batch_benchmark()
