#!/usr/bin/env python3
"""
Benchmark不同batch size的推理速度
"""
import pathlib
import time
import numpy as np
from inference.onnx_api import load_onnx_model, enforce_max_chunk_size, extract_batch_generator
from inference.slicer2 import Slicer
import librosa

def benchmark_batch_size(audio_path: str, batch_sizes: list[int]):
    """测试不同batch size的推理速度"""
    
    # 加载模型
    model_dir = pathlib.Path("experiments/GAME-1.0-small-onnx")
    print(f"加载模型: {model_dir}")
    model = load_onnx_model(model_dir, device="dml")
    
    # 加载音频
    print(f"\n加载音频: {audio_path}")
    waveform, _ = librosa.load(audio_path, sr=model.samplerate, mono=True)
    duration = len(waveform) / model.samplerate
    print(f"音频时长: {duration:.2f}秒")
    
    # 切片
    slicer = Slicer(
        sr=model.samplerate,
        threshold=-40.,
        min_length=1000,
        min_interval=200,
        max_sil_kept=100,
    )
    initial_chunks = slicer.slice(waveform)
    chunks = enforce_max_chunk_size(initial_chunks, 15.0, model.samplerate)
    print(f"切片数量: {len(chunks)}")
    
    # 测试参数
    seg_threshold = 0.2
    seg_radius = 0.02
    est_threshold = 0.2
    boundary_radius_frames = round(seg_radius / model.timestep)
    ts = [0.0 + i * ((1 - 0.0) / 8) for i in range(8)]
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"测试 batch_size={batch_size}")
        print(f"{'='*60}")
        
        all_notes = []
        batch_count = 0
        
        # 预热
        print("预热中...")
        for batch_wavs, batch_durations, batch_chunks in extract_batch_generator(chunks[:2], batch_size, model.samplerate):
            model.infer_batch(
                waveforms=batch_wavs,
                durations=batch_durations,
                known_durations=None,
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=0,
                ts=ts,
            )
            break
        
        # 正式测试
        print("开始计时...")
        start_time = time.time()
        
        for batch_wavs, batch_durations, batch_chunks in extract_batch_generator(chunks, batch_size, model.samplerate):
            batch_results = model.infer_batch(
                waveforms=batch_wavs,
                durations=batch_durations,
                known_durations=None,
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=0,
                ts=ts,
            )
            
            batch_count += 1
            
            # 处理结果
            for chunk_result, chunk_info in zip(batch_results, batch_chunks):
                durations, presence, scores = chunk_result
                chunk_offset = chunk_info['offset']
                
                note_onset = np.concatenate([[0], np.cumsum(durations[:-1])]) + chunk_offset
                note_offset = np.cumsum(durations) + chunk_offset
                
                for onset, offset, score, is_present in zip(note_onset, note_offset, scores, presence):
                    if offset - onset > 0 and is_present:
                        all_notes.append((onset, offset, score))
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"总耗时: {elapsed:.2f}秒")
        print(f"处理batch数: {batch_count}")
        print(f"提取音符数: {len(all_notes)}")
        print(f"平均每batch耗时: {elapsed/batch_count:.3f}秒")
        print(f"处理速度: {duration/elapsed:.2f}x 实时")
        
        results[batch_size] = {
            'elapsed': elapsed,
            'batch_count': batch_count,
            'note_count': len(all_notes),
            'speed_ratio': duration / elapsed
        }
    
    # 对比结果
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}")
    
    baseline = results[batch_sizes[0]]
    print(f"\n基准 (batch_size={batch_sizes[0]}):")
    print(f"  总耗时: {baseline['elapsed']:.2f}秒")
    print(f"  处理速度: {baseline['speed_ratio']:.2f}x 实时")
    print(f"  音符数: {baseline['note_count']}")
    
    for batch_size in batch_sizes[1:]:
        result = results[batch_size]
        speedup = baseline['elapsed'] / result['elapsed']
        note_diff = result['note_count'] - baseline['note_count']
        
        print(f"\nbatch_size={batch_size}:")
        print(f"  总耗时: {result['elapsed']:.2f}秒")
        print(f"  处理速度: {result['speed_ratio']:.2f}x 实时")
        print(f"  音符数: {result['note_count']} (差异: {note_diff:+d})")
        print(f"  相对加速: {speedup:.2f}x")
        
        if speedup > 1.1:
            print(f"  ✓ 显著提升 ({(speedup-1)*100:.1f}%)")
        elif speedup > 1.0:
            print(f"  ✓ 轻微提升 ({(speedup-1)*100:.1f}%)")
        else:
            print(f"  ✗ 性能下降 ({(1-speedup)*100:.1f}%)")

if __name__ == "__main__":
    audio_path = r"E:\创作文件夹\God Knows\God knows(去和声有混响).wav"
    batch_sizes = [1, 2, 4]
    
    benchmark_batch_size(audio_path, batch_sizes)
