#!/usr/bin/env python3
"""
API for pure ONNX inference of GAME models.
This API is designed to be a drop-in replacement for the PyTorch-based API
for users who want to use the ONNX-exported models.
"""
import csv
import json
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import librosa
import mido
import numpy as np
import onnxruntime as ort

from inference.slicer2 import Slicer


@dataclass
class NoteInfo:
    onset: float
    offset: float
    pitch: float


class ONNXInferenceModel:
    """
    Manages ONNX inference sessions and the inference process for a single chunk.
    This class is intended for internal use by the API functions.
    """
    
    def __init__(self, model_dir: pathlib.Path, use_dml: bool = True):
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.samplerate = self.config["samplerate"]
        self.timestep = self.config["timestep"]
        self.loop = self.config.get("loop", True)
        self.languages = self.config.get("languages", None)
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        
        if use_dml:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.encoder = ort.InferenceSession(
            (model_dir / "encoder.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
        self.segmenter = ort.InferenceSession(
            (model_dir / "segmenter.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
        self.estimator = ort.InferenceSession(
            (model_dir / "estimator.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
        self.dur2bd = ort.InferenceSession(
            (model_dir / "dur2bd.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
        self.bd2dur = ort.InferenceSession(
            (model_dir / "bd2dur.onnx").as_posix(), sess_options=sess_options, providers=providers
        )
    
    def infer_chunk(
        self,
        waveform: np.ndarray,
        known_durations: np.ndarray,
        boundary_threshold: float,
        boundary_radius: int,
        score_threshold: float,
        language: int = 0,
        ts: list[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if ts is None:
            ts = []
        
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        
        duration = waveform.shape[1] / self.samplerate
        
        enc_out = self.encoder.run(None, {"waveform": waveform, "duration": np.array([duration], dtype=np.float32)})
        x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
        
        if known_durations is not None and known_durations.size > 0:
            if known_durations.ndim == 1:
                known_durations = known_durations[np.newaxis, :]
            known_boundaries = self.dur2bd.run(None, {"durations": known_durations.astype(np.float32), "maskT": maskT})[0]
        else:
            known_boundaries = np.zeros_like(maskT, dtype=bool)
        
        boundaries = known_boundaries
        if self.loop and len(ts) > 0:
            for t in ts:
                boundaries = self.segmenter.run(None, {
                    "x_seg": x_seg, "language": np.array([language], dtype=np.int64),
                    "known_boundaries": known_boundaries, "prev_boundaries": boundaries,
                    "t": np.array(t, dtype=np.float32), "maskT": maskT,
                    "threshold": np.array(boundary_threshold, dtype=np.float32),
                    "radius": np.array(boundary_radius, dtype=np.int64),
                })[0]
        else:
             boundaries = self.segmenter.run(None, {
                "x_seg": x_seg, "language": np.array([language], dtype=np.int64),
                "known_boundaries": known_boundaries, "prev_boundaries": boundaries,
                "t": np.array(0.0, dtype=np.float32), "maskT": maskT,
                "threshold": np.array(boundary_threshold, dtype=np.float32),
                "radius": np.array(boundary_radius, dtype=np.int64),
            })[0]
        
        durations, maskN = self.bd2dur.run(None, {"boundaries": boundaries, "maskT": maskT})
        
        presence, scores = self.estimator.run(None, {
            "x_est": x_est, "boundaries": boundaries, "maskT": maskT, "maskN": maskN,
            "threshold": np.array(score_threshold, dtype=np.float32)
        })
        
        valid = maskN[0].astype(bool)
        return durations[0][valid], presence[0][valid], scores[0][valid]


def load_onnx_model(
    model_dir: pathlib.Path,
    device: Literal["dml", "cpu"] = "dml"
) -> ONNXInferenceModel:
    """
    Loads the ONNX models from a directory and initializes the inference engine.
    """
    print("Initializing ONNX sessions...")
    use_dml = (device == "dml")
    return ONNXInferenceModel(model_dir, use_dml=use_dml)


def _save_midi(notes: list[NoteInfo], filepath: pathlib.Path, tempo: int = 120):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    
    sorted_notes = sorted(notes, key=lambda n: n.onset)
    
    last_abs_ticks = 0
    for note in sorted_notes:
        abs_onset_ticks = round(note.onset * tempo * 8)
        abs_offset_ticks = round(note.offset * tempo * 8)
        
        if abs_offset_ticks <= abs_onset_ticks:
            abs_offset_ticks = abs_onset_ticks + 1
            
        midi_pitch = round(note.pitch)

        delta_onset_ticks = abs_onset_ticks - last_abs_ticks
        note_duration_ticks = abs_offset_ticks - abs_onset_ticks
        
        track.append(mido.Message("note_on", note=midi_pitch, velocity=100, time=delta_onset_ticks))
        track.append(mido.Message("note_off", note=midi_pitch, velocity=100, time=note_duration_ticks))
        
        last_abs_ticks = abs_offset_ticks
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with mido.MidiFile(charset="utf8") as midi_file:
        midi_file.tracks.append(track)
        midi_file.save(filepath)
    print(f"Saved MIDI file: {filepath}")


def _save_text(
    notes: list[NoteInfo],
    filepath: pathlib.Path,
    file_format: Literal["txt", "csv"],
    pitch_format: Literal["number", "name"],
    round_pitch: bool,
):
    onset_list = [f"{note.onset:.3f}" for note in notes]
    offset_list = [f"{note.offset:.3f}" for note in notes]
    pitch_list = []
    for note in notes:
        pitch = note.pitch
        if round_pitch:
            pitch = round(pitch)
        pitch_txt = librosa.midi_to_note(pitch, unicode=False, cents=not round_pitch) if pitch_format == "name" else f"{pitch:.3f}"
        pitch_list.append(pitch_txt)
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "txt":
        with filepath.open(encoding="utf8", mode="w") as f:
            for onset, offset, pitch in zip(onset_list, offset_list, pitch_list):
                f.write(f"{onset}\t{offset}\t{pitch}\n")
    elif file_format == "csv":
        with filepath.open(encoding="utf8", mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["onset", "offset", "pitch"])
            writer.writeheader()
            for row in zip(onset_list, offset_list, pitch_list):
                writer.writerow({"onset": row[0], "offset": row[1], "pitch": row[2]})
    print(f"Saved {file_format.upper()} file: {filepath}")


def infer_from_files(
    model: ONNXInferenceModel,
    filemap: dict[str, pathlib.Path],
    output_dir: pathlib.Path,
    output_formats: set[str],
    language_id: int,
    seg_threshold: float,
    seg_radius: float,
    ts: list[float],
    est_threshold: float,
    pitch_format: str,
    round_pitch: bool,
    tempo: float,
):
    slicer = Slicer(sr=model.samplerate, threshold=-40., min_length=1000, min_interval=200, max_sil_kept=100)
    boundary_radius_frames = round(seg_radius / model.timestep)

    for key, filepath in filemap.items():
        print(f"\nProcessing: {key}")
        waveform, _ = librosa.load(filepath, sr=model.samplerate, mono=True)
        chunks = slicer.slice(waveform)
        print(f"  Sliced into {len(chunks)} chunks")

        all_notes = []
        for chunk in chunks:
            durations, presence, scores = model.infer_chunk(
                waveform=chunk["waveform"],
                known_durations=None,
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=language_id,
                ts=ts,
            )
            note_onset = np.concatenate([[0], np.cumsum(durations[:-1])]) + chunk["offset"]
            note_offset = np.cumsum(durations) + chunk["offset"]
            
            for onset, offset, score, is_present in zip(note_onset, note_offset, scores, presence):
                if offset - onset > 0 and is_present:
                    all_notes.append(NoteInfo(onset=onset, offset=offset, pitch=score))

        all_notes.sort(key=lambda x: x.onset)
        print(f"  Extracted {len(all_notes)} notes")

        output_key = pathlib.Path(key).stem
        if "mid" in output_formats:
            _save_midi(all_notes, output_dir / f"{output_key}.mid", int(tempo))
        if "txt" in output_formats:
            _save_text(all_notes, output_dir / f"{output_key}.txt", "txt", pitch_format, round_pitch)
        if "csv" in output_formats:
            _save_text(all_notes, output_dir / f"{output_key}.csv", "csv", pitch_format, round_pitch)

def align_with_transcriptions(
    model: ONNXInferenceModel,
    transcription_paths: list[pathlib.Path],
    language_id: int,
    seg_threshold: float,
    seg_radius: float,
    ts: list[float],
    est_threshold: float,
    use_wb: bool,
    inplace: bool,
    save_dir: pathlib.Path,
    save_name: str,
):
    boundary_radius_frames = round(seg_radius / model.timestep)
    extensions = [".wav", ".flac"]

    for index_path in transcription_paths:
        print(f"\nProcessing: {index_path.as_posix()}")
        
        with open(index_path, "r", encoding="utf8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            items = list(reader)
        
        updated_items = []
        for item in items:
            name = item["name"]
            
            candidate_wav_fns = [index_path.parent / "wavs" / f"{name}{ext}" for ext in extensions]
            wav_fn = next((fn for fn in candidate_wav_fns if fn.is_file()), None)
            
            if wav_fn is None:
                print(f"  - WARNING: Waveform file not found for item '{name}'. Skipping.")
                updated_items.append(item)
                continue

            ph_dur = [float(d) for d in item["ph_dur"].split()]
            word_dur = []
            if use_wb and "ph_num" in item:
                ph_num = [int(n) for n in item["ph_num"].split()]
                if sum(ph_num) == len(ph_dur):
                    idx = 0
                    for num in ph_num:
                        word_dur.append(sum(ph_dur[idx : idx + num]))
                        idx += num
                else:
                    print(f"  - WARNING: ph_num and ph_dur length mismatch for item '{name}'. Using total duration.")
                    word_dur = [sum(ph_dur)]
            else:
                word_dur = [sum(ph_dur)]
            
            waveform, _ = librosa.load(wav_fn, sr=model.samplerate, mono=True)
            
            durations, presence, scores = model.infer_chunk(
                waveform=waveform,
                known_durations=np.array([word_dur]),
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=language_id,
                ts=ts,
            )
            
            note_seq = [
                librosa.midi_to_note(score, unicode=False, cents=True) if pres else "rest"
                for score, pres in zip(scores, presence)
            ]
            note_dur = [f"{dur:.3f}" for dur in durations]
            
            item["note_seq"] = " ".join(note_seq)
            item["note_dur"] = " ".join(note_dur)
            item.pop("note_glide", None)
            updated_items.append(item)
            print(f"  - Processed item: {name}")

        if inplace:
            output_path = index_path
        elif save_dir is not None and save_name is not None:
             output_path = save_dir / save_name
        else:
            output_path = index_path.parent / (save_name or f"{index_path.stem}-midi.csv")
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf8", newline="") as f:
            # Ensure all required fields are present
            if "note_seq" not in fieldnames:
                fieldnames.append("note_seq")
            if "note_dur" not in fieldnames:
                fieldnames.append("note_dur")
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_items)
        print(f"  Saved updated transcriptions to: {output_path.as_posix()}")
