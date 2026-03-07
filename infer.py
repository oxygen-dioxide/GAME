#!/usr/bin/env python3
"""
Pure ONNX inference script for GAME (Generative Adaptive MIDI Extractor)
No PyTorch dependency - uses only onnxruntime and numpy
"""

import csv
import glob
import json
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import click
import librosa
import mido
import numpy as np
import onnxruntime as ort

from inference.slicer2 import Slicer

_OPT_KEY_BATCH_SIZE = "batch_size"
_OPT_KEY_NUM_WORKERS = "num_workers"
_OPT_KEY_SEG_THRESHOLD = "seg_threshold"
_OPT_KEY_SEG_RADIUS = "seg_radius"
_OPT_KEY_SEG_D3PM_T0 = "t0"
_OPT_KEY_SEG_D3PM_NSTEPS = "nsteps"
_OPT_KEY_EST_THRESHOLD = "est_threshold"


@dataclass
class NoteInfo:
    onset: float
    offset: float
    pitch: float


class ONNXInferenceModel:
    """Pure ONNX inference model without torch dependency"""
    
    def __init__(self, model_dir: pathlib.Path, use_dml: bool = True):
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.samplerate = self.config["samplerate"]
        self.timestep = self.config["timestep"]
        self.loop = self.config.get("loop", True)
        self.languages = self.config.get("languages", None)
        
        # --- Create Optimized Session Options ---
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        
        if use_dml:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        print("Initializing ONNX sessions...")
        print(f"  - Provider: {providers[0]}")
        print(f"  - Graph Optimization: ORT_ENABLE_ALL")
        print(f"  - Memory Pattern: Enabled")
        
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
    
    def infer(
        self,
        waveform: np.ndarray,
        known_durations: np.ndarray,
        boundary_threshold: float,
        boundary_radius: int,
        score_threshold: float,
        language: int = 0,
        ts: list[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on a single waveform chunk
        """
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


def _validate_d3pm_ts(ctx, param, value) -> list[float] | None:
    if value is None:
        return None
    try:
        ts = [float(t.strip()) for t in value.split(",")]
        if not ts:
            raise ValueError("At least one T value must be provided.")
        if any(t < 0 or t >= 1 for t in ts):
            raise ValueError("All T values must be in the range (0, 1).")
        return ts
    except Exception as e:
        raise click.BadParameter(f"Invalid T values: {e}")


def _validate_exts(ctx, param, value) -> set[str]:
    try:
        exts = {"." + ext.strip().lower() for ext in value.split(",")}
        if not exts:
            raise ValueError("At least one extension must be provided.")
        return exts
    except Exception as e:
        raise click.BadParameter(f"Invalid extensions: {e}")


def _validate_output_formats(ctx, param, value) -> set[str]:
    try:
        formats = {fmt.strip().lower() for fmt in value.split(",")}
        supported_formats = {"mid", "txt", "csv"}
        if not formats.issubset(supported_formats):
            raise ValueError(
                f"Unsupported formats: {formats - supported_formats}. "
                f"Supported formats: {supported_formats}"
            )
        return formats
    except Exception as e:
        raise click.BadParameter(f"Invalid output formats: {e}")


def _validate_path_or_glob(ctx, param, value) -> list[pathlib.Path]:
    try:
        paths = []
        for v in value:
            if glob.has_magic(v):
                paths.extend(glob.glob(v, recursive=True))
            else:
                paths.append(v)
        if not paths:
            raise FileNotFoundError(f"No files found for paths: {value}")
        paths = [pathlib.Path(p) for p in paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Path does not exist: {p}")
            if not p.is_file():
                raise FileNotFoundError(f"Path is not a file: {p}")
        return paths
    except Exception as e:
        raise click.BadParameter(f"Invalid path or glob: {e}")


def _t0_nstep_to_ts(t0: float, nsteps: int) -> list[float]:
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]


def _get_language_id(language: str, lang_map: dict[str, int]) -> int:
    if language and lang_map:
        if language not in lang_map:
            raise ValueError(
                f"Language '{language}' not supported by the segmentation model. "
                f"Supported languages: {', '.join(lang_map.keys())}"
            )
        language_id = lang_map[language]
    else:
        language_id = 0
    return language_id


def _parse_filemap(path: pathlib.Path, exts: set[str], glb: str | None) -> dict[str, pathlib.Path]:
    if path.is_file():
        return {path.name: path}
    elif path.is_dir():
        if glb:
            files = [f for f in path.rglob(glb) if f.is_file()]
        else:
            files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in exts]
        filemap = {f.relative_to(path).as_posix(): f for f in files}
        if not filemap:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return filemap
    else:
        raise ValueError(f"Invalid path: {path}")


def save_midi(notes: list[NoteInfo], filepath: pathlib.Path, tempo: int = 120):
    """Save notes to MIDI file, ensuring no cumulative precision errors."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    
    # Sort notes by onset time to process them in order
    sorted_notes = sorted(notes, key=lambda n: n.onset)
    
    last_abs_ticks = 0
    for note in sorted_notes:
        # 1. Convert absolute seconds to absolute ticks and then round
        abs_onset_ticks = round(note.onset * tempo * 8)
        abs_offset_ticks = round(note.offset * tempo * 8)
        
        # Ensure note has a duration after rounding
        if abs_offset_ticks <= abs_onset_ticks:
            abs_offset_ticks = abs_onset_ticks + 1 # Give it a minimal duration of 1 tick
            
        midi_pitch = round(note.pitch)

        # 2. Calculate delta time based on the rounded absolute ticks
        delta_onset_ticks = abs_onset_ticks - last_abs_ticks
        note_duration_ticks = abs_offset_ticks - abs_onset_ticks
        
        # Mido handles delta times correctly
        track.append(mido.Message("note_on", note=midi_pitch, velocity=100, time=delta_onset_ticks))
        track.append(mido.Message("note_off", note=midi_pitch, velocity=100, time=note_duration_ticks))
        
        # 3. Update the last absolute time marker
        last_abs_ticks = abs_offset_ticks
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with mido.MidiFile(charset="utf8") as midi_file:
        midi_file.tracks.append(track)
        midi_file.save(filepath)
    print(f"Saved MIDI file: {filepath}")


def save_text(
    notes: list[NoteInfo],
    filepath: pathlib.Path,
    file_format: Literal["txt", "csv"],
    pitch_format: Literal["number", "name"],
    round_pitch: bool,
):
    """Save notes to text file"""
    onset_list = [f"{note.onset:.3f}" for note in notes]
    offset_list = [f"{note.offset:.3f}" for note in notes]
    pitch_list = []
    for note in notes:
        pitch = note.pitch
        if round_pitch:
            pitch = round(pitch)
            pitch_txt = str(pitch)
        else:
            pitch_txt = f"{pitch:.3f}"
        if pitch_format == "name":
            pitch_txt = librosa.midi_to_note(pitch, unicode=False, cents=not round_pitch)
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
            for onset, offset, pitch in zip(onset_list, offset_list, pitch_list):
                writer.writerow({"onset": onset, "offset": offset, "pitch": pitch})
    print(f"Saved {file_format.upper()} file: {filepath}")


@click.group()
def main():
    pass


def shared_options(func=None, *, defaults: dict[str, Any] = None):
    if defaults is None:
        defaults = {}
    
    options = [
        click.option(
            "-m", "--model", type=click.Path(
                exists=True, dir_okay=True, file_okay=False, readable=True, path_type=pathlib.Path
            ),
            required=True,
            help="Path to the ONNX model directory."
        ),
        click.option(
            "-d", "--device", type=click.Choice(["dml", "cpu"]), default="dml", show_default=True,
            help="Execution provider: 'dml' for DirectML (GPU), 'cpu' for CPU."
        ),
        click.option(
            "-l", "--language", type=str, default=None, show_default=False,
            help="Language code for better segmentation if supported."
        ),
        click.option(
            "--batch-size", type=click.IntRange(min=1), show_default=True,
            default=defaults.get(_OPT_KEY_BATCH_SIZE, 4),
            help="Batch size for inference (currently unused in pure ONNX mode)."
        ),
        click.option(
            "--num-workers", type=click.IntRange(min=0), show_default=True,
            default=defaults.get(_OPT_KEY_NUM_WORKERS, 0),
            help="Number of worker processes (currently unused in pure ONNX mode)."
        ),
        click.option(
            "--seg-threshold", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_THRESHOLD, 0.2),
            help="Boundary decoding threshold for segmentation model."
        ),
        click.option(
            "--seg-radius", type=click.FloatRange(min=0.01), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_RADIUS, 0.02),
            help="Boundary decoding radius for segmentation model."
        ),
        click.option(
            "--t0", "--seg-d3pm-t0", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_D3PM_T0, 0.0),
            help="Starting T value (t0) of D3PM for segmentation model."
        ),
        click.option(
            "--nsteps", "--seg-d3pm-nsteps", type=click.IntRange(min=1), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_D3PM_NSTEPS, 8),
            help="Number of D3PM sampling steps for segmentation model."
        ),
        click.option(
            "--ts", "--seg-d3pm-ts", type=str, default=None, show_default=False,
            callback=_validate_d3pm_ts,
            help=(
                "Custom T values for D3PM sampling in segmentation model, separated by commas. "
                "Overrides --t0 and --nsteps if provided."
            )
        ),
        click.option(
            "--est-threshold", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), show_default=True,
            default=defaults.get(_OPT_KEY_EST_THRESHOLD, 0.2),
            help="Presence detecting threshold for estimation model."
        ),
    ]
    
    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f
    
    if func is None:
        return decorator
    else:
        return decorator(func)


@main.command(
    name="extract",
    help="Extract MIDI from single or multiple audio files."
)
@click.argument(
    "path", type=click.Path(
        exists=True, dir_okay=True, file_okay=True, readable=True, path_type=pathlib.Path
    ),
)
@shared_options(defaults={
    _OPT_KEY_SEG_D3PM_T0: 0.0,
    _OPT_KEY_SEG_D3PM_NSTEPS: 8,
})
@click.option(
    "--input-formats", type=str, default="wav,flac,mp3,aac,ogg", show_default=True,
    callback=_validate_exts,
    help=(
        "List of audio file extensions to process, separated by commas. "
        "Ignored if a single audio file is provided."
    )
)
@click.option(
    "--glob", "glb", type=str, default=None, show_default=False,
    help=(
        "Glob pattern to filter audio files (i.e. *.wav). "
        "Overrides --exts if provided. Ignored if a single audio file is provided."
    )
)
@click.option(
    "--output-formats", type=str, default="mid", show_default=True,
    callback=_validate_output_formats,
    help=(
        "List of output formats to save the extracted results, separated by commas. "
        "Supported formats: mid, txt, csv."
    )
)
@click.option(
    "--tempo", type=click.FloatRange(min=0, min_open=True), default=120, show_default=True,
    help="Tempo (in BPM) to save MIDI files with. Ignored for non-MIDI output formats."
)
@click.option(
    "--pitch-format", type=click.Choice(["number", "name"]), default="name", show_default=True,
    help=(
        "Format to save pitch in text-based output formats. "
        "Ignored for MIDI output. 'number' saves MIDI pitch numbers, while 'name' saves note names (e.g. C4+10)."
    )
)
@click.option(
    "--round-pitch", is_flag=True, default=False, show_default=True,
    help=(
        "Whether to round pitch values to the nearest integer in text-based output formats. "
        "Ignored for MIDI output. If not set, pitch values will be saved with decimals or cents."
    )
)
@click.option(
    "--output-dir", type=click.Path(
        exists=False, dir_okay=True, file_okay=False, writable=True, path_type=pathlib.Path
    ),
    default=None, show_default=False,
    help=(
        "Directory to save the extracted results. "
        "If not provided, results will be saved in the same directory as the input."
    )
)
def extract(
    path: pathlib.Path,
    model: pathlib.Path,
    device: str,
    language: str,
    batch_size: int,
    num_workers: int,
    seg_threshold: float,
    seg_radius: float,
    t0: float,
    nsteps: int,
    ts: list[float],
    est_threshold: float,
    input_formats: set[str],
    glb: str,
    output_formats: set[str],
    pitch_format: str,
    round_pitch: bool,
    tempo: float,
    output_dir: pathlib.Path,
):
    if ts is None:
        ts = _t0_nstep_to_ts(t0, nsteps)
    filemap = _parse_filemap(path, input_formats, glb)
    if output_dir is None:
        output_dir = path if path.is_dir() else path.parent
    
    # Load model
    use_dml = (device == "dml")
    print(f"Loading ONNX model from {model}...")
    inference_model = ONNXInferenceModel(model, use_dml=use_dml)
    language_id = _get_language_id(language, inference_model.languages)
    
    # Initialize slicer
    slicer = Slicer(
        sr=inference_model.samplerate,
        threshold=-40.,
        min_length=1000,
        min_interval=200,
        max_sil_kept=100,
    )
    
    boundary_radius_frames = round(seg_radius / inference_model.timestep)
    
    # Process each file
    for key, filepath in filemap.items():
        print(f"\nProcessing: {key}")
        
        # Load audio
        waveform, _ = librosa.load(filepath, sr=inference_model.samplerate, mono=True)
        
        # Slice audio
        chunks = slicer.slice(waveform)
        print(f"  Sliced into {len(chunks)} chunks")
        
        # Process each chunk
        all_notes = []
        for chunk in chunks:
            chunk_wav = chunk["waveform"]
            chunk_offset = chunk["offset"]
            chunk_duration = len(chunk_wav) / inference_model.samplerate
            
            # Run inference
            durations, presence, scores = inference_model.infer(
                waveform=chunk_wav,
                known_durations=None, # For extract, we have no known durations
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=language_id,
                ts=ts,
            )
            
            # Convert to notes with absolute timing
            note_onset = np.concatenate([[0], np.cumsum(durations[:-1])])
            note_offset = np.cumsum(durations)
            
            for onset, offset, score, valid in zip(note_onset, note_offset, scores, presence):
                if offset - onset <= 0:
                    continue
                if not valid:
                    continue
                all_notes.append(NoteInfo(
                    onset=onset + chunk_offset,
                    offset=offset + chunk_offset,
                    pitch=score,
                ))
        
        # Sort and clean notes
        all_notes.sort(key=lambda x: (x.onset, x.offset, x.pitch))
        last_time = 0
        i = 0
        while i < len(all_notes):
            note = all_notes[i]
            note.onset = max(note.onset, last_time)
            note.offset = max(note.offset, note.onset)
            if note.offset <= note.onset:
                all_notes.pop(i)
            else:
                last_time = note.offset
                i += 1
        
        print(f"  Extracted {len(all_notes)} notes")
        
        # Save results
        output_key = pathlib.Path(key).stem
        if "mid" in output_formats:
            save_midi(all_notes, output_dir / f"{output_key}.mid", tempo=int(tempo))
        if "txt" in output_formats:
            save_text(all_notes, output_dir / f"{output_key}.txt", "txt", pitch_format, round_pitch)
        if "csv" in output_formats:
            save_text(all_notes, output_dir / f"{output_key}.csv", "csv", pitch_format, round_pitch)
    
    print("\nInference completed.")


@main.command(
    name="align",
    help="Generate aligned note labels with word boundaries in DiffSinger transcriptions."
)
@click.argument(
    "paths", type=str, nargs=-1, metavar="PATH_OR_GLOB",
    callback=_validate_path_or_glob,
)
@shared_options(defaults={
    _OPT_KEY_SEG_D3PM_T0: 0.5,
    _OPT_KEY_SEG_D3PM_NSTEPS: 4,
})
@click.option(
    "--save-path", type=click.Path(
        file_okay=True, dir_okay=False, writable=True, path_type=pathlib.Path
    ),
    help=(
        "Path of the output file to save the updated transcriptions. "
        "Ignored if multiple input files are provided. Overrides --save-name if provided."
    )
)
@click.option(
    "--save-name", type=str, default=None, show_default=False,
    help=(
        "Name of the updated transcription files to save. "
        "Ignored if input is a single file and --save-path is provided. "
        "If both --save-path and --save-name are not provided, the original files will be overwritten."
    )
)
@click.option(
    "--overwrite", is_flag=True, default=False, show_default=True,
    help="Whether to overwrite existing files when saving updated transcriptions."
)
@click.option(
    "--no-wb", is_flag=True, default=False, show_default=True,
    help=(
        "Whether to disable word boundaries for better alignment. "
        "If set, 'ph_num' field will not be checked and used. Not recommended."
    )
)
def align(
    paths: list[pathlib.Path],
    model: pathlib.Path,
    device: str,
    language: str,
    batch_size: int,
    num_workers: int,
    seg_threshold: float,
    seg_radius: float,
    t0: float,
    nsteps: int,
    ts: list[float],
    est_threshold: float,
    save_path: pathlib.Path,
    save_name: str,
    overwrite: bool,
    no_wb: bool,
):
    if ts is None:
        ts = _t0_nstep_to_ts(t0, nsteps)
    if len(paths) > 1:
        save_path = None
    if save_path is not None:
        save_name = save_path.name
    inplace = (save_path is None) and (save_name is None)
    if inplace and not overwrite:
        raise ValueError("In-place saving requires --overwrite flag to be set.")
    if not overwrite:
        for p in paths:
            if save_path is not None and save_path.exists():
                raise FileExistsError(f"Output file already exists: {save_path}")
            if save_name is not None:
                output_file = p.parent / save_name
                if output_file.exists():
                    raise FileExistsError(f"Output file already exists: {output_file}")
    save_dir = save_path.parent if save_path else None
    use_wb = not no_wb
    
    # Load model
    use_dml = (device == "dml")
    print(f"Loading ONNX model from {model}...")
    inference_model = ONNXInferenceModel(model, use_dml=use_dml)
    language_id = _get_language_id(language, inference_model.languages)
    
    boundary_radius_frames = round(seg_radius / inference_model.timestep)
    extensions = [".wav", ".flac"]
    
    # Process each transcription file
    for index_path in paths:
        print(f"\nProcessing: {index_path}")
        
        # Load transcription
        with open(index_path, "r", encoding="utf8") as f:
            items = list(csv.DictReader(f))
        
        updated_items = OrderedDict()
        for item in items:
            name = item["name"]
            
            # Find waveform file
            candidate_wav_fns = [
                index_path.parent / "wavs" / f"{name}{ext}"
                for ext in extensions
            ]
            wav_fn = None
            for fn in candidate_wav_fns:
                if fn.is_file():
                    wav_fn = fn
                    break
            if wav_fn is None:
                raise FileNotFoundError(
                    f"Waveform file not found for item '{name}' in index '{index_path}'. "
                    f"Tried candidates: {[fn.as_posix() for fn in candidate_wav_fns]}"
                )
            
            # Parse durations
            ph_dur = [float(d) for d in item["ph_dur"].split()]
            if use_wb:
                ph_num = [int(n) for n in item["ph_num"].split()]
                if sum(ph_num) != len(ph_dur):
                    raise ValueError(
                        f"Length mismatch in item '{name}' in index '{index_path}': "
                        f"sum(ph_num) = {sum(ph_num)}, len(ph_dur) = {len(ph_dur)}."
                    )
                word_dur = []
                idx = 0
                for num in ph_num:
                    word_dur.append(sum(ph_dur[idx: idx + num]))
                    idx += num
            else:
                word_dur = [sum(ph_dur)]
            
            # Load audio
            waveform, _ = librosa.load(wav_fn, sr=inference_model.samplerate, mono=True)
            
            # Run inference
            durations, presence, scores = inference_model.infer(
                waveform=waveform,
                known_durations=np.array([word_dur]),
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=language_id,
                ts=ts,
            )
            
            # Filter valid notes
            valid = durations > 0
            durations = durations[valid]
            scores = scores[valid]
            presence = presence[valid]
            
            # Convert to note sequence
            note_seq = [
                librosa.midi_to_note(score, unicode=False, cents=True) if pres else "rest"
                for score, pres in zip(scores, presence)
            ]
            note_dur = [f"{dur:.3f}" for dur in durations]
            
            # Update item
            item["note_seq"] = " ".join(note_seq)
            item["note_dur"] = " ".join(note_dur)
            item.pop("note_glide", None)
            
            updated_items[name] = item
            print(f"  Processed: {name}")
        
        # Save updated transcription
        if inplace:
            output_path = index_path
        elif save_dir is not None:
            output_path = save_dir / save_name
        else:
            output_path = index_path.parent / save_name
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(encoding="utf8", mode="w", newline="") as f:
            fieldnames = list(items[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_items.values())
        
        print(f"Saved transcriptions: {output_path}")
    
    print("\nInference completed.")


if __name__ == '__main__':
    main()
