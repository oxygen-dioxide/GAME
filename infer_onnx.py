import glob
import pathlib
from typing import Any
import json
import numpy as np
import torch

import click

from lib import logging
from lib.config.schema import ValidationConfig

_OPT_KEY_BATCH_SIZE = "batch_size"
_OPT_KEY_NUM_WORKERS = "num_workers"
_OPT_KEY_SEG_THRESHOLD = "seg_threshold"
_OPT_KEY_SEG_RADIUS = "seg_radius"
_OPT_KEY_SEG_D3PM_T0 = "t0"
_OPT_KEY_SEG_D3PM_NSTEPS = "nsteps"
_OPT_KEY_EST_THRESHOLD = "est_threshold"


# noinspection PyUnusedLocal
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


# noinspection PyUnusedLocal
def _validate_exts(ctx, param, value) -> set[str]:
    try:
        exts = {"." + ext.strip().lower() for ext in value.split(",")}
        if not exts:
            raise ValueError("At least one extension must be provided.")
        return exts
    except Exception as e:
        raise click.BadParameter(f"Invalid extensions: {e}")


# noinspection PyUnusedLocal
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


# noinspection PyUnusedLocal
def _validate_path_or_glob(ctx, param, value) -> str:
    try:
        paths = []
        for v in value:
            if glob.has_magic(v):
                paths.extend(glob.glob(v, recursive=True))
            else:
                paths.append(v)
        if not paths:
            raise FileNotFoundError(f"No files found for paths: {value}")
        paths = [
            pathlib.Path(p)
            for p in paths
        ]
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
    return [
        t0 + i * step
        for i in range(nsteps)
    ]


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
        filemap = {
            f.relative_to(path).as_posix(): f
            for f in files
        }
        if not filemap:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return filemap
    else:
        raise ValueError(f"Invalid path: {path}")


class ONNXInferenceModel(torch.nn.Module):
    def __init__(self, model_dir: pathlib.Path):
        super().__init__()
        import onnxruntime as ort
        
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            self.onnx_config = json.load(f)
            
        class Features:
            def __init__(self, sr):
                self.audio_sample_rate = sr
        class InfConfig:
            def __init__(self, sr):
                self.features = Features(sr)
            
        self.inference_config = InfConfig(self.onnx_config["samplerate"])
        self.timestep = self.onnx_config["timestep"]
        self.loop = self.onnx_config.get("loop", True)
        self.languages = self.onnx_config.get("languages", None)
        
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.encoder = ort.InferenceSession((model_dir / "encoder.onnx").as_posix(), providers=providers)
        self.segmenter = ort.InferenceSession((model_dir / "segmenter.onnx").as_posix(), providers=providers)
        self.estimator = ort.InferenceSession((model_dir / "estimator.onnx").as_posix(), providers=providers)
        self.dur2bd = ort.InferenceSession((model_dir / "dur2bd.onnx").as_posix(), providers=providers)
        self.bd2dur = ort.InferenceSession((model_dir / "bd2dur.onnx").as_posix(), providers=providers)

    def forward(
            self, waveform: torch.Tensor,
            known_durations: torch.Tensor,
            boundary_threshold: torch.Tensor,
            boundary_radius: torch.Tensor,
            score_threshold: torch.Tensor,
            language: torch.Tensor = None,
            t: torch.Tensor = None,
    ):
        device = waveform.device
        
        # Static shapes constants based on exporter.py
        L_STATIC = 441000  # 10 seconds at 44100 Hz
        T_STATIC = 1000    # 10 seconds / 0.01 timestep
        N_STATIC = 200     # Fixed max notes per chunk
        
        waveform_np = waveform.detach().cpu().numpy()
        known_durations_np = known_durations.detach().cpu().numpy()
        boundary_threshold_np = boundary_threshold.detach().cpu().numpy()
        boundary_radius_np = boundary_radius.detach().cpu().numpy()
        score_threshold_np = score_threshold.detach().cpu().numpy()
        
        if language is not None:
            language_np = language.detach().cpu().numpy()
        else:
            language_np = np.zeros(waveform_np.shape[0], dtype=np.int64)
            
        if t is not None:
            t_np = t.detach().cpu().numpy()
        else:
            t_np = np.array([], dtype=np.float32)

        B = waveform_np.shape[0]
        L = waveform_np.shape[1]
        
        all_durations = []
        all_presence = []
        all_scores = []
        
        # We must chunk the input to L_STATIC and batch size 1
        for b in range(B):
            b_durations = []
            b_presence = []
            b_scores = []
            
            w = waveform_np[b]
            durs = known_durations_np[b]
            
            num_chunks = int(np.ceil(L / L_STATIC))
            if num_chunks == 0:
                num_chunks = 1
                
            for c in range(num_chunks):
                start_idx = c * L_STATIC
                end_idx = min((c + 1) * L_STATIC, L)
                
                chunk_w = w[start_idx:end_idx]
                chunk_len = len(chunk_w)
                
                # Pad waveform chunk
                padded_w = np.zeros((1, L_STATIC), dtype=np.float32)
                padded_w[0, :chunk_len] = chunk_w
                
                chunk_duration_sec = chunk_len / self.inference_config.features.audio_sample_rate
                
                # Assuming known_durations just has [duration] for inference
                padded_known_durs = np.zeros((1, N_STATIC), dtype=np.float32)
                padded_known_durs[0, 0] = chunk_duration_sec
                
                enc_out = self.encoder.run(None, {
                    "waveform": padded_w,
                    "duration": np.array([chunk_duration_sec], dtype=np.float32)
                })
                x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
                
                dur2bd_out = self.dur2bd.run(None, {
                    "durations": padded_known_durs
                })
                known_boundaries = dur2bd_out[0]
                
                boundaries = known_boundaries
                seg_inputs = [i.name for i in self.segmenter.get_inputs()]
                
                if self.loop and len(t_np) > 0:
                    for ti in t_np:
                        seg_args = {
                            "x_seg": x_seg,
                            "maskT": maskT,
                            "threshold": boundary_threshold_np.astype(np.float32),
                            "radius": boundary_radius_np.astype(np.int64)
                        }
                        if "language" in seg_inputs:
                            seg_args["language"] = np.array([language_np[b]], dtype=np.int64)
                        if "known_boundaries" in seg_inputs:
                            seg_args["known_boundaries"] = known_boundaries
                        if "prev_boundaries" in seg_inputs:
                            seg_args["prev_boundaries"] = boundaries
                        if "t" in seg_inputs:
                            seg_args["t"] = np.array(ti, dtype=np.float32)
                        
                        seg_out = self.segmenter.run(None, seg_args)
                        boundaries = seg_out[0]
                else:
                    seg_args = {
                        "x_seg": x_seg,
                        "maskT": maskT,
                        "threshold": boundary_threshold_np.astype(np.float32),
                        "radius": boundary_radius_np.astype(np.int64)
                    }
                    if "language" in seg_inputs:
                        seg_args["language"] = np.array([language_np[b]], dtype=np.int64)
                    if "known_boundaries" in seg_inputs:
                        seg_args["known_boundaries"] = known_boundaries
                    if "prev_boundaries" in seg_inputs:
                        seg_args["prev_boundaries"] = boundaries
                    if "t" in seg_inputs:
                        seg_args["t"] = np.array(0.0, dtype=np.float32)
                        
                    seg_out = self.segmenter.run(None, seg_args)
                    boundaries = seg_out[0]
                    
                bd2dur_out = self.bd2dur.run(None, {
                    "boundaries": boundaries,
                    "maskT": maskT
                })
                durations, maskN = bd2dur_out[0], bd2dur_out[1]
                
                # In bd2dur, maskN and durations output shapes might be dynamic based on the actual number of notes.
                # However, for estimator we MUST pass maskN as [1, 200]
                actual_n = maskN.shape[1]
                if actual_n < N_STATIC:
                    pad_maskN = np.zeros((1, N_STATIC), dtype=bool)
                    pad_maskN[0, :actual_n] = maskN[0]
                    padded_maskN = pad_maskN
                else:
                    padded_maskN = maskN[:, :N_STATIC]
                    
                if actual_n < N_STATIC:
                    pad_durs = np.zeros((1, N_STATIC), dtype=np.float32)
                    pad_durs[0, :actual_n] = durations[0]
                    padded_durations = pad_durs
                else:
                    padded_durations = durations[:, :N_STATIC]
                
                est_inputs = [i.name for i in self.estimator.get_inputs()]
                est_args = {
                    "x_est": x_est,
                    "boundaries": boundaries,
                    "maskT": maskT,
                    "maskN": padded_maskN,
                    "threshold": score_threshold_np.astype(np.float32)
                }
                est_args = {k: v for k, v in est_args.items() if k in est_inputs}
                
                est_out = self.estimator.run(None, est_args)
                presence, scores = est_out[0], est_out[1]
                
                # Extract valid notes using maskN
                valid_notes = padded_maskN[0].astype(bool)
                b_durations.extend(padded_durations[0][valid_notes].tolist())
                b_presence.extend(presence[0][valid_notes].tolist())
                b_scores.extend(scores[0][valid_notes].tolist())
                
            all_durations.append(b_durations)
            all_presence.append(b_presence)
            all_scores.append(b_scores)
            
        # Pad back to batch max N to return tensors
        max_n = max([len(d) for d in all_durations] + [1])
        
        out_dur = np.zeros((B, max_n), dtype=np.float32)
        out_pres = np.zeros((B, max_n), dtype=bool)
        out_score = np.zeros((B, max_n), dtype=np.float32)
        
        for b in range(B):
            n = len(all_durations[b])
            if n > 0:
                out_dur[b, :n] = all_durations[b]
                out_pres[b, :n] = all_presence[b]
                out_score[b, :n] = all_scores[b]
                
        return torch.from_numpy(out_dur).to(device), torch.from_numpy(out_pres).to(device), torch.from_numpy(out_score).to(device)


def load_onnx_inference_model(path: pathlib.Path):
    from lightning_utilities.core.rank_zero import rank_zero_info
    model = ONNXInferenceModel(path)
    model.eval()
    logging.info(f"Loaded ONNX model from '{path}'.", callback=rank_zero_info)
    return model, model.languages


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
            "-l", "--language", type=str, default=None, show_default=False,
            help="Language code for better segmentation if supported."
        ),
        click.option(
            "--batch-size", type=click.IntRange(min=1), show_default=True,
            default=defaults.get(_OPT_KEY_BATCH_SIZE, 4),
            help="Batch size for inference."
        ),
        click.option(
            "--num-workers", type=click.IntRange(min=0), show_default=True,
            default=defaults.get(_OPT_KEY_NUM_WORKERS, 0),
            help="Number of worker processes for dataloader."
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

    from lightning_utilities.core.rank_zero import rank_zero_info
    from inference.api import infer_model
    from inference.slicer2 import Slicer
    from inference.data import SlicedAudioFileIterableDataset
    from inference.callbacks import SaveCombinedMidiFileCallback, SaveCombinedTextFileCallback

    model, lang_map = load_onnx_inference_model(model)
    language_id = _get_language_id(language, lang_map)

    sr = model.inference_config.features.audio_sample_rate
    dataset = SlicedAudioFileIterableDataset(
        filemap=filemap,
        samplerate=sr,
        slicer=Slicer(
            sr=sr,
            threshold=-40.,
            min_length=1000,
            min_interval=200,
            max_sil_kept=100,
        ),
        language=language_id,
    )
    callbacks = []
    if "mid" in output_formats:
        callbacks.append(SaveCombinedMidiFileCallback(
            output_dir=output_dir,
            tempo=tempo,
        ))
    if "txt" in output_formats:
        callbacks.append(SaveCombinedTextFileCallback(
            output_dir=output_dir,
            file_format="txt",
            pitch_format=pitch_format,
            round_pitch=round_pitch,
        ))
    if "csv" in output_formats:
        callbacks.append(SaveCombinedTextFileCallback(
            output_dir=output_dir,
            file_format="csv",
            pitch_format=pitch_format,
            round_pitch=round_pitch,
        ))
    # noinspection PyArgumentList
    infer_model(
        model=model,
        dataset=dataset,
        config=ValidationConfig(
            d3pm_sample_ts=ts,
            boundary_decoding_threshold=seg_threshold,
            boundary_decoding_radius=round(seg_radius / model.timestep),
            note_presence_threshold=est_threshold,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        callbacks=callbacks,
    )
    logging.success("Inference completed.", callback=rank_zero_info)


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
    help=(
        "Whether to overwrite existing files when saving updated transcriptions."
    )
)
@click.option(
    "--no-wb", is_flag=True, default=False, show_default=True,
    help=(
            "Whether to disable word boundaries for better alignment. "
            "If set, \'ph_num\' field will not be checked and used. Not recommended."
    )
)
def align(
        paths: list[pathlib.Path],
        model: pathlib.Path,
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

    from lightning_utilities.core.rank_zero import rank_zero_info
    from inference.api import infer_model
    from inference.data import DiffSingerTranscriptionsDataset
    from inference.callbacks import UpdateDiffSingerTranscriptionsCallback

    model, lang_map = load_onnx_inference_model(model)
    language_id = _get_language_id(language, lang_map)

    sr = model.inference_config.features.audio_sample_rate
    dataset = DiffSingerTranscriptionsDataset(
        filelist=paths,
        samplerate=sr,
        language=language_id,
        use_wb=use_wb,
    )
    callbacks = [
        UpdateDiffSingerTranscriptionsCallback(
            filelist=paths,
            overwrite=inplace,
            save_dir=save_dir,
            save_filename=save_name,
        )
    ]
    # noinspection PyArgumentList
    infer_model(
        model=model,
        dataset=dataset,
        config=ValidationConfig(
            d3pm_sample_ts=ts,
            boundary_decoding_threshold=seg_threshold,
            boundary_decoding_radius=round(seg_radius / model.timestep),
            note_presence_threshold=est_threshold,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        callbacks=callbacks,
    )
    logging.success("Inference completed.", callback=rank_zero_info)


if __name__ == '__main__':
    main()
