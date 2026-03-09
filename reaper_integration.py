import reapy
import librosa
import infer_onnx
import pathlib
from inference.slicer2 import Slicer
from typing import Any, Callable, Literal
from lib.config.schema import ValidationConfig
import click
import numpy as np
from inference.onnx_api import (
    NoteInfo,
    load_onnx_model,
    infer_from_files,
    align_with_transcriptions,
    enforce_max_chunk_size,
    extract_batch_generator,
)

max_chunk_duration_s = 15.0

# Mapping from original input Reaper track index to output reapy.Track object for saving MIDI output to correct track
outputTrackMap: dict[int, reapy.Track] = {}
partmap = {}

class ReaperItemInfo():
    """Extracts the audio path, the start and end times of the clip in use from the audio file,
        track index, and position in reaper project (in seconds) from a Reaper item."""
    def __init__(self, item: reapy.Item):
        take = item.takes[-1]
        self.audioPath = pathlib.Path(take.source.filename)
        self.start: float = take.start_offset
        self.end: float = item.length + self.start
        self.track_index: int = item.track.index
        self.position: float = item.position


@click.command()
@infer_onnx.shared_options
def main(
        model: pathlib.Path,
        device: str,
        language: str | None = None,
        batch_size: int = 2,
        seg_threshold: float = 0.2,
        seg_radius: float = 0.02,
        t0: float = 0,
        nsteps: int = 8,
        ts: str | None = None,
        est_threshold: float = 0.2,
        **kwargs
    ):
    project = reapy.Project()
    ts = ts or infer_onnx._t0_nstep_to_ts(t0, nsteps)
    model = load_onnx_model(model, device)
    language_id = infer_onnx._get_language_id(language, model.languages)

    sr = model.samplerate
    slicer = Slicer(
        sr=model.samplerate,
        threshold=-40.,
        min_length=1000,
        min_interval=200,
        max_sil_kept=100,
    )
    boundary_radius_frames = round(seg_radius / model.timestep)
    
    for item in project.selected_items:
        itemInfo = ReaperItemInfo(item)
        print(f"Processing item on track {itemInfo.track_index} from {itemInfo.start}s to {itemInfo.end}s in file {itemInfo.audioPath}...")
        if(not itemInfo.audioPath.is_file()):
            print(f"  Skipping item because audio file {itemInfo.audioPath} does not exist.")
            continue
        waveform, _ = librosa.load(itemInfo.audioPath, sr=sr, mono=True)
        #clip the waveform to the part of the audio file corresponding to the item
        waveform = waveform[int(itemInfo.start*sr):int(itemInfo.end*sr)]
        initial_chunks = slicer.slice(waveform)
        # Enforce max chunk size to prevent VRAM OOM
        chunks = enforce_max_chunk_size(initial_chunks, max_chunk_duration_s, model.samplerate)
        print(f"  Sliced into {len(chunks)} chunks, batch size: {batch_size}")

        all_notes = []
        for batch_wavs, batch_durations, batch_chunks in extract_batch_generator(chunks, batch_size, model.samplerate):
            
            # Run inference on batch
            batch_results = model.infer_batch(
                waveforms=batch_wavs,
                durations=batch_durations,
                known_durations=None,
                boundary_threshold=seg_threshold,
                boundary_radius=boundary_radius_frames,
                score_threshold=est_threshold,
                language=language_id,
                ts=ts,
            )
            # Process results
            for chunk_result, chunk_info in zip(batch_results, batch_chunks):
                durations, presence, scores = chunk_result
                chunk_offset = chunk_info['offset']
                
                note_onset = np.concatenate([[0], np.cumsum(durations[:-1])]) + chunk_offset
                note_offset = np.cumsum(durations) + chunk_offset
                
                for onset, offset, score, is_present in zip(note_onset, note_offset, scores, presence):
                    if offset - onset > 0 and is_present:
                        all_notes.append(NoteInfo(onset=onset, offset=offset, pitch=score))

        all_notes.sort(key=lambda x: x.onset)
        print(f"  Extracted {len(all_notes)} notes")

        track_index = itemInfo.track_index
        if track_index not in outputTrackMap:
            # create new track for output if it doesn't exist already
            outputTrackMap[track_index] = project.add_track(len(project.tracks))
        outputTrack = outputTrackMap[track_index]
        outputPart = outputTrack.add_midi_item(
            start = itemInfo.position, 
            end = itemInfo.end - itemInfo.start + itemInfo.position)
        outputTake = outputPart.takes[0]
        for note in all_notes:
            outputTake.add_note(
                start = float(note.onset),
                end = float(note.offset),
                pitch = round(note.pitch),
                sort = False
            )
        outputTake.sort_events()
        
        

if __name__ == '__main__':
    main()