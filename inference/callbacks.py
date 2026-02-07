import abc
import csv
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Literal

import librosa
import lightning.pytorch.callbacks
import mido
import torch
import torch.nn.functional as F

from lib import logging

__all__ = [
    "SaveFileCallback",
    "SaveMidiCallback",
    "SaveTextCallback",
]


@dataclass
class _NoteInfo:
    onset: float
    offset: float
    pitch: float


class SaveFileCallback(lightning.pytorch.callbacks.Callback, abc.ABC):
    def __init__(self, output_dir: str | pathlib.Path):
        super().__init__()
        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)
        self.output_dir = output_dir
        self.counters: dict[str, int] = {}
        self.notes: dict[str, list[_NoteInfo]] = {}

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, Any],
            *args, **kwargs
    ) -> None:
        batch_size = batch["size"]
        for i in range(batch_size):
            key: str = batch["key"][i]
            num_parts = batch["num_parts"][i]
            if key not in self.counters:
                self.counters[key] = 0
                self.notes[key] = []
            offset: float = batch["offset"][i]
            length: float = batch["length"][i]
            durations = outputs["durations"][i]
            scores = outputs["scores"][i]
            presence = outputs["presence"][i]
            note_onset = F.pad(
                durations, (1, 0), mode="constant", value=0
            ).cumsum(dim=0).clamp(max=length).add(offset)
            note_offset = durations.cumsum(dim=0).clamp(max=length).add(offset)
            for onset, offset, score, valid in zip(
                    note_onset.tolist(),
                    note_offset.tolist(),
                    scores.tolist(),
                    presence.tolist(),
            ):
                if offset - onset <= 0:
                    continue
                if not valid:
                    continue
                self.notes[key].append(_NoteInfo(
                    onset=onset,
                    offset=offset,
                    pitch=score,
                ))
            self.counters[key] += 1
            if self.counters[key] >= num_parts:
                self.save_file(key, logger_fn=trainer.progress_bar_callback.print)

    def on_predict_epoch_end(self, trainer: lightning.pytorch.Trainer, *args, **kwargs) -> None:
        for key in list(self.counters.keys()):
            self.save_file(key, logger_fn=trainer.progress_bar_callback.print)

    def save_file(self, key: str, logger_fn: Callable) -> None:
        sorted_notes = sorted(self.notes[key], key=lambda x: (x.onset, x.offset, x.pitch))
        last_time = 0
        i = 0
        while i < len(sorted_notes):
            note = sorted_notes[i]
            note.onset = max(note.onset, last_time)
            note.offset = max(note.offset, note.onset)
            if note.offset <= note.onset:
                sorted_notes.pop(i)
            else:
                last_time = note.offset
                i += 1
        self.flush(key, sorted_notes, logger_fn)
        del self.counters[key]
        del self.notes[key]

    @abc.abstractmethod
    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        pass


class SaveMidiCallback(SaveFileCallback):
    def __init__(
            self, output_dir: str | pathlib.Path,
            tempo: int = 120,
    ):
        super().__init__(output_dir)
        self.tempo = tempo

    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(self.tempo), time=0))
        last_time = 0
        for note in notes:
            onset_ticks = round(note.onset * self.tempo * 8)
            offset_ticks = round(note.offset * self.tempo * 8)
            midi_pitch = round(note.pitch)
            if offset_ticks <= onset_ticks:
                continue
            track.append(mido.Message(
                "note_on", note=midi_pitch, time=onset_ticks - last_time
            ))
            track.append(mido.Message(
                "note_off", note=midi_pitch, time=offset_ticks - onset_ticks
            ))
            last_time = offset_ticks

        filepath = (self.output_dir / key).with_suffix(".mid")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with mido.MidiFile(charset="utf8") as midi_file:
            midi_file.tracks.append(track)
            midi_file.save(filepath)

        logging.info(f"Saved MIDI file: {filepath.as_posix()}", callback=logger_fn)


class SaveTextCallback(SaveFileCallback):
    def __init__(
            self, output_dir: str | pathlib.Path,
            file_format: Literal["txt", "csv"] = "csv",
            pitch_format: Literal["number", "name"] = "name",
            round_pitch: bool = False,
    ):
        super().__init__(output_dir)
        self.file_format = file_format
        self.pitch_format = pitch_format
        self.round_pitch = round_pitch

    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        onset_list = [
            f"{note.onset:.3f}" for note in notes
        ]
        offset_list = [
            f"{note.offset:.3f}" for note in notes
        ]
        pitch_list = []
        for note in notes:
            pitch = note.pitch
            if self.round_pitch:
                pitch = round(pitch)
                pitch_txt = str(pitch)
            else:
                pitch_txt = f"{pitch:.3f}"
            if self.pitch_format == "name":
                pitch_txt = librosa.midi_to_note(pitch, unicode=False, cents=not self.round_pitch)
            pitch_list.append(pitch_txt)

        if self.file_format == "txt":
            filepath = (self.output_dir / key).with_suffix(".txt")
            with filepath.open(encoding="utf8", mode="w") as f:
                for onset, offset, pitch in zip(onset_list, offset_list, pitch_list):
                    f.write(f"{onset}\t{offset}\t{pitch}\n")
            logging.info(f"Saved text file: {filepath.as_posix()}", callback=logger_fn)
        elif self.file_format == "csv":
            filepath = (self.output_dir / key).with_suffix(".csv")
            with filepath.open(encoding="utf8", mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["onset", "offset", "pitch"])
                writer.writeheader()
                for onset, offset, pitch in zip(onset_list, offset_list, pitch_list):
                    writer.writerow({
                        "onset": onset,
                        "offset": offset,
                        "pitch": pitch,
                    })
            logging.info(f"Saved CSV file: {filepath.as_posix()}", callback=logger_fn)
