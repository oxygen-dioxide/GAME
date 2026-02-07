import pathlib
from dataclasses import dataclass
from typing import Any, Callable

import lightning.pytorch.callbacks
import mido
import torch
import torch.nn.functional as F
from lightning.fabric.utilities import rank_zero_only

from lib import logging

__all__ = [
    "SaveMidiCallback",
]


@dataclass
class _MidiNote:
    onset: int
    offset: int
    pitch: int


class SaveMidiCallback(lightning.pytorch.callbacks.Callback):
    def __init__(
            self, output_dir: str | pathlib.Path,
            tempo: int = 120,
    ):
        super().__init__()
        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)
        self.output_dir = output_dir
        self.tempo = tempo
        self.counters: dict[str, int] = {}
        self.notes: dict[str, list[_MidiNote]] = {}

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
            scores = outputs["scores"][i].round().long()
            presence = outputs["presence"][i]
            note_onset = F.pad(
                durations, (1, 0), mode="constant", value=0
            ).cumsum(dim=0).clamp(max=length).add(offset).mul(self.tempo * 8).round().long()
            note_offset = durations.cumsum(dim=0).clamp(max=length).add(offset).mul(self.tempo * 8).round().long()
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
                self.notes[key].append(_MidiNote(
                    onset=onset,
                    offset=offset,
                    pitch=score,
                ))
            self.counters[key] += 1
            if self.counters[key] >= num_parts:
                self.flush(key, logger_fn=trainer.progress_bar_callback.print)

    def on_predict_epoch_end(self, *args, **kwargs) -> None:
        for key in list(self.counters.keys()):
            self.flush(key, logger_fn=print)

    def flush(self, key: str, logger_fn: Callable) -> None:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(self.tempo), time=0))
        last_time = 0
        sorted_notes = sorted(self.notes[key], key=lambda x: (x.onset, x.offset, x.pitch))
        for note in sorted_notes:
            onset = max(note.onset, last_time)
            offset = max(note.offset, onset)
            if offset - onset <= 0:
                continue
            track.append(mido.Message(
                "note_on", note=note.pitch, time=onset - last_time
            ))
            track.append(mido.Message(
                "note_off", note=note.pitch, time=offset - onset
            ))
            last_time = offset

        filepath = (self.output_dir / key).with_suffix(".mid")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with mido.MidiFile(charset="utf8") as midi_file:
            midi_file.tracks.append(track)
            midi_file.save(filepath)
        del self.counters[key]
        del self.notes[key]

        @rank_zero_only
        def log(s):
            logging.info(s, callback=logger_fn)

        log("Saved MIDI file: " + filepath.as_posix())
