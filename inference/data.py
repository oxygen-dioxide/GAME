import pathlib
from typing import Any

import librosa
import torch.utils.data

from inference.slicer2 import Slicer
from training.data import collate_nd


class SlicedAudioFileIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
            self, filemap: dict[str, pathlib.Path],
            samplerate: int,
            slicer: Slicer,
            language: int = 0,
    ):
        assert samplerate == slicer.sr, "Samplerate mismatches between dataset and slicer!"
        self.filemap = filemap
        self.samplerate = samplerate
        self.slicer = slicer
        self.language = language

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        items = list(self.filemap.items())
        if worker is None:
            start, step = 0, 1
        else:
            start = worker.id
            step = worker.num_workers
        for key, filepath in items[start::step]:
            waveform, _ = librosa.load(filepath, sr=self.samplerate, mono=True)
            chunks = self.slicer.slice(waveform)
            num_parts = len(chunks)
            for chunk in chunks:
                chunk_wav = torch.from_numpy(chunk["waveform"]).float()
                yield {
                    "key": key,
                    "offset": chunk["offset"],
                    "num_parts": num_parts,
                    "waveform": chunk_wav,
                    "samplerate": self.samplerate,
                    "duration": chunk_wav.shape[0] / self.samplerate,
                    "language": self.language,
                }

    @classmethod
    def collate(cls, samples: list[dict[str, Any]]) -> dict[str, Any]:
        sr = {s["samplerate"] for s in samples}
        if len(sr) > 1:
            raise ValueError(f"Multiple samplerate values found in batch: {sr}")
        samplerate = sr.pop()
        batch = {
            "size": len(samples),
            "samplerate": samplerate,
            "key": [s["key"] for s in samples],
            "offset": [s["offset"] for s in samples],
            "length": [s["duration"] for s in samples],
            "num_parts": [s["num_parts"] for s in samples],
            "waveform": collate_nd(
                [s["waveform"] for s in samples], pad_value=0.,
            ),
            "known_durations": torch.FloatTensor([[s["duration"]] for s in samples]),
            "language": torch.LongTensor([s["language"] for s in samples]),
        }
        return batch
