import csv
import pathlib
from dataclasses import dataclass

import dask
import numpy

from lib import logging
from .binarizer_base import BaseBinarizer, MetadataItem, DataSample, ACCEPTED_AUDIO_FORMATS


SYLLABLES_ITEM_ATTRIBUTES = [
    "language_id",  # int64
    "frame2syllable",  # int64 [T,]
    "spectrogram",  # float32 [T, C]
]


@dataclass
class SyllablesMetadataItem(MetadataItem):
    syllable_durations: list[float]


class SyllablesBinarizer(BaseBinarizer):
    __data_attrs__ = SYLLABLES_ITEM_ATTRIBUTES

    def load_metadata(self, subset_dir: pathlib.Path) -> list[MetadataItem]:
        with open(subset_dir / "index.csv", "r", encoding="utf8") as f:
            items = list(csv.DictReader(f))
        metadata_items = []
        waveform_fn = None
        for item in items:
            item_name = item["name"]
            language = item.get("language")
            for ext in ACCEPTED_AUDIO_FORMATS:
                searched_wav_fn = subset_dir / "waveforms" / f"{item_name}{ext}"
                if searched_wav_fn.exists():
                    waveform_fn = searched_wav_fn
                    break
            if waveform_fn is None:
                logging.error(
                    f"Waveform file missing in raw dataset \'{subset_dir.as_posix()}\':\n"
                    f"item {item_name}, searched extensions: {', '.join(ACCEPTED_AUDIO_FORMATS)}."
                )
            syllable_durations = [
                float(dur) for dur in item["syllables"].split()
            ]
            estimated_duration = sum(syllable_durations)
            metadata_items.append(SyllablesMetadataItem(
                item_name=item_name,
                language=language,
                waveform_fn=waveform_fn,
                estimated_duration=estimated_duration,
                syllable_durations=syllable_durations,
            ))
        return metadata_items

    def process_item(self, item: SyllablesMetadataItem) -> DataSample:
        language_id = numpy.array(self.lang_map.get(item.language, 0), dtype=numpy.int64)
        waveform = self.load_waveform(item.waveform_fn)
        spectrogram, length = self.get_mel(waveform)
        syllable_dur_sec = numpy.array(item.syllable_durations, dtype=numpy.float32)
        syllable_dur_frames = self.sec_dur_to_frame_dur(syllable_dur_sec, length)
        frame2syllable = self.length_regulator(syllable_dur_frames)
        data = {
            "language_id": language_id,
            "frame2syllable": frame2syllable,
            "spectrogram": spectrogram,
        }
        data, length = dask.compute(data, length)
        return DataSample(
            path=item.waveform_fn.relative_to(self.data_dir).as_posix(),
            name=item.item_name,
            length=length,
            data=data,
        )
