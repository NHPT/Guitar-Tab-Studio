from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 22050
    hop_length: int = 256
    bins_per_octave: int = 36
    n_bins: int = 216
    fmin_hz: float = 82.4068892282175
    minimum_db: float = -80.0

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FeatureConfig":
        return cls(
            sample_rate=int(value.get("sample_rate", 22050)),
            hop_length=int(value.get("hop_length", 256)),
            bins_per_octave=int(value.get("bins_per_octave", 36)),
            n_bins=int(value.get("n_bins", 216)),
            fmin_hz=float(value.get("fmin_hz", 82.4068892282175)),
            minimum_db=float(value.get("minimum_db", -80.0)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length


def load_audio_segment(
    path: Path,
    config: FeatureConfig,
    *,
    offset: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    audio, _sample_rate = librosa.load(
        path,
        sr=config.sample_rate,
        mono=True,
        offset=max(0.0, offset),
        duration=duration,
    )
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1:
        audio = audio / peak
    return audio.astype(np.float32)


def extract_cqt(
    audio: np.ndarray,
    config: FeatureConfig,
) -> np.ndarray:
    minimum_samples = max(config.hop_length * 4, 2048)
    if audio.size < minimum_samples:
        audio = np.pad(audio, (0, minimum_samples - audio.size))
    spectrum = np.abs(
        librosa.cqt(
            audio,
            sr=config.sample_rate,
            hop_length=config.hop_length,
            fmin=config.fmin_hz,
            n_bins=config.n_bins,
            bins_per_octave=config.bins_per_octave,
        )
    )
    decibels = librosa.amplitude_to_db(spectrum, ref=np.max)
    clipped = np.clip(decibels, config.minimum_db, 0.0)
    normalized = (clipped - config.minimum_db) / abs(config.minimum_db)
    return normalized.T.astype(np.float32)


def extract_file_cqt(
    path: Path,
    config: FeatureConfig,
    *,
    offset: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    return extract_cqt(
        load_audio_segment(path, config, offset=offset, duration=duration),
        config,
    )
