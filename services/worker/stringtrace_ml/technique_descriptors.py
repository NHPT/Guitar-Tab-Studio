from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .technique_schema import TECHNIQUE_LABELS, load_technique_manifest


@dataclass(frozen=True)
class TechniqueDescriptorConfig:
    sample_rate: int = 22050
    hop_length: int = 256
    fft_size: int = 1024
    mel_bands: int = 64
    mfcc_count: int = 20
    window_seconds: float = 0.40

    @property
    def frame_count(self) -> int:
        return round(
            self.window_seconds * self.sample_rate / self.hop_length
        )

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> "TechniqueDescriptorConfig":
        return cls(
            sample_rate=int(value.get("sample_rate", 22050)),
            hop_length=int(value.get("hop_length", 256)),
            fft_size=int(value.get("fft_size", 1024)),
            mel_bands=int(value.get("mel_bands", 64)),
            mfcc_count=int(value.get("mfcc_count", 20)),
            window_seconds=float(value.get("window_seconds", 0.40)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_track_descriptors(
    audio_path: Path,
    config: TechniqueDescriptorConfig,
) -> np.ndarray:
    audio, _sample_rate = librosa.load(
        audio_path,
        sr=config.sample_rate,
        mono=True,
    )
    magnitude = np.abs(
        librosa.stft(
            audio,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
        )
    )
    power = np.square(magnitude)
    mel_power = librosa.feature.melspectrogram(
        S=power,
        sr=config.sample_rate,
        n_mels=config.mel_bands,
        fmin=50,
        fmax=config.sample_rate / 2,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    mfcc = librosa.feature.mfcc(
        S=mel_db,
        n_mfcc=config.mfcc_count,
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mel_summary = mel_db.reshape(
        config.mel_bands // 2,
        2,
        mel_db.shape[1],
    ).mean(axis=1)
    chroma = librosa.feature.chroma_stft(
        S=power,
        sr=config.sample_rate,
    )
    spectral = np.concatenate(
        (
            librosa.feature.spectral_centroid(
                S=magnitude,
                sr=config.sample_rate,
            )
            / (config.sample_rate / 2),
            librosa.feature.spectral_bandwidth(
                S=magnitude,
                sr=config.sample_rate,
            )
            / (config.sample_rate / 2),
            librosa.feature.spectral_rolloff(
                S=magnitude,
                sr=config.sample_rate,
                roll_percent=0.85,
            )
            / (config.sample_rate / 2),
            librosa.feature.spectral_flatness(S=power),
            librosa.feature.rms(
                S=magnitude,
                frame_length=config.fft_size,
            ),
            librosa.feature.zero_crossing_rate(
                audio,
                frame_length=config.fft_size,
                hop_length=config.hop_length,
            ),
        ),
        axis=0,
    )
    frame_count = min(
        mfcc.shape[1],
        mfcc_delta.shape[1],
        mel_summary.shape[1],
        chroma.shape[1],
        spectral.shape[1],
    )
    return np.concatenate(
        (
            mfcc[:, :frame_count],
            mfcc_delta[:, :frame_count],
            mel_summary[:, :frame_count],
            chroma[:, :frame_count],
            spectral[:, :frame_count],
        ),
        axis=0,
    ).T.astype(np.float32)


def summarize_event_descriptors(
    descriptors: np.ndarray,
    *,
    onset: float,
    config: TechniqueDescriptorConfig,
) -> np.ndarray:
    start = max(
        0,
        round(onset * config.sample_rate / config.hop_length),
    )
    end = min(descriptors.shape[0], start + config.frame_count)
    patch = np.zeros(
        (config.frame_count, descriptors.shape[1]),
        dtype=np.float32,
    )
    if start < end:
        patch[:end - start] = descriptors[start:end]
    return np.concatenate(
        (patch.mean(axis=0), patch.std(axis=0))
    ).astype(np.float32)


class TechniqueDescriptorDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        config: TechniqueDescriptorConfig | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.tracks = load_technique_manifest(self.manifest_path, split)
        self.config = config or TechniqueDescriptorConfig()
        self.cache_directory = (
            cache_directory.resolve()
            if cache_directory is not None
            else None
        )
        self.events = [
            (track_index, event_index)
            for track_index, track in enumerate(self.tracks)
            for event_index, _event in enumerate(track.events)
        ]
        if self.cache_directory is not None:
            self.cache_directory.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.events)

    def _cache_path(self, track_index: int) -> Path:
        if self.cache_directory is None:
            raise RuntimeError("Technique descriptor cache is not configured")
        track = self.tracks[track_index]
        audio_path = track.resolve_audio(self.manifest_path)
        stat = audio_path.stat()
        payload = {
            "format_version": 1,
            "audio": track.audio,
            "audio_size": stat.st_size,
            "audio_mtime_ns": stat.st_mtime_ns,
            "config": self.config.to_dict(),
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.cache_directory / f"{key}.npy"

    def _track_descriptors(self, track_index: int) -> np.ndarray:
        track = self.tracks[track_index]
        if self.cache_directory is None:
            return extract_track_descriptors(
                track.resolve_audio(self.manifest_path),
                self.config,
            )
        path = self._cache_path(track_index)
        if path.is_file():
            return np.load(path, allow_pickle=False, mmap_mode="r")
        descriptors = extract_track_descriptors(
            track.resolve_audio(self.manifest_path),
            self.config,
        )
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, descriptors, allow_pickle=False)
        os.replace(temporary, path)
        return np.load(path, allow_pickle=False, mmap_mode="r")

    def prepare_feature_cache(self) -> dict[str, int]:
        created = 0
        for track_index in range(len(self.tracks)):
            path = (
                self._cache_path(track_index)
                if self.cache_directory is not None
                else None
            )
            existed = path is not None and path.is_file()
            self._track_descriptors(track_index)
            created += not existed
        return {"tracks": len(self.tracks), "tracks_created": created}

    def balanced_sample_weights(
        self,
        *,
        exponent: float = 1.0,
        balance_dataset: bool = False,
    ) -> np.ndarray:
        if not 0 <= exponent <= 1:
            raise ValueError("balance exponent must be between 0 and 1")
        counts: dict[tuple[str, str], int] = {}
        for track_index, event_index in self.events:
            track = self.tracks[track_index]
            dataset = (
                str(track.provenance.get("dataset", "unknown"))
                if balance_dataset
                else "all"
            )
            technique = track.events[event_index].technique
            counts[(dataset, technique)] = (
                counts.get((dataset, technique), 0) + 1
            )
        weights = np.asarray(
            [
                counts[
                    (
                        str(
                            self.tracks[track_index].provenance.get(
                                "dataset",
                                "unknown",
                            )
                        )
                        if balance_dataset
                        else "all",
                        self.tracks[track_index]
                        .events[event_index]
                        .technique,
                    )
                ]
                ** -exponent
                for track_index, event_index in self.events
            ],
            dtype=np.float64,
        )
        return weights / weights.mean()

    def dataset_names(self) -> np.ndarray:
        return np.asarray(
            [
                str(
                    self.tracks[track_index].provenance.get(
                        "dataset",
                        "unknown",
                    )
                )
                for track_index, _event_index in self.events
            ]
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        track_index, event_index = self.events[index]
        event = self.tracks[track_index].events[event_index]
        summary = summarize_event_descriptors(
            self._track_descriptors(track_index),
            onset=event.onset,
            config=self.config,
        )
        return {
            "summary": torch.from_numpy(summary),
            "label": torch.tensor(
                TECHNIQUE_LABELS.index(event.technique),
                dtype=torch.long,
            ),
        }
