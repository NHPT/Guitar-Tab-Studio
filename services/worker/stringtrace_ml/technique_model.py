from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .technique_schema import TECHNIQUE_LABELS, load_technique_manifest


TECHNIQUE_DYNAMICS_SIZE = 8


@dataclass(frozen=True)
class TechniqueFeatureConfig:
    sample_rate: int = 22050
    hop_length: int = 256
    bins_per_octave: int = 36
    n_bins: int = 216
    fmin_hz: float = 82.4068892282175
    relative_bins: int = 144
    below_pitch_bins: int = 9
    pre_onset_seconds: float = 0.10
    post_onset_seconds: float = 1.50
    truncate_at_event_end: bool = True

    @property
    def frame_count(self) -> int:
        return round(
            (self.pre_onset_seconds + self.post_onset_seconds)
            * self.sample_rate
            / self.hop_length
        )

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> "TechniqueFeatureConfig":
        return cls(
            sample_rate=int(value.get("sample_rate", 22050)),
            hop_length=int(value.get("hop_length", 256)),
            bins_per_octave=int(value.get("bins_per_octave", 36)),
            n_bins=int(value.get("n_bins", 216)),
            fmin_hz=float(value.get("fmin_hz", 82.4068892282175)),
            relative_bins=int(value.get("relative_bins", 144)),
            below_pitch_bins=int(value.get("below_pitch_bins", 9)),
            pre_onset_seconds=float(value.get("pre_onset_seconds", 0.10)),
            post_onset_seconds=float(value.get("post_onset_seconds", 1.50)),
            truncate_at_event_end=bool(
                value.get("truncate_at_event_end", False)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TechniqueModelConfig:
    channels: tuple[int, int, int] = (16, 32, 64)
    hidden_size: int = 96
    dropout: float = 0.35
    architecture: str = "global-summary-dynamics-v1"

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TechniqueModelConfig":
        return cls(
            channels=tuple(
                int(channel)
                for channel in value.get("channels", (16, 32, 64))
            ),
            hidden_size=int(value.get("hidden_size", 96)),
            dropout=float(value.get("dropout", 0.35)),
            architecture=str(
                value.get("architecture", "global-summary-dynamics-v1")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["channels"] = list(self.channels)
        return result


def extract_log_cqt(
    audio_path: Path,
    config: TechniqueFeatureConfig,
) -> np.ndarray:
    audio, _sample_rate = librosa.load(
        audio_path,
        sr=config.sample_rate,
        mono=True,
    )
    spectrum = np.abs(
        librosa.cqt(
            audio,
            sr=config.sample_rate,
            hop_length=config.hop_length,
            bins_per_octave=config.bins_per_octave,
            n_bins=config.n_bins,
            fmin=config.fmin_hz,
        )
    )
    decibels = librosa.amplitude_to_db(
        spectrum,
        ref=np.max,
    )
    return np.clip(
        (decibels + 80.0) / 80.0,
        0,
        1,
    ).T.astype(np.float32)


def extract_technique_dynamics(
    patch: np.ndarray,
    config: TechniqueFeatureConfig,
) -> np.ndarray:
    if patch.shape != (config.frame_count, config.relative_bins):
        raise ValueError(
            "patch must have shape "
            f"[{config.frame_count}, {config.relative_bins}]"
        )

    amplitude = np.power(
        10.0,
        (np.asarray(patch, dtype=np.float32) * 80.0 - 80.0) / 20.0,
    )
    amplitude[patch <= 0] = 0
    fundamental_bin = config.below_pitch_bins

    def band_peak(
        source: np.ndarray,
        center: int,
        radius: int = 3,
    ) -> np.ndarray:
        start = max(0, center - radius)
        end = min(config.relative_bins, center + radius + 1)
        if start >= end:
            return np.zeros(config.frame_count, dtype=np.float32)
        return np.max(source[:, start:end], axis=1)

    fundamental_db = band_peak(patch, fundamental_bin)
    octave_db = band_peak(
        patch,
        fundamental_bin + config.bins_per_octave,
    )
    fundamental_amplitude = np.maximum(
        1e-4,
        band_peak(amplitude, fundamental_bin).astype(np.float32),
    )
    octave_amplitude = band_peak(
        amplitude,
        fundamental_bin + config.bins_per_octave
    ).astype(np.float32)
    harmonic_ratio = np.clip(
        octave_amplitude / fundamental_amplitude,
        0,
        4,
    ) / 4

    search_radius = round(config.bins_per_octave * 5 / 12)
    search_start = max(0, fundamental_bin - search_radius)
    search_end = min(
        config.relative_bins,
        fundamental_bin + search_radius + 1,
    )
    local_amplitude = amplitude[:, search_start:search_end]
    local_weights = np.square(local_amplitude)
    local_bins = np.arange(
        search_start,
        search_end,
        dtype=np.float32,
    )
    pitch_bins = (
        (local_weights * local_bins[None, :]).sum(axis=1)
        / np.maximum(1e-8, local_weights.sum(axis=1))
    )
    pitch_offset = (
        (pitch_bins - fundamental_bin)
        * 12
        / config.bins_per_octave
    )
    quiet = local_weights.sum(axis=1) < 1e-7
    pitch_offset[quiet] = 0
    pitch_delta = np.diff(pitch_offset, prepend=pitch_offset[:1])

    high_start = min(
        config.relative_bins - 1,
        fundamental_bin + config.bins_per_octave,
    )
    high_energy = amplitude[:, high_start:].mean(axis=1)
    total_energy = amplitude.mean(axis=1)
    high_ratio = np.clip(
        high_energy / np.maximum(1e-5, total_energy),
        0,
        4,
    ) / 4

    spectral_flux = np.maximum(
        0,
        np.diff(amplitude, axis=0, prepend=amplitude[:1]),
    ).mean(axis=1)
    envelope_delta = np.diff(
        fundamental_db,
        prepend=fundamental_db[:1],
    )
    return np.stack(
        (
            fundamental_db,
            octave_db,
            harmonic_ratio,
            high_ratio,
            np.clip(pitch_offset / 5, -1, 1),
            np.clip(pitch_delta / 2, -1, 1),
            np.clip(spectral_flux * 12, 0, 1),
            np.clip(envelope_delta * 4, -1, 1),
        ),
        axis=1,
    ).astype(np.float32)


def extract_event_patch(
    features: np.ndarray,
    *,
    onset: float,
    offset: float | None,
    pitch: int,
    config: TechniqueFeatureConfig,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    frames_per_second = config.sample_rate / config.hop_length
    start = round((onset - config.pre_onset_seconds) * frames_per_second)
    if augment:
        start += random.randint(-2, 2)
    source_start = max(0, start)
    source_end = min(features.shape[0], start + config.frame_count)
    patch = np.zeros(
        (config.frame_count, config.relative_bins),
        dtype=np.float32,
    )
    destination_start = max(0, -start)
    destination_end = destination_start + max(0, source_end - source_start)
    pitch_bin = round((pitch - 40) * config.bins_per_octave / 12)
    frequency_start = pitch_bin - config.below_pitch_bins
    frequency_end = frequency_start + config.relative_bins
    available_start = max(0, frequency_start)
    available_end = min(features.shape[1], frequency_end)
    frequency_destination_start = max(0, -frequency_start)
    frequency_destination_end = (
        frequency_destination_start
        + max(0, available_end - available_start)
    )
    patch[
        destination_start:destination_end,
        frequency_destination_start:frequency_destination_end,
    ] = features[
        source_start:source_end,
        available_start:available_end,
    ]
    event_end_frame = (
        round(
            (
                offset
                + 0.05
                - onset
                + config.pre_onset_seconds
            )
            * frames_per_second
        )
        if offset is not None and config.truncate_at_event_end
        else None
    )
    if augment:
        spectral_tilt = np.linspace(
            random.uniform(-0.10, 0.10),
            random.uniform(-0.10, 0.10),
            config.relative_bins,
            dtype=np.float32,
        )
        patch = np.clip(
            patch + spectral_tilt[None, :],
            0,
            1,
        )
        patch = np.clip(
            patch + np.random.normal(0, 0.015, patch.shape),
            0,
            1,
        ).astype(np.float32)
    if event_end_frame is not None:
        patch[max(0, event_end_frame):] = 0
    dynamics = extract_technique_dynamics(patch, config)
    patch = (
        (patch - float(patch.mean()))
        / max(1e-5, float(patch.std()))
    ).astype(np.float32)
    return patch, dynamics


def summarize_technique_features(
    patch: np.ndarray,
    dynamics: np.ndarray,
) -> np.ndarray:
    if patch.ndim != 2:
        raise ValueError("patch must have shape [frames, bins]")
    if dynamics.shape != (patch.shape[0], TECHNIQUE_DYNAMICS_SIZE):
        raise ValueError("dynamics must have shape [frames, 8]")
    return np.concatenate(
        (
            patch.mean(axis=0),
            patch.std(axis=0),
            patch.max(axis=0),
            patch.mean(axis=1),
            patch.std(axis=1),
            patch.max(axis=1),
            dynamics.mean(axis=0),
            dynamics.std(axis=0),
            dynamics.max(axis=0),
            dynamics.min(axis=0),
        )
    ).astype(np.float32)


class TechniqueEventDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        feature_config: TechniqueFeatureConfig | None = None,
        cache_directory: Path | None = None,
        augment: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.tracks = load_technique_manifest(self.manifest_path, split)
        self.feature_config = feature_config or TechniqueFeatureConfig()
        self.cache_directory = (
            cache_directory.resolve()
            if cache_directory is not None
            else None
        )
        self.augment = augment
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
            raise RuntimeError("Technique feature cache is not configured")
        track = self.tracks[track_index]
        audio_path = track.resolve_audio(self.manifest_path)
        stat = audio_path.stat()
        payload = {
            "format_version": 3,
            "audio": track.audio,
            "audio_size": stat.st_size,
            "audio_mtime_ns": stat.st_mtime_ns,
            "spectral_config": {
                "sample_rate": self.feature_config.sample_rate,
                "hop_length": self.feature_config.hop_length,
                "bins_per_octave": self.feature_config.bins_per_octave,
                "n_bins": self.feature_config.n_bins,
                "fmin_hz": self.feature_config.fmin_hz,
            },
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.cache_directory / f"{key}.npy"

    def _track_features(self, track_index: int) -> np.ndarray:
        track = self.tracks[track_index]
        if self.cache_directory is None:
            return extract_log_cqt(
                track.resolve_audio(self.manifest_path),
                self.feature_config,
            )
        path = self._cache_path(track_index)
        if path.is_file():
            return np.load(path, allow_pickle=False, mmap_mode="r")
        features = extract_log_cqt(
            track.resolve_audio(self.manifest_path),
            self.feature_config,
        )
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, features.astype(np.float16), allow_pickle=False)
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
            self._track_features(track_index)
            created += not existed
        return {"tracks": len(self.tracks), "tracks_created": created}

    def class_counts(self) -> list[int]:
        counts = [0 for _label in TECHNIQUE_LABELS]
        for track_index, event_index in self.events:
            technique = self.tracks[track_index].events[event_index].technique
            counts[TECHNIQUE_LABELS.index(technique)] += 1
        return counts

    def balanced_sample_weights(
        self,
        *,
        exponent: float = 1.0,
        balance_dataset: bool = False,
    ) -> list[float]:
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
            key = (dataset, track.events[event_index].technique)
            counts[key] = counts.get(key, 0) + 1
        return [
            max(
                1,
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
                ],
            )
            ** -exponent
            for track_index, event_index in self.events
        ]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        track_index, event_index = self.events[index]
        track = self.tracks[track_index]
        event = track.events[event_index]
        features = self._track_features(track_index)
        next_onset = next(
            (
                candidate.onset
                for candidate in track.events[event_index + 1:]
                if candidate.onset > event.onset + 0.05
            ),
            None,
        )
        patch, dynamics = extract_event_patch(
            features,
            onset=event.onset,
            offset=min(event.offset, next_onset)
            if next_onset is not None
            else event.offset,
            pitch=event.pitch,
            config=self.feature_config,
            augment=self.augment,
        )
        return {
            "features": torch.from_numpy(patch),
            "dynamics": torch.from_numpy(dynamics),
            "summary": torch.from_numpy(
                summarize_technique_features(patch, dynamics)
            ),
            "label": torch.tensor(
                TECHNIQUE_LABELS.index(event.technique),
                dtype=torch.long,
            ),
            "domain": str(track.provenance.get("dataset", "unknown")),
        }


class TechniqueNet(nn.Module):
    def __init__(
        self,
        feature_config: TechniqueFeatureConfig | None = None,
        model_config: TechniqueModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.feature_config = feature_config or TechniqueFeatureConfig()
        self.model_config = model_config or TechniqueModelConfig()
        if self.model_config.architecture not in {
            "global-average-v4",
            "global-summary-dynamics-v1",
        }:
            raise ValueError(
                "Unsupported technique model architecture: "
                f"{self.model_config.architecture}"
            )
        channels = (1, *self.model_config.channels)
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            layers.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GroupNorm(
                        num_groups=min(8, output_channels),
                        num_channels=output_channels,
                    ),
                    nn.SiLU(),
                    nn.MaxPool2d(2),
                )
            )
        self.encoder = nn.Sequential(*layers)
        if self.model_config.architecture == "global-average-v4":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(self.model_config.dropout),
                nn.Linear(
                    self.model_config.channels[-1],
                    self.model_config.hidden_size,
                ),
                nn.SiLU(),
                nn.Dropout(self.model_config.dropout),
                nn.Linear(
                    self.model_config.hidden_size,
                    len(TECHNIQUE_LABELS),
                ),
            )
            return
        pooled_frames = self.feature_config.frame_count // (
            2 ** len(self.model_config.channels)
        )
        pooled_bins = self.feature_config.relative_bins // (
            2 ** len(self.model_config.channels)
        )
        self.spatial_average = nn.AvgPool2d(
            kernel_size=(pooled_frames, pooled_bins),
        )
        self.spatial_maximum = nn.MaxPool2d(
            kernel_size=(pooled_frames, pooled_bins),
        )
        summary_size = (
            self.model_config.channels[-1] * 2
            + TECHNIQUE_DYNAMICS_SIZE * 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.model_config.dropout),
            nn.Linear(
                summary_size,
                self.model_config.hidden_size,
            ),
            nn.LayerNorm(self.model_config.hidden_size),
            nn.SiLU(),
            nn.Dropout(self.model_config.dropout),
            nn.Linear(
                self.model_config.hidden_size,
                len(TECHNIQUE_LABELS),
            ),
        )

    def forward(
        self,
        features: Tensor,
        dynamics: Tensor | None = None,
    ) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, frames, bins]")
        if dynamics is None:
            dynamics = features.new_zeros(
                features.shape[0],
                features.shape[1],
                TECHNIQUE_DYNAMICS_SIZE,
            )
        if dynamics.shape != (
            features.shape[0],
            features.shape[1],
            TECHNIQUE_DYNAMICS_SIZE,
        ):
            raise ValueError(
                "dynamics must have shape [batch, frames, 8]"
            )
        encoded = self.encoder(features.unsqueeze(1))
        if self.model_config.architecture == "global-average-v4":
            return self.classifier(encoded)
        spectral_summary = torch.cat(
            (
                self.spatial_average(encoded).flatten(start_dim=1),
                self.spatial_maximum(encoded).flatten(start_dim=1),
            ),
            dim=-1,
        )
        dynamic_summary = torch.cat(
            (
                dynamics.mean(dim=1),
                dynamics.std(dim=1),
                dynamics.amax(dim=1),
                dynamics.amin(dim=1),
            ),
            dim=-1,
        )
        return self.classifier(
            torch.cat((spectral_summary, dynamic_summary), dim=-1)
        )
