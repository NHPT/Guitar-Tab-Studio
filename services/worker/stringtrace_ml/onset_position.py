from __future__ import annotations

import itertools
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import Dataset

from .dataset import GuitarTabDataset
from .features import FeatureConfig
from .model import (
    FRET_STATE_COUNT,
    OPEN_STRING_PITCHES,
    PITCH_COUNT,
    PITCH_MIN,
    STRING_COUNT,
)


@dataclass(frozen=True)
class OnsetPositionConfig:
    frames_before: int = 2
    frames_after: int = 12
    group_tolerance: float = 0.045
    convolution_channels: tuple[int, int, int] = (32, 64, 96)
    hidden_size: int = 256
    dropout: float = 0.15
    use_candidate_confidence: bool = False
    adapter_size: int = 0
    adapter_scale: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "OnsetPositionConfig":
        channels = value.get("convolution_channels", [32, 64, 96])
        return cls(
            frames_before=int(value.get("frames_before", 2)),
            frames_after=int(value.get("frames_after", 12)),
            group_tolerance=float(value.get("group_tolerance", 0.045)),
            convolution_channels=tuple(int(item) for item in channels),
            hidden_size=int(value.get("hidden_size", 256)),
            dropout=float(value.get("dropout", 0.15)),
            use_candidate_confidence=bool(
                value.get("use_candidate_confidence", False)
            ),
            adapter_size=int(value.get("adapter_size", 0)),
            adapter_scale=float(value.get("adapter_scale", 1.0)),
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["convolution_channels"] = list(self.convolution_channels)
        return result

    @property
    def frame_count(self) -> int:
        return self.frames_before + self.frames_after


@dataclass(frozen=True)
class OnsetGroupIndex:
    annotation_index: int
    event_indexes: tuple[int, ...]
    onset: float | None = None
    input_pitches: tuple[int, ...] | None = None
    input_confidences: tuple[float, ...] | None = None


def basic_pitch_cache_path(
    audio_path: Path,
    cache_directory: Path,
    *,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length: float,
) -> Path:
    stat = audio_path.stat()
    identity = json.dumps(
        {
            "version": 1,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "name": audio_path.name,
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
            "minimum_note_length": minimum_note_length,
        },
        sort_keys=True,
    )
    return cache_directory / (
        hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json"
    )


def group_event_indexes(
    onsets: list[float],
    strings: list[int] | None = None,
    *,
    tolerance: float,
) -> list[tuple[int, ...]]:
    groups: list[list[int]] = []
    for event_index in sorted(range(len(onsets)), key=onsets.__getitem__):
        if not groups:
            groups.append([event_index])
            continue
        current = groups[-1]
        duplicate_string = (
            strings is not None
            and strings[event_index] in {strings[index] for index in current}
        )
        if (
            onsets[event_index] - onsets[current[0]] > tolerance
            or duplicate_string
        ):
            groups.append([event_index])
        else:
            current.append(event_index)
    return [tuple(group) for group in groups]


def extract_onset_patch(
    features: np.ndarray,
    *,
    onset: float,
    feature_config: FeatureConfig,
    config: OnsetPositionConfig,
) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != feature_config.n_bins:
        raise ValueError("features must have shape [frames, frequency_bins]")
    center = round(onset * feature_config.frames_per_second)
    indexes = np.clip(
        np.arange(
            center - config.frames_before,
            center + config.frames_after,
        ),
        0,
        max(0, features.shape[0] - 1),
    )
    patch = np.asarray(features[indexes], dtype=np.float32)
    previous_indexes = np.maximum(0, indexes - 1)
    previous = np.asarray(features[previous_indexes], dtype=np.float32)
    positive_flux = np.maximum(0.0, patch - previous)
    signed_delta = patch - previous
    return np.stack((patch, positive_flux, signed_delta), axis=0)


def pitch_mask(
    pitches: list[int],
    confidences: list[float] | None = None,
) -> np.ndarray:
    if confidences is not None and len(confidences) != len(pitches):
        raise ValueError("pitch confidences must align with pitches")
    result = np.zeros(PITCH_COUNT, dtype=np.float32)
    values = confidences or [1.0] * len(pitches)
    for pitch, confidence in zip(pitches, values):
        if PITCH_MIN <= pitch < PITCH_MIN + PITCH_COUNT:
            result[pitch - PITCH_MIN] = max(
                result[pitch - PITCH_MIN],
                max(0.0, min(1.0, confidence)),
            )
    return result


def augment_pitch_candidates(
    input_pitches: list[int],
    input_confidences: list[float],
    reference_pitches: list[int],
    *,
    drop_probability: float,
    add_probability: float,
) -> tuple[list[int], list[float]]:
    if len(input_pitches) != len(input_confidences):
        raise ValueError("pitch confidences must align with pitches")
    if not 0 <= drop_probability <= 1:
        raise ValueError("pitch drop probability must be in 0..1")
    if not 0 <= add_probability <= 1:
        raise ValueError("pitch add probability must be in 0..1")

    retained = (
        list(zip(input_pitches, input_confidences))
        if drop_probability == 0
        else [
            (pitch, confidence)
            for pitch, confidence in zip(input_pitches, input_confidences)
            if random.random() >= drop_probability
        ]
    )
    if not retained and reference_pitches:
        retained.append((random.choice(reference_pitches), 1.0))

    source_pitches = reference_pitches or [pitch for pitch, _confidence in retained]
    if source_pitches and random.random() < add_probability:
        source_pitch = random.choice(source_pitches)
        added_pitch = source_pitch + random.choice((-12, -2, -1, 1, 2, 12))
        if PITCH_MIN <= added_pitch < PITCH_MIN + PITCH_COUNT:
            retained.append((added_pitch, 1.0))

    return (
        [pitch for pitch, _confidence in retained],
        [confidence for _pitch, confidence in retained],
    )


class OnsetPositionDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        feature_config: FeatureConfig | None = None,
        config: OnsetPositionConfig | None = None,
        feature_cache_directory: Path | None = None,
        onset_jitter_seconds: float = 0.0,
        pitch_drop_probability: float = 0.0,
        pitch_add_probability: float = 0.0,
        candidate_cache_directory: Path | None = None,
        candidate_onset_threshold: float = 0.5,
        candidate_frame_threshold: float = 0.3,
        candidate_minimum_note_length: float = 127.7,
        candidate_match_tolerance: float = 0.05,
        positive_only_datasets: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.feature_config = feature_config or FeatureConfig()
        self.config = config or OnsetPositionConfig()
        if onset_jitter_seconds < 0:
            raise ValueError("onset jitter must be non-negative")
        if not 0 <= pitch_drop_probability <= 1:
            raise ValueError("pitch drop probability must be in 0..1")
        if not 0 <= pitch_add_probability <= 1:
            raise ValueError("pitch add probability must be in 0..1")
        self.onset_jitter_seconds = onset_jitter_seconds
        self.pitch_drop_probability = pitch_drop_probability
        self.pitch_add_probability = pitch_add_probability
        self.positive_only_datasets = frozenset(
            positive_only_datasets or ()
        )
        self.filtered_negative_groups = 0
        self.candidate_cache_directory = (
            candidate_cache_directory.resolve()
            if candidate_cache_directory is not None
            else None
        )
        self.track_dataset = GuitarTabDataset(
            self.manifest_path,
            split=split,
            feature_config=self.feature_config,
            feature_cache_directory=feature_cache_directory,
        )
        self.annotations = self.track_dataset.annotations
        self.groups: list[OnsetGroupIndex] = []
        for annotation_index, annotation in enumerate(self.annotations):
            if self.candidate_cache_directory is None:
                grouped_indexes = group_event_indexes(
                    [event.onset for event in annotation.events],
                    [event.string for event in annotation.events],
                    tolerance=self.config.group_tolerance,
                )
                self.groups.extend(
                    OnsetGroupIndex(annotation_index, indexes)
                    for indexes in grouped_indexes
                )
                continue

            audio_path = annotation.resolve_audio(self.manifest_path)
            candidate_path = basic_pitch_cache_path(
                audio_path,
                self.candidate_cache_directory,
                onset_threshold=candidate_onset_threshold,
                frame_threshold=candidate_frame_threshold,
                minimum_note_length=candidate_minimum_note_length,
            )
            if not candidate_path.is_file():
                raise FileNotFoundError(
                    f"Missing Basic Pitch candidate cache: {candidate_path.name}"
                )
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidates = list(payload.get("events", []))
            candidate_groups = group_event_indexes(
                [float(event["onset"]) for event in candidates],
                tolerance=self.config.group_tolerance,
            )
            references_by_group: list[list[int]] = [
                [] for _group in candidate_groups
            ]
            reference_groups = group_event_indexes(
                [event.onset for event in annotation.events],
                [event.string for event in annotation.events],
                tolerance=self.config.group_tolerance,
            )
            for reference_indexes in reference_groups:
                reference_onset = annotation.events[
                    reference_indexes[0]
                ].onset
                matches = [
                    (
                        group_index,
                        abs(
                            float(candidates[indexes[0]]["onset"])
                            - reference_onset
                        ),
                    )
                    for group_index, indexes in enumerate(candidate_groups)
                    if abs(
                        float(candidates[indexes[0]]["onset"])
                        - reference_onset
                    )
                    <= candidate_match_tolerance
                ]
                if matches:
                    closest_group = min(matches, key=lambda item: item[1])[0]
                    references_by_group[closest_group].extend(
                        reference_indexes
                    )
            for indexes, target_indexes in zip(
                candidate_groups,
                references_by_group,
            ):
                candidate_onset = float(candidates[indexes[0]]["onset"])
                unique_targets: list[int] = []
                used_strings: set[int] = set()
                for target_index in sorted(
                    target_indexes,
                    key=lambda item: abs(
                        annotation.events[item].onset - candidate_onset
                    ),
                ):
                    string = annotation.events[target_index].string
                    if string in used_strings:
                        continue
                    unique_targets.append(target_index)
                    used_strings.add(string)
                dataset = str(
                    annotation.provenance.get("dataset", "unknown")
                )
                if (
                    not unique_targets
                    and dataset in self.positive_only_datasets
                ):
                    self.filtered_negative_groups += 1
                    continue
                self.groups.append(
                    OnsetGroupIndex(
                        annotation_index,
                        tuple(unique_targets),
                        onset=candidate_onset,
                        input_pitches=tuple(
                            int(candidates[index]["pitch"])
                            for index in indexes
                        ),
                        input_confidences=tuple(
                            float(candidates[index].get("confidence", 1.0))
                            for index in indexes
                        ),
                    )
                )

    def prepare_feature_cache(self) -> dict[str, int]:
        return self.track_dataset.prepare_feature_cache()

    def repeat_sample_weights(
        self,
        *,
        interval: float = 0.75,
        bonus: float = 0.0,
        maximum_weight: float = 4.0,
    ) -> list[float]:
        if interval <= 0:
            raise ValueError("repeat interval must be positive")
        if bonus < 0:
            raise ValueError("repeat bonus must be non-negative")
        repeated_indexes: set[tuple[int, int]] = set()
        for annotation_index, annotation in enumerate(self.annotations):
            previous_by_position: dict[tuple[int, int], float] = {}
            for event_index, event in enumerate(annotation.events):
                position = (event.string, event.fret)
                previous = previous_by_position.get(position)
                if (
                    previous is not None
                    and 0 < event.onset - previous <= interval
                ):
                    repeated_indexes.add((annotation_index, event_index))
                previous_by_position[position] = event.onset
        return [
            min(
                maximum_weight,
                1.0
                + sum(
                    (group.annotation_index, event_index) in repeated_indexes
                    for event_index in group.event_indexes
                )
                * bonus,
            )
            for group in self.groups
        ]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        group = self.groups[index]
        annotation = self.annotations[group.annotation_index]
        features = self.track_dataset._load_cached_track(
            group.annotation_index
        )
        if features is None:
            self.track_dataset._prepare_annotation(group.annotation_index)
            features = self.track_dataset._load_cached_track(
                group.annotation_index
            )
        if features is None:
            raise RuntimeError(
                f"Could not create feature cache for {annotation.track_id}"
            )

        events = [annotation.events[event_index] for event_index in group.event_indexes]
        pitches = [
            OPEN_STRING_PITCHES[event.string - 1] + event.fret
            for event in events
        ]
        input_pitches = (
            list(group.input_pitches)
            if group.input_pitches is not None
            else list(pitches)
        )
        input_confidences = (
            list(group.input_confidences or [1.0] * len(input_pitches))
            if (
                group.input_pitches is not None
                and self.config.use_candidate_confidence
            )
            else [1.0] * len(input_pitches)
        )
        input_pitches, input_confidences = augment_pitch_candidates(
            input_pitches,
            input_confidences,
            pitches,
            drop_probability=self.pitch_drop_probability,
            add_probability=self.pitch_add_probability,
        )
        onset = (
            group.onset
            if group.onset is not None
            else events[0].onset
        )
        if self.onset_jitter_seconds:
            onset += random.uniform(
                -self.onset_jitter_seconds,
                self.onset_jitter_seconds,
            )
        fret_targets = np.zeros(STRING_COUNT, dtype=np.int64)
        event_pitches = np.full(STRING_COUNT, -1, dtype=np.int64)
        event_strings = np.full(STRING_COUNT, -1, dtype=np.int64)
        for event_slot, (event, pitch) in enumerate(zip(events, pitches)):
            fret_targets[event.string - 1] = event.fret + 1
            event_pitches[event_slot] = pitch
            event_strings[event_slot] = event.string - 1
        return {
            "features": torch.from_numpy(
                extract_onset_patch(
                    features,
                    onset=onset,
                    feature_config=self.feature_config,
                    config=self.config,
                )
            ),
            "pitch_mask": torch.from_numpy(
                pitch_mask(input_pitches, input_confidences)
            ),
            "pitch_targets": torch.from_numpy(pitch_mask(pitches)),
            "fret_targets": torch.from_numpy(fret_targets),
            "event_pitches": torch.from_numpy(event_pitches),
            "event_strings": torch.from_numpy(event_strings),
        }


class OnsetPositionNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        config: OnsetPositionConfig | None = None,
    ) -> None:
        super().__init__()
        self.feature_config = feature_config or FeatureConfig()
        self.config = config or OnsetPositionConfig()
        if self.config.adapter_size < 0:
            raise ValueError("adapter size must be non-negative")
        if (
            not math.isfinite(self.config.adapter_scale)
            or self.config.adapter_scale < 0
        ):
            raise ValueError("adapter scale must be finite and non-negative")
        self.adapter_scale = self.config.adapter_scale
        channels = (3, *self.config.convolution_channels)
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GroupNorm(
                        min(8, output_channels),
                        output_channels,
                    ),
                    nn.SiLU(),
                    nn.MaxPool2d(kernel_size=(1, 3)),
                ]
            )
        self.spectral_encoder = nn.Sequential(*layers)
        encoded_frequency_bins = self.feature_config.n_bins // 27
        encoded_size = (
            self.config.convolution_channels[-1]
            * self.config.frame_count
            * encoded_frequency_bins
        )
        self.spectral_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoded_size, self.config.hidden_size),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
        )
        self.pitch_projection = nn.Sequential(
            nn.Linear(PITCH_COUNT, self.config.hidden_size // 2),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(
                self.config.hidden_size + self.config.hidden_size // 2,
                self.config.hidden_size,
            ),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.hidden_size,
                STRING_COUNT * FRET_STATE_COUNT,
            ),
        )
        self.adapter = (
            nn.Sequential(
                nn.Linear(
                    self.config.hidden_size
                    + self.config.hidden_size // 2,
                    self.config.adapter_size,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.config.adapter_size,
                    STRING_COUNT * FRET_STATE_COUNT,
                ),
            )
            if self.config.adapter_size > 0
            else None
        )
        if self.adapter is not None:
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def encode(self, features: Tensor, pitches: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != 3:
            raise ValueError(
                "features must have shape [batch, 3, frames, bins]"
            )
        spectral = self.spectral_projection(self.spectral_encoder(features))
        pitch_context = self.pitch_projection(pitches)
        return torch.cat((spectral, pitch_context), dim=-1)

    def classify(self, encoded: Tensor) -> Tensor:
        logits = self.output(encoded)
        if self.adapter is not None:
            logits = logits + self.adapter_scale * self.adapter(encoded)
        return logits.reshape(
            encoded.shape[0],
            STRING_COUNT,
            FRET_STATE_COUNT,
        )

    def forward(self, features: Tensor, pitches: Tensor) -> Tensor:
        return self.classify(self.encode(features, pitches))


def tablature_pitch_probabilities(
    fret_probabilities: Tensor,
) -> Tensor:
    pitch_not_active = torch.ones(
        *fret_probabilities.shape[:-2],
        PITCH_COUNT,
        device=fret_probabilities.device,
        dtype=fret_probabilities.dtype,
    )
    for string_index, open_pitch in enumerate(OPEN_STRING_PITCHES):
        pitch_indexes = (
            torch.arange(
                FRET_STATE_COUNT - 1,
                device=fret_probabilities.device,
            )
            + open_pitch
            - PITCH_MIN
        )
        string_not_active = torch.ones_like(pitch_not_active)
        string_not_active[..., pitch_indexes] = (
            1 - fret_probabilities[..., string_index, 1:]
        )
        pitch_not_active = pitch_not_active * string_not_active
    return 1 - pitch_not_active


def onset_position_loss(
    logits: Tensor,
    fret_targets: Tensor,
    pitches: Tensor,
    *,
    silence_weight: float = 0.2,
    pitch_loss_weight: float = 0.25,
) -> tuple[Tensor, dict[str, Tensor]]:
    class_weights = torch.ones(
        FRET_STATE_COUNT,
        device=logits.device,
        dtype=logits.dtype,
    )
    class_weights[0] = silence_weight
    tablature_loss = functional.cross_entropy(
        logits.flatten(0, 1),
        fret_targets.flatten(),
        weight=class_weights,
    )
    probabilities = torch.softmax(logits, dim=-1)
    pitch_probabilities = tablature_pitch_probabilities(probabilities)
    pitch_weights = 1 + pitches * 4
    pitch_loss = (
        functional.binary_cross_entropy(
            pitch_probabilities.clamp(1e-6, 1 - 1e-6),
            pitches,
            reduction="none",
        )
        * pitch_weights
    ).mean()
    total = tablature_loss + pitch_loss_weight * pitch_loss
    return total, {
        "loss": total.detach(),
        "tablature_loss": tablature_loss.detach(),
        "pitch_loss": pitch_loss.detach(),
    }


def candidate_positions(pitch: int) -> list[tuple[int, int]]:
    return [
        (string_index, pitch - open_pitch)
        for string_index, open_pitch in enumerate(OPEN_STRING_PITCHES)
        if 0 <= pitch - open_pitch < FRET_STATE_COUNT - 1
    ]


def assign_pitch_group(
    probabilities: np.ndarray,
    pitches: list[int],
) -> list[tuple[int, int]]:
    if probabilities.shape != (STRING_COUNT, FRET_STATE_COUNT):
        raise ValueError("probabilities must have shape [6, 26]")
    candidate_sets = [candidate_positions(pitch) for pitch in pitches]
    if any(not candidates for candidates in candidate_sets):
        return []
    combinations = [
        positions
        for positions in itertools.product(*candidate_sets)
        if len({string for string, _fret in positions}) == len(positions)
    ]
    if not combinations:
        combinations = list(itertools.product(*candidate_sets))
    return list(
        max(
            combinations,
            key=lambda positions: sum(
                math.log(
                    max(
                        1e-8,
                        float(probabilities[string, fret + 1]),
                    )
                )
                for string, fret in positions
            ),
        )
    )


def reference_position_counts(
    probabilities: np.ndarray,
    event_pitches: np.ndarray,
    event_strings: np.ndarray,
) -> tuple[int, int]:
    correct = 0
    total = 0
    for group_probabilities, padded_pitches, padded_strings in zip(
        probabilities,
        event_pitches,
        event_strings,
    ):
        valid = padded_pitches >= 0
        pitches = [int(value) for value in padded_pitches[valid]]
        strings = [int(value) for value in padded_strings[valid]]
        assignments = assign_pitch_group(group_probabilities, pitches)
        correct += sum(
            predicted_string == reference_string
            for (predicted_string, _fret), reference_string in zip(
                assignments,
                strings,
            )
        )
        total += len(strings)
    return correct, total
