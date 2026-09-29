from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .features import FeatureConfig
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    OnsetGroupIndex,
)
from .schema import TrackAnnotation


@dataclass(frozen=True)
class CandidateActivityConfig:
    hidden_size: int = 64
    dropout: float = 0.1
    repeat_interval: float = 0.75
    temporal_feature_count: int = 0
    temporal_max_interval: float = 2.0

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> "CandidateActivityConfig":
        return cls(
            hidden_size=int(value.get("hidden_size", 64)),
            dropout=float(value.get("dropout", 0.1)),
            repeat_interval=float(value.get("repeat_interval", 0.75)),
            temporal_feature_count=int(
                value.get("temporal_feature_count", 0)
            ),
            temporal_max_interval=float(
                value.get("temporal_max_interval", 2.0)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CandidateActivityHead(nn.Module):
    def __init__(
        self,
        position_config: OnsetPositionConfig,
        config: CandidateActivityConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CandidateActivityConfig()
        if self.config.hidden_size <= 0:
            raise ValueError("activity hidden size must be positive")
        if not 0 <= self.config.dropout < 1:
            raise ValueError("activity dropout must be in [0, 1)")
        if self.config.repeat_interval <= 0:
            raise ValueError("repeat interval must be positive")
        if self.config.temporal_feature_count not in (0, 6):
            raise ValueError("temporal feature count must be 0 or 6")
        if self.config.temporal_max_interval <= 0:
            raise ValueError("temporal max interval must be positive")
        encoded_size = (
            position_config.hidden_size
            + position_config.hidden_size // 2
            + self.config.temporal_feature_count
        )
        self.network = nn.Sequential(
            nn.Linear(encoded_size, self.config.hidden_size),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, 2),
        )

    def forward(
        self,
        encoded: Tensor,
        temporal_context: Tensor | None = None,
    ) -> Tensor:
        if self.config.temporal_feature_count:
            if (
                temporal_context is None
                or temporal_context.shape
                != (
                    encoded.shape[0],
                    self.config.temporal_feature_count,
                )
            ):
                raise ValueError(
                    "temporal context does not match activity configuration"
                )
            encoded = torch.cat((encoded, temporal_context), dim=-1)
        return self.network(encoded)


def candidate_temporal_context(
    onsets: list[float],
    pitches: list[list[int]],
    confidences: list[list[float]],
    *,
    repeat_interval: float,
    maximum_interval: float,
) -> np.ndarray:
    if not (len(onsets) == len(pitches) == len(confidences)):
        raise ValueError("candidate context inputs must have equal lengths")
    features = np.zeros((len(onsets), 6), dtype=np.float32)
    previous_onset: float | None = None
    previous_pitches: set[int] = set()
    last_onset_by_pitch: dict[int, float] = {}
    for index, (onset, group_pitches, group_confidences) in enumerate(
        zip(onsets, pitches, confidences)
    ):
        current_pitches = set(group_pitches)
        previous_gap = (
            maximum_interval
            if previous_onset is None
            else max(0.0, onset - previous_onset)
        )
        same_pitch_gaps = [
            onset - last_onset_by_pitch[pitch]
            for pitch in current_pitches
            if pitch in last_onset_by_pitch
            and onset > last_onset_by_pitch[pitch]
        ]
        same_pitch_gap = (
            min(same_pitch_gaps)
            if same_pitch_gaps
            else maximum_interval
        )
        features[index] = np.asarray(
            (
                min(previous_gap, maximum_interval) / maximum_interval,
                min(same_pitch_gap, maximum_interval) / maximum_interval,
                float(
                    bool(same_pitch_gaps)
                    and same_pitch_gap <= repeat_interval
                ),
                float(bool(current_pitches & previous_pitches)),
                min(len(group_pitches), 6) / 6,
                (
                    sum(group_confidences) / len(group_confidences)
                    if group_confidences
                    else 1.0
                ),
            ),
            dtype=np.float32,
        )
        previous_onset = onset
        previous_pitches = current_pitches
        for pitch in current_pitches:
            last_onset_by_pitch[pitch] = onset
    return features


def candidate_sampling_weights(
    annotations: list[TrackAnnotation],
    groups: list[OnsetGroupIndex],
    dataset_weights: dict[str, float],
) -> list[float]:
    if any(
        not math.isfinite(weight) or weight <= 0
        for weight in dataset_weights.values()
    ):
        raise ValueError("dataset sampling weights must be positive and finite")
    return [
        dataset_weights.get(
            str(
                annotations[group.annotation_index].provenance.get(
                    "dataset",
                    "unknown",
                )
            ),
            1.0,
        )
        for group in groups
    ]


class CandidateActivityDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        feature_config: FeatureConfig,
        position_config: OnsetPositionConfig,
        activity_config: CandidateActivityConfig,
        feature_cache_directory: Path,
        candidate_cache_directory: Path,
        candidate_onset_threshold: float,
        candidate_frame_threshold: float,
        candidate_minimum_note_length: float,
        positive_only_datasets: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.position_dataset = OnsetPositionDataset(
            manifest_path,
            split=split,
            feature_config=feature_config,
            config=position_config,
            feature_cache_directory=feature_cache_directory,
            candidate_cache_directory=candidate_cache_directory,
            candidate_onset_threshold=candidate_onset_threshold,
            candidate_frame_threshold=candidate_frame_threshold,
            candidate_minimum_note_length=candidate_minimum_note_length,
            positive_only_datasets=positive_only_datasets,
        )
        self.annotations = self.position_dataset.annotations
        repeated_indexes: set[tuple[int, int]] = set()
        for annotation_index, annotation in enumerate(self.annotations):
            previous_by_position: dict[tuple[int, int], float] = {}
            for event_index, event in enumerate(annotation.events):
                position = (event.string, event.fret)
                previous = previous_by_position.get(position)
                if (
                    previous is not None
                    and 0
                    < event.onset - previous
                    <= activity_config.repeat_interval
                ):
                    repeated_indexes.add((annotation_index, event_index))
                previous_by_position[position] = event.onset
        self.targets = [
            (
                float(bool(group.event_indexes)),
                float(
                    any(
                        (group.annotation_index, event_index)
                        in repeated_indexes
                        for event_index in group.event_indexes
                    )
                ),
            )
            for group in self.position_dataset.groups
        ]
        self.temporal_contexts = np.zeros(
            (
                len(self.position_dataset.groups),
                activity_config.temporal_feature_count,
            ),
            dtype=np.float32,
        )
        if activity_config.temporal_feature_count:
            group_indexes_by_annotation: dict[int, list[int]] = {}
            for group_index, group in enumerate(
                self.position_dataset.groups
            ):
                group_indexes_by_annotation.setdefault(
                    group.annotation_index,
                    [],
                ).append(group_index)
            for group_indexes in group_indexes_by_annotation.values():
                groups = [
                    self.position_dataset.groups[index]
                    for index in group_indexes
                ]
                contexts = candidate_temporal_context(
                    [float(group.onset or 0) for group in groups],
                    [list(group.input_pitches or ()) for group in groups],
                    [
                        list(
                            group.input_confidences
                            or (1.0,) * len(group.input_pitches or ())
                        )
                        for group in groups
                    ],
                    repeat_interval=activity_config.repeat_interval,
                    maximum_interval=activity_config.temporal_max_interval,
                )
                self.temporal_contexts[group_indexes] = contexts

    @property
    def filtered_negative_groups(self) -> int:
        return self.position_dataset.filtered_negative_groups

    def prepare_feature_cache(self) -> dict[str, int]:
        return self.position_dataset.prepare_feature_cache()

    def target_counts(self) -> dict[str, int]:
        return {
            "groups": len(self.targets),
            "active": sum(activity > 0 for activity, _repeat in self.targets),
            "inactive": sum(
                activity == 0 for activity, _repeat in self.targets
            ),
            "repeated": sum(repeat > 0 for _activity, repeat in self.targets),
        }

    def sampling_weights(
        self,
        dataset_weights: dict[str, float],
    ) -> list[float]:
        return candidate_sampling_weights(
            self.annotations,
            self.position_dataset.groups,
            dataset_weights,
        )

    def __len__(self) -> int:
        return len(self.position_dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.position_dataset[index]
        activity, repeat = self.targets[index]
        return {
            **item,
            "activity_target": torch.tensor(activity, dtype=torch.float32),
            "repeat_target": torch.tensor(repeat, dtype=torch.float32),
            "temporal_context": torch.from_numpy(
                self.temporal_contexts[index]
            ),
        }
