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
    candidate_position_head: bool = False
    candidate_position_objective: str = "binary"
    candidate_pitch_embedding_size: int = 24
    candidate_spectral_evidence: bool = False
    candidate_spectral_harmonics: int = 5
    candidate_spectral_radius: int = 1
    candidate_spectral_size: int = 128
    candidate_string_embedding_size: int = 16
    candidate_position_hidden_size: int = 128
    candidate_activity_residual: bool = False
    candidate_activity_hidden_size: int = 128
    candidate_temporal_string_residual: bool = False
    candidate_temporal_feature_count: int = 20
    candidate_temporal_hidden_size: int = 64
    candidate_temporal_max_interval: float = 2.0
    candidate_temporal_repeat_interval: float = 0.75

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
            candidate_position_head=bool(
                value.get("candidate_position_head", False)
            ),
            candidate_position_objective=str(
                value.get("candidate_position_objective", "binary")
            ),
            candidate_pitch_embedding_size=int(
                value.get("candidate_pitch_embedding_size", 24)
            ),
            candidate_spectral_evidence=bool(
                value.get("candidate_spectral_evidence", False)
            ),
            candidate_spectral_harmonics=int(
                value.get("candidate_spectral_harmonics", 5)
            ),
            candidate_spectral_radius=int(
                value.get("candidate_spectral_radius", 1)
            ),
            candidate_spectral_size=int(
                value.get("candidate_spectral_size", 128)
            ),
            candidate_string_embedding_size=int(
                value.get("candidate_string_embedding_size", 16)
            ),
            candidate_position_hidden_size=int(
                value.get("candidate_position_hidden_size", 128)
            ),
            candidate_activity_residual=bool(
                value.get("candidate_activity_residual", False)
            ),
            candidate_activity_hidden_size=int(
                value.get("candidate_activity_hidden_size", 128)
            ),
            candidate_temporal_string_residual=bool(
                value.get("candidate_temporal_string_residual", False)
            ),
            candidate_temporal_feature_count=int(
                value.get("candidate_temporal_feature_count", 20)
            ),
            candidate_temporal_hidden_size=int(
                value.get("candidate_temporal_hidden_size", 64)
            ),
            candidate_temporal_max_interval=float(
                value.get("candidate_temporal_max_interval", 2.0)
            ),
            candidate_temporal_repeat_interval=float(
                value.get("candidate_temporal_repeat_interval", 0.75)
            ),
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
    input_durations: tuple[float, ...] | None = None


def onset_encoder_signature(
    config: OnsetPositionConfig,
) -> tuple[object, ...]:
    return (
        config.frames_before,
        config.frames_after,
        config.group_tolerance,
        config.convolution_channels,
        config.hidden_size,
        config.dropout,
        config.use_candidate_confidence,
    )


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


def candidate_string_temporal_context(
    onsets: list[float],
    pitches: list[list[int]],
    confidences: list[list[float]],
    durations: list[list[float]],
    *,
    repeat_interval: float,
    maximum_interval: float,
    feature_count: int = 20,
) -> list[dict[int, np.ndarray]]:
    if not (
        len(onsets)
        == len(pitches)
        == len(confidences)
        == len(durations)
    ):
        raise ValueError("candidate trajectory inputs must have equal lengths")
    if repeat_interval <= 0 or maximum_interval <= 0:
        raise ValueError("candidate trajectory intervals must be positive")
    if feature_count != 20:
        raise ValueError("candidate string temporal feature count must be 20")
    for group_pitches, group_confidences, group_durations in zip(
        pitches,
        confidences,
        durations,
    ):
        if not (
            len(group_pitches)
            == len(group_confidences)
            == len(group_durations)
        ):
            raise ValueError(
                "candidate pitch, confidence and duration groups must align"
            )

    records: list[dict[int, tuple[float, float]]] = []
    for group_pitches, group_confidences, group_durations in zip(
        pitches,
        confidences,
        durations,
    ):
        group_records: dict[int, tuple[float, float]] = {}
        for pitch, confidence, duration in zip(
            group_pitches,
            group_confidences,
            group_durations,
        ):
            normalized = (
                max(0.0, min(1.0, confidence)),
                max(0.0, duration),
            )
            previous = group_records.get(pitch)
            if previous is None or normalized[0] > previous[0]:
                group_records[pitch] = normalized
        records.append(group_records)

    previous_same: list[
        dict[int, tuple[float, float, float]]
    ] = []
    previous_by_pitch: dict[int, tuple[float, float, float]] = {}
    for onset, group in zip(onsets, records):
        previous_same.append(
            {
                pitch: previous_by_pitch[pitch]
                for pitch in group
                if pitch in previous_by_pitch
            }
        )
        for pitch, (confidence, duration) in group.items():
            previous_by_pitch[pitch] = (onset, confidence, duration)

    next_same: list[dict[int, tuple[float, float, float]]] = [
        {} for _group in records
    ]
    next_by_pitch: dict[int, tuple[float, float, float]] = {}
    for index in range(len(records) - 1, -1, -1):
        group = records[index]
        next_same[index] = {
            pitch: next_by_pitch[pitch]
            for pitch in group
            if pitch in next_by_pitch
        }
        for pitch, (confidence, duration) in group.items():
            next_by_pitch[pitch] = (
                onsets[index],
                confidence,
                duration,
            )

    def closest_pitch(
        pitch: int,
        group: dict[int, tuple[float, float]] | None,
    ) -> int | None:
        if not group:
            return None
        return min(
            group,
            key=lambda value: (
                abs(value - pitch),
                -group[value][0],
                value,
            ),
        )

    def movement(
        pitch: int,
        string_index: int,
        neighbor_pitch: int | None,
    ) -> tuple[float, float]:
        if neighbor_pitch is None:
            return 1.0, 1.0
        fret = pitch - OPEN_STRING_PITCHES[string_index]
        if not 0 <= fret < FRET_STATE_COUNT - 1:
            return 1.0, 1.0
        positions = candidate_positions(neighbor_pitch)
        if not positions:
            return 1.0, 1.0
        neighbor_string, neighbor_fret = min(
            positions,
            key=lambda position: (
                abs(position[0] - string_index) / (STRING_COUNT - 1)
                + abs(position[1] - fret) / (FRET_STATE_COUNT - 2),
                abs(position[0] - string_index),
                abs(position[1] - fret),
            ),
        )
        return (
            abs(neighbor_string - string_index) / (STRING_COUNT - 1),
            min(
                abs(neighbor_fret - fret),
                FRET_STATE_COUNT - 2,
            )
            / (FRET_STATE_COUNT - 2),
        )

    result: list[dict[int, np.ndarray]] = []
    for index, (onset, group) in enumerate(zip(onsets, records)):
        previous_group = records[index - 1] if index > 0 else None
        following_group = (
            records[index + 1] if index + 1 < len(records) else None
        )
        previous_onset = onsets[index - 1] if index > 0 else None
        following_onset = (
            onsets[index + 1] if index + 1 < len(onsets) else None
        )
        group_contexts: dict[int, np.ndarray] = {}
        for pitch, (confidence, duration) in group.items():
            previous_pitch = closest_pitch(pitch, previous_group)
            following_pitch = closest_pitch(pitch, following_group)
            previous_record = previous_same[index].get(pitch)
            following_record = next_same[index].get(pitch)
            previous_gap = (
                onset - previous_onset
                if previous_onset is not None
                else maximum_interval
            )
            following_gap = (
                following_onset - onset
                if following_onset is not None
                else maximum_interval
            )
            previous_same_gap = (
                onset - previous_record[0]
                if previous_record is not None
                else maximum_interval
            )
            following_same_gap = (
                following_record[0] - onset
                if following_record is not None
                else maximum_interval
            )
            shared = np.asarray(
                (
                    min(max(previous_gap, 0.0), maximum_interval)
                    / maximum_interval,
                    min(max(following_gap, 0.0), maximum_interval)
                    / maximum_interval,
                    min(max(previous_same_gap, 0.0), maximum_interval)
                    / maximum_interval,
                    min(max(following_same_gap, 0.0), maximum_interval)
                    / maximum_interval,
                    float(
                        previous_record is not None
                        and previous_same_gap <= repeat_interval
                    ),
                    float(
                        following_record is not None
                        and following_same_gap <= repeat_interval
                    ),
                    (
                        max(-24, min(24, previous_pitch - pitch)) / 24
                        if previous_pitch is not None
                        else 0.0
                    ),
                    (
                        max(-24, min(24, following_pitch - pitch)) / 24
                        if following_pitch is not None
                        else 0.0
                    ),
                    (
                        sum(value[0] for value in previous_group.values())
                        / len(previous_group)
                        if previous_group
                        else 0.0
                    ),
                    (
                        sum(value[0] for value in following_group.values())
                        / len(following_group)
                        if following_group
                        else 0.0
                    ),
                    previous_record[1] if previous_record is not None else 0.0,
                    following_record[1]
                    if following_record is not None
                    else 0.0,
                    min(len(previous_group or {}), STRING_COUNT)
                    / STRING_COUNT,
                    min(len(following_group or {}), STRING_COUNT)
                    / STRING_COUNT,
                    min(duration, maximum_interval) / maximum_interval,
                    confidence,
                ),
                dtype=np.float32,
            )
            temporal = np.zeros(
                (STRING_COUNT, feature_count),
                dtype=np.float32,
            )
            temporal[:, : shared.shape[0]] = shared
            for string_index in range(STRING_COUNT):
                temporal[string_index, 16:18] = movement(
                    pitch,
                    string_index,
                    previous_pitch,
                )
                temporal[string_index, 18:20] = movement(
                    pitch,
                    string_index,
                    following_pitch,
                )
            group_contexts[pitch] = temporal
        result.append(group_contexts)
    return result


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
        candidate_ambiguity_weight: float = 1.0,
        candidate_rare_pair_max_weight: float = 1.0,
        candidate_cross_string_weight: float = 1.0,
        candidate_cross_string_interval: float = 2.0,
        candidate_max_supervision_weight: float = 4.0,
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
        for name, value in (
            ("candidate ambiguity weight", candidate_ambiguity_weight),
            (
                "candidate rare-pair maximum weight",
                candidate_rare_pair_max_weight,
            ),
            (
                "candidate cross-string weight",
                candidate_cross_string_weight,
            ),
            (
                "candidate maximum supervision weight",
                candidate_max_supervision_weight,
            ),
        ):
            if not math.isfinite(value) or value < 1:
                raise ValueError(f"{name} must be finite and at least one")
        if (
            not math.isfinite(candidate_cross_string_interval)
            or candidate_cross_string_interval <= 0
        ):
            raise ValueError(
                "candidate cross-string interval must be positive and finite"
            )
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
                        input_durations=tuple(
                            max(
                                0.0,
                                float(
                                    candidates[index].get(
                                        "offset",
                                        candidate_onset,
                                    )
                                )
                                - float(candidates[index]["onset"]),
                            )
                            for index in indexes
                        ),
                    )
                )
        self.candidate_temporal_contexts: list[
            dict[int, np.ndarray]
        ] = [{} for _group in self.groups]
        if self.config.candidate_temporal_string_residual:
            group_indexes_by_annotation: dict[int, list[int]] = {}
            for group_index, group in enumerate(self.groups):
                group_indexes_by_annotation.setdefault(
                    group.annotation_index,
                    [],
                ).append(group_index)
            for annotation_index, group_indexes in (
                group_indexes_by_annotation.items()
            ):
                annotation = self.annotations[annotation_index]
                track_onsets: list[float] = []
                track_pitches: list[list[int]] = []
                track_confidences: list[list[float]] = []
                track_durations: list[list[float]] = []
                for group_index in group_indexes:
                    group = self.groups[group_index]
                    events = [
                        annotation.events[event_index]
                        for event_index in group.event_indexes
                    ]
                    reference_pitches = [
                        OPEN_STRING_PITCHES[event.string - 1] + event.fret
                        for event in events
                    ]
                    input_pitches = (
                        list(group.input_pitches)
                        if group.input_pitches is not None
                        else reference_pitches
                    )
                    track_onsets.append(
                        float(
                            group.onset
                            if group.onset is not None
                            else events[0].onset
                        )
                    )
                    track_pitches.append(input_pitches)
                    track_confidences.append(
                        list(
                            group.input_confidences
                            or (1.0,) * len(input_pitches)
                        )
                    )
                    track_durations.append(
                        list(
                            group.input_durations
                            or tuple(
                                max(0.0, event.offset - event.onset)
                                for event in events
                            )
                        )
                    )
                contexts = candidate_string_temporal_context(
                    track_onsets,
                    track_pitches,
                    track_confidences,
                    track_durations,
                    repeat_interval=(
                        self.config.candidate_temporal_repeat_interval
                    ),
                    maximum_interval=(
                        self.config.candidate_temporal_max_interval
                    ),
                    feature_count=(
                        self.config.candidate_temporal_feature_count
                    ),
                )
                for group_index, context in zip(group_indexes, contexts):
                    self.candidate_temporal_contexts[group_index] = context

        pair_counts: dict[tuple[int, int], int] = {}
        matched_events: set[tuple[int, int]] = set()
        for group in self.groups:
            annotation = self.annotations[group.annotation_index]
            for event_index in group.event_indexes:
                key = (group.annotation_index, event_index)
                if key in matched_events:
                    continue
                matched_events.add(key)
                event = annotation.events[event_index]
                pitch = OPEN_STRING_PITCHES[event.string - 1] + event.fret
                pair = (pitch, event.string)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        cross_string_events: set[tuple[int, int]] = set()
        for annotation_index, annotation in enumerate(self.annotations):
            events_by_pitch: dict[int, list[int]] = {}
            for event_index, event in enumerate(annotation.events):
                if (annotation_index, event_index) not in matched_events:
                    continue
                pitch = OPEN_STRING_PITCHES[event.string - 1] + event.fret
                events_by_pitch.setdefault(pitch, []).append(event_index)
            for pitch_events in events_by_pitch.values():
                ordered = sorted(
                    pitch_events,
                    key=lambda index: annotation.events[index].onset,
                )
                for offset, event_index in enumerate(ordered):
                    event = annotation.events[event_index]
                    neighbors = (
                        next(
                            (
                                candidate
                                for candidate in reversed(ordered[:offset])
                                if annotation.events[candidate].onset
                                < event.onset
                            ),
                            None,
                        ),
                        next(
                            (
                                candidate
                                for candidate in ordered[offset + 1 :]
                                if annotation.events[candidate].onset
                                > event.onset
                            ),
                            None,
                        ),
                    )
                    if any(
                        neighbor is not None
                        and 0
                        < abs(
                            annotation.events[neighbor].onset
                            - event.onset
                        )
                        <= candidate_cross_string_interval
                        and annotation.events[neighbor].string
                        != event.string
                        for neighbor in neighbors
                    ):
                        cross_string_events.add(
                            (annotation_index, event_index)
                        )

        self.candidate_target_weights: dict[tuple[int, int], float] = {}
        for annotation_index, event_index in matched_events:
            event = self.annotations[annotation_index].events[event_index]
            pitch = OPEN_STRING_PITCHES[event.string - 1] + event.fret
            positions = candidate_positions(pitch)
            playable_counts = [
                pair_counts.get((pitch, string_index + 1), 0)
                for string_index, _fret in positions
            ]
            maximum_pair_count = max(playable_counts, default=1)
            pair_count = max(1, pair_counts.get((pitch, event.string), 0))
            ambiguity = (
                candidate_ambiguity_weight
                if len(positions) >= 2
                else 1.0
            )
            rarity = min(
                candidate_rare_pair_max_weight,
                max(1.0, math.sqrt(maximum_pair_count / pair_count)),
            )
            cross_string = (
                candidate_cross_string_weight
                if (annotation_index, event_index) in cross_string_events
                else 1.0
            )
            self.candidate_target_weights[
                (annotation_index, event_index)
            ] = min(
                candidate_max_supervision_weight,
                ambiguity * rarity * cross_string,
            )
        self.candidate_supervision = {
            "matched_events": len(matched_events),
            "ambiguous_events": sum(
                len(
                    candidate_positions(
                        OPEN_STRING_PITCHES[
                            self.annotations[annotation_index]
                            .events[event_index]
                            .string
                            - 1
                        ]
                        + self.annotations[annotation_index]
                        .events[event_index]
                        .fret
                    )
                )
                >= 2
                for annotation_index, event_index in matched_events
            ),
            "cross_string_events": len(cross_string_events),
            "weighted_events": sum(
                weight > 1
                for weight in self.candidate_target_weights.values()
            ),
            "maximum_weight": max(
                self.candidate_target_weights.values(),
                default=1.0,
            ),
            "mean_weight": (
                sum(self.candidate_target_weights.values())
                / max(1, len(self.candidate_target_weights))
            ),
        }

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
        candidate_target_weights = np.ones(STRING_COUNT, dtype=np.float32)
        for event_slot, (event_index, event, pitch) in enumerate(
            zip(group.event_indexes, events, pitches)
        ):
            fret_targets[event.string - 1] = event.fret + 1
            event_pitches[event_slot] = pitch
            event_strings[event_slot] = event.string - 1
            candidate_target_weights[event_slot] = (
                self.candidate_target_weights.get(
                    (group.annotation_index, event_index),
                    1.0,
                )
            )
        temporal_context = np.zeros(
            (
                PITCH_COUNT,
                STRING_COUNT,
                (
                    self.config.candidate_temporal_feature_count
                    if self.config.candidate_temporal_string_residual
                    else 0
                ),
            ),
            dtype=np.float32,
        )
        for pitch, context in self.candidate_temporal_contexts[index].items():
            if PITCH_MIN <= pitch < PITCH_MIN + PITCH_COUNT:
                temporal_context[pitch - PITCH_MIN] = context
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
            "candidate_target_weights": torch.from_numpy(
                candidate_target_weights
            ),
            "candidate_temporal_context": torch.from_numpy(
                temporal_context
            ),
        }


def candidate_harmonic_cqt_indexes(
    feature_config: FeatureConfig,
    *,
    harmonic_count: int,
    radius: int,
) -> tuple[Tensor, Tensor]:
    if harmonic_count <= 0:
        raise ValueError("candidate harmonic count must be positive")
    if radius < 0:
        raise ValueError("candidate spectral radius must be non-negative")
    minimum_midi = 69 + 12 * math.log2(feature_config.fmin_hz / 440)
    bins_per_semitone = feature_config.bins_per_octave / 12
    indexes = torch.zeros(
        PITCH_COUNT,
        harmonic_count,
        radius * 2 + 1,
        dtype=torch.long,
    )
    valid = torch.zeros_like(indexes, dtype=torch.bool)
    offsets = range(-radius, radius + 1)
    for pitch_index in range(PITCH_COUNT):
        pitch = PITCH_MIN + pitch_index
        for harmonic_index in range(harmonic_count):
            harmonic = harmonic_index + 1
            harmonic_midi = pitch + 12 * math.log2(harmonic)
            center = round(
                (harmonic_midi - minimum_midi) * bins_per_semitone
            )
            for offset_index, offset in enumerate(offsets):
                index = center + offset
                indexes[pitch_index, harmonic_index, offset_index] = min(
                    feature_config.n_bins - 1,
                    max(0, index),
                )
                valid[pitch_index, harmonic_index, offset_index] = (
                    0 <= index < feature_config.n_bins
                )
    return indexes, valid


class FactorizedCandidatePositionHead(nn.Module):
    def __init__(
        self,
        *,
        encoded_size: int,
        pitch_embedding_size: int,
        spectral_input_size: int,
        spectral_size: int,
        string_embedding_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectral_projection = nn.Sequential(
            nn.Linear(spectral_input_size, spectral_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        context_size = (
            encoded_size
            + pitch_embedding_size
            + spectral_size
            + 1
        )
        self.activity_head = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.string_embedding = nn.Embedding(
            STRING_COUNT,
            string_embedding_size,
        )
        self.string_head = nn.Sequential(
            nn.Linear(
                context_size + string_embedding_size + 3,
                hidden_size,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        encoded: Tensor,
        pitch_embeddings: Tensor,
        spectral_evidence: Tensor,
        candidate_confidences: Tensor,
        frets: Tensor,
        valid_strings: Tensor,
    ) -> Tensor:
        spectral = self.spectral_projection(spectral_evidence)
        context = torch.cat(
            (
                encoded,
                pitch_embeddings,
                spectral,
                candidate_confidences.unsqueeze(1),
            ),
            dim=1,
        )
        activity_logits = self.activity_head(context).squeeze(1)
        string_indexes = torch.arange(
            STRING_COUNT,
            device=context.device,
        ).expand(context.shape[0], -1)
        normalized_frets = frets.clamp(0, FRET_STATE_COUNT - 2) / (
            FRET_STATE_COUNT - 2
        )
        position_features = torch.stack(
            (
                normalized_frets,
                normalized_frets.square(),
                (frets == 0).to(context.dtype),
            ),
            dim=2,
        )
        string_inputs = torch.cat(
            (
                context.unsqueeze(1).expand(-1, STRING_COUNT, -1),
                self.string_embedding(string_indexes),
                position_features,
            ),
            dim=2,
        )
        string_logits = self.string_head(string_inputs).squeeze(2)
        string_log_probabilities = functional.log_softmax(
            string_logits.masked_fill(~valid_strings, -30.0),
            dim=1,
        )
        return torch.cat(
            (
                functional.logsigmoid(-activity_logits).unsqueeze(1),
                functional.logsigmoid(activity_logits).unsqueeze(1)
                + string_log_probabilities,
            ),
            dim=1,
        )


class SpectralCandidateStringResidualHead(nn.Module):
    def __init__(
        self,
        *,
        encoded_size: int,
        pitch_embedding_size: int,
        spectral_input_size: int,
        spectral_size: int,
        string_embedding_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectral_projection = nn.Sequential(
            nn.Linear(spectral_input_size, spectral_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        context_size = (
            encoded_size
            + pitch_embedding_size
            + spectral_size
            + 1
        )
        self.string_embedding = nn.Embedding(
            STRING_COUNT,
            string_embedding_size,
        )
        self.string_head = nn.Sequential(
            nn.Linear(
                context_size + string_embedding_size + 3,
                hidden_size,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.string_head[-1].weight)
        nn.init.zeros_(self.string_head[-1].bias)

    def forward(
        self,
        encoded: Tensor,
        pitch_embeddings: Tensor,
        spectral_evidence: Tensor,
        candidate_confidences: Tensor,
        frets: Tensor,
    ) -> Tensor:
        spectral = self.spectral_projection(spectral_evidence)
        context = torch.cat(
            (
                encoded,
                pitch_embeddings,
                spectral,
                candidate_confidences.unsqueeze(1),
            ),
            dim=1,
        )
        string_indexes = torch.arange(
            STRING_COUNT,
            device=context.device,
        ).expand(context.shape[0], -1)
        normalized_frets = frets.clamp(0, FRET_STATE_COUNT - 2) / (
            FRET_STATE_COUNT - 2
        )
        position_features = torch.stack(
            (
                normalized_frets,
                normalized_frets.square(),
                (frets == 0).to(context.dtype),
            ),
            dim=2,
        )
        string_inputs = torch.cat(
            (
                context.unsqueeze(1).expand(-1, STRING_COUNT, -1),
                self.string_embedding(string_indexes),
                position_features,
            ),
            dim=2,
        )
        return self.string_head(string_inputs).squeeze(2)


class SpectralCandidateActivityResidualHead(nn.Module):
    def __init__(
        self,
        *,
        encoded_size: int,
        pitch_embedding_size: int,
        spectral_input_size: int,
        spectral_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectral_projection = nn.Sequential(
            nn.Linear(spectral_input_size, spectral_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.activity_head = nn.Sequential(
            nn.Linear(
                encoded_size
                + pitch_embedding_size
                + spectral_size
                + 1,
                hidden_size,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.activity_head[-1].weight)
        nn.init.zeros_(self.activity_head[-1].bias)

    def forward(
        self,
        encoded: Tensor,
        pitch_embeddings: Tensor,
        spectral_evidence: Tensor,
        candidate_confidences: Tensor,
    ) -> Tensor:
        spectral = self.spectral_projection(spectral_evidence)
        context = torch.cat(
            (
                encoded,
                pitch_embeddings,
                spectral,
                candidate_confidences.unsqueeze(1),
            ),
            dim=1,
        )
        return self.activity_head(context).squeeze(1)


class TemporalCandidateStringResidualHead(nn.Module):
    def __init__(
        self,
        *,
        temporal_feature_count: int,
        string_embedding_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_feature_count = temporal_feature_count
        self.string_embedding = nn.Embedding(
            STRING_COUNT,
            string_embedding_size,
        )
        self.network = nn.Sequential(
            nn.Linear(
                temporal_feature_count
                + string_embedding_size
                + 5,
                hidden_size,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        base_string_logits: Tensor,
        temporal_context: Tensor,
        candidate_confidences: Tensor,
        frets: Tensor,
    ) -> Tensor:
        if base_string_logits.ndim != 2 or base_string_logits.shape[1] != (
            STRING_COUNT
        ):
            raise ValueError(
                "base candidate string logits must have shape [items, 6]"
            )
        if temporal_context.shape != (
            base_string_logits.shape[0],
            STRING_COUNT,
            self.temporal_feature_count,
        ):
            raise ValueError(
                "candidate temporal context must have shape "
                "[items, 6, temporal_features]"
            )
        if candidate_confidences.shape != (base_string_logits.shape[0],):
            raise ValueError(
                "candidate confidences must align with temporal candidates"
            )
        if frets.shape != base_string_logits.shape:
            raise ValueError("candidate frets must align with string logits")
        string_indexes = torch.arange(
            STRING_COUNT,
            device=base_string_logits.device,
        ).expand(base_string_logits.shape[0], -1)
        normalized_frets = frets.clamp(0, FRET_STATE_COUNT - 2) / (
            FRET_STATE_COUNT - 2
        )
        position_features = torch.stack(
            (
                normalized_frets,
                normalized_frets.square(),
                (frets == 0).to(base_string_logits.dtype),
            ),
            dim=2,
        )
        inputs = torch.cat(
            (
                temporal_context.to(base_string_logits.dtype),
                self.string_embedding(string_indexes),
                base_string_logits.unsqueeze(2),
                candidate_confidences[:, None, None].expand(
                    -1,
                    STRING_COUNT,
                    -1,
                ),
                position_features,
            ),
            dim=2,
        )
        return self.network(inputs).squeeze(2)


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
        if self.config.candidate_pitch_embedding_size <= 0:
            raise ValueError(
                "candidate pitch embedding size must be positive"
            )
        if self.config.candidate_position_objective not in (
            "binary",
            "categorical",
            "factorized",
            "spectral_residual",
        ):
            raise ValueError(
                "candidate position objective must be binary, categorical "
                "factorized or spectral_residual"
            )
        if (
            self.config.candidate_spectral_evidence
            != (
                self.config.candidate_position_objective
                in ("factorized", "spectral_residual")
            )
        ):
            raise ValueError(
                "candidate spectral evidence requires a spectral "
                "candidate objective"
            )
        if self.config.candidate_spectral_harmonics <= 0:
            raise ValueError(
                "candidate spectral harmonics must be positive"
            )
        if self.config.candidate_spectral_radius < 0:
            raise ValueError(
                "candidate spectral radius must be non-negative"
            )
        if min(
            self.config.candidate_spectral_size,
            self.config.candidate_string_embedding_size,
            self.config.candidate_position_hidden_size,
            self.config.candidate_activity_hidden_size,
            self.config.candidate_temporal_feature_count,
            self.config.candidate_temporal_hidden_size,
        ) <= 0:
            raise ValueError("candidate evidence dimensions must be positive")
        if (
            not math.isfinite(self.config.candidate_temporal_max_interval)
            or self.config.candidate_temporal_max_interval <= 0
            or not math.isfinite(
                self.config.candidate_temporal_repeat_interval
            )
            or self.config.candidate_temporal_repeat_interval <= 0
        ):
            raise ValueError(
                "candidate temporal intervals must be positive and finite"
            )
        if self.config.candidate_temporal_feature_count != 20:
            raise ValueError(
                "candidate temporal string feature count must be 20"
            )
        if (
            self.config.candidate_activity_residual
            and self.config.candidate_position_objective
            != "spectral_residual"
        ):
            raise ValueError(
                "candidate activity residual requires the spectral "
                "residual position objective"
            )
        if (
            self.config.candidate_temporal_string_residual
            and self.config.candidate_position_objective
            != "spectral_residual"
        ):
            raise ValueError(
                "candidate temporal string residual requires the spectral "
                "residual position objective"
            )
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
        if self.config.candidate_position_head:
            self.candidate_pitch_embedding: nn.Module | None = nn.Embedding(
                PITCH_COUNT,
                self.config.candidate_pitch_embedding_size,
            )
            encoded_size = (
                self.config.hidden_size
                + self.config.hidden_size // 2
            )
            spectral_input_size = (
                3
                * self.config.frame_count
                * self.config.candidate_spectral_harmonics
                * (self.config.candidate_spectral_radius * 2 + 1)
            )
            if self.config.candidate_position_objective == "factorized":
                self.candidate_head = FactorizedCandidatePositionHead(
                    encoded_size=encoded_size,
                    pitch_embedding_size=(
                        self.config.candidate_pitch_embedding_size
                    ),
                    spectral_input_size=spectral_input_size,
                    spectral_size=self.config.candidate_spectral_size,
                    string_embedding_size=(
                        self.config.candidate_string_embedding_size
                    ),
                    hidden_size=(
                        self.config.candidate_position_hidden_size
                    ),
                    dropout=self.config.dropout,
                )
            else:
                self.candidate_head = nn.Sequential(
                    nn.Linear(
                        encoded_size
                        + self.config.candidate_pitch_embedding_size,
                        self.config.hidden_size,
                    ),
                    nn.SiLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(
                        self.config.hidden_size,
                        (
                            STRING_COUNT + 1
                            if (
                                self.config.candidate_position_objective
                                in (
                                    "categorical",
                                    "spectral_residual",
                                )
                            )
                            else STRING_COUNT
                        ),
                    ),
                )
            if (
                self.config.candidate_position_objective
                == "spectral_residual"
            ):
                self.candidate_residual_head: nn.Module | None = (
                    SpectralCandidateStringResidualHead(
                        encoded_size=encoded_size,
                        pitch_embedding_size=(
                            self.config.candidate_pitch_embedding_size
                        ),
                        spectral_input_size=spectral_input_size,
                        spectral_size=self.config.candidate_spectral_size,
                        string_embedding_size=(
                            self.config.candidate_string_embedding_size
                        ),
                        hidden_size=(
                            self.config.candidate_position_hidden_size
                        ),
                        dropout=self.config.dropout,
                    )
                )
            else:
                self.candidate_residual_head = None
            if self.config.candidate_activity_residual:
                self.candidate_activity_residual_head: nn.Module | None = (
                    SpectralCandidateActivityResidualHead(
                        encoded_size=encoded_size,
                        pitch_embedding_size=(
                            self.config.candidate_pitch_embedding_size
                        ),
                        spectral_input_size=spectral_input_size,
                        spectral_size=self.config.candidate_spectral_size,
                        hidden_size=(
                            self.config.candidate_activity_hidden_size
                        ),
                        dropout=self.config.dropout,
                    )
                )
            else:
                self.candidate_activity_residual_head = None
            if self.config.candidate_temporal_string_residual:
                self.candidate_temporal_residual_head: nn.Module | None = (
                    TemporalCandidateStringResidualHead(
                        temporal_feature_count=(
                            self.config.candidate_temporal_feature_count
                        ),
                        string_embedding_size=(
                            self.config.candidate_string_embedding_size
                        ),
                        hidden_size=(
                            self.config.candidate_temporal_hidden_size
                        ),
                        dropout=self.config.dropout,
                    )
                )
            else:
                self.candidate_temporal_residual_head = None
            valid_positions = torch.zeros(
                PITCH_COUNT,
                (
                    STRING_COUNT + 1
                    if (
                        self.config.candidate_position_objective
                        in (
                            "categorical",
                            "factorized",
                            "spectral_residual",
                        )
                    )
                    else STRING_COUNT
                ),
                dtype=torch.bool,
            )
            if self.config.candidate_position_objective in (
                "categorical",
                "factorized",
                "spectral_residual",
            ):
                valid_positions[:, 0] = True
            for pitch_index in range(PITCH_COUNT):
                pitch = PITCH_MIN + pitch_index
                for string_index, open_pitch in enumerate(
                    OPEN_STRING_PITCHES
                ):
                    class_index = (
                        string_index + 1
                        if (
                            self.config.candidate_position_objective
                            in (
                                "categorical",
                                "factorized",
                                "spectral_residual",
                            )
                        )
                        else string_index
                    )
                    valid_positions[pitch_index, class_index] = (
                        0 <= pitch - open_pitch < FRET_STATE_COUNT - 1
                    )
            self.register_buffer(
                "valid_candidate_positions",
                valid_positions,
                persistent=False,
            )
            if self.config.candidate_position_objective in (
                "factorized",
                "spectral_residual",
            ):
                harmonic_indexes, harmonic_valid = (
                    candidate_harmonic_cqt_indexes(
                        self.feature_config,
                        harmonic_count=(
                            self.config.candidate_spectral_harmonics
                        ),
                        radius=self.config.candidate_spectral_radius,
                    )
                )
                frets = torch.tensor(
                    [
                        [
                            PITCH_MIN + pitch_index - open_pitch
                            for open_pitch in OPEN_STRING_PITCHES
                        ]
                        for pitch_index in range(PITCH_COUNT)
                    ],
                    dtype=torch.float32,
                )
                self.register_buffer(
                    "candidate_harmonic_indexes",
                    harmonic_indexes,
                    persistent=False,
                )
                self.register_buffer(
                    "candidate_harmonic_valid",
                    harmonic_valid,
                    persistent=False,
                )
                self.register_buffer(
                    "candidate_frets",
                    frets,
                    persistent=False,
                )
        else:
            self.candidate_pitch_embedding = None
            self.candidate_head = None
            self.candidate_residual_head = None
            self.candidate_activity_residual_head = None
            self.candidate_temporal_residual_head = None

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

    def classify_candidate_positions(
        self,
        encoded: Tensor,
        pitch_indexes: Tensor,
        *,
        features: Tensor | None = None,
        candidate_confidences: Tensor | None = None,
        temporal_context: Tensor | None = None,
    ) -> Tensor:
        if (
            self.candidate_pitch_embedding is None
            or self.candidate_head is None
        ):
            raise ValueError("candidate position head is not enabled")
        if encoded.ndim != 2:
            raise ValueError("encoded candidates must have shape [items, channels]")
        if pitch_indexes.ndim != 1 or pitch_indexes.shape[0] != encoded.shape[0]:
            raise ValueError("pitch indexes must align with encoded candidates")
        if (
            bool((pitch_indexes < 0).any())
            or bool((pitch_indexes >= PITCH_COUNT).any())
        ):
            raise ValueError("candidate pitch indexes are out of range")
        spectral_objective = (
            self.config.candidate_position_objective
            in ("factorized", "spectral_residual")
        )
        spectral_evidence = None
        if spectral_objective:
            if features is None:
                raise ValueError(
                    "spectral candidate positions require spectral features"
                )
            if (
                features.ndim != 4
                or features.shape[0] != encoded.shape[0]
                or features.shape[1:] != (
                    3,
                    self.config.frame_count,
                    self.feature_config.n_bins,
                )
            ):
                raise ValueError(
                    "candidate features must have shape "
                    "[items, 3, frames, bins]"
                )
            if candidate_confidences is None:
                candidate_confidences = torch.ones(
                    encoded.shape[0],
                    device=encoded.device,
                    dtype=encoded.dtype,
                )
            if candidate_confidences.shape != (encoded.shape[0],):
                raise ValueError(
                    "candidate confidences must align with candidates"
                )
            harmonic_indexes = self.candidate_harmonic_indexes[
                pitch_indexes
            ]
            harmonic_valid = self.candidate_harmonic_valid[pitch_indexes]
            flattened_indexes = harmonic_indexes.flatten(1)
            gathered = torch.gather(
                features,
                3,
                flattened_indexes[:, None, None, :].expand(
                    -1,
                    features.shape[1],
                    features.shape[2],
                    -1,
                ),
            )
            gathered = gathered * harmonic_valid.flatten(1)[
                :, None, None, :
            ]
            spectral_evidence = gathered.flatten(1)
        pitch_embeddings = self.candidate_pitch_embedding(pitch_indexes)
        if self.config.candidate_position_objective == "factorized":
            return self.candidate_head(
                encoded,
                pitch_embeddings,
                spectral_evidence,
                candidate_confidences.to(encoded.dtype),
                self.candidate_frets[pitch_indexes],
                self.valid_candidate_positions[pitch_indexes, 1:],
            )
        inputs = torch.cat(
            (
                encoded,
                pitch_embeddings,
            ),
            dim=-1,
        )
        base_logits = self.candidate_head(inputs).masked_fill(
            ~self.valid_candidate_positions[pitch_indexes],
            -30.0,
        )
        if (
            self.config.candidate_position_objective
            == "spectral_residual"
        ):
            if self.candidate_residual_head is None:
                raise RuntimeError("candidate residual head is not enabled")
            string_residuals = self.candidate_residual_head(
                encoded,
                pitch_embeddings,
                spectral_evidence,
                candidate_confidences.to(encoded.dtype),
                self.candidate_frets[pitch_indexes],
            )
            valid_strings = self.valid_candidate_positions[
                pitch_indexes,
                1:,
            ]
            base_string_logits = base_logits[:, 1:].masked_fill(
                ~valid_strings,
                -30.0,
            )
            activity_logits = (
                torch.logsumexp(base_string_logits, dim=1)
                - base_logits[:, 0]
            )
            combined_string_logits = base_string_logits + string_residuals
            if self.candidate_temporal_residual_head is not None:
                if temporal_context is None:
                    raise ValueError(
                        "candidate temporal residual requires temporal context"
                    )
                temporal_residuals = (
                    self.candidate_temporal_residual_head(
                        combined_string_logits,
                        temporal_context,
                        candidate_confidences.to(encoded.dtype),
                        self.candidate_frets[pitch_indexes],
                    )
                )
                combined_string_logits = (
                    combined_string_logits + temporal_residuals
                )
            string_log_probabilities = functional.log_softmax(
                combined_string_logits.masked_fill(
                    ~valid_strings,
                    -30.0,
                ),
                dim=1,
            )
            return torch.cat(
                (
                    functional.logsigmoid(-activity_logits).unsqueeze(1),
                    functional.logsigmoid(activity_logits).unsqueeze(1)
                    + string_log_probabilities,
                ),
                dim=1,
            )
        return base_logits

    def classify_candidate_activity(
        self,
        encoded: Tensor,
        pitch_indexes: Tensor,
        *,
        features: Tensor | None = None,
        candidate_confidences: Tensor | None = None,
        candidate_position_logits: Tensor | None = None,
        temporal_context: Tensor | None = None,
    ) -> Tensor:
        if candidate_position_logits is None:
            candidate_position_logits = self.classify_candidate_positions(
                encoded,
                pitch_indexes,
                features=features,
                candidate_confidences=candidate_confidences,
                temporal_context=temporal_context,
            )
        if (
            candidate_position_logits.ndim != 2
            or candidate_position_logits.shape
            != (encoded.shape[0], STRING_COUNT + 1)
        ):
            raise ValueError(
                "candidate position logits must have shape [items, 7]"
            )
        activity_logits = (
            torch.logsumexp(candidate_position_logits[:, 1:], dim=1)
            - candidate_position_logits[:, 0]
        )
        if self.candidate_activity_residual_head is None:
            return activity_logits
        if self.candidate_pitch_embedding is None:
            raise RuntimeError("candidate pitch embedding is not enabled")
        if features is None:
            raise ValueError(
                "candidate activity residual requires spectral features"
            )
        if (
            features.ndim != 4
            or features.shape[0] != encoded.shape[0]
            or features.shape[1:] != (
                3,
                self.config.frame_count,
                self.feature_config.n_bins,
            )
        ):
            raise ValueError(
                "candidate features must have shape "
                "[items, 3, frames, bins]"
            )
        if candidate_confidences is None:
            candidate_confidences = torch.ones(
                encoded.shape[0],
                device=encoded.device,
                dtype=encoded.dtype,
            )
        if candidate_confidences.shape != (encoded.shape[0],):
            raise ValueError(
                "candidate confidences must align with candidates"
            )
        harmonic_indexes = self.candidate_harmonic_indexes[pitch_indexes]
        harmonic_valid = self.candidate_harmonic_valid[pitch_indexes]
        flattened_indexes = harmonic_indexes.flatten(1)
        spectral_evidence = torch.gather(
            features,
            3,
            flattened_indexes[:, None, None, :].expand(
                -1,
                features.shape[1],
                features.shape[2],
                -1,
            ),
        )
        spectral_evidence = spectral_evidence * harmonic_valid.flatten(1)[
            :, None, None, :
        ]
        residual = self.candidate_activity_residual_head(
            encoded,
            self.candidate_pitch_embedding(pitch_indexes),
            spectral_evidence.flatten(1),
            candidate_confidences.to(encoded.dtype),
        )
        return activity_logits + residual

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


def candidate_position_queries(
    event_pitches: Tensor,
    event_strings: Tensor,
    candidate_pitch_masks: Tensor | None = None,
    *,
    categorical: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    if event_pitches.shape != event_strings.shape or event_pitches.ndim != 2:
        raise ValueError(
            "event pitches and strings must have shape [batch, slots]"
        )
    if (
        candidate_pitch_masks is not None
        and candidate_pitch_masks.shape
        != (event_pitches.shape[0], PITCH_COUNT)
    ):
        raise ValueError(
            "candidate pitch masks must have shape [batch, 49]"
        )
    records: dict[tuple[int, int], set[int]] = {}
    if candidate_pitch_masks is not None:
        for batch_index in range(candidate_pitch_masks.shape[0]):
            for pitch_index in torch.nonzero(
                candidate_pitch_masks[batch_index] > 0,
                as_tuple=False,
            ).flatten():
                records[
                    (batch_index, PITCH_MIN + int(pitch_index))
                ] = set()
    for batch_index in range(event_pitches.shape[0]):
        for slot_index in range(event_pitches.shape[1]):
            pitch = int(event_pitches[batch_index, slot_index])
            string = int(event_strings[batch_index, slot_index])
            if pitch < 0:
                continue
            if not PITCH_MIN <= pitch < PITCH_MIN + PITCH_COUNT:
                raise ValueError("candidate pitch is out of range")
            if (
                candidate_pitch_masks is not None
                and (batch_index, pitch) not in records
            ):
                continue
            if not 0 <= string < STRING_COUNT:
                raise ValueError("candidate string is out of range")
            records.setdefault((batch_index, pitch), set()).add(string)
    device = event_pitches.device
    if not records:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(
                (0, STRING_COUNT + 1 if categorical else STRING_COUNT),
                dtype=torch.float32,
                device=device,
            ),
        )
    ordered = sorted(records)
    batch_indexes = torch.tensor(
        [batch_index for batch_index, _pitch in ordered],
        dtype=torch.long,
        device=device,
    )
    pitch_indexes = torch.tensor(
        [pitch - PITCH_MIN for _batch_index, pitch in ordered],
        dtype=torch.long,
        device=device,
    )
    targets = torch.zeros(
        (
            len(ordered),
            STRING_COUNT + 1 if categorical else STRING_COUNT,
        ),
        dtype=torch.float32,
        device=device,
    )
    for row, key in enumerate(ordered):
        strings = sorted(records[key])
        if categorical:
            if strings:
                targets[row, [string + 1 for string in strings]] = (
                    1 / len(strings)
                )
            else:
                targets[row, 0] = 1
        else:
            targets[row, strings] = 1
    return batch_indexes, pitch_indexes, targets


def candidate_position_query_weights(
    event_pitches: Tensor,
    event_weights: Tensor,
    candidate_pitch_masks: Tensor | None = None,
) -> Tensor:
    if event_pitches.shape != event_weights.shape or event_pitches.ndim != 2:
        raise ValueError(
            "event pitches and weights must have shape [batch, slots]"
        )
    if (
        candidate_pitch_masks is not None
        and candidate_pitch_masks.shape
        != (event_pitches.shape[0], PITCH_COUNT)
    ):
        raise ValueError(
            "candidate pitch masks must have shape [batch, 49]"
        )
    records: dict[tuple[int, int], list[float]] = {}
    if candidate_pitch_masks is not None:
        for batch_index in range(candidate_pitch_masks.shape[0]):
            for pitch_index in torch.nonzero(
                candidate_pitch_masks[batch_index] > 0,
                as_tuple=False,
            ).flatten():
                records[
                    (batch_index, PITCH_MIN + int(pitch_index))
                ] = []
    for batch_index in range(event_pitches.shape[0]):
        for slot_index in range(event_pitches.shape[1]):
            pitch = int(event_pitches[batch_index, slot_index])
            if pitch < 0:
                continue
            key = (batch_index, pitch)
            if candidate_pitch_masks is not None and key not in records:
                continue
            weight = float(event_weights[batch_index, slot_index])
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    "candidate supervision weights must be positive and finite"
                )
            records.setdefault(key, []).append(weight)
    return torch.tensor(
        [
            sum(records[key]) / len(records[key])
            if records[key]
            else 1.0
            for key in sorted(records)
        ],
        dtype=event_weights.dtype,
        device=event_weights.device,
    )


def candidate_position_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    objective: str | None = None,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, int, int]:
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError(
            "candidate logits and targets must have shape [items, classes]"
        )
    if logits.shape[1] not in (STRING_COUNT, STRING_COUNT + 1):
        raise ValueError("candidate classes must be six binary or seven categorical")
    if logits.shape[0] == 0:
        return logits.sum(), 0, 0
    if sample_weights is not None:
        if sample_weights.shape != (logits.shape[0],):
            raise ValueError(
                "candidate sample weights must align with candidate rows"
            )
        if (
            not bool(torch.isfinite(sample_weights).all())
            or bool((sample_weights <= 0).any())
        ):
            raise ValueError(
                "candidate sample weights must be positive and finite"
            )
    if objective is None:
        objective = (
            "categorical"
            if logits.shape[1] == STRING_COUNT + 1
            else "binary"
        )
    if objective not in (
        "binary",
        "categorical",
        "factorized",
        "spectral_residual",
    ):
        raise ValueError("unknown candidate position objective")
    if objective in (
        "categorical",
        "factorized",
        "spectral_residual",
    ):
        if logits.shape[1] != STRING_COUNT + 1:
            raise ValueError(
                "categorical candidate objectives require seven classes"
            )
        if not torch.allclose(
            targets.sum(dim=1),
            torch.ones(
                targets.shape[0],
                dtype=targets.dtype,
                device=targets.device,
            ),
        ):
            raise ValueError("categorical candidate targets must sum to one")
        if objective == "factorized":
            log_probabilities = functional.log_softmax(logits, dim=1)
            active = targets[:, 0] < 0.5
            activity_loss = -torch.where(
                active,
                torch.logsumexp(log_probabilities[:, 1:], dim=1),
                log_probabilities[:, 0],
            ).mean()
            string_loss = (
                functional.cross_entropy(
                    log_probabilities[active, 1:],
                    targets[active, 1:],
                )
                if bool(active.any())
                else logits.sum() * 0
            )
            loss = activity_loss + string_loss
        elif objective == "spectral_residual":
            active = targets[:, 0] < 0.5
            active_losses = (
                functional.cross_entropy(
                    logits[active, 1:],
                    targets[active, 1:],
                    reduction="none",
                )
                if bool(active.any())
                else torch.empty(
                    0,
                    dtype=logits.dtype,
                    device=logits.device,
                )
            )
            loss = (
                (
                    active_losses * sample_weights[active]
                ).sum()
                / sample_weights[active].sum()
                if sample_weights is not None and bool(active.any())
                else active_losses.mean()
                if bool(active.any())
                else logits.sum() * 0
            )
        else:
            losses = functional.cross_entropy(
                logits,
                targets,
                reduction="none",
            )
            loss = (
                (losses * sample_weights).sum() / sample_weights.sum()
                if sample_weights is not None
                else losses.mean()
            )
        predictions = logits.argmax(dim=1)
        active = targets[:, 0] < 0.5
        correct = int(
            (
                active
                & (
                    targets.gather(1, predictions.unsqueeze(1)).squeeze(1)
                    > 0
                )
            ).sum().detach().cpu()
        )
        total = int(active.sum().detach().cpu())
        return loss, correct, total
    if objective != "binary":
        raise ValueError("binary candidate objectives require six classes")
    loss = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.full(
            (STRING_COUNT,),
            float(STRING_COUNT - 1),
            dtype=logits.dtype,
            device=logits.device,
        ),
    )
    correct = 0
    total = int(targets.sum().detach().cpu())
    for row in range(targets.shape[0]):
        target_count = int(targets[row].sum().detach().cpu())
        if target_count == 0:
            continue
        predicted = torch.topk(logits[row], k=target_count).indices
        correct += int(targets[row, predicted].sum().detach().cpu())
    return loss, correct, total


def candidate_position_probabilities(
    logits: Tensor,
    *,
    objective: str,
    none_logit_bias: float = 0.0,
) -> Tensor:
    if not math.isfinite(none_logit_bias):
        raise ValueError("candidate none logit bias must be finite")
    if objective in (
        "categorical",
        "factorized",
        "spectral_residual",
    ):
        if logits.ndim != 2 or logits.shape[1] != STRING_COUNT + 1:
            raise ValueError(
                "categorical candidate logits must have shape [items, 7]"
            )
        adjusted = logits.clone()
        adjusted[:, 0] += none_logit_bias
        return torch.softmax(adjusted, dim=1)
    if objective == "binary":
        if none_logit_bias != 0:
            raise ValueError(
                "candidate none logit bias requires a categorical objective"
            )
        if logits.ndim != 2 or logits.shape[1] != STRING_COUNT:
            raise ValueError(
                "binary candidate logits must have shape [items, 6]"
            )
        return torch.sigmoid(logits)
    raise ValueError("candidate position objective must be binary or categorical")


def hierarchical_candidate_probabilities(
    logits: Tensor,
    *,
    objective: str,
    activity_logits: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if objective not in (
        "categorical",
        "factorized",
        "spectral_residual",
    ):
        raise ValueError(
            "hierarchical candidate decisions require a categorical objective"
        )
    joint_probabilities = candidate_position_probabilities(
        logits,
        objective=objective,
    )
    if activity_logits is not None:
        if activity_logits.shape != (logits.shape[0],):
            raise ValueError(
                "candidate activity logits must align with candidates"
            )
        active_probabilities = torch.sigmoid(activity_logits)
    else:
        active_probabilities = joint_probabilities[:, 1:].sum(dim=1)
    joint_active_probabilities = joint_probabilities[:, 1:].sum(dim=1)
    conditional_string_probabilities = (
        joint_probabilities[:, 1:]
        / joint_active_probabilities.clamp_min(
            torch.finfo(joint_probabilities.dtype).tiny
        ).unsqueeze(1)
    )
    return active_probabilities, conditional_string_probabilities


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


def assign_candidate_pitch_group(
    probabilities: np.ndarray,
    pitches: list[int],
) -> list[tuple[int, int] | None]:
    if probabilities.shape not in (
        (len(pitches), STRING_COUNT),
        (len(pitches), STRING_COUNT + 1),
    ):
        raise ValueError(
            "candidate probabilities must have shape [events, 6 or 7]"
        )
    candidate_sets = [candidate_positions(pitch) for pitch in pitches]
    if any(not candidates for candidates in candidate_sets):
        return []
    categorical = probabilities.shape[1] == STRING_COUNT + 1
    if categorical:
        options: list[list[tuple[int, int] | None]] = [
            [None, *candidates] for candidates in candidate_sets
        ]
        combinations = [
            positions
            for positions in itertools.product(*options)
            if len(
                {
                    position[0]
                    for position in positions
                    if position is not None
                }
            )
            == sum(position is not None for position in positions)
        ]
        return list(
            max(
                combinations,
                key=lambda positions: sum(
                    math.log(
                        max(
                            1e-8,
                            float(
                                probabilities[
                                    index,
                                    (
                                        0
                                        if position is None
                                        else position[0] + 1
                                    ),
                                ]
                            ),
                        )
                    )
                    for index, position in enumerate(positions)
                ),
            )
        )
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
                        float(probabilities[index, string]),
                    )
                )
                for index, (string, _fret) in enumerate(positions)
            ),
        )
    )


def assign_hierarchical_candidate_pitch_group(
    activity_probabilities: np.ndarray,
    conditional_string_probabilities: np.ndarray,
    pitches: list[int],
    *,
    minimum_activity_probability: float,
    minimum_string_margin: float = 0.0,
) -> list[tuple[int, int] | None]:
    if activity_probabilities.shape != (len(pitches),):
        raise ValueError(
            "candidate activity probabilities must align with pitches"
        )
    if conditional_string_probabilities.shape != (
        len(pitches),
        STRING_COUNT,
    ):
        raise ValueError(
            "conditional string probabilities must have shape [events, 6]"
        )
    if not 0 <= minimum_activity_probability <= 1:
        raise ValueError(
            "minimum candidate activity probability must be in 0..1"
        )
    if not 0 <= minimum_string_margin <= 1:
        raise ValueError("minimum candidate string margin must be in 0..1")
    sorted_probabilities = np.sort(
        conditional_string_probabilities,
        axis=1,
    )
    string_margins = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )
    accepted_indexes = np.flatnonzero(
        (
            activity_probabilities >= minimum_activity_probability
        )
        & (string_margins >= minimum_string_margin)
    )
    assignments: list[tuple[int, int] | None] = [None] * len(pitches)
    if not len(accepted_indexes):
        return assignments
    accepted_assignments = assign_candidate_pitch_group(
        conditional_string_probabilities[accepted_indexes],
        [pitches[index] for index in accepted_indexes],
    )
    for event_index, position in zip(
        accepted_indexes.tolist(),
        accepted_assignments,
    ):
        assignments[event_index] = position
    return assignments


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
