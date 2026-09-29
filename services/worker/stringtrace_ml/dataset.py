from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .features import FeatureConfig, extract_file_cqt
from .model import STRING_COUNT
from .schema import TrackAnnotation, load_manifest


CACHE_STORAGE_DTYPE = np.float16


@dataclass(frozen=True)
class WindowIndex:
    annotation_index: int
    start: float


class GuitarTabDataset(Dataset[dict[str, Tensor | str | float]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        feature_config: FeatureConfig | None = None,
        window_seconds: float = 4.0,
        window_hop_seconds: float | None = None,
        feature_cache_directory: Path | None = None,
        pitch_shift_probability: float = 0.0,
        pitch_shift_min: int = 1,
        pitch_shift_max: int = 3,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.feature_config = feature_config or FeatureConfig()
        self.window_seconds = window_seconds
        self.window_hop_seconds = window_hop_seconds or window_seconds
        if not 0 <= pitch_shift_probability <= 1:
            raise ValueError("pitch_shift_probability must be in 0..1")
        if pitch_shift_min < 1 or pitch_shift_max < pitch_shift_min:
            raise ValueError("pitch shift range must contain positive semitones")
        self.pitch_shift_probability = pitch_shift_probability
        self.pitch_shift_min = pitch_shift_min
        self.pitch_shift_max = pitch_shift_max
        self.feature_cache_directory = (
            feature_cache_directory.resolve()
            if feature_cache_directory is not None
            else None
        )
        self.annotations = load_manifest(self.manifest_path, split)
        self.windows: list[WindowIndex] = []
        self._prepared_annotations: set[int] = set()
        self._track_cache_keys: dict[int, str] = {}
        for annotation_index, annotation in enumerate(self.annotations):
            maximum_start = max(0.0, annotation.duration - self.window_seconds)
            window_count = max(
                1,
                math.floor(maximum_start / self.window_hop_seconds) + 1,
            )
            starts = [
                min(index * self.window_hop_seconds, maximum_start)
                for index in range(window_count)
            ]
            if starts[-1] < maximum_start - 0.05:
                starts.append(maximum_start)
            for start in starts:
                self.windows.append(
                    WindowIndex(annotation_index, round(start, 6))
                )
        self.frame_count = (
            round(
                self.window_seconds
                * self.feature_config.frames_per_second
            )
            + 1
        )
        if self.feature_cache_directory is not None:
            self.feature_cache_directory.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.windows)

    def active_string_weights(
        self,
        *,
        exponent: float = 0.0,
        minimum_weight: float = 0.5,
        maximum_weight: float = 2.0,
    ) -> list[float]:
        if exponent < 0:
            raise ValueError("string balance exponent must be non-negative")
        counts = [0 for _string in range(STRING_COUNT)]
        for annotation in self.annotations:
            for event in annotation.events:
                counts[event.string - 1] += 1
        if exponent == 0 or not any(counts):
            return [1.0 for _string in range(STRING_COUNT)]
        mean_count = sum(counts) / STRING_COUNT
        raw_weights = [
            min(
                maximum_weight,
                max(
                    minimum_weight,
                    (mean_count / max(1, count)) ** exponent,
                ),
            )
            for count in counts
        ]
        mean_weight = sum(raw_weights) / STRING_COUNT
        return [
            weight / mean_weight
            for weight in raw_weights
        ]

    def difficulty_sample_weights(
        self,
        *,
        fast_interval: float = 0.14,
        high_fret: int = 12,
        fast_bonus: float = 0.0,
        high_fret_bonus: float = 0.0,
        maximum_weight: float = 4.0,
    ) -> list[float]:
        previous_onset_by_event: dict[tuple[int, int], float] = {}
        for annotation_index, annotation in enumerate(self.annotations):
            previous_by_string: dict[int, float] = {}
            for event_index, event in enumerate(annotation.events):
                if event.string in previous_by_string:
                    previous_onset_by_event[
                        (annotation_index, event_index)
                    ] = previous_by_string[event.string]
                previous_by_string[event.string] = event.onset

        weights: list[float] = []
        for window in self.windows:
            annotation = self.annotations[window.annotation_index]
            window_end = window.start + self.window_seconds
            fast_events = 0
            high_fret_events = 0
            for event_index, event in enumerate(annotation.events):
                if not window.start <= event.onset < window_end:
                    continue
                previous_onset = previous_onset_by_event.get(
                    (window.annotation_index, event_index)
                )
                if (
                    previous_onset is not None
                    and 0 < event.onset - previous_onset <= fast_interval
                ):
                    fast_events += 1
                if event.fret >= high_fret:
                    high_fret_events += 1
            weight = (
                1.0
                + min(4, fast_events) * fast_bonus
                + min(2, high_fret_events) * high_fret_bonus
            )
            weights.append(min(maximum_weight, weight))
        return weights

    def dataset_sample_weights(
        self,
        dataset_weights: dict[str, float],
    ) -> list[float]:
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in dataset_weights.values()
        ):
            raise ValueError(
                "dataset sampling weights must be positive and finite"
            )
        return [
            dataset_weights.get(
                str(
                    self.annotations[
                        window.annotation_index
                    ].provenance.get("dataset", "unknown")
                ),
                1.0,
            )
            for window in self.windows
        ]

    def _track_cache_key(self, annotation_index: int) -> str:
        existing = self._track_cache_keys.get(annotation_index)
        if existing is not None:
            return existing
        annotation = self.annotations[annotation_index]
        audio_path = annotation.resolve_audio(self.manifest_path)
        stat = audio_path.stat()
        value = {
            "format_version": 2,
            "audio_path": str(audio_path),
            "audio_size": stat.st_size,
            "audio_mtime_ns": stat.st_mtime_ns,
            "feature_config": self.feature_config.to_dict(),
        }
        key = hashlib.sha256(
            json.dumps(value, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._track_cache_keys[annotation_index] = key
        return key

    def _track_cache_path(self, annotation_index: int) -> Path:
        if self.feature_cache_directory is None:
            raise RuntimeError("Feature cache is not configured")
        return self.feature_cache_directory / (
            f"{self._track_cache_key(annotation_index)}.npy"
        )

    def _load_cached_track(
        self,
        annotation_index: int,
    ) -> np.ndarray | None:
        path = self._track_cache_path(annotation_index)
        if not path.is_file():
            return None
        try:
            features = np.load(
                path,
                allow_pickle=False,
                mmap_mode="r",
            )
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if (
            features.ndim != 2
            or features.shape[0] <= 0
            or features.shape[1] != self.feature_config.n_bins
            or features.dtype not in (np.float16, np.float32)
        ):
            del features
            path.unlink(missing_ok=True)
            return None
        return features

    def _save_cached_track(
        self,
        annotation_index: int,
        features: np.ndarray,
    ) -> None:
        path = self._track_cache_path(annotation_index)
        temporary = path.with_suffix(f".{os.getpid()}.download")
        with temporary.open("wb") as stream:
            np.save(
                stream,
                features.astype(CACHE_STORAGE_DTYPE),
                allow_pickle=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _prepare_annotation(self, annotation_index: int) -> int:
        if self.feature_cache_directory is None:
            return 0
        if self._load_cached_track(annotation_index) is not None:
            self._prepared_annotations.add(annotation_index)
            return 0

        annotation = self.annotations[annotation_index]
        full_features = extract_file_cqt(
            annotation.resolve_audio(self.manifest_path),
            self.feature_config,
        )
        if full_features.shape[0] == 0:
            raise RuntimeError(
                f"No feature frames available for {annotation.track_id}"
            )
        self._save_cached_track(annotation_index, full_features)
        self._prepared_annotations.add(annotation_index)
        return 1

    def prepare_feature_cache(self) -> dict[str, int]:
        if self.feature_cache_directory is None:
            return {"tracks": 0, "tracks_created": 0}
        tracks_created = 0
        for annotation_index in range(len(self.annotations)):
            tracks_created += self._prepare_annotation(annotation_index)
        return {
            "tracks": len(self.annotations),
            "tracks_created": tracks_created,
        }

    def load_track_features(self, annotation_index: int) -> np.ndarray:
        if self.feature_cache_directory is None:
            annotation = self.annotations[annotation_index]
            return extract_file_cqt(
                annotation.resolve_audio(self.manifest_path),
                self.feature_config,
            )
        features = self._load_cached_track(annotation_index)
        if features is None:
            self._prepare_annotation(annotation_index)
            features = self._load_cached_track(annotation_index)
        if features is None:
            raise RuntimeError(
                f"Could not create feature cache for track {annotation_index}"
            )
        return np.asarray(features, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | float]:
        window = self.windows[index]
        annotation = self.annotations[window.annotation_index]
        if self.feature_cache_directory is not None:
            track_features = self._load_cached_track(
                window.annotation_index
            )
            if track_features is None:
                self._prepare_annotation(window.annotation_index)
                track_features = self._load_cached_track(
                    window.annotation_index
                )
            if track_features is None:
                raise RuntimeError(f"Could not create feature cache for window {index}")
            start_frame = round(
                window.start * self.feature_config.frames_per_second
            )
            features = np.asarray(
                track_features[
                    start_frame : start_frame + self.frame_count
                ],
                dtype=np.float32,
            )
        else:
            features = extract_file_cqt(
                annotation.resolve_audio(self.manifest_path),
                self.feature_config,
                offset=window.start,
                duration=self.window_seconds,
            )
        valid_frame_count = min(features.shape[0], self.frame_count)
        padded_features = np.zeros(
            (self.frame_count, features.shape[1]),
            dtype=np.float32,
        )
        padded_features[:valid_frame_count] = features[:valid_frame_count]

        fret_targets = np.zeros(
            (self.frame_count, STRING_COUNT),
            dtype=np.int64,
        )
        onset_targets = np.zeros(
            (self.frame_count, STRING_COUNT),
            dtype=np.float32,
        )
        frames_per_second = self.feature_config.frames_per_second
        window_end = window.start + self.window_seconds
        for event in annotation.events:
            if event.offset <= window.start or event.onset >= window_end:
                continue
            string_index = event.string - 1
            active_start = max(0, round((event.onset - window.start) * frames_per_second))
            active_end = min(
                valid_frame_count,
                max(
                    active_start + 1,
                    math.ceil((event.offset - window.start) * frames_per_second),
                ),
            )
            fret_targets[active_start:active_end, string_index] = event.fret + 1
            if window.start <= event.onset < window_end:
                onset_frame = min(
                    valid_frame_count - 1,
                    max(0, round((event.onset - window.start) * frames_per_second)),
                )
                onset_targets[onset_frame, string_index] = event.confidence

        valid_frames = np.zeros(self.frame_count, dtype=np.bool_)
        valid_frames[:valid_frame_count] = True
        pitch_shift = 0
        active_targets = fret_targets[fret_targets > 0]
        maximum_fret = (
            int(active_targets.max()) - 1
            if active_targets.size
            else 24
        )
        maximum_shift = min(
            self.pitch_shift_max,
            24 - maximum_fret,
        )
        if (
            active_targets.size
            and maximum_shift >= self.pitch_shift_min
            and random.random() < self.pitch_shift_probability
        ):
            pitch_shift = random.randint(
                self.pitch_shift_min,
                maximum_shift,
            )
            bin_shift = round(
                pitch_shift * self.feature_config.bins_per_octave / 12
            )
            shifted_features = np.zeros_like(padded_features)
            shifted_features[:, bin_shift:] = padded_features[:, :-bin_shift]
            padded_features = shifted_features
            fret_targets[fret_targets > 0] += pitch_shift
        return {
            "features": torch.from_numpy(padded_features),
            "fret_targets": torch.from_numpy(fret_targets),
            "onset_targets": torch.from_numpy(onset_targets),
            "valid_frames": torch.from_numpy(valid_frames),
            "track_id": annotation.track_id,
            "window_start": window.start,
            "pitch_shift": pitch_shift,
        }
