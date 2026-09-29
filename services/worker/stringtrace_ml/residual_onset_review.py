from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile
import torch

from .evaluate_onset_position import (
    assign_detected_events,
    basic_pitch_events,
    load_model,
)
from .features import extract_file_cqt
from .model import OPEN_STRING_PITCHES
from .onset_position import basic_pitch_cache_path
from .reference_review import (
    review_content_digest,
    sha256_file,
    validate_review,
)
from .schema import (
    ExcludedRange,
    LabeledEvent,
    TrackAnnotation,
    load_manifest,
    write_manifest,
)
from .train import choose_device


REVIEW_KIND = "residual-onset-training"
REVIEW_NAME_PREFIX = "residual-onset"
TECHNIQUE_LABELS = {
    "bend": "推弦",
    "harmonic": "泛音",
    "palm-mute": "掌根制音",
    "pick": "拨弦",
    "pinch-harmonic": "人工泛音",
    "vibrato": "揉弦",
}
MODE_LABELS = {
    "directinput": "直录",
    "micamp": "麦克风/音箱",
}


@dataclass(frozen=True)
class ReviewWindow:
    annotation_index: int
    start: float
    end: float
    score: float
    reference_events: int
    repeated_events: int
    missing_candidate_events: int = 0
    weak_candidate_events: int = 0


def event_pitch(event: LabeledEvent, tuning: list[int]) -> int:
    return tuning[event.string - 1] + event.fret


def _window_start(
    onset: float,
    duration: float,
    clip_seconds: float,
) -> float:
    maximum_start = max(0.0, duration - clip_seconds)
    return min(maximum_start, max(0.0, onset - clip_seconds * 0.25))


def _window_score(
    annotation: TrackAnnotation,
    start: float,
    end: float,
    *,
    repeat_interval: float,
    missing_candidate_indexes: frozenset[int] = frozenset(),
    weak_candidate_indexes: frozenset[int] = frozenset(),
    repeat_weight: float = 4.0,
    missing_candidate_weight: float = 8.0,
    weak_candidate_weight: float = 2.0,
) -> tuple[float, int, int, int, int]:
    event_indexes = [
        index
        for index, event in enumerate(annotation.events)
        if start <= event.onset < end
    ]
    events = [annotation.events[index] for index in event_indexes]
    previous_by_position: dict[tuple[int, int], float] = {}
    repeated = 0
    for event in events:
        position = (event.string, event.fret)
        previous = previous_by_position.get(position)
        if (
            previous is not None
            and 0 < event.onset - previous <= repeat_interval
        ):
            repeated += 1
        previous_by_position[position] = event.onset
    missing = sum(
        index in missing_candidate_indexes
        for index in event_indexes
    )
    weak = sum(
        index in weak_candidate_indexes
        for index in event_indexes
    )
    score = (
        len(events)
        + repeated * repeat_weight
        + missing * missing_candidate_weight
        + weak * weak_candidate_weight
    )
    return score, len(events), repeated, missing, weak


def reference_candidate_difficulty(
    annotation: TrackAnnotation,
    candidates: list[dict[str, object]],
    *,
    onset_tolerance: float,
    weak_confidence_threshold: float,
) -> tuple[frozenset[int], frozenset[int]]:
    if onset_tolerance <= 0:
        raise ValueError("onset_tolerance must be positive")
    if not 0 <= weak_confidence_threshold <= 1:
        raise ValueError("weak_confidence_threshold must be in [0, 1]")
    candidates_by_pitch: dict[int, list[dict[str, object]]] = {}
    for candidate in candidates:
        candidates_by_pitch.setdefault(
            int(candidate["pitch"]),
            [],
        ).append(candidate)

    missing: set[int] = set()
    weak: set[int] = set()
    for index, event in enumerate(annotation.events):
        matches = [
            candidate
            for candidate in candidates_by_pitch.get(
                event_pitch(event, annotation.tuning),
                [],
            )
            if abs(float(candidate["onset"]) - event.onset)
            <= onset_tolerance
        ]
        if not matches:
            missing.add(index)
            continue
        if max(
            float(candidate.get("confidence", 1.0))
            for candidate in matches
        ) < weak_confidence_threshold:
            weak.add(index)
    return frozenset(missing), frozenset(weak)


def best_review_window(
    annotation: TrackAnnotation,
    *,
    clip_seconds: float,
    repeat_interval: float,
    candidates: list[dict[str, object]] | None = None,
    candidate_onset_tolerance: float = 0.05,
    weak_confidence_threshold: float = 0.75,
    repeat_weight: float = 4.0,
    missing_candidate_weight: float = 8.0,
    weak_candidate_weight: float = 2.0,
) -> ReviewWindow | None:
    if not annotation.events:
        return None
    missing_candidate_indexes: frozenset[int] = frozenset()
    weak_candidate_indexes: frozenset[int] = frozenset()
    if candidates is not None:
        (
            missing_candidate_indexes,
            weak_candidate_indexes,
        ) = reference_candidate_difficulty(
            annotation,
            candidates,
            onset_tolerance=candidate_onset_tolerance,
            weak_confidence_threshold=weak_confidence_threshold,
        )
    candidate_starts = {
        round(
            _window_start(event.onset, annotation.duration, clip_seconds),
            6,
        )
        for event in annotation.events
    }
    best: ReviewWindow | None = None
    for start in sorted(candidate_starts):
        end = min(annotation.duration, start + clip_seconds)
        score, event_count, repeated, missing, weak = _window_score(
            annotation,
            start,
            end,
            repeat_interval=repeat_interval,
            missing_candidate_indexes=missing_candidate_indexes,
            weak_candidate_indexes=weak_candidate_indexes,
            repeat_weight=repeat_weight,
            missing_candidate_weight=missing_candidate_weight,
            weak_candidate_weight=weak_candidate_weight,
        )
        candidate = ReviewWindow(
            annotation_index=-1,
            start=start,
            end=end,
            score=score,
            reference_events=event_count,
            repeated_events=repeated,
            missing_candidate_events=missing,
            weak_candidate_events=weak,
        )
        if best is None or (
            candidate.score,
            candidate.missing_candidate_events,
            candidate.repeated_events,
            candidate.reference_events,
            -candidate.start,
        ) > (
            best.score,
            best.missing_candidate_events,
            best.repeated_events,
            best.reference_events,
            -best.start,
        ):
            best = candidate
    return best


def select_review_windows(
    annotations: list[TrackAnnotation],
    *,
    dataset: str,
    audio_condition: str,
    clip_seconds: float,
    repeat_interval: float,
    limit: int,
    candidate_events_by_track_id: (
        dict[str, list[dict[str, object]]] | None
    ) = None,
    candidate_onset_tolerance: float = 0.05,
    weak_confidence_threshold: float = 0.75,
    repeat_weight: float = 4.0,
    missing_candidate_weight: float = 8.0,
    weak_candidate_weight: float = 2.0,
    maximum_per_group: int | None = None,
) -> list[ReviewWindow]:
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be positive")
    if repeat_interval <= 0:
        raise ValueError("repeat_interval must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if maximum_per_group is not None and maximum_per_group <= 0:
        raise ValueError("maximum_per_group must be positive")
    if any(
        weight < 0
        for weight in (
            repeat_weight,
            missing_candidate_weight,
            weak_candidate_weight,
        )
    ):
        raise ValueError("selection weights must be non-negative")

    candidates: list[ReviewWindow] = []
    for index, annotation in enumerate(annotations):
        if annotation.split != "train":
            continue
        if str(annotation.provenance.get("dataset", "")) != dataset:
            continue
        if (
            str(annotation.provenance.get("audio_condition", ""))
            != audio_condition
        ):
            continue
        window = best_review_window(
            annotation,
            clip_seconds=clip_seconds,
            repeat_interval=repeat_interval,
            candidates=(
                candidate_events_by_track_id.get(annotation.track_id, [])
                if candidate_events_by_track_id is not None
                else None
            ),
            candidate_onset_tolerance=candidate_onset_tolerance,
            weak_confidence_threshold=weak_confidence_threshold,
            repeat_weight=repeat_weight,
            missing_candidate_weight=missing_candidate_weight,
            weak_candidate_weight=weak_candidate_weight,
        )
        if window is not None:
            candidates.append(
                ReviewWindow(
                    annotation_index=index,
                    start=window.start,
                    end=window.end,
                    score=window.score,
                    reference_events=window.reference_events,
                    repeated_events=window.repeated_events,
                    missing_candidate_events=(
                        window.missing_candidate_events
                    ),
                    weak_candidate_events=window.weak_candidate_events,
                )
            )

    candidates.sort(
        key=lambda item: (
            -item.score,
            -item.missing_candidate_events,
            -item.repeated_events,
            -item.weak_candidate_events,
            annotations[item.annotation_index].track_id,
        )
    )
    selected: list[ReviewWindow] = []
    selected_group_counts: dict[str, int] = {}
    for candidate in candidates:
        group_id = annotations[candidate.annotation_index].group_id
        if selected_group_counts.get(group_id, 0):
            continue
        selected.append(candidate)
        selected_group_counts[group_id] = 1
        if len(selected) >= limit:
            return selected
    for candidate in candidates:
        if candidate in selected:
            continue
        group_id = annotations[candidate.annotation_index].group_id
        if (
            maximum_per_group is not None
            and selected_group_counts.get(group_id, 0)
            >= maximum_per_group
        ):
            continue
        selected.append(candidate)
        selected_group_counts[group_id] = (
            selected_group_counts.get(group_id, 0) + 1
        )
        if len(selected) >= limit:
            break
    return selected


def load_cached_selection_candidates(
    annotations: list[TrackAnnotation],
    manifest_path: Path,
    cache_directory: Path,
    *,
    dataset: str,
    audio_condition: str,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length: float,
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for annotation in annotations:
        if annotation.split != "train":
            continue
        if str(annotation.provenance.get("dataset", "")) != dataset:
            continue
        if (
            str(annotation.provenance.get("audio_condition", ""))
            != audio_condition
        ):
            continue
        cache_path = basic_pitch_cache_path(
            annotation.resolve_audio(manifest_path),
            cache_directory,
            onset_threshold=float(onset_threshold),
            frame_threshold=float(frame_threshold),
            minimum_note_length=float(minimum_note_length),
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                "Missing selection candidate cache for "
                f"{annotation.track_id}: {cache_path.name}"
            )
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        result[annotation.track_id] = [
            dict(candidate)
            for candidate in payload.get("events", [])
        ]
    return result


def paired_review_source(
    annotation: TrackAnnotation,
    annotations_by_track_id: dict[str, TrackAnnotation],
) -> TrackAnnotation:
    source_track_id = annotation.provenance.get("derived_from_track_id")
    if not isinstance(source_track_id, str) or not source_track_id:
        raise ValueError(
            f"{annotation.track_id} does not identify its clean source track"
        )
    source = annotations_by_track_id.get(source_track_id)
    if source is None:
        raise ValueError(
            f"Clean source track {source_track_id} is missing from the manifest"
        )
    if source.group_id != annotation.group_id:
        raise ValueError(
            f"Clean source group differs for {annotation.track_id}"
        )
    if source.split != annotation.split:
        raise ValueError(
            f"Clean source split differs for {annotation.track_id}"
        )
    if abs(source.duration - annotation.duration) > 1e-5:
        raise ValueError(
            f"Clean source duration differs for {annotation.track_id}"
        )
    if (
        annotation.provenance.get("annotation_alignment")
        != "sample-aligned-no-offset"
    ):
        raise ValueError(
            f"{annotation.track_id} is not declared sample-aligned"
        )
    return source


def normalize_review_audio(
    audio: np.ndarray,
    *,
    target_rms_dbfs: float = -20.0,
    target_peak_dbfs: float = -1.0,
    maximum_gain_db: float = 60.0,
) -> tuple[np.ndarray, float]:
    rms = (
        float(np.sqrt(np.mean(np.square(audio))))
        if audio.size
        else 0.0
    )
    if rms <= 1e-8:
        raise ValueError("Review source audio is silent")
    current_rms_dbfs = 20.0 * math.log10(rms)
    gain_db = min(maximum_gain_db, target_rms_dbfs - current_rms_dbfs)
    gain = 10 ** (gain_db / 20.0)
    target_peak = 10 ** (target_peak_dbfs / 20.0)
    normalized = target_peak * np.tanh((audio * gain) / target_peak)
    return np.asarray(normalized, dtype=np.float32), gain_db


def _relative_reference_events(
    annotation: TrackAnnotation,
    start: float,
    duration: float,
) -> list[tuple[LabeledEvent, dict[str, str]]]:
    end = start + duration
    result: list[tuple[LabeledEvent, dict[str, str]]] = []
    for index, event in enumerate(annotation.events):
        if not start <= event.onset < end:
            continue
        onset = max(0.0, event.onset - start)
        offset = min(duration, max(onset + 0.01, event.offset - start))
        result.append(
            (
                LabeledEvent(
                    onset=round(onset, 6),
                    offset=round(offset, 6),
                    string=event.string,
                    fret=event.fret,
                    technique=event.technique,
                    confidence=1.0,
                ),
                {
                    "project_note_id": f"licensed-{index + 1}",
                    "position_source": "licensed-technique-label",
                    "technique_source": "licensed-technique-label",
                },
            )
        )
    return result


def _seed_review_events(
    predictions: list[LabeledEvent],
    references: list[tuple[LabeledEvent, dict[str, str]]],
    tuning: list[int],
    *,
    onset_tolerance: float,
    minimum_candidate_confidence: float = 0.0,
) -> list[dict[str, object]]:
    if not 0 <= minimum_candidate_confidence <= 1:
        raise ValueError("minimum_candidate_confidence must be in [0, 1]")
    events: list[tuple[LabeledEvent, dict[str, str]]] = [
        (
            prediction,
            {
                "project_note_id": f"candidate-{index + 1}",
                "position_source": "v12-constrained-candidate",
                "technique_source": "unknown",
            },
        )
        for index, prediction in enumerate(predictions)
        if prediction.confidence >= minimum_candidate_confidence
    ]
    for reference, provenance in references:
        reference_pitch = event_pitch(reference, tuning)
        events = [
            (event, existing_provenance)
            for event, existing_provenance in events
            if not (
                event_pitch(event, tuning) == reference_pitch
                and abs(event.onset - reference.onset) <= onset_tolerance
            )
        ]
        events.append((reference, provenance))
    events.sort(key=lambda item: (item[0].onset, item[0].string))
    return [
        {
            "onset": event.onset,
            "offset": event.offset,
            "string": event.string,
            "fret": event.fret,
            "technique": event.technique,
            "confidence": event.confidence,
            "candidate_provenance": provenance,
        }
        for event, provenance in events
    ]


def _track_labels(annotation: TrackAnnotation) -> tuple[str, str]:
    track_id = annotation.track_id
    technique = next(
        (
            name
            for name in sorted(
                TECHNIQUE_LABELS,
                key=len,
                reverse=True,
            )
            if f"-{name}-" in track_id
        ),
        "unknown",
    )
    mode = next(
        (
            name
            for name in MODE_LABELS
            if f"-{name}-" in track_id
        ),
        "unknown",
    )
    return (
        TECHNIQUE_LABELS.get(technique, technique),
        MODE_LABELS.get(mode, mode),
    )


def _review_display_title(annotation: TrackAnnotation) -> str:
    technique_label, mode_label = _track_labels(annotation)
    if technique_label == "unknown" and annotation.events:
        technique_label = TECHNIQUE_LABELS.get(
            annotation.events[0].technique,
            annotation.events[0].technique,
        )
    player_id = annotation.provenance.get("player_id")
    if player_id is None:
        player_label = next(
            (
                part
                for part in annotation.track_id.split("-")
                if part.startswith("P") and part[1:].isdigit()
            ),
            annotation.group_id,
        )
    else:
        player_label = f"P{player_id}"
    dataset = str(annotation.provenance.get("dataset", "unknown"))
    labels = [dataset, player_label, technique_label]
    if mode_label != "unknown":
        labels.append(mode_label)
    return " · ".join(labels)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_residual_onset_reviews(
    manifest_path: Path,
    output_directory: Path,
    position_checkpoint: Path,
    *,
    basic_pitch_cache: Path,
    dataset: str = "Guitar-TECHS",
    audio_condition: str = "procedural-mixture-demucs-guitar-stem",
    clip_seconds: float = 12.0,
    repeat_interval: float = 0.75,
    limit: int = 8,
    onset_threshold: float = 0.25,
    frame_threshold: float = 0.2,
    minimum_note_length: float = 50.0,
    onset_tolerance: float = 0.05,
    minimum_seed_confidence: float = 0.0,
    selection_candidate_cache: Path | None = None,
    selection_onset_threshold: float = 0.35,
    selection_frame_threshold: float = 0.25,
    selection_minimum_note_length: float = 70.0,
    selection_onset_tolerance: float = 0.05,
    weak_confidence_threshold: float = 0.75,
    repeat_weight: float = 4.0,
    missing_candidate_weight: float = 8.0,
    weak_candidate_weight: float = 2.0,
    maximum_per_group: int | None = None,
    device_name: str = "auto",
) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    output_directory = output_directory.resolve()
    existing = (
        list(output_directory.glob(f"{REVIEW_NAME_PREFIX}-*.json"))
        if output_directory.exists()
        else []
    )
    if existing:
        raise FileExistsError(
            "Residual-onset review queue already contains review packages"
        )
    annotations = load_manifest(manifest_path)
    annotations_by_track_id = {
        annotation.track_id: annotation
        for annotation in annotations
    }
    selection_candidates = (
        load_cached_selection_candidates(
            annotations,
            manifest_path,
            selection_candidate_cache.resolve(),
            dataset=dataset,
            audio_condition=audio_condition,
            onset_threshold=selection_onset_threshold,
            frame_threshold=selection_frame_threshold,
            minimum_note_length=selection_minimum_note_length,
        )
        if selection_candidate_cache is not None
        else None
    )
    windows = select_review_windows(
        annotations,
        dataset=dataset,
        audio_condition=audio_condition,
        clip_seconds=clip_seconds,
        repeat_interval=repeat_interval,
        limit=limit,
        candidate_events_by_track_id=selection_candidates,
        candidate_onset_tolerance=selection_onset_tolerance,
        weak_confidence_threshold=weak_confidence_threshold,
        repeat_weight=repeat_weight,
        missing_candidate_weight=missing_candidate_weight,
        weak_candidate_weight=weak_candidate_weight,
        maximum_per_group=maximum_per_group,
    )
    if len(windows) < limit:
        raise RuntimeError(
            f"Only {len(windows)} eligible review windows were available"
        )

    device = choose_device(device_name)
    model, feature_config, model_config, model_metadata = load_model(
        position_checkpoint,
        device,
    )
    manifest_hash = sha256_file(manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    packages: list[dict[str, object]] = []

    for rank, window in enumerate(windows, start=1):
        annotation = annotations[window.annotation_index]
        source_annotation = paired_review_source(
            annotation,
            annotations_by_track_id,
        )
        training_source_audio = annotation.resolve_audio(manifest_path)
        review_source_audio = source_annotation.resolve_audio(manifest_path)
        training_audio_samples, _sample_rate = librosa.load(
            training_source_audio,
            sr=feature_config.sample_rate,
            mono=True,
            offset=window.start,
            duration=window.end - window.start,
        )
        review_audio_samples, _sample_rate = librosa.load(
            review_source_audio,
            sr=feature_config.sample_rate,
            mono=True,
            offset=window.start,
            duration=window.end - window.start,
        )
        if len(review_audio_samples) != len(training_audio_samples):
            raise RuntimeError(
                f"Review and training clips differ in length for "
                f"{annotation.track_id}"
            )
        review_audio_samples, review_audio_gain_db = normalize_review_audio(
            review_audio_samples
        )
        duration = len(training_audio_samples) / feature_config.sample_rate
        if duration <= 0:
            raise RuntimeError(f"Empty review clip for {annotation.track_id}")
        name = f"{REVIEW_NAME_PREFIX}-{rank:02d}"
        review_audio = output_directory / f"{name}.wav"
        training_audio = output_directory / f"{name}.training.wav"
        soundfile.write(
            review_audio,
            review_audio_samples,
            feature_config.sample_rate,
            subtype="PCM_16",
        )
        soundfile.write(
            training_audio,
            training_audio_samples,
            feature_config.sample_rate,
            subtype="PCM_16",
        )
        candidates, dropped = basic_pitch_events(
            training_audio,
            basic_pitch_cache,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
        )
        features = extract_file_cqt(training_audio, feature_config)
        predictions = assign_detected_events(
            candidates,
            features,
            model,
            feature_config,
            model_config,
            device,
            mode="constrained",
        )
        references = _relative_reference_events(
            annotation,
            window.start,
            duration,
        )
        seeded_events = _seed_review_events(
            predictions,
            references,
            annotation.tuning,
            onset_tolerance=onset_tolerance,
            minimum_candidate_confidence=minimum_seed_confidence,
        )
        review: dict[str, Any] = {
            "schema_version": 1,
            "status": "draft",
            "review_kind": REVIEW_KIND,
            "display_title": _review_display_title(annotation),
            "display_subtitle": (
                f"可听源音频 · {window.start:.1f}-"
                f"{window.start + duration:.1f} 秒"
            ),
            "review_id": f"{annotation.track_id}-clip-{rank:02d}",
            "project_id": annotation.track_id,
            "project_sha256": manifest_hash,
            "source_audio_sha256": sha256_file(review_source_audio),
            "review_audio": review_audio.name,
            "review_audio_sha256": sha256_file(review_audio),
            "training_audio": training_audio.name,
            "training_audio_sha256": sha256_file(training_audio),
            "source_kind": REVIEW_KIND,
            "measure_numbers": [rank],
            "source_start": round(window.start, 6),
            "source_end": round(window.start + duration, 6),
            "duration": round(duration, 6),
            "capo": annotation.capo,
            "bpm": 60.0,
            "beats": [
                float(index)
                for index in range(max(1, math.ceil(duration)))
                if index < duration
            ],
            "time_signature": [4, 4],
            "events": seeded_events,
            "excluded_ranges": [],
            "checks": {
                "timing": False,
                "string_fret": False,
                "completeness": False,
                "technique": False,
            },
            "approval": None,
            "training_provenance": {
                "source_manifest": (
                    f"{manifest_path.parent.name}/{manifest_path.name}"
                ),
                "source_manifest_sha256": manifest_hash,
                "source_track_id": annotation.track_id,
                "review_source_track_id": source_annotation.track_id,
                "source_group_id": annotation.group_id,
                "source_split": annotation.split,
                "source_dataset": str(
                    annotation.provenance.get("dataset", "unknown")
                ),
                "source_license": str(
                    annotation.provenance.get("license", "unknown")
                ),
                "audio_condition": str(
                    annotation.provenance.get(
                        "audio_condition",
                        "unknown",
                    )
                ),
                "training_source_audio_sha256": sha256_file(
                    training_source_audio
                ),
                "audio_alignment": "sample-aligned-no-offset",
                "review_audio_processing": (
                    "rms-normalized--20-dbfs-soft-limited--1-dbfs"
                ),
                "review_audio_gain_db": round(review_audio_gain_db, 6),
            },
        }
        validate_review(review, require_approved=False)
        review_path = output_directory / f"{name}.json"
        _write_json(review_path, review)
        packages.append(
            {
                "name": name,
                "review_sha256": sha256_file(review_path),
                "review_audio_sha256": review["review_audio_sha256"],
                "training_audio_sha256": review["training_audio_sha256"],
                "source_track_id": annotation.track_id,
                "review_source_track_id": source_annotation.track_id,
                "review_audio_gain_db": round(review_audio_gain_db, 6),
                "source_group_id": annotation.group_id,
                "source_start": review["source_start"],
                "source_end": review["source_end"],
                "reference_events": len(references),
                "seeded_events": len(seeded_events),
                "dropped_out_of_range_candidates": dropped,
                "selection_score": window.score,
                "repeated_reference_events": window.repeated_events,
                "missing_candidate_events": (
                    window.missing_candidate_events
                ),
                "weak_candidate_events": window.weak_candidate_events,
            }
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "residual-onset-manual-review-queue",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_only": True,
        "evaluation_data_included": False,
        "source_manifest": {
            "name": manifest_path.name,
            "sha256": manifest_hash,
        },
        "position_checkpoint": {
            "name": position_checkpoint.name,
            "sha256": sha256_file(position_checkpoint),
            "metadata": model_metadata,
        },
        "candidate_frontend": {
            "name": "spotify-basic-pitch-0.4",
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
            "minimum_note_length": minimum_note_length,
            "minimum_seed_confidence": minimum_seed_confidence,
            "assignment": "v12-constrained",
            "audio_role": "training_audio",
        },
        "audio_roles": {
            "review_audio": (
                "rms-normalized-soft-limited-clean-source-listening"
            ),
            "training_audio": "residual-stem-training",
            "alignment": "sample-aligned-no-offset",
        },
        "selection": {
            "dataset": dataset,
            "audio_condition": audio_condition,
            "clip_seconds": clip_seconds,
            "repeat_interval": repeat_interval,
            "limit": limit,
            "group_diversity_first": True,
            "maximum_per_group": maximum_per_group,
            "candidate_priority": (
                {
                    "cache": selection_candidate_cache.name,
                    "onset_threshold": selection_onset_threshold,
                    "frame_threshold": selection_frame_threshold,
                    "minimum_note_length": (
                        selection_minimum_note_length
                    ),
                    "onset_tolerance": selection_onset_tolerance,
                    "weak_confidence_threshold": (
                        weak_confidence_threshold
                    ),
                    "weights": {
                        "reference": 1.0,
                        "repeat": repeat_weight,
                        "missing_candidate": missing_candidate_weight,
                        "weak_candidate": weak_candidate_weight,
                    },
                }
                if selection_candidate_cache is not None
                else None
            ),
        },
        "summary": {
            "groups": len(
                {
                    annotations[window.annotation_index].group_id
                    for window in windows
                }
            ),
            "reference_events": sum(
                window.reference_events for window in windows
            ),
            "repeated_reference_events": sum(
                window.repeated_events for window in windows
            ),
            "missing_candidate_events": sum(
                window.missing_candidate_events for window in windows
            ),
            "weak_candidate_events": sum(
                window.weak_candidate_events for window in windows
            ),
        },
        "packages": packages,
    }
    _write_json(output_directory / "queue.json", report)
    return report


def export_approved_residual_reviews(
    review_directory: Path,
    source_manifest: Path,
    output_manifest: Path,
    output_report: Path,
) -> tuple[list[TrackAnnotation], dict[str, object]]:
    review_directory = review_directory.resolve()
    source_manifest = source_manifest.resolve()
    output_manifest = output_manifest.resolve()
    expected_manifest_hash = sha256_file(source_manifest)
    annotations: list[TrackAnnotation] = []
    source_datasets: dict[str, int] = {}
    review_paths = sorted(
        review_directory.glob(f"{REVIEW_NAME_PREFIX}-*.json")
    )
    if not review_paths:
        raise RuntimeError("No residual-onset review packages are available")
    pending: list[str] = []
    for review_path in review_paths:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        validate_review(review, require_approved=False)
        if review.get("status") != "approved":
            pending.append(review_path.stem)
    if pending:
        raise RuntimeError(
            "All residual-onset reviews must be approved before export: "
            + ", ".join(pending)
        )

    for review_path in review_paths:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        validate_review(review, require_approved=True)
        if review.get("review_kind") != REVIEW_KIND:
            raise ValueError(f"Unexpected review kind in {review_path.name}")
        provenance = dict(review.get("training_provenance", {}))
        if provenance.get("source_split") != "train":
            raise ValueError("Residual-onset reviews must come from train split")
        if provenance.get("source_manifest_sha256") != expected_manifest_hash:
            raise ValueError("Residual-onset source manifest changed")
        review_audio = review_directory / str(review["review_audio"])
        if sha256_file(review_audio) != review["review_audio_sha256"]:
            raise ValueError(f"Review audio hash mismatch: {review_audio.name}")
        training_audio = review_directory / str(review["training_audio"])
        if sha256_file(training_audio) != review["training_audio_sha256"]:
            raise ValueError(
                f"Training audio hash mismatch: {training_audio.name}"
            )
        dataset = str(provenance["source_dataset"])
        source_datasets[dataset] = source_datasets.get(dataset, 0) + 1
        relative_audio = os.path.relpath(
            training_audio,
            output_manifest.parent,
        )
        annotations.append(
            TrackAnnotation(
                track_id=str(review["review_id"]),
                group_id=str(provenance["source_group_id"]),
                audio=relative_audio,
                split="train",
                duration=float(review["duration"]),
                events=[
                    LabeledEvent.from_dict(value)
                    for value in review["events"]
                ],
                excluded_ranges=[
                    ExcludedRange.from_dict(value)
                    for value in review.get("excluded_ranges", [])
                ],
                tuning=[64, 59, 55, 50, 45, 40],
                capo=int(review["capo"]),
                bpm=float(review["bpm"]),
                beats=[float(value) for value in review["beats"]],
                time_signature=(
                    int(review["time_signature"][0]),
                    int(review["time_signature"][1]),
                ),
                provenance={
                    "dataset": "Residual-Onset-Manual",
                    "source_dataset": dataset,
                    "source_license": str(provenance["source_license"]),
                    "source_track_id": str(provenance["source_track_id"]),
                    "source_manifest_sha256": expected_manifest_hash,
                    "audio_condition": str(provenance["audio_condition"]),
                    "review_kind": REVIEW_KIND,
                    "review_content_sha256": review_content_digest(review),
                    "review_audio_sha256": review["review_audio_sha256"],
                    "training_audio_sha256": (
                        review["training_audio_sha256"]
                    ),
                    "review_source_track_id": str(
                        provenance["review_source_track_id"]
                    ),
                    "audio_alignment": str(provenance["audio_alignment"]),
                    "human_verified_onset_complete": True,
                    "human_verified_string_fret": bool(
                        review["checks"]["string_fret"]
                    ),
                    "training_only": True,
                },
            )
        )

    annotations.sort(key=lambda annotation: annotation.track_id)
    write_manifest(output_manifest, annotations)
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "approved-residual-onset-training-export",
        "training_only": True,
        "source_manifest": {
            "name": source_manifest.name,
            "sha256": expected_manifest_hash,
        },
        "output_manifest": {
            "name": output_manifest.name,
            "sha256": sha256_file(output_manifest),
        },
        "approved_reviews": len(annotations),
        "events": sum(len(annotation.events) for annotation in annotations),
        "groups": len({annotation.group_id for annotation in annotations}),
        "datasets": dict(sorted(source_datasets.items())),
        "test_tracks": 0,
    }
    _write_json(output_report, report)
    return annotations, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or export residual-onset manual review packages.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    prepare.add_argument("--position-checkpoint", type=Path, required=True)
    prepare.add_argument("--basic-pitch-cache", type=Path, required=True)
    prepare.add_argument("--dataset", default="Guitar-TECHS")
    prepare.add_argument(
        "--audio-condition",
        default="procedural-mixture-demucs-guitar-stem",
    )
    prepare.add_argument("--clip-seconds", type=float, default=12.0)
    prepare.add_argument("--repeat-interval", type=float, default=0.75)
    prepare.add_argument("--limit", type=int, default=8)
    prepare.add_argument("--onset-threshold", type=float, default=0.25)
    prepare.add_argument("--frame-threshold", type=float, default=0.2)
    prepare.add_argument("--minimum-note-length", type=float, default=50.0)
    prepare.add_argument(
        "--minimum-seed-confidence",
        type=float,
        default=0.0,
    )
    prepare.add_argument("--selection-candidate-cache", type=Path)
    prepare.add_argument(
        "--selection-onset-threshold",
        type=float,
        default=0.35,
    )
    prepare.add_argument(
        "--selection-frame-threshold",
        type=float,
        default=0.25,
    )
    prepare.add_argument(
        "--selection-minimum-note-length",
        type=float,
        default=70.0,
    )
    prepare.add_argument(
        "--selection-onset-tolerance",
        type=float,
        default=0.05,
    )
    prepare.add_argument(
        "--weak-confidence-threshold",
        type=float,
        default=0.75,
    )
    prepare.add_argument("--repeat-weight", type=float, default=4.0)
    prepare.add_argument(
        "--missing-candidate-weight",
        type=float,
        default=8.0,
    )
    prepare.add_argument(
        "--weak-candidate-weight",
        type=float,
        default=2.0,
    )
    prepare.add_argument("--maximum-per-group", type=int)
    prepare.add_argument("--device", default="auto")

    export = commands.add_parser("export")
    export.add_argument("--review-directory", type=Path, required=True)
    export.add_argument("--source-manifest", type=Path, required=True)
    export.add_argument("--output-manifest", type=Path, required=True)
    export.add_argument("--output-report", type=Path, required=True)

    arguments = parser.parse_args()
    if arguments.command == "prepare":
        report = prepare_residual_onset_reviews(
            arguments.manifest,
            arguments.output_directory,
            arguments.position_checkpoint,
            basic_pitch_cache=arguments.basic_pitch_cache,
            dataset=arguments.dataset,
            audio_condition=arguments.audio_condition,
            clip_seconds=arguments.clip_seconds,
            repeat_interval=arguments.repeat_interval,
            limit=arguments.limit,
            onset_threshold=arguments.onset_threshold,
            frame_threshold=arguments.frame_threshold,
            minimum_note_length=arguments.minimum_note_length,
            minimum_seed_confidence=arguments.minimum_seed_confidence,
            selection_candidate_cache=(
                arguments.selection_candidate_cache
            ),
            selection_onset_threshold=(
                arguments.selection_onset_threshold
            ),
            selection_frame_threshold=(
                arguments.selection_frame_threshold
            ),
            selection_minimum_note_length=(
                arguments.selection_minimum_note_length
            ),
            selection_onset_tolerance=(
                arguments.selection_onset_tolerance
            ),
            weak_confidence_threshold=(
                arguments.weak_confidence_threshold
            ),
            repeat_weight=arguments.repeat_weight,
            missing_candidate_weight=(
                arguments.missing_candidate_weight
            ),
            weak_candidate_weight=arguments.weak_candidate_weight,
            maximum_per_group=arguments.maximum_per_group,
            device_name=arguments.device,
        )
        print(
            json.dumps(
                {
                    "packages": len(report["packages"]),
                    "output": arguments.output_directory.name,
                },
                ensure_ascii=False,
            )
        )
    else:
        annotations, report = export_approved_residual_reviews(
            arguments.review_directory,
            arguments.source_manifest,
            arguments.output_manifest,
            arguments.output_report,
        )
        print(
            json.dumps(
                {
                    "tracks": len(annotations),
                    "events": report["events"],
                    "output": arguments.output_manifest.name,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
