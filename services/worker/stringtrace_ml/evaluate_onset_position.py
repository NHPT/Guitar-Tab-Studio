from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from worker import transcribe_basic_pitch

from .candidate_activity import (
    CandidateActivityConfig,
    CandidateActivityHead,
    candidate_temporal_context,
)
from .dataset import GuitarTabDataset
from .evaluate import aggregate, aggregate_decoder_errors
from .features import FeatureConfig
from .inference import (
    LoadedModel,
    decode_events,
    load_checkpoint as load_frame_checkpoint,
    predict_probabilities,
)
from .metrics import evaluate_events
from .model import OPEN_STRING_PITCHES, PITCH_COUNT, PITCH_MIN
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionNet,
    assign_pitch_group,
    extract_onset_patch,
    group_event_indexes,
    pitch_mask,
)
from .schema import LabeledEvent, load_manifest, reject_quarantined_references


BASIC_PITCH_CACHE_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    OnsetPositionNet,
    FeatureConfig,
    OnsetPositionConfig,
    dict[str, object],
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported onset-position checkpoint format")
    feature_config = FeatureConfig.from_dict(
        dict(checkpoint.get("feature_config", {}))
    )
    model_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("model_config", {}))
    )
    model = OnsetPositionNet(feature_config, model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return (
        model,
        feature_config,
        model_config,
        dict(checkpoint.get("metadata", {})),
    )


def load_candidate_activity_head(
    checkpoint_path: Path,
    position_checkpoint_path: Path,
    position_config: OnsetPositionConfig,
    device: torch.device,
) -> tuple[CandidateActivityHead, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported candidate-activity checkpoint format")
    metadata = dict(checkpoint.get("metadata", {}))
    base_checkpoint = dict(metadata.get("base_checkpoint", {}))
    expected_hash = str(base_checkpoint.get("sha256", ""))
    actual_hash = sha256_file(position_checkpoint_path)
    if expected_hash != actual_hash:
        raise ValueError(
            "Candidate-activity head was trained against a different "
            "position checkpoint"
        )
    expected_position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("position_config", {}))
    )
    if expected_position_config != position_config:
        raise ValueError(
            "Candidate-activity and position model configurations differ"
        )
    activity_config = CandidateActivityConfig.from_dict(
        dict(checkpoint.get("activity_config", {}))
    )
    model = CandidateActivityHead(
        position_config,
        activity_config,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, metadata


def _cache_path(
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
            "version": BASIC_PITCH_CACHE_VERSION,
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


def basic_pitch_events(
    audio_path: Path,
    cache_directory: Path | None,
    *,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length: float,
) -> tuple[list[dict[str, float | int]], int]:
    cache_path = (
        _cache_path(
            audio_path,
            cache_directory,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
        )
        if cache_directory is not None
        else None
    )
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if int(cached.get("version", 0)) == BASIC_PITCH_CACHE_VERSION:
            return list(cached["events"]), int(cached["dropped"])

    with contextlib.redirect_stdout(sys.stderr):
        notes, dropped = transcribe_basic_pitch(
            audio_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
        )
    events = [
        {
            "onset": note.start,
            "offset": note.end,
            "pitch": note.pitch,
            "confidence": note.confidence,
        }
        for note in notes
    ]
    if cache_path is not None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "version": BASIC_PITCH_CACHE_VERSION,
                    "events": events,
                    "dropped": dropped,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return events, dropped


def assign_detected_events(
    events: list[dict[str, float | int]],
    features: np.ndarray,
    model: OnsetPositionNet,
    feature_config: FeatureConfig,
    model_config: OnsetPositionConfig,
    device: torch.device,
    *,
    mode: str = "constrained",
    minimum_direct_probability: float = 0.5,
    minimum_output_confidence: float = 0.0,
    activity_head: CandidateActivityHead | None = None,
    minimum_activity_probability: float = 0.0,
    minimum_repeat_probability: float = 0.5,
    repeat_minimum_direct_probability: float = 0.45,
    batch_size: int = 256,
) -> list[LabeledEvent]:
    groups = group_event_indexes(
        [float(event["onset"]) for event in events],
        tolerance=model_config.group_tolerance,
    )
    grouped_events = [
        [events[index] for index in indexes]
        for indexes in groups
    ]
    patches = [
        extract_onset_patch(
            features,
            onset=float(group[0]["onset"]),
            feature_config=feature_config,
            config=model_config,
        )
        for group in grouped_events
    ]
    masks = [
        pitch_mask(
            [int(event["pitch"]) for event in group],
            (
                [
                    float(event.get("confidence", 1.0))
                    for event in group
                ]
                if model_config.use_candidate_confidence
                else None
            ),
        )
        for group in grouped_events
    ]
    temporal_contexts = (
        candidate_temporal_context(
            [float(group[0]["onset"]) for group in grouped_events],
            [
                [int(event["pitch"]) for event in group]
                for group in grouped_events
            ],
            [
                [
                    float(event.get("confidence", 1.0))
                    for event in group
                ]
                for group in grouped_events
            ],
            repeat_interval=activity_head.config.repeat_interval,
            maximum_interval=activity_head.config.temporal_max_interval,
        )
        if (
            activity_head is not None
            and activity_head.config.temporal_feature_count
        )
        else np.zeros((len(groups), 0), dtype=np.float32)
    )
    probability_batches = []
    activity_batches = []
    with torch.no_grad():
        for start in range(0, len(groups), batch_size):
            feature_batch = torch.from_numpy(
                np.stack(patches[start : start + batch_size])
            ).to(device)
            pitch_batch = torch.from_numpy(
                np.stack(masks[start : start + batch_size])
            ).to(device)
            encoded = model.encode(feature_batch, pitch_batch)
            probability_batches.append(
                torch.softmax(
                    model.classify(encoded),
                    dim=-1,
                ).cpu().numpy()
            )
            if activity_head is not None:
                temporal_batch = torch.from_numpy(
                    temporal_contexts[start : start + batch_size]
                ).to(device)
                activity_batches.append(
                    torch.sigmoid(
                        activity_head(encoded, temporal_batch)
                    ).cpu().numpy()
                )
    all_probabilities = (
        np.concatenate(probability_batches)
        if probability_batches
        else np.zeros((0, 6, 26), dtype=np.float32)
    )
    all_activity_probabilities = (
        np.concatenate(activity_batches)
        if activity_batches
        else np.ones((len(groups), 2), dtype=np.float32)
    )
    predictions: list[LabeledEvent] = []
    for grouped, probabilities, activity_probabilities in zip(
        grouped_events,
        all_probabilities,
        all_activity_probabilities,
    ):
        if (
            activity_head is not None
            and float(activity_probabilities[0])
            < minimum_activity_probability
        ):
            continue
        pitches = [int(event["pitch"]) for event in grouped]
        if mode == "constrained":
            resolved = [
                (event, string_index, fret)
                for event, (string_index, fret) in zip(
                    grouped,
                    assign_pitch_group(probabilities, pitches),
                )
            ]
        elif mode == "direct":
            resolved = []
            direct_probability = (
                repeat_minimum_direct_probability
                if (
                    activity_head is not None
                    and float(activity_probabilities[1])
                    >= minimum_repeat_probability
                )
                else minimum_direct_probability
            )
            for string_index, string_probabilities in enumerate(
                probabilities
            ):
                state = int(string_probabilities.argmax())
                if (
                    state == 0
                    or float(string_probabilities[state])
                    < direct_probability
                ):
                    continue
                fret = state - 1
                pitch = OPEN_STRING_PITCHES[string_index] + fret
                source = min(
                    grouped,
                    key=lambda event: abs(int(event["pitch"]) - pitch),
                )
                resolved.append((source, string_index, fret))
        else:
            raise ValueError(f"Unknown assignment mode: {mode}")
        for event, string_index, fret in resolved:
            confidence = (
                max(
                    0.0,
                    min(1.0, float(event.get("confidence", 1.0))),
                )
                * max(
                    0.0,
                    min(
                        1.0,
                        float(probabilities[string_index, fret + 1]),
                    ),
                )
            ) ** 0.5
            if confidence < minimum_output_confidence:
                continue
            predictions.append(
                LabeledEvent(
                    onset=float(event["onset"]),
                    offset=float(event["offset"]),
                    string=string_index + 1,
                    fret=fret,
                    technique="unknown",
                    confidence=round(confidence, 4),
                )
            )
    return sorted(predictions, key=lambda event: (event.onset, event.string))


def merge_predictions(
    primary: list[LabeledEvent],
    supplemental: list[LabeledEvent],
    *,
    onset_tolerance: float = 0.05,
) -> list[LabeledEvent]:
    merged = list(primary)
    for candidate in supplemental:
        duplicate = next(
            (
                existing
                for existing in merged
                if existing.string == candidate.string
                and existing.fret == candidate.fret
                and abs(existing.onset - candidate.onset) <= onset_tolerance
            ),
            None,
        )
        if duplicate is None:
            merged.append(candidate)
    return sorted(merged, key=lambda event: (event.onset, event.string))


def direct_onset_candidates(
    features: np.ndarray,
    loaded: LoadedModel,
    device: torch.device,
    *,
    onset_threshold: float,
    activity_threshold: float,
    minimum_onset_gap: float,
    activity_context: float,
    activity_delay: float,
    minimum_event_confidence: float,
) -> list[dict[str, float | int]]:
    fret_probabilities, onset_probabilities = predict_probabilities(
        loaded,
        features,
        device=device,
    )
    predictions = decode_events(
        fret_probabilities,
        onset_probabilities,
        loaded.feature_config,
        onset_threshold=onset_threshold,
        activity_threshold=activity_threshold,
        minimum_onset_gap=minimum_onset_gap,
        activity_context=activity_context,
        activity_delay=activity_delay,
        minimum_event_confidence=minimum_event_confidence,
    )
    return [
        {
            "onset": event.onset,
            "offset": event.offset,
            "pitch": (
                OPEN_STRING_PITCHES[event.string - 1] + event.fret
            ),
            "confidence": event.confidence,
        }
        for event in predictions
        if (
            PITCH_MIN
            <= OPEN_STRING_PITCHES[event.string - 1] + event.fret
            < PITCH_MIN + PITCH_COUNT
        )
    ]


def unmatched_onset_candidates(
    primary: list[dict[str, float | int]],
    candidates: list[dict[str, float | int]],
    *,
    group_tolerance: float,
    match_tolerance: float,
) -> list[dict[str, float | int]]:
    primary_onsets = [float(event["onset"]) for event in primary]
    candidate_groups = group_event_indexes(
        [float(event["onset"]) for event in candidates],
        tolerance=group_tolerance,
    )
    unmatched: list[dict[str, float | int]] = []
    for indexes in candidate_groups:
        if any(
            abs(float(candidates[index]["onset"]) - primary_onset)
            <= match_tolerance
            for index in indexes
            for primary_onset in primary_onsets
        ):
            continue
        unmatched.extend(candidates[index] for index in indexes)
    return sorted(
        unmatched,
        key=lambda event: (float(event["onset"]), int(event["pitch"])),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Basic Pitch events with the onset-window position model."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mode",
        choices=("constrained", "direct"),
        default="constrained",
    )
    parser.add_argument(
        "--minimum-direct-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--minimum-output-confidence",
        type=float,
        default=0.0,
    )
    parser.add_argument("--adapter-scale", type=float)
    parser.add_argument("--candidate-activity-checkpoint", type=Path)
    parser.add_argument(
        "--minimum-activity-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-repeat-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--repeat-minimum-direct-probability",
        type=float,
        default=0.45,
    )
    parser.add_argument("--basic-pitch-cache", type=Path)
    parser.add_argument("--cqt-cache", type=Path)
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--pitch-onset-threshold", type=float, default=0.5)
    parser.add_argument("--pitch-frame-threshold", type=float, default=0.3)
    parser.add_argument(
        "--pitch-minimum-note-length",
        type=float,
        default=127.7,
    )
    parser.add_argument("--supplemental-pitch-onset-threshold", type=float)
    parser.add_argument("--supplemental-pitch-frame-threshold", type=float)
    parser.add_argument("--supplemental-pitch-minimum-note-length", type=float)
    parser.add_argument(
        "--supplemental-minimum-direct-probability",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--supplemental-repeat-minimum-direct-probability",
        type=float,
    )
    parser.add_argument(
        "--onset-fusion-checkpoint",
        type=Path,
        help=(
            "Experimental frame model used only to propose onset groups "
            "missing from the pitch frontend."
        ),
    )
    parser.add_argument(
        "--onset-fusion-onset-threshold",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--onset-fusion-activity-threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--onset-fusion-minimum-onset-gap",
        type=float,
        default=0.046,
    )
    parser.add_argument(
        "--onset-fusion-activity-context",
        type=float,
        default=0.046,
    )
    parser.add_argument(
        "--onset-fusion-activity-delay",
        type=float,
        default=0.058,
    )
    parser.add_argument(
        "--onset-fusion-minimum-event-confidence",
        type=float,
        default=0.46,
    )
    parser.add_argument(
        "--onset-fusion-match-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--onset-fusion-minimum-direct-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def parse_arguments(arguments: list[str] | None = None) -> Namespace:
    parser = build_parser()
    parsed = parser.parse_args(arguments)

    if parsed.start < 0:
        parser.error("--start must be non-negative")
    if parsed.limit is not None and parsed.limit <= 0:
        parser.error("--limit must be positive")
    if (
        parsed.adapter_scale is not None
        and (
            not np.isfinite(parsed.adapter_scale)
            or parsed.adapter_scale < 0
        )
    ):
        parser.error("--adapter-scale must be finite and non-negative")
    probability_arguments = (
        ("--minimum-direct-probability", parsed.minimum_direct_probability),
        ("--minimum-output-confidence", parsed.minimum_output_confidence),
        (
            "--supplemental-minimum-direct-probability",
            parsed.supplemental_minimum_direct_probability,
        ),
        (
            "--minimum-activity-probability",
            parsed.minimum_activity_probability,
        ),
        (
            "--minimum-repeat-probability",
            parsed.minimum_repeat_probability,
        ),
        (
            "--repeat-minimum-direct-probability",
            parsed.repeat_minimum_direct_probability,
        ),
        (
            "--onset-fusion-onset-threshold",
            parsed.onset_fusion_onset_threshold,
        ),
        (
            "--onset-fusion-activity-threshold",
            parsed.onset_fusion_activity_threshold,
        ),
        (
            "--onset-fusion-minimum-event-confidence",
            parsed.onset_fusion_minimum_event_confidence,
        ),
        (
            "--onset-fusion-minimum-direct-probability",
            parsed.onset_fusion_minimum_direct_probability,
        ),
    )
    for name, value in probability_arguments:
        if not 0 <= value <= 1:
            parser.error(f"{name} must be in 0..1")
    if (
        parsed.supplemental_repeat_minimum_direct_probability is not None
        and not 0
        <= parsed.supplemental_repeat_minimum_direct_probability
        <= 1
    ):
        parser.error(
            "--supplemental-repeat-minimum-direct-probability "
            "must be in 0..1"
        )
    positive_arguments = (
        (
            "--onset-fusion-minimum-onset-gap",
            parsed.onset_fusion_minimum_onset_gap,
        ),
        (
            "--onset-fusion-activity-context",
            parsed.onset_fusion_activity_context,
        ),
        (
            "--onset-fusion-match-tolerance",
            parsed.onset_fusion_match_tolerance,
        ),
    )
    for name, value in positive_arguments:
        if value <= 0:
            parser.error(f"{name} must be positive")
    if parsed.onset_fusion_activity_delay < 0:
        parser.error("--onset-fusion-activity-delay must be non-negative")
    return parsed


def main() -> None:
    arguments = parse_arguments()
    all_annotations = load_manifest(arguments.manifest, arguments.split)
    annotations = all_annotations[arguments.start :]
    if arguments.limit is not None:
        annotations = annotations[: arguments.limit]
    if not annotations:
        raise RuntimeError(f"The manifest contains no {arguments.split} tracks")
    reject_quarantined_references(annotations)
    device = torch.device(arguments.device)
    model, feature_config, model_config, metadata = load_model(
        arguments.checkpoint,
        device,
    )
    if arguments.adapter_scale is not None:
        if model.adapter is None:
            raise ValueError(
                "--adapter-scale requires a checkpoint with an adapter"
            )
        model.adapter_scale = arguments.adapter_scale
    activity_head = None
    activity_metadata: dict[str, object] = {}
    if arguments.candidate_activity_checkpoint is not None:
        activity_head, activity_metadata = load_candidate_activity_head(
            arguments.candidate_activity_checkpoint,
            arguments.checkpoint,
            model_config,
            device,
        )
    cqt_dataset = GuitarTabDataset(
        arguments.manifest,
        split=arguments.split,
        feature_config=feature_config,
        feature_cache_directory=(
            arguments.cqt_cache
            if arguments.cqt_cache is not None
            else arguments.manifest.resolve().parent / ".feature-cache"
        ),
    )
    if [
        annotation.track_id
        for annotation in cqt_dataset.annotations[
            arguments.start : arguments.start + len(annotations)
        ]
    ] != [annotation.track_id for annotation in annotations]:
        raise ValueError("CQT cache dataset is not aligned with evaluation data")
    cqt_dataset.prepare_feature_cache()
    frame_model = None
    frame_cqt_dataset = None
    if arguments.onset_fusion_checkpoint is not None:
        frame_model = load_frame_checkpoint(
            arguments.onset_fusion_checkpoint,
            device,
        )
        if frame_model.feature_config == feature_config:
            frame_cqt_dataset = cqt_dataset
        else:
            frame_cqt_dataset = GuitarTabDataset(
                arguments.manifest,
                split=arguments.split,
                feature_config=frame_model.feature_config,
                feature_cache_directory=(
                    arguments.cqt_cache
                    if arguments.cqt_cache is not None
                    else arguments.manifest.resolve().parent
                    / ".feature-cache"
                ),
            )
            if [
                annotation.track_id
                for annotation in frame_cqt_dataset.annotations[
                    arguments.start : arguments.start + len(annotations)
                ]
            ] != [annotation.track_id for annotation in annotations]:
                raise ValueError(
                    "Frame-model CQT dataset is not aligned with "
                    "evaluation data"
                )
            frame_cqt_dataset.prepare_feature_cache()
    track_results: list[dict[str, object]] = []
    for index, annotation in enumerate(annotations, start=1):
        audio_path = annotation.resolve_audio(arguments.manifest.resolve())
        events, dropped = basic_pitch_events(
            audio_path,
            arguments.basic_pitch_cache,
            onset_threshold=arguments.pitch_onset_threshold,
            frame_threshold=arguments.pitch_frame_threshold,
            minimum_note_length=arguments.pitch_minimum_note_length,
        )
        features = cqt_dataset.load_track_features(
            arguments.start + index - 1
        )
        predictions = assign_detected_events(
            events,
            features,
            model,
            feature_config,
            model_config,
            device,
            mode=arguments.mode,
            minimum_direct_probability=arguments.minimum_direct_probability,
            minimum_output_confidence=arguments.minimum_output_confidence,
            activity_head=activity_head,
            minimum_activity_probability=(
                arguments.minimum_activity_probability
            ),
            minimum_repeat_probability=(
                arguments.minimum_repeat_probability
            ),
            repeat_minimum_direct_probability=(
                arguments.repeat_minimum_direct_probability
            ),
        )
        pitch_frontend_events = list(events)
        if arguments.supplemental_pitch_onset_threshold is not None:
            supplemental_events, supplemental_dropped = basic_pitch_events(
                audio_path,
                arguments.basic_pitch_cache,
                onset_threshold=(
                    arguments.supplemental_pitch_onset_threshold
                ),
                frame_threshold=(
                    arguments.supplemental_pitch_frame_threshold
                    if arguments.supplemental_pitch_frame_threshold is not None
                    else arguments.pitch_frame_threshold
                ),
                minimum_note_length=(
                    arguments.supplemental_pitch_minimum_note_length
                    if (
                        arguments.supplemental_pitch_minimum_note_length
                        is not None
                    )
                    else arguments.pitch_minimum_note_length
                ),
            )
            supplemental_predictions = assign_detected_events(
                supplemental_events,
                features,
                model,
                feature_config,
                model_config,
                device,
                mode=arguments.mode,
                minimum_direct_probability=(
                    arguments.supplemental_minimum_direct_probability
                ),
                minimum_output_confidence=(
                    arguments.minimum_output_confidence
                ),
                activity_head=activity_head,
                minimum_activity_probability=(
                    arguments.minimum_activity_probability
                ),
                minimum_repeat_probability=(
                    arguments.minimum_repeat_probability
                    if (
                        arguments
                        .supplemental_repeat_minimum_direct_probability
                        is not None
                    )
                    else 1.0
                ),
                repeat_minimum_direct_probability=(
                    arguments
                    .supplemental_repeat_minimum_direct_probability
                    if (
                        arguments
                        .supplemental_repeat_minimum_direct_probability
                        is not None
                    )
                    else arguments.supplemental_minimum_direct_probability
                ),
            )
            predictions = merge_predictions(
                predictions,
                supplemental_predictions,
            )
            pitch_frontend_events.extend(supplemental_events)
            dropped += supplemental_dropped
        fusion_counts = None
        if frame_model is not None and frame_cqt_dataset is not None:
            frame_features = frame_cqt_dataset.load_track_features(
                arguments.start + index - 1
            )
            direct_candidates = direct_onset_candidates(
                frame_features,
                frame_model,
                device,
                onset_threshold=arguments.onset_fusion_onset_threshold,
                activity_threshold=arguments.onset_fusion_activity_threshold,
                minimum_onset_gap=(
                    arguments.onset_fusion_minimum_onset_gap
                ),
                activity_context=arguments.onset_fusion_activity_context,
                activity_delay=arguments.onset_fusion_activity_delay,
                minimum_event_confidence=(
                    arguments.onset_fusion_minimum_event_confidence
                ),
            )
            unmatched_candidates = unmatched_onset_candidates(
                pitch_frontend_events,
                direct_candidates,
                group_tolerance=model_config.group_tolerance,
                match_tolerance=arguments.onset_fusion_match_tolerance,
            )
            fusion_predictions = assign_detected_events(
                unmatched_candidates,
                features,
                model,
                feature_config,
                model_config,
                device,
                mode=arguments.mode,
                minimum_direct_probability=(
                    arguments.onset_fusion_minimum_direct_probability
                ),
                minimum_output_confidence=(
                    arguments.minimum_output_confidence
                ),
            )
            predictions = merge_predictions(
                predictions,
                fusion_predictions,
                onset_tolerance=arguments.onset_fusion_match_tolerance,
            )
            fusion_counts = {
                "direct_candidate_events": len(direct_candidates),
                "unmatched_candidate_events": len(unmatched_candidates),
                "refined_prediction_events": len(fusion_predictions),
            }
        track_result: dict[str, object] = {
            "track_id": annotation.track_id,
            "reference_events": len(annotation.events),
            "predicted_events": len(predictions),
            "dropped_out_of_range_pitch_events": dropped,
            "metrics": evaluate_events(
                annotation.events,
                predictions,
                onset_tolerance=arguments.onset_tolerance,
                beats=annotation.beats,
                excluded_ranges=annotation.excluded_ranges,
            ),
        }
        if fusion_counts is not None:
            track_result["onset_fusion"] = fusion_counts
        track_results.append(track_result)
        print(
            json.dumps(
                {
                    "stage": "onset_position_evaluation",
                    "tracks": index,
                    "total_tracks": len(annotations),
                    "track_id": annotation.track_id,
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    metric_names = [
        "onset",
        "pitch",
        "tablature",
        "repeated_tablature",
    ]
    if all(
        "beat_relative_onset" in result["metrics"]
        for result in track_results
    ):
        metric_names.append("beat_relative_onset")
    report = {
        "split": arguments.split,
        "track_count": len(track_results),
        "pitch_frontend": {
            "name": "spotify-basic-pitch",
            "version": "0.4",
            "onset_threshold": arguments.pitch_onset_threshold,
            "frame_threshold": arguments.pitch_frame_threshold,
            "minimum_note_length": arguments.pitch_minimum_note_length,
        },
        "position_model": {
            **metadata,
            "checkpoint": {
                "name": arguments.checkpoint.name,
                "sha256": sha256_file(arguments.checkpoint),
            },
        },
        "assignment": {
            "mode": arguments.mode,
            "minimum_direct_probability": (
                arguments.minimum_direct_probability
            ),
            "minimum_output_confidence": (
                arguments.minimum_output_confidence
            ),
            "adapter_scale": (
                arguments.adapter_scale
                if arguments.adapter_scale is not None
                else model.adapter_scale
            ),
            "candidate_activity": (
                {
                    **activity_metadata,
                    "checkpoint": {
                        "name": (
                            arguments.candidate_activity_checkpoint.name
                        ),
                        "sha256": sha256_file(
                            arguments.candidate_activity_checkpoint
                        ),
                    },
                    "minimum_activity_probability": (
                        arguments.minimum_activity_probability
                    ),
                    "minimum_repeat_probability": (
                        arguments.minimum_repeat_probability
                    ),
                    "repeat_minimum_direct_probability": (
                        arguments.repeat_minimum_direct_probability
                    ),
                }
                if arguments.candidate_activity_checkpoint is not None
                else None
            ),
            "supplemental": (
                {
                    "onset_threshold": (
                        arguments.supplemental_pitch_onset_threshold
                    ),
                    "frame_threshold": (
                        arguments.supplemental_pitch_frame_threshold
                        if (
                            arguments.supplemental_pitch_frame_threshold
                            is not None
                        )
                        else arguments.pitch_frame_threshold
                    ),
                    "minimum_note_length": (
                        arguments.supplemental_pitch_minimum_note_length
                        if (
                            arguments.supplemental_pitch_minimum_note_length
                            is not None
                        )
                        else arguments.pitch_minimum_note_length
                    ),
                    "minimum_direct_probability": (
                        arguments.supplemental_minimum_direct_probability
                    ),
                    "repeat_minimum_direct_probability": (
                        arguments
                        .supplemental_repeat_minimum_direct_probability
                    ),
                }
                if arguments.supplemental_pitch_onset_threshold is not None
                else None
            ),
            "onset_fusion": (
                {
                    "status": "experimental",
                    "candidate_model": {
                        **frame_model.metadata,
                        "checkpoint": {
                            "name": arguments.onset_fusion_checkpoint.name,
                            "sha256": sha256_file(
                                arguments.onset_fusion_checkpoint
                            ),
                        },
                    },
                    "decoder": {
                        "onset_threshold": (
                            arguments.onset_fusion_onset_threshold
                        ),
                        "activity_threshold": (
                            arguments.onset_fusion_activity_threshold
                        ),
                        "minimum_onset_gap": (
                            arguments.onset_fusion_minimum_onset_gap
                        ),
                        "activity_context": (
                            arguments.onset_fusion_activity_context
                        ),
                        "activity_delay": (
                            arguments.onset_fusion_activity_delay
                        ),
                        "minimum_event_confidence": (
                            arguments.onset_fusion_minimum_event_confidence
                        ),
                    },
                    "match_tolerance": (
                        arguments.onset_fusion_match_tolerance
                    ),
                    "minimum_direct_probability": (
                        arguments.onset_fusion_minimum_direct_probability
                    ),
                    "totals": {
                        name: sum(
                            int(result["onset_fusion"][name])
                            for result in track_results
                        )
                        for name in (
                            "direct_candidate_events",
                            "unmatched_candidate_events",
                            "refined_prediction_events",
                        )
                    },
                }
                if (
                    frame_model is not None
                    and arguments.onset_fusion_checkpoint is not None
                )
                else None
            ),
        },
        "aggregate": {
            metric_name: aggregate(track_results, metric_name)
            for metric_name in metric_names
        }
        | {
            "decoder_errors": aggregate_decoder_errors(track_results),
        },
        "tracks": track_results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
