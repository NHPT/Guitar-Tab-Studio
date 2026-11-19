from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from .candidate_sequence import (
    CandidateSequenceStringModel,
    infer_candidate_sequence_residuals,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import sha256_file
from .features import FeatureConfig
from .model import OPEN_STRING_PITCHES, PITCH_MIN
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    OnsetPositionNet,
    candidate_position_queries,
    candidate_positions,
    group_event_indexes,
)
from .train import choose_device


@dataclass(frozen=True)
class EventContext:
    previous_group_gap: float | None
    next_group_gap: float | None
    previous_string_distance: int | None
    next_string_distance: int | None
    previous_same_pitch_gap: float | None
    next_same_pitch_gap: float | None
    previous_same_pitch_string: int | None
    next_same_pitch_string: int | None


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[OnsetPositionNet, FeatureConfig, OnsetPositionConfig]:
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
    return model, feature_config, model_config


def _event_contexts(
    annotation: object,
    *,
    group_tolerance: float,
) -> dict[int, EventContext]:
    events = annotation.events
    groups = group_event_indexes(
        [event.onset for event in events],
        [event.string for event in events],
        tolerance=group_tolerance,
    )
    group_by_event = {
        event_index: group_index
        for group_index, group in enumerate(groups)
        for event_index in group
    }
    events_by_pitch: dict[int, list[int]] = defaultdict(list)
    for event_index, event in enumerate(events):
        pitch = OPEN_STRING_PITCHES[event.string - 1] + event.fret
        events_by_pitch[pitch].append(event_index)
    same_pitch_neighbors: dict[
        int,
        tuple[int | None, int | None],
    ] = {}
    for pitch_events in events_by_pitch.values():
        ordered = sorted(
            pitch_events,
            key=lambda event_index: events[event_index].onset,
        )
        for offset, event_index in enumerate(ordered):
            current_onset = events[event_index].onset
            previous = next(
                (
                    candidate
                    for candidate in reversed(ordered[:offset])
                    if events[candidate].onset < current_onset
                ),
                None,
            )
            following = next(
                (
                    candidate
                    for candidate in ordered[offset + 1 :]
                    if events[candidate].onset > current_onset
                ),
                None,
            )
            same_pitch_neighbors[event_index] = (previous, following)

    result: dict[int, EventContext] = {}
    for event_index, event in enumerate(events):
        group_index = group_by_event[event_index]
        current_onset = event.onset
        current_string = event.string
        previous_group = groups[group_index - 1] if group_index > 0 else None
        next_group = (
            groups[group_index + 1]
            if group_index + 1 < len(groups)
            else None
        )
        previous_event, next_event = same_pitch_neighbors[event_index]
        result[event_index] = EventContext(
            previous_group_gap=(
                abs(current_onset - events[previous_group[0]].onset)
                if previous_group is not None
                else None
            ),
            next_group_gap=(
                abs(current_onset - events[next_group[0]].onset)
                if next_group is not None
                else None
            ),
            previous_string_distance=(
                min(
                    abs(current_string - events[index].string)
                    for index in previous_group
                )
                if previous_group is not None
                else None
            ),
            next_string_distance=(
                min(
                    abs(current_string - events[index].string)
                    for index in next_group
                )
                if next_group is not None
                else None
            ),
            previous_same_pitch_gap=(
                event.onset - events[previous_event].onset
                if previous_event is not None
                else None
            ),
            next_same_pitch_gap=(
                events[next_event].onset - event.onset
                if next_event is not None
                else None
            ),
            previous_same_pitch_string=(
                events[previous_event].string
                if previous_event is not None
                else None
            ),
            next_same_pitch_string=(
                events[next_event].string
                if next_event is not None
                else None
            ),
        )
    return result


def _gap_bucket(value: float | None) -> str:
    if value is None:
        return "none"
    if value <= 0.25:
        return "0-0.25"
    if value <= 0.75:
        return "0.25-0.75"
    if value <= 2.0:
        return "0.75-2.0"
    return "2.0+"


def _fret_bucket(fret: int) -> str:
    if fret == 0:
        return "open"
    if fret <= 4:
        return "1-4"
    if fret <= 8:
        return "5-8"
    if fret <= 12:
        return "9-12"
    if fret <= 18:
        return "13-18"
    return "19-24"


def _distance_bucket(value: int | None) -> str:
    if value is None:
        return "none"
    if value <= 1:
        return str(value)
    return "2+"


def _update(
    accumulator: dict[str, dict[str, int]],
    key: str,
    *,
    correct: bool,
) -> None:
    bucket = accumulator.setdefault(key, {"count": 0, "correct": 0})
    bucket["count"] += 1
    bucket["correct"] += int(correct)


def _finalize(
    accumulator: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            **value,
            "errors": value["count"] - value["correct"],
            "accuracy": value["correct"] / max(1, value["count"]),
        }
        for key, value in sorted(accumulator.items())
    }


def analyze(
    dataset: OnsetPositionDataset,
    model: OnsetPositionNet,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    contexts = [
        _event_contexts(
            annotation,
            group_tolerance=model.config.group_tolerance,
        )
        for annotation in dataset.annotations
    ]
    buckets: dict[str, dict[str, dict[str, int]]] = {
        name: {}
        for name in (
            "playable_positions",
            "true_string",
            "true_predicted_string",
            "fret",
            "previous_group_ioi",
            "next_group_ioi",
            "previous_string_movement",
            "next_string_movement",
            "previous_same_pitch",
            "next_same_pitch",
        )
    }
    total = 0
    correct_total = 0
    all_target_total = 0
    all_target_correct = 0
    multi_target_candidates = 0
    group_offset = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pitch_masks = batch["pitch_mask"].to(device)
            encoded = model.encode(features, pitch_masks)
            (
                query_batch_indexes,
                query_pitch_indexes,
                targets,
            ) = candidate_position_queries(
                batch["event_pitches"].to(device),
                batch["event_strings"].to(device),
                pitch_masks,
                categorical=True,
            )
            logits = model.classify_candidate_positions(
                encoded[query_batch_indexes],
                query_pitch_indexes,
                features=features[query_batch_indexes],
                candidate_confidences=pitch_masks[
                    query_batch_indexes,
                    query_pitch_indexes,
                ],
                temporal_context=(
                    batch["candidate_temporal_context"]
                    .to(device)[
                        query_batch_indexes,
                        query_pitch_indexes,
                    ]
                    if model.config.candidate_temporal_string_residual
                    else None
                ),
            )
            predictions = logits[:, 1:].argmax(dim=1) + 1
            for row in range(targets.shape[0]):
                true_strings = (
                    torch.nonzero(
                        targets[row, 1:] > 0,
                        as_tuple=False,
                    ).flatten()
                    + 1
                ).tolist()
                if not true_strings:
                    continue
                predicted_string = int(predictions[row])
                all_target_total += 1
                all_target_correct += int(
                    predicted_string in true_strings
                )
                if len(true_strings) != 1:
                    multi_target_candidates += 1
                    continue
                true_string = int(true_strings[0])
                correct = predicted_string == true_string
                local_group_index = int(query_batch_indexes[row])
                group_index = group_offset + local_group_index
                group = dataset.groups[group_index]
                annotation = dataset.annotations[group.annotation_index]
                pitch = PITCH_MIN + int(query_pitch_indexes[row])
                matching_indexes = [
                    event_index
                    for event_index in group.event_indexes
                    if (
                        annotation.events[event_index].string == true_string
                        and OPEN_STRING_PITCHES[true_string - 1]
                        + annotation.events[event_index].fret
                        == pitch
                    )
                ]
                if not matching_indexes:
                    raise RuntimeError(
                        "Could not map candidate target to reference event"
                    )
                event_index = matching_indexes[0]
                event = annotation.events[event_index]
                context = contexts[group.annotation_index][event_index]
                total += 1
                correct_total += int(correct)
                _update(
                    buckets["playable_positions"],
                    str(len(candidate_positions(pitch))),
                    correct=correct,
                )
                _update(
                    buckets["true_string"],
                    str(true_string),
                    correct=correct,
                )
                _update(
                    buckets["true_predicted_string"],
                    f"{true_string}->{predicted_string}",
                    correct=correct,
                )
                _update(
                    buckets["fret"],
                    _fret_bucket(event.fret),
                    correct=correct,
                )
                _update(
                    buckets["previous_group_ioi"],
                    _gap_bucket(context.previous_group_gap),
                    correct=correct,
                )
                _update(
                    buckets["next_group_ioi"],
                    _gap_bucket(context.next_group_gap),
                    correct=correct,
                )
                _update(
                    buckets["previous_string_movement"],
                    _distance_bucket(context.previous_string_distance),
                    correct=correct,
                )
                _update(
                    buckets["next_string_movement"],
                    _distance_bucket(context.next_string_distance),
                    correct=correct,
                )
                for direction, gap, string in (
                    (
                        "previous_same_pitch",
                        context.previous_same_pitch_gap,
                        context.previous_same_pitch_string,
                    ),
                    (
                        "next_same_pitch",
                        context.next_same_pitch_gap,
                        context.next_same_pitch_string,
                    ),
                ):
                    relationship = (
                        "none"
                        if string is None
                        else (
                            "same_string"
                            if string == true_string
                            else "different_string"
                        )
                    )
                    _update(
                        buckets[direction],
                        f"{_gap_bucket(gap)}:{relationship}",
                        correct=correct,
                    )
            group_offset += features.shape[0]
    return {
        "groups": len(dataset),
        "active_candidates": all_target_total,
        "single_target_active_candidates": total,
        "multi_target_active_candidates": multi_target_candidates,
        "correct": all_target_correct,
        "errors": all_target_total - all_target_correct,
        "oracle_string_accuracy": (
            all_target_correct / max(1, all_target_total)
        ),
        "single_target_correct": correct_total,
        "single_target_errors": total - correct_total,
        "single_target_oracle_string_accuracy": (
            correct_total / max(1, total)
        ),
        "breakdowns": {
            name: _finalize(values)
            for name, values in buckets.items()
        },
    }


def analyze_sequence(
    dataset: OnsetPositionDataset,
    position_model: OnsetPositionNet,
    sequence_model: CandidateSequenceStringModel,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, object]:
    groups = precompute_candidate_sequence_groups(
        dataset,
        position_model,
        device,
        batch_size=batch_size,
    )
    groups_by_annotation: dict[int, list[int]] = defaultdict(list)
    for group_index, group in enumerate(groups):
        groups_by_annotation[group.annotation_index].append(group_index)
    residuals: list[torch.Tensor | None] = [None] * len(groups)
    for group_indexes in groups_by_annotation.values():
        annotation_groups = [groups[index] for index in group_indexes]
        annotation_residuals = infer_candidate_sequence_residuals(
            sequence_model,
            annotation_groups,
            device,
            batch_size=batch_size,
        )
        for group_index, residual in zip(
            group_indexes,
            annotation_residuals,
        ):
            residuals[group_index] = residual

    correct = 0
    total = 0
    multi_target_candidates = 0
    for group, residual in zip(groups, residuals):
        if residual is None:
            raise RuntimeError("candidate sequence residual is missing")
        active = group.string_targets.sum(dim=1) > 0
        if not bool(active.any()):
            continue
        refined = functional.log_softmax(
            (
                group.base_string_logits + residual
            ).masked_fill(~group.valid_strings, -30.0),
            dim=1,
        )
        predictions = refined.argmax(dim=1)
        matches = torch.gather(
            group.string_targets,
            1,
            predictions.unsqueeze(1),
        ).squeeze(1) > 0
        correct += int((matches & active).sum())
        total += int(active.sum())
        multi_target_candidates += int(
            (
                active
                & ((group.string_targets > 0).sum(dim=1) > 1)
            ).sum()
        )
    return {
        "groups": len(groups),
        "active_candidates": total,
        "multi_target_active_candidates": multi_target_candidates,
        "correct": correct,
        "errors": total - correct,
        "oracle_string_accuracy": correct / max(1, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze candidate string errors against temporal performance "
            "context without reading a frozen promotion gate."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-checkpoint", type=Path)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-onset-threshold", type=float, default=0.35)
    parser.add_argument("--candidate-frame-threshold", type=float, default=0.25)
    parser.add_argument(
        "--candidate-minimum-note-length",
        type=float,
        default=70.0,
    )
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")

    device = choose_device(arguments.device)
    model, feature_config, model_config = _load_model(
        arguments.checkpoint,
        device,
    )
    sequence_model = None
    sequence_metadata: dict[str, object] = {}
    if arguments.sequence_checkpoint is not None:
        sequence_model, sequence_metadata = load_candidate_sequence_model(
            arguments.sequence_checkpoint,
            arguments.checkpoint,
            model_config,
            feature_config,
            device,
        )
    dataset = OnsetPositionDataset(
        arguments.manifest,
        split=arguments.split,
        feature_config=feature_config,
        config=model_config,
        feature_cache_directory=arguments.feature_cache,
        candidate_cache_directory=arguments.candidate_cache,
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=(
            arguments.candidate_minimum_note_length
        ),
    )
    cache_result = dataset.prepare_feature_cache()
    report = {
        "manifest": arguments.manifest.name,
        "checkpoint": arguments.checkpoint.name,
        "split": arguments.split,
        "candidate_frontend": {
            "onset_threshold": arguments.candidate_onset_threshold,
            "frame_threshold": arguments.candidate_frame_threshold,
            "minimum_note_length": (
                arguments.candidate_minimum_note_length
            ),
        },
        "feature_cache": cache_result,
        "sequence_model": (
            {
                **sequence_metadata,
                "checkpoint": {
                    "name": arguments.sequence_checkpoint.name,
                    "sha256": sha256_file(
                        arguments.sequence_checkpoint
                    ),
                },
            }
            if arguments.sequence_checkpoint is not None
            else None
        ),
        "analysis": (
            analyze_sequence(
                dataset,
                model,
                sequence_model,
                device,
                batch_size=arguments.batch_size,
            )
            if sequence_model is not None
            else analyze(
                dataset,
                model,
                device,
                batch_size=arguments.batch_size,
            )
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "device": str(device),
                **report["analysis"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
