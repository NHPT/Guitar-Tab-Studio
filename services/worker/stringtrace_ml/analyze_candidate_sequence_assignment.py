from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from .candidate_sequence import (
    CandidateSequenceGroup,
    candidate_group_structured_objective,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .candidate_margin import (
    calibration_partition_digest,
    stratified_calibration_partition,
)
from .data_sources import manifest_identity, sha256_file
from .model import PITCH_MIN
from .onset_position import OnsetPositionDataset
from .schema import TrackAnnotation, load_manifest
from .train import choose_device
from .train_candidate_sequence import load_position_model
from .train_candidate_sequence_activity import (
    refine_groups_with_string_sequence,
)


def _empty_counts() -> Counter[str]:
    return Counter(
        {
            "groups": 0,
            "groups_with_activity_accepted": 0,
            "groups_with_multiple_activity_accepted": 0,
            "groups_with_independent_string_collision": 0,
            "candidates": 0,
            "activity_accepted": 0,
            "accepted_reference_active": 0,
            "accepted_reference_inactive": 0,
            "correct_string": 0,
            "wrong_string": 0,
            "repeat_reference": 0,
            "repeat_correct_string": 0,
            "assignment_changed_from_independent_top1": 0,
            "assignment_change_corrected": 0,
            "assignment_change_harmed": 0,
        }
    )


def _finalize_counts(counts: Counter[str]) -> dict[str, float | int]:
    active = counts["accepted_reference_active"]
    repeat = counts["repeat_reference"]
    result: dict[str, float | int] = dict(sorted(counts.items()))
    result["accepted_string_accuracy"] = (
        counts["correct_string"] / max(1, active)
    )
    result["repeat_string_accuracy"] = (
        counts["repeat_correct_string"] / max(1, repeat)
    )
    return result


def assignment_audit(
    groups: list[CandidateSequenceGroup],
    annotations: list[TrackAnnotation],
) -> dict[str, object]:
    total = _empty_counts()
    by_dataset: dict[str, Counter[str]] = {}
    by_accepted_group_size: dict[str, Counter[str]] = {}
    wrong_string_pairs: Counter[str] = Counter()
    wrong_pitch_string_pairs: Counter[str] = Counter()

    for group in groups:
        annotation = annotations[group.annotation_index]
        dataset = str(annotation.provenance.get("dataset", "unknown"))
        dataset_counts = by_dataset.setdefault(dataset, _empty_counts())
        for counts in (total, dataset_counts):
            counts["groups"] += 1
            counts["candidates"] += group.pitches.shape[0]

        accepted_indexes = torch.nonzero(
            group.base_activity_logits >= 0,
            as_tuple=False,
        ).flatten()
        if not accepted_indexes.numel():
            continue
        accepted_list = accepted_indexes.tolist()
        accepted_size = len(accepted_list)
        size_key = str(min(accepted_size, 7))
        size_counts = by_accepted_group_size.setdefault(
            size_key,
            _empty_counts(),
        )
        for counts in (total, dataset_counts, size_counts):
            counts["groups"] += int(counts is size_counts)
            counts["groups_with_activity_accepted"] += 1
            counts["groups_with_multiple_activity_accepted"] += int(
                accepted_size > 1
            )
            counts["activity_accepted"] += accepted_size

        logits = group.base_string_logits.masked_fill(
            ~group.valid_strings,
            -torch.inf,
        )
        independent = logits.argmax(dim=1)
        accepted_independent = independent[accepted_indexes]
        has_collision = (
            len(set(accepted_independent.tolist())) < accepted_size
        )
        if has_collision:
            for counts in (total, dataset_counts, size_counts):
                counts["groups_with_independent_string_collision"] += 1

        _loss, assigned_strings, _used_unique = (
            candidate_group_structured_objective(
                logits[accepted_indexes],
                group.valid_strings[accepted_indexes],
                torch.zeros_like(group.string_targets[accepted_indexes]),
                torch.zeros(accepted_size, dtype=torch.bool),
            )
        )
        for local_index, predicted_string_value in enumerate(
            assigned_strings.tolist()
        ):
            candidate_index = accepted_list[local_index]
            predicted_string = int(predicted_string_value)
            reference_mask = group.string_targets[candidate_index] > 0
            reference_active = bool(reference_mask.any())
            independent_string = int(independent[candidate_index])
            independent_correct = (
                reference_active
                and bool(reference_mask[independent_string])
            )
            assigned_correct = (
                reference_active
                and bool(reference_mask[predicted_string])
            )
            repeated = bool(group.repeat_targets[candidate_index] > 0)
            changed = predicted_string != independent_string
            for counts in (total, dataset_counts, size_counts):
                counts["accepted_reference_active"] += int(reference_active)
                counts["accepted_reference_inactive"] += int(
                    not reference_active
                )
                counts["correct_string"] += int(assigned_correct)
                counts["wrong_string"] += int(
                    reference_active and not assigned_correct
                )
                counts["repeat_reference"] += int(
                    reference_active and repeated
                )
                counts["repeat_correct_string"] += int(
                    assigned_correct and repeated
                )
                counts[
                    "assignment_changed_from_independent_top1"
                ] += int(changed)
                counts["assignment_change_corrected"] += int(
                    changed and assigned_correct and not independent_correct
                )
                counts["assignment_change_harmed"] += int(
                    changed and independent_correct and not assigned_correct
                )
            if reference_active and not assigned_correct:
                reference_string = int(
                    group.string_targets[candidate_index].argmax()
                )
                wrong_string_pairs[
                    f"{reference_string + 1}->{predicted_string + 1}"
                ] += 1
                pitch = int(group.pitches[candidate_index]) + PITCH_MIN
                wrong_pitch_string_pairs[
                    (
                        f"{pitch}:"
                        f"{reference_string + 1}->{predicted_string + 1}"
                    )
                ] += 1

    return {
        "aggregate": _finalize_counts(total),
        "by_dataset": {
            dataset: _finalize_counts(counts)
            for dataset, counts in sorted(by_dataset.items())
        },
        "by_activity_accepted_group_size": {
            ("7+" if size == "7" else size): _finalize_counts(counts)
            for size, counts in sorted(
                by_accepted_group_size.items(),
                key=lambda item: int(item[0]),
            )
        },
        "top_wrong_string_pairs": dict(
            wrong_string_pairs.most_common(20)
        ),
        "top_wrong_pitch_string_pairs": dict(
            wrong_pitch_string_pairs.most_common(30)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit full-group candidate string assignments on training data."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--string-sequence-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--repeat-interval", type=float, default=0.75)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-seed", type=int, default=20261122)
    parser.add_argument(
        "--stratify-by-group-dataset",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--positive-only-dataset",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--candidate-onset-threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--candidate-frame-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--candidate-minimum-note-length",
        type=float,
        default=70.0,
    )
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    for name in ("precompute_batch_size", "inference_batch_size"):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "candidate_onset_threshold",
        "candidate_frame_threshold",
    ):
        value = getattr(arguments, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in 0..1")
    for name in ("candidate_minimum_note_length", "repeat_interval"):
        value = getattr(arguments, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        not math.isfinite(arguments.calibration_fraction)
        or not 0 < arguments.calibration_fraction < 1
    ):
        parser.error("--calibration-fraction must be between zero and one")
    stratified_datasets = set(arguments.stratify_by_group_dataset)
    if not stratified_datasets:
        stratified_datasets = {"GuitarSet"}

    device = choose_device(arguments.device)
    (
        position_model,
        feature_config,
        position_config,
        _position_metadata,
    ) = load_position_model(arguments.position_checkpoint, device)
    string_model, string_metadata = load_candidate_sequence_model(
        arguments.string_sequence_checkpoint,
        arguments.position_checkpoint,
        position_config,
        feature_config,
        device,
    )
    dataset = OnsetPositionDataset(
        arguments.manifest,
        split="train",
        feature_config=feature_config,
        config=position_config,
        feature_cache_directory=arguments.feature_cache,
        candidate_cache_directory=arguments.candidate_cache,
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=(
            arguments.candidate_minimum_note_length
        ),
        positive_only_datasets=set(arguments.positive_only_dataset),
    )
    cache_report = dataset.prepare_feature_cache()
    (
        selected_annotation_indexes,
        selected_keys,
        partition_counts,
    ) = stratified_calibration_partition(
        dataset.annotations,
        fraction=arguments.calibration_fraction,
        seed=arguments.calibration_seed,
        stratify_by_group_datasets=stratified_datasets,
    )
    selected_group_indexes = [
        index
        for index, group in enumerate(dataset.groups)
        if group.annotation_index in selected_annotation_indexes
    ]
    all_group_count = len(dataset)
    dataset.groups = [
        dataset.groups[index] for index in selected_group_indexes
    ]
    dataset.candidate_temporal_contexts = [
        dataset.candidate_temporal_contexts[index]
        for index in selected_group_indexes
    ]
    print(
        json.dumps(
            {
                "stage": "precompute_train_groups",
                "all_groups": all_group_count,
                "groups": len(dataset),
                "selected_annotations": len(selected_annotation_indexes),
                "device": str(device),
            }
        ),
        flush=True,
    )
    base_groups = precompute_candidate_sequence_groups(
        dataset,
        position_model,
        device,
        batch_size=arguments.precompute_batch_size,
        repeat_interval=arguments.repeat_interval,
    )
    print(
        json.dumps(
            {
                "stage": "apply_string_sequence",
                "groups": len(base_groups),
            }
        ),
        flush=True,
    )
    refined_groups = refine_groups_with_string_sequence(
        base_groups,
        string_model,
        device,
        batch_size=arguments.inference_batch_size,
    )
    base_audit = assignment_audit(base_groups, dataset.annotations)
    refined_audit = assignment_audit(refined_groups, dataset.annotations)
    base_aggregate = dict(base_audit["aggregate"])
    refined_aggregate = dict(refined_audit["aggregate"])
    report = {
        "schema_version": 1,
        "kind": "candidate-sequence-assignment-audit",
        "version": "candidate-sequence-string-v2-input-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_only": True,
        "governed_validation_used": False,
        "cross_source_validation_used": False,
        "manifest": manifest_identity(
            arguments.manifest.resolve(),
            load_manifest(arguments.manifest.resolve()),
            purpose="training",
        ),
        "position_checkpoint": {
            "name": arguments.position_checkpoint.name,
            "sha256": sha256_file(arguments.position_checkpoint),
        },
        "string_sequence_checkpoint": {
            "name": arguments.string_sequence_checkpoint.name,
            "sha256": sha256_file(
                arguments.string_sequence_checkpoint
            ),
            "metadata_version": string_metadata.get("version"),
        },
        "candidate_frontend": {
            "kind": "spotify-basic-pitch-0.4",
            "onset_threshold": arguments.candidate_onset_threshold,
            "frame_threshold": arguments.candidate_frame_threshold,
            "minimum_note_length": (
                arguments.candidate_minimum_note_length
            ),
        },
        "positive_only_datasets": sorted(
            set(arguments.positive_only_dataset)
        ),
        "partition": {
            "fraction": arguments.calibration_fraction,
            "seed": arguments.calibration_seed,
            "stratify_by_group_datasets": sorted(stratified_datasets),
            "selected_key_sha256": calibration_partition_digest(
                selected_keys
            ),
            "selected_annotations": len(selected_annotation_indexes),
            "datasets": partition_counts,
        },
        "cache": cache_report,
        "baseline": base_audit,
        "sequence_v1": refined_audit,
        "sequence_v1_delta": {
            "correct_string": (
                int(refined_aggregate["correct_string"])
                - int(base_aggregate["correct_string"])
            ),
            "wrong_string": (
                int(refined_aggregate["wrong_string"])
                - int(base_aggregate["wrong_string"])
            ),
            "repeat_correct_string": (
                int(refined_aggregate["repeat_correct_string"])
                - int(base_aggregate["repeat_correct_string"])
            ),
            "accepted_string_accuracy": (
                float(refined_aggregate["accepted_string_accuracy"])
                - float(base_aggregate["accepted_string_accuracy"])
            ),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "baseline": base_audit["aggregate"],
                "sequence_v1": refined_audit["aggregate"],
                "delta": report["sequence_v1_delta"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
