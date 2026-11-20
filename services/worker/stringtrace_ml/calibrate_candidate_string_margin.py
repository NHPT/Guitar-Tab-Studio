from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .candidate_margin import (
    calibration_partition_digest,
    calibration_partition_keys,
    collect_margin_calibration_values,
    select_margin_threshold,
    stratified_calibration_partition,
)
from .candidate_sequence import (
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import manifest_identity, sha256_file
from .onset_position import OnsetPositionDataset
from .schema import load_manifest
from .train import choose_device
from .train_candidate_sequence import load_position_model
from .train_candidate_sequence_activity import (
    group_target_counts,
    refine_groups_with_string_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate one candidate string probability-margin rejection "
            "threshold on a deterministic training-only partition."
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
    parser.add_argument(
        "--version",
        choices=(
            "candidate-sequence-margin-v1",
            "candidate-sequence-margin-v2",
        ),
        default="candidate-sequence-margin-v1",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-seed", type=int)
    parser.add_argument(
        "--stratify-by-group-dataset",
        action="append",
        default=[],
        help=(
            "Select tracks independently inside every group for this "
            "dataset instead of selecting whole groups."
        ),
    )
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--repeat-interval", type=float, default=0.75)
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
    if arguments.calibration_seed is None:
        arguments.calibration_seed = (
            20261121
            if arguments.version == "candidate-sequence-margin-v2"
            else 20261120
        )
    if (
        arguments.version == "candidate-sequence-margin-v2"
        and not arguments.stratify_by_group_dataset
    ):
        arguments.stratify_by_group_dataset = ["GuitarSet"]

    if (
        not math.isfinite(arguments.calibration_fraction)
        or not 0 < arguments.calibration_fraction < 1
    ):
        parser.error("--calibration-fraction must be between zero and one")
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

    device = choose_device(arguments.device)
    (
        position_model,
        feature_config,
        position_config,
        _position_metadata,
    ) = load_position_model(arguments.position_checkpoint, device)
    string_model, _string_metadata = load_candidate_sequence_model(
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
    if arguments.version == "candidate-sequence-margin-v2":
        (
            selected_annotation_indexes,
            selected_keys,
            partition_counts,
        ) = stratified_calibration_partition(
            dataset.annotations,
            fraction=arguments.calibration_fraction,
            seed=arguments.calibration_seed,
            stratify_by_group_datasets=set(
                arguments.stratify_by_group_dataset
            ),
        )
        partition_unit = (
            "track_within_group_for_stratified_datasets_else_"
            "dataset_and_group_id"
        )
        partition_method = "smallest_sha256_rank_per_partition"
        assignment_order = "post_assignment"
        calibration_match_mode = "activity_then_full_group_assignment"
    else:
        selected_keys, partition_counts = calibration_partition_keys(
            dataset.annotations,
            fraction=arguments.calibration_fraction,
            seed=arguments.calibration_seed,
        )
        selected_annotation_indexes = {
            index
            for index, annotation in enumerate(dataset.annotations)
            if (
                str(annotation.provenance.get("dataset", "unknown")),
                annotation.group_id,
            )
            in selected_keys
        }
        partition_unit = "dataset_and_group_id"
        partition_method = "smallest_sha256_rank_per_dataset"
        assignment_order = "pre_assignment"
        calibration_match_mode = "independent_top1"
    all_group_count = len(dataset)
    selected_group_indexes = [
        index
        for index, group in enumerate(dataset.groups)
        if group.annotation_index in selected_annotation_indexes
    ]
    dataset.groups = [
        dataset.groups[index] for index in selected_group_indexes
    ]
    dataset.candidate_temporal_contexts = [
        dataset.candidate_temporal_contexts[index]
        for index in selected_group_indexes
    ]
    if not dataset.groups:
        raise RuntimeError("Calibration partition contains no candidate groups")
    print(
        json.dumps(
            {
                "stage": "precompute_train_groups",
                "all_groups": all_group_count,
                "calibration_groups": len(dataset),
                "selected_annotations": len(
                    selected_annotation_indexes
                ),
                "device": str(device),
            }
        ),
        flush=True,
    )
    groups = precompute_candidate_sequence_groups(
        dataset,
        position_model,
        device,
        batch_size=arguments.precompute_batch_size,
        repeat_interval=arguments.repeat_interval,
    )
    print(
        json.dumps(
            {
                "stage": "apply_frozen_string_sequence",
                "calibration_groups": len(groups),
            }
        ),
        flush=True,
    )
    calibration_groups = refine_groups_with_string_sequence(
        groups,
        string_model,
        device,
        batch_size=arguments.inference_batch_size,
    )
    selection = select_margin_threshold(
        collect_margin_calibration_values(
            calibration_groups,
            assignment_order=(
                "post_assignment"
                if arguments.version == "candidate-sequence-margin-v2"
                else "independent_top1"
            ),
        )
    )
    annotations = load_manifest(arguments.manifest.resolve())
    selected = dict(selection["selected"])
    report = {
        "schema_version": (
            2
            if arguments.version == "candidate-sequence-margin-v2"
            else 1
        ),
        "kind": "candidate-string-margin-calibration",
        "version": arguments.version,
        "assignment_order": assignment_order,
        "calibration_match_mode": calibration_match_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_only": True,
        "governed_validation_used": False,
        "cross_source_used": False,
        "manifest": manifest_identity(
            arguments.manifest.resolve(),
            annotations,
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
        },
        "candidate_frontend": {
            "kind": "spotify-basic-pitch-0.4",
            "onset_threshold": arguments.candidate_onset_threshold,
            "frame_threshold": arguments.candidate_frame_threshold,
            "minimum_note_length": (
                arguments.candidate_minimum_note_length
            ),
        },
        "partition": {
            "unit": partition_unit,
            "method": partition_method,
            "fraction": arguments.calibration_fraction,
            "seed": arguments.calibration_seed,
            "stratify_by_group_datasets": sorted(
                set(arguments.stratify_by_group_dataset)
            ),
            "selected_key_sha256": calibration_partition_digest(
                selected_keys
            ),
            "datasets": partition_counts,
            "selected_annotations": len(selected_annotation_indexes),
        },
        "positive_only_datasets": sorted(
            set(arguments.positive_only_dataset)
        ),
        "filtered_negative_groups": dataset.filtered_negative_groups,
        "calibration_targets": group_target_counts(calibration_groups),
        "repeat_interval": arguments.repeat_interval,
        "cache": cache_report,
        **selection,
        "selected_threshold": float(selected["threshold"]),
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
                "calibration_gate_passed": report[
                    "calibration_gate_passed"
                ],
                "selected_threshold": report["selected_threshold"],
                "baseline": report["baseline"],
                "selected": report["selected"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
