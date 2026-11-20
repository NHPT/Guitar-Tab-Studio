from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, WeightedRandomSampler

from .candidate_sequence import (
    CandidateSequenceActivityModel,
    CandidateSequenceGroup,
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    candidate_sequence_activity_loss,
    collate_candidate_sequence_windows,
    infer_candidate_sequence_activity_residuals,
    infer_candidate_sequence_residuals,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import manifest_identity, sha256_file
from .onset_position import OnsetPositionDataset
from .schema import TrackAnnotation, load_manifest
from .train import choose_device
from .train_candidate_activity import parse_dataset_sampling_weights
from .train_candidate_sequence import load_position_model, move_batch


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    return (
        2
        * true_positive
        / max(1, 2 * true_positive + false_positive + false_negative)
    )


def _metrics(counts: dict[str, int], loss: float) -> dict[str, float | int]:
    repeat_total = (
        counts["repeat_true_positive"] + counts["repeat_false_negative"]
    )
    return {
        "loss": loss,
        "activity_f1": _f1(
            counts["true_positive"],
            counts["false_positive"],
            counts["false_negative"],
        ),
        "activity_true_positive": counts["true_positive"],
        "activity_false_positive": counts["false_positive"],
        "activity_false_negative": counts["false_negative"],
        "repeat_recall": counts["repeat_true_positive"] / max(1, repeat_total),
        "repeat_true_positive": counts["repeat_true_positive"],
        "repeat_false_negative": counts["repeat_false_negative"],
        "candidates": counts["candidates"],
    }


def _add_counts(
    destination: dict[str, int],
    source: dict[str, int],
) -> None:
    for key, value in source.items():
        destination[key] += value


def refine_groups_with_string_sequence(
    groups: list[CandidateSequenceGroup],
    string_model: CandidateSequenceStringModel,
    device: torch.device,
    *,
    batch_size: int,
) -> list[CandidateSequenceGroup]:
    groups_by_annotation: dict[int, list[tuple[int, CandidateSequenceGroup]]] = (
        defaultdict(list)
    )
    for group_index, group in enumerate(groups):
        groups_by_annotation[group.annotation_index].append(
            (group_index, group)
        )
    refined = list(groups)
    for annotation_groups in groups_by_annotation.values():
        indexes = [group_index for group_index, _group in annotation_groups]
        local_groups = [group for _group_index, group in annotation_groups]
        residuals = infer_candidate_sequence_residuals(
            string_model,
            local_groups,
            device,
            batch_size=batch_size,
        )
        for group_index, group, residual in zip(
            indexes,
            local_groups,
            residuals,
        ):
            refined[group_index] = replace(
                group,
                base_string_logits=functional.log_softmax(
                    (group.base_string_logits + residual).masked_fill(
                        ~group.valid_strings,
                        -30.0,
                    ),
                    dim=1,
                ),
            )
    return refined


def group_target_counts(
    groups: list[CandidateSequenceGroup],
) -> dict[str, int]:
    return {
        "groups": len(groups),
        "candidates": sum(group.pitches.shape[0] for group in groups),
        "active": sum(
            int(group.activity_targets.sum())
            for group in groups
        ),
        "inactive": sum(
            int((group.activity_targets == 0).sum())
            for group in groups
        ),
        "repeated": sum(
            int(group.repeat_targets.sum())
            for group in groups
        ),
    }


def window_sampling_plan(
    windows: CandidateSequenceWindowDataset,
    annotations: list[TrackAnnotation],
    dataset_weights: dict[str, float],
) -> tuple[list[float], dict[str, object]]:
    counts: dict[str, int] = defaultdict(int)
    names: list[str] = []
    for window in windows.windows:
        annotation_index = windows.groups[window[0]].annotation_index
        annotation = annotations[annotation_index]
        dataset = str(
            annotation.provenance.get("dataset", "unknown")
        )
        names.append(dataset)
        counts[dataset] += 1
    weights = [dataset_weights.get(name, 1.0) for name in names]
    weighted_total = sum(
        count * dataset_weights.get(dataset, 1.0)
        for dataset, count in counts.items()
    )
    return weights, {
        "replacement": bool(dataset_weights),
        "num_samples": len(windows),
        "window_counts": dict(sorted(counts.items())),
        "dataset_weights": dict(sorted(dataset_weights.items())),
        "expected_dataset_share": {
            dataset: (
                count * dataset_weights.get(dataset, 1.0)
                / max(weighted_total, 1.0)
            )
            for dataset, count in sorted(counts.items())
        },
    }


def run_training_epoch(
    model: CandidateSequenceActivityModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    activity_positive_weight: float,
    repeat_positive_weight: float,
) -> dict[str, float | int]:
    model.train()
    loss_total = 0.0
    batches = 0
    counts = defaultdict(int)
    for source_batch in loader:
        batch = move_batch(source_batch, device)
        optimizer.zero_grad(set_to_none=True)
        residuals = model(
            batch["pitches"],
            batch["confidences"],
            batch["durations"],
            batch["base_string_logits"],
            batch["valid_strings"],
            batch["base_activity_logits"],
            batch["candidate_mask"],
            batch["group_mask"],
            batch["group_onsets"],
        )
        loss, batch_counts = candidate_sequence_activity_loss(
            batch["base_activity_logits"],
            residuals,
            batch["activity_targets"],
            batch["repeat_targets"],
            batch["candidate_mask"],
            activity_positive_weight=activity_positive_weight,
            repeat_positive_weight=repeat_positive_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_total += float(loss.detach().cpu())
        _add_counts(counts, batch_counts)
        batches += 1
    return _metrics(counts, loss_total / max(1, batches))


def evaluate_groups(
    model: CandidateSequenceActivityModel,
    groups: list[CandidateSequenceGroup],
    device: torch.device,
    *,
    inference_batch_size: int,
    activity_positive_weight: float,
    repeat_positive_weight: float,
) -> dict[str, float | int]:
    groups_by_annotation: dict[int, list[CandidateSequenceGroup]] = (
        defaultdict(list)
    )
    for group in groups:
        groups_by_annotation[group.annotation_index].append(group)
    counts = defaultdict(int)
    weighted_loss_total = 0.0
    candidate_total = 0
    for annotation_groups in groups_by_annotation.values():
        residuals = infer_candidate_sequence_activity_residuals(
            model,
            annotation_groups,
            device,
            batch_size=inference_batch_size,
        )
        for group, residual in zip(annotation_groups, residuals):
            logits = group.base_activity_logits + residual
            targets = group.activity_targets
            weights = torch.where(
                targets > 0,
                torch.full_like(targets, activity_positive_weight),
                torch.ones_like(targets),
            )
            weights = torch.where(
                group.repeat_targets > 0,
                weights * repeat_positive_weight,
                weights,
            )
            losses = functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
            weighted_loss_total += float((losses * weights).sum())
            candidate_total += targets.numel()
            predictions = logits >= 0
            references = targets > 0
            repeated = group.repeat_targets > 0
            counts["candidates"] += targets.numel()
            counts["true_positive"] += int(
                (predictions & references).sum()
            )
            counts["false_positive"] += int(
                (predictions & ~references).sum()
            )
            counts["false_negative"] += int(
                (~predictions & references).sum()
            )
            counts["repeat_true_positive"] += int(
                (predictions & repeated).sum()
            )
            counts["repeat_false_negative"] += int(
                (~predictions & repeated).sum()
            )
    return _metrics(
        counts,
        weighted_loss_total / max(1, candidate_total),
    )


def append_log(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a candidate sequence activity residual above frozen "
            "position/activity and sequence-string checkpoints."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--string-sequence-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--train-candidate-cache", type=Path, required=True)
    parser.add_argument(
        "--validation-candidate-cache",
        type=Path,
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--activity-positive-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--repeat-positive-weight",
        type=float,
        default=4.0,
    )
    parser.add_argument("--repeat-interval", type=float, default=0.75)
    parser.add_argument(
        "--positive-only-dataset",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--dataset-sampling-weight",
        action="append",
        default=[],
        metavar="DATASET=WEIGHT",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("activity_f1", "repeat_recall"),
        default="activity_f1",
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
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    for name in (
        "epochs",
        "batch_size",
        "precompute_batch_size",
        "inference_batch_size",
    ):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "activity_positive_weight"):
        if not math.isfinite(getattr(arguments, name)) or getattr(
            arguments, name
        ) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if (
        not math.isfinite(arguments.repeat_positive_weight)
        or arguments.repeat_positive_weight < 1
    ):
        parser.error("--repeat-positive-weight must be at least one")
    if arguments.repeat_interval <= 0:
        parser.error("--repeat-interval must be positive")
    try:
        dataset_sampling_weights = parse_dataset_sampling_weights(
            arguments.dataset_sampling_weight
        )
    except ValueError as error:
        parser.error(str(error))

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = choose_device(arguments.device)
    (
        position_model,
        feature_config,
        position_config,
        position_metadata,
    ) = load_position_model(arguments.position_checkpoint, device)
    string_model, string_metadata = load_candidate_sequence_model(
        arguments.string_sequence_checkpoint,
        arguments.position_checkpoint,
        position_config,
        feature_config,
        device,
    )
    for parameter in string_model.parameters():
        parameter.requires_grad = False
    sequence_config = string_model.config
    validation_manifest = (
        arguments.validation_manifest
        if arguments.validation_manifest is not None
        else arguments.manifest
    )
    train_dataset = OnsetPositionDataset(
        arguments.manifest,
        split="train",
        feature_config=feature_config,
        config=position_config,
        feature_cache_directory=arguments.feature_cache,
        candidate_cache_directory=arguments.train_candidate_cache,
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=(
            arguments.candidate_minimum_note_length
        ),
        positive_only_datasets=set(arguments.positive_only_dataset),
    )
    validation_dataset = OnsetPositionDataset(
        validation_manifest,
        split="validation",
        feature_config=feature_config,
        config=position_config,
        feature_cache_directory=arguments.feature_cache,
        candidate_cache_directory=arguments.validation_candidate_cache,
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=(
            arguments.candidate_minimum_note_length
        ),
    )
    cache_report = {
        "train": train_dataset.prepare_feature_cache(),
        "validation": validation_dataset.prepare_feature_cache(),
    }
    print(
        json.dumps(
            {
                "stage": "precompute_train_groups",
                "groups": len(train_dataset),
                "device": str(device),
            }
        ),
        flush=True,
    )
    train_groups = precompute_candidate_sequence_groups(
        train_dataset,
        position_model,
        device,
        batch_size=arguments.precompute_batch_size,
        repeat_interval=arguments.repeat_interval,
    )
    validation_groups = precompute_candidate_sequence_groups(
        validation_dataset,
        position_model,
        device,
        batch_size=arguments.precompute_batch_size,
        repeat_interval=arguments.repeat_interval,
    )
    print(
        json.dumps({"stage": "apply_frozen_string_sequence"}),
        flush=True,
    )
    train_groups = refine_groups_with_string_sequence(
        train_groups,
        string_model,
        device,
        batch_size=arguments.inference_batch_size,
    )
    validation_groups = refine_groups_with_string_sequence(
        validation_groups,
        string_model,
        device,
        batch_size=arguments.inference_batch_size,
    )
    train_windows = CandidateSequenceWindowDataset(
        train_groups,
        sequence_config,
    )
    validation_windows = CandidateSequenceWindowDataset(
        validation_groups,
        sequence_config,
    )
    sampling_weights, sampling_plan = window_sampling_plan(
        train_windows,
        train_dataset.annotations,
        dataset_sampling_weights,
    )
    train_loader = DataLoader(
        train_windows,
        batch_size=arguments.batch_size,
        shuffle=not dataset_sampling_weights,
        sampler=(
            WeightedRandomSampler(
                sampling_weights,
                num_samples=len(train_windows),
                replacement=True,
                generator=torch.Generator().manual_seed(arguments.seed),
            )
            if dataset_sampling_weights
            else None
        ),
        generator=torch.Generator().manual_seed(arguments.seed),
        num_workers=0,
        collate_fn=collate_candidate_sequence_windows,
    )
    model = CandidateSequenceActivityModel(sequence_config).to(device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, arguments.epochs),
        eta_min=arguments.learning_rate * 0.05,
    )

    arguments.log.parent.mkdir(parents=True, exist_ok=True)
    arguments.log.write_text("", encoding="utf-8")
    baseline = evaluate_groups(
        model,
        validation_groups,
        device,
        inference_batch_size=arguments.inference_batch_size,
        activity_positive_weight=arguments.activity_positive_weight,
        repeat_positive_weight=arguments.repeat_positive_weight,
    )
    append_log(
        arguments.log,
        {
            "epoch": 0,
            "stage": "zero_initialized_baseline",
            "validation": baseline,
        },
    )
    print(json.dumps({"epoch": 0, "validation": baseline}), flush=True)

    training_data_identity = manifest_identity(
        arguments.manifest.resolve(),
        load_manifest(arguments.manifest.resolve()),
        purpose="training",
    )
    validation_data_identity = (
        training_data_identity
        if validation_manifest.resolve() == arguments.manifest.resolve()
        else manifest_identity(
            validation_manifest.resolve(),
            load_manifest(validation_manifest.resolve()),
            purpose="training",
        )
    )
    base_identity = {
        "name": arguments.position_checkpoint.name,
        "sha256": sha256_file(arguments.position_checkpoint),
    }
    string_identity = {
        "name": arguments.string_sequence_checkpoint.name,
        "sha256": sha256_file(arguments.string_sequence_checkpoint),
    }
    train_targets = group_target_counts(train_groups)
    validation_targets = group_target_counts(validation_groups)
    best_metric = float("-inf")
    best_repeat_recall = float("-inf")
    best_epoch = 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_training_epoch(
            model,
            train_loader,
            device,
            optimizer,
            activity_positive_weight=arguments.activity_positive_weight,
            repeat_positive_weight=arguments.repeat_positive_weight,
        )
        validation_metrics = evaluate_groups(
            model,
            validation_groups,
            device,
            inference_batch_size=arguments.inference_batch_size,
            activity_positive_weight=arguments.activity_positive_weight,
            repeat_positive_weight=arguments.repeat_positive_weight,
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        append_log(arguments.log, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        metric = float(validation_metrics[arguments.selection_metric])
        repeat_recall = float(validation_metrics["repeat_recall"])
        if metric > best_metric or (
            metric == best_metric and repeat_recall > best_repeat_recall
        ):
            best_metric = metric
            best_repeat_recall = repeat_recall
            best_epoch = epoch
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "sequence_config": sequence_config.to_dict(),
                    "position_config": position_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "name": (
                            "stringtrace-candidate-sequence-activity"
                        ),
                        "version": "candidate-sequence-activity-v1",
                        "kind": "trained",
                        "created_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "base_checkpoint": base_identity,
                        "base_model_metadata": position_metadata,
                        "string_sequence_checkpoint": string_identity,
                        "string_sequence_metadata": string_metadata,
                        "training_data": training_data_identity,
                        "validation_data": validation_data_identity,
                        "seed": arguments.seed,
                        "epoch": epoch,
                        "selection_metric": arguments.selection_metric,
                        "selection_metric_value": metric,
                        "selection_repeat_recall": repeat_recall,
                        "trainable_scope": (
                            "candidate-sequence-activity-residual"
                        ),
                        "trainable_parameters": trainable_parameters,
                        "candidate_frontend": {
                            "kind": "spotify-basic-pitch-0.4",
                            "onset_threshold": (
                                arguments.candidate_onset_threshold
                            ),
                            "frame_threshold": (
                                arguments.candidate_frame_threshold
                            ),
                            "minimum_note_length": (
                                arguments.candidate_minimum_note_length
                            ),
                        },
                        "sequence_data": {
                            "train_groups": len(train_groups),
                            "validation_groups": len(validation_groups),
                            "train_windows": len(train_windows),
                            "validation_windows": len(validation_windows),
                            "window_length": (
                                sequence_config.sequence_length
                            ),
                            "window_stride": (
                                sequence_config.sequence_stride
                            ),
                            "no_candidate_truncation": True,
                        },
                        "positive_only_datasets": sorted(
                            set(arguments.positive_only_dataset)
                        ),
                        "filtered_negative_groups": (
                            train_dataset.filtered_negative_groups
                        ),
                        "sampling_plan": sampling_plan,
                        "train_targets": train_targets,
                        "validation_targets": validation_targets,
                        "loss_weights": {
                            "activity_positive": (
                                arguments.activity_positive_weight
                            ),
                            "repeat_positive": (
                                arguments.repeat_positive_weight
                            ),
                        },
                        "repeat_interval": arguments.repeat_interval,
                        "feature_cache": cache_report,
                    },
                },
                arguments.output,
            )
        scheduler.step()
    print(
        json.dumps(
            {
                "checkpoint": arguments.output.name,
                "device": str(device),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "selection_metric": arguments.selection_metric,
                "best_validation_metric": best_metric,
                "best_repeat_recall": best_repeat_recall,
                "best_epoch": best_epoch,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
