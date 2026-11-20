from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .candidate_margin import (
    calibration_partition_digest,
    stratified_calibration_partition,
)
from .candidate_sequence import (
    CandidateSequenceGroup,
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    candidate_group_structured_objective,
    candidate_sequence_structured_loss,
    collate_candidate_sequence_windows,
    infer_candidate_sequence_residuals,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import manifest_identity, sha256_file
from .onset_position import OnsetPositionDataset
from .schema import load_manifest
from .train import choose_device
from .train_candidate_activity import parse_dataset_sampling_weights
from .train_candidate_sequence import load_position_model, move_batch
from .train_candidate_sequence_activity import window_sampling_plan


def _add_counts(
    destination: defaultdict[str, int],
    source: dict[str, int | float],
) -> None:
    for key, value in source.items():
        if key == "loss_weight_millis":
            destination[key] += round(float(value))
        else:
            destination[key] += int(value)


def _metrics(
    counts: dict[str, int],
    loss: float,
) -> dict[str, float | int]:
    candidates = counts.get("supervised_candidates", 0)
    groups = counts.get("supervised_groups", 0)
    repeat_total = (
        counts.get("repeat_true_positive", 0)
        + counts.get("repeat_false_negative", 0)
    )
    return {
        "loss": loss,
        "assignment_candidate_accuracy": (
            counts.get("correct_candidates", 0) / max(1, candidates)
        ),
        "assignment_candidate_correct": counts.get(
            "correct_candidates",
            0,
        ),
        "assignment_candidate_total": candidates,
        "wrong_string": counts.get("wrong_string", 0),
        "exact_group_rate": (
            counts.get("exact_groups", 0) / max(1, groups)
        ),
        "exact_groups": counts.get("exact_groups", 0),
        "supervised_groups": groups,
        "repeat_string_recall": (
            counts.get("repeat_true_positive", 0)
            / max(1, repeat_total)
        ),
        "repeat_true_positive": counts.get("repeat_true_positive", 0),
        "repeat_false_negative": counts.get(
            "repeat_false_negative",
            0,
        ),
        "groups_with_multiple_accepted": counts.get(
            "groups_with_multiple_accepted",
            0,
        ),
        "unique_assignment_groups": counts.get(
            "unique_assignment_groups",
            0,
        ),
        "fallback_independent_groups": counts.get(
            "fallback_independent_groups",
            0,
        ),
        "unsatisfiable_groups": counts.get("unsatisfiable_groups", 0),
    }


def run_training_epoch(
    model: CandidateSequenceStringModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    multi_candidate_group_weight: float,
    repeat_group_weight: float,
) -> dict[str, float | int]:
    model.train()
    loss_total = 0.0
    batches = 0
    counts: defaultdict[str, int] = defaultdict(int)
    for source_batch in loader:
        batch = move_batch(source_batch, device)
        optimizer.zero_grad(set_to_none=True)
        residuals = model(
            batch["pitches"],
            batch["confidences"],
            batch["durations"],
            batch["base_string_logits"],
            batch["valid_strings"],
            batch["candidate_mask"],
            batch["group_mask"],
            batch["group_onsets"],
        )
        loss, batch_counts = candidate_sequence_structured_loss(
            batch["base_string_logits"],
            residuals,
            batch["string_targets"],
            batch["target_weights"],
            batch["base_activity_logits"],
            batch["repeat_targets"],
            batch["candidate_mask"],
            batch["group_mask"],
            batch["valid_strings"],
            multi_candidate_group_weight=multi_candidate_group_weight,
            repeat_group_weight=repeat_group_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_total += float(loss.detach().cpu())
        _add_counts(counts, batch_counts)
        batches += 1
    return _metrics(counts, loss_total / max(1, batches))


def _evaluate_group(
    group: CandidateSequenceGroup,
    residual: torch.Tensor,
    *,
    multi_candidate_group_weight: float,
    repeat_group_weight: float,
) -> tuple[torch.Tensor | None, float, dict[str, int]]:
    accepted_indexes = torch.nonzero(
        group.base_activity_logits >= 0,
        as_tuple=False,
    ).flatten()
    if not accepted_indexes.numel():
        return None, 0.0, {}
    targets = group.string_targets[accepted_indexes]
    supervised = targets.sum(dim=1) > 0
    if not bool(supervised.any()):
        return None, 0.0, {}
    repeats = group.repeat_targets[accepted_indexes] > 0
    loss, predictions, used_unique = candidate_group_structured_objective(
        group.base_string_logits[accepted_indexes]
        + residual[accepted_indexes],
        group.valid_strings[accepted_indexes],
        targets,
        supervised,
    )
    matches = torch.gather(
        targets > 0,
        1,
        predictions.unsqueeze(1),
    ).squeeze(1)
    supervised_matches = matches[supervised]
    repeat_supervised = supervised & repeats
    counts = {
        "supervised_groups": 1,
        "groups_with_multiple_accepted": int(
            accepted_indexes.numel() > 1
        ),
        "unique_assignment_groups": int(used_unique),
        "fallback_independent_groups": int(not used_unique),
        "supervised_candidates": int(supervised.sum()),
        "correct_candidates": int(supervised_matches.sum()),
        "wrong_string": int((~supervised_matches).sum()),
        "exact_groups": int(bool(supervised_matches.all())),
        "repeat_true_positive": int(
            (matches & repeat_supervised).sum()
        ),
        "repeat_false_negative": int(
            (~matches & repeat_supervised).sum()
        ),
        "unsatisfiable_groups": int(loss is None),
    }
    if loss is None:
        return None, 0.0, counts
    group_weight = float(
        group.target_weights[accepted_indexes][supervised].mean()
    )
    if accepted_indexes.numel() > 1:
        group_weight *= multi_candidate_group_weight
    if bool(repeat_supervised.any()):
        group_weight *= repeat_group_weight
    return loss, group_weight, counts


def evaluate_groups(
    model: CandidateSequenceStringModel,
    groups: list[CandidateSequenceGroup],
    device: torch.device,
    *,
    inference_batch_size: int,
    multi_candidate_group_weight: float,
    repeat_group_weight: float,
) -> dict[str, float | int]:
    groups_by_annotation: dict[int, list[CandidateSequenceGroup]] = (
        defaultdict(list)
    )
    for group in groups:
        groups_by_annotation[group.annotation_index].append(group)

    counts: defaultdict[str, int] = defaultdict(int)
    weighted_loss = 0.0
    loss_weight = 0.0
    for annotation_groups in groups_by_annotation.values():
        residuals = infer_candidate_sequence_residuals(
            model,
            annotation_groups,
            device,
            batch_size=inference_batch_size,
        )
        for group, residual in zip(annotation_groups, residuals):
            loss, weight, group_counts = _evaluate_group(
                group,
                residual,
                multi_candidate_group_weight=(
                    multi_candidate_group_weight
                ),
                repeat_group_weight=repeat_group_weight,
            )
            _add_counts(counts, group_counts)
            if loss is not None:
                weighted_loss += float(loss) * weight
                loss_weight += weight
    return _metrics(counts, weighted_loss / max(loss_weight, 1e-12))


def append_log(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune candidate sequence string logits with the deployed "
            "full-group assignment objective."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initial-string-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--multi-candidate-group-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--repeat-group-weight",
        type=float,
        default=1.5,
    )
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
        "--dataset-sampling-weight",
        action="append",
        default=[],
        metavar="DATASET=WEIGHT",
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
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-device", default="cpu")
    arguments = parser.parse_args()

    for name in (
        "epochs",
        "batch_size",
        "precompute_batch_size",
        "inference_batch_size",
    ):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "learning_rate",
        "multi_candidate_group_weight",
        "repeat_group_weight",
    ):
        value = getattr(arguments, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if (
        not 0 < arguments.calibration_fraction < 1
        or not math.isfinite(arguments.calibration_fraction)
    ):
        parser.error("--calibration-fraction must be between zero and one")
    try:
        dataset_sampling_weights = parse_dataset_sampling_weights(
            arguments.dataset_sampling_weight
        )
    except ValueError as error:
        parser.error(str(error))
    stratified_datasets = set(arguments.stratify_by_group_dataset)
    if not stratified_datasets:
        stratified_datasets = {"GuitarSet"}

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    precompute_device = choose_device(arguments.device)
    training_device = choose_device(arguments.training_device)
    (
        position_model,
        feature_config,
        position_config,
        position_metadata,
    ) = load_position_model(
        arguments.position_checkpoint,
        precompute_device,
    )
    model, initial_string_metadata = load_candidate_sequence_model(
        arguments.initial_string_checkpoint,
        arguments.position_checkpoint,
        position_config,
        feature_config,
        training_device,
    )
    model.train()
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
    print(
        json.dumps(
            {
                "stage": "precompute_train_groups",
                "groups": len(dataset),
                "precompute_device": str(precompute_device),
                "training_device": str(training_device),
            }
        ),
        flush=True,
    )
    groups = precompute_candidate_sequence_groups(
        dataset,
        position_model,
        precompute_device,
        batch_size=arguments.precompute_batch_size,
        repeat_interval=arguments.repeat_interval,
    )
    (
        calibration_annotation_indexes,
        calibration_keys,
        partition_counts,
    ) = stratified_calibration_partition(
        dataset.annotations,
        fraction=arguments.calibration_fraction,
        seed=arguments.calibration_seed,
        stratify_by_group_datasets=stratified_datasets,
    )
    train_groups = [
        group
        for group in groups
        if group.annotation_index not in calibration_annotation_indexes
    ]
    calibration_groups = [
        group
        for group in groups
        if group.annotation_index in calibration_annotation_indexes
    ]
    train_windows = CandidateSequenceWindowDataset(
        train_groups,
        model.config,
    )
    sampling_weights, sampling_plan = window_sampling_plan(
        train_windows,
        dataset.annotations,
        dataset_sampling_weights,
    )
    print(
        json.dumps(
            {
                "stage": "partition_and_sampling",
                "train_groups": len(train_groups),
                "calibration_groups": len(calibration_groups),
                "train_windows": len(train_windows),
                "selected_annotations": len(
                    calibration_annotation_indexes
                ),
                "sampling": sampling_plan,
            },
            ensure_ascii=False,
        ),
        flush=True,
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
        calibration_groups,
        training_device,
        inference_batch_size=arguments.inference_batch_size,
        multi_candidate_group_weight=(
            arguments.multi_candidate_group_weight
        ),
        repeat_group_weight=arguments.repeat_group_weight,
    )
    append_log(
        arguments.log,
        {
            "epoch": 0,
            "stage": "frozen_string_v1_baseline",
            "calibration": baseline,
        },
    )
    print(
        json.dumps({"epoch": 0, "calibration": baseline}),
        flush=True,
    )

    training_data_identity = manifest_identity(
        arguments.manifest.resolve(),
        load_manifest(arguments.manifest.resolve()),
        purpose="training",
    )
    base_identity = {
        "name": arguments.position_checkpoint.name,
        "sha256": sha256_file(arguments.position_checkpoint),
    }
    initial_string_identity = {
        "name": arguments.initial_string_checkpoint.name,
        "sha256": sha256_file(arguments.initial_string_checkpoint),
    }
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    best_key: tuple[float, int, float, float] | None = None
    best_epoch = 0
    best_metrics: dict[str, float | int] | None = None
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_training_epoch(
            model,
            train_loader,
            training_device,
            optimizer,
            multi_candidate_group_weight=(
                arguments.multi_candidate_group_weight
            ),
            repeat_group_weight=arguments.repeat_group_weight,
        )
        calibration_metrics = evaluate_groups(
            model,
            calibration_groups,
            training_device,
            inference_batch_size=arguments.inference_batch_size,
            multi_candidate_group_weight=(
                arguments.multi_candidate_group_weight
            ),
            repeat_group_weight=arguments.repeat_group_weight,
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "calibration": calibration_metrics,
        }
        append_log(arguments.log, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        eligible = (
            float(
                calibration_metrics["assignment_candidate_accuracy"]
            )
            > float(baseline["assignment_candidate_accuracy"])
            and int(calibration_metrics["wrong_string"])
            < int(baseline["wrong_string"])
            and int(calibration_metrics["repeat_true_positive"])
            >= int(baseline["repeat_true_positive"])
        )
        selection_key = (
            float(calibration_metrics["assignment_candidate_accuracy"]),
            int(calibration_metrics["repeat_true_positive"]),
            float(calibration_metrics["exact_group_rate"]),
            -float(calibration_metrics["loss"]),
        )
        if eligible and (best_key is None or selection_key > best_key):
            best_key = selection_key
            best_epoch = epoch
            best_metrics = calibration_metrics
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "sequence_config": model.config.to_dict(),
                    "position_config": position_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "name": "stringtrace-candidate-sequence-string",
                        "version": "candidate-sequence-string-v2",
                        "kind": "trained",
                        "created_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "base_checkpoint": base_identity,
                        "base_model_metadata": position_metadata,
                        "initialized_string_checkpoint": (
                            initial_string_identity
                        ),
                        "initialized_string_metadata": (
                            initial_string_metadata
                        ),
                        "training_data": training_data_identity,
                        "seed": arguments.seed,
                        "training_device": str(training_device),
                        "epoch": epoch,
                        "selection_metric": (
                            "training_only_structured_assignment_accuracy"
                        ),
                        "selection_metric_value": selection_key[0],
                        "calibration_gate_passed": True,
                        "calibration_baseline": baseline,
                        "calibration_selected": calibration_metrics,
                        "trainable_scope": (
                            "candidate-sequence-string-full-model"
                        ),
                        "trainable_parameters": trainable_parameters,
                        "objective": {
                            "kind": "full_group_assignment_nll",
                            "activity_threshold": 0.5,
                            "multi_candidate_group_weight": (
                                arguments.multi_candidate_group_weight
                            ),
                            "repeat_group_weight": (
                                arguments.repeat_group_weight
                            ),
                            "unique_string_assignment": True,
                            "decoder_fallback_reproduced": True,
                        },
                        "partition": {
                            "fraction": arguments.calibration_fraction,
                            "seed": arguments.calibration_seed,
                            "stratify_by_group_datasets": sorted(
                                stratified_datasets
                            ),
                            "selected_key_sha256": (
                                calibration_partition_digest(
                                    calibration_keys
                                )
                            ),
                            "selected_annotations": len(
                                calibration_annotation_indexes
                            ),
                            "datasets": partition_counts,
                        },
                        "sampling": sampling_plan,
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
                            "calibration_groups": len(
                                calibration_groups
                            ),
                            "train_windows": len(train_windows),
                        },
                        "positive_only_datasets": sorted(
                            set(arguments.positive_only_dataset)
                        ),
                        "cache": cache_report,
                    },
                },
                arguments.output,
            )
        scheduler.step()

    gate_passed = best_metrics is not None
    result: dict[str, object] = {
        "output": str(arguments.output) if gate_passed else None,
        "calibration_gate_passed": gate_passed,
        "best_epoch": best_epoch,
        "baseline": baseline,
        "selected": best_metrics,
        "trainable_parameters": trainable_parameters,
    }
    if gate_passed:
        result["sha256"] = sha256_file(arguments.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
