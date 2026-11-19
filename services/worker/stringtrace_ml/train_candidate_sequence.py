from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from .candidate_sequence import (
    CandidateSequenceConfig,
    CandidateSequenceGroup,
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    candidate_sequence_loss,
    collate_candidate_sequence_windows,
    infer_candidate_sequence_residuals,
    precompute_candidate_sequence_groups,
)
from .data_sources import manifest_identity, sha256_file
from .features import FeatureConfig
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    OnsetPositionNet,
)
from .schema import load_manifest
from .train import choose_device


def load_position_model(
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
    position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("model_config", {}))
    )
    if (
        position_config.candidate_position_objective
        != "spectral_residual"
        or not position_config.candidate_position_head
    ):
        raise ValueError(
            "Candidate sequence training requires a spectral residual "
            "position checkpoint"
        )
    if position_config.candidate_temporal_string_residual:
        raise ValueError(
            "Candidate sequence training requires a local position checkpoint"
        )
    model = OnsetPositionNet(feature_config, position_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return (
        model,
        feature_config,
        position_config,
        dict(checkpoint.get("metadata", {})),
    )


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def run_training_epoch(
    model: CandidateSequenceStringModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    loss_total = 0.0
    correct_total = 0
    candidate_total = 0
    batches = 0
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
        loss, correct, total = candidate_sequence_loss(
            batch["base_string_logits"],
            residuals,
            batch["string_targets"],
            batch["target_weights"],
            batch["candidate_mask"],
            batch["valid_strings"],
        )
        loss.backward()
        optimizer.step()
        loss_total += float(loss.detach().cpu())
        correct_total += correct
        candidate_total += total
        batches += 1
    return {
        "loss": loss_total / max(1, batches),
        "window_candidate_oracle_string_accuracy": (
            correct_total / max(1, candidate_total)
        ),
        "window_active_candidates": candidate_total,
    }


def evaluate_groups(
    model: CandidateSequenceStringModel,
    groups: list[CandidateSequenceGroup],
    device: torch.device,
    *,
    inference_batch_size: int,
) -> dict[str, float]:
    groups_by_annotation: dict[int, list[CandidateSequenceGroup]] = (
        defaultdict(list)
    )
    for group in groups:
        groups_by_annotation[group.annotation_index].append(group)

    losses: list[torch.Tensor] = []
    correct_total = 0
    candidate_total = 0
    for annotation_groups in groups_by_annotation.values():
        residuals = infer_candidate_sequence_residuals(
            model,
            annotation_groups,
            device,
            batch_size=inference_batch_size,
        )
        for group, residual in zip(annotation_groups, residuals):
            active = group.string_targets.sum(dim=1) > 0
            if not bool(active.any()):
                continue
            refined = functional.log_softmax(
                (
                    group.base_string_logits
                    + residual
                ).masked_fill(~group.valid_strings, -30.0),
                dim=1,
            )
            losses.append(
                -(
                    group.string_targets[active] * refined[active]
                ).sum(dim=1)
            )
            predictions = refined.argmax(dim=1)
            matches = torch.gather(
                group.string_targets,
                1,
                predictions.unsqueeze(1),
            ).squeeze(1) > 0
            correct_total += int((matches & active).sum())
            candidate_total += int(active.sum())
    loss = (
        float(torch.cat(losses).mean())
        if losses
        else 0.0
    )
    return {
        "loss": loss,
        "candidate_oracle_string_accuracy": (
            correct_total / max(1, candidate_total)
        ),
        "candidate_oracle_string_correct": correct_total,
        "candidate_oracle_string_events": candidate_total,
    }


def append_log(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an explicit candidate-group sequence residual while "
            "freezing position and activity decisions."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument(
        "--position-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument(
        "--train-candidate-cache",
        type=Path,
        required=True,
    )
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
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-stride", type=int, default=16)
    parser.add_argument("--token-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--output-hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
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
    parser.add_argument(
        "--candidate-ambiguity-weight",
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "--candidate-rare-pair-max-weight",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--candidate-cross-string-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--candidate-cross-string-interval",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--candidate-max-supervision-weight",
        type=float,
        default=4.0,
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    if arguments.epochs <= 0:
        parser.error("--epochs must be positive")
    for name in (
        "batch_size",
        "precompute_batch_size",
        "inference_batch_size",
    ):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if arguments.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")

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
    sequence_config = CandidateSequenceConfig(
        sequence_length=arguments.sequence_length,
        sequence_stride=arguments.sequence_stride,
        token_size=arguments.token_size,
        hidden_size=arguments.hidden_size,
        output_hidden_size=arguments.output_hidden_size,
        dropout=arguments.dropout,
    )
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
        candidate_ambiguity_weight=(
            arguments.candidate_ambiguity_weight
        ),
        candidate_rare_pair_max_weight=(
            arguments.candidate_rare_pair_max_weight
        ),
        candidate_cross_string_weight=(
            arguments.candidate_cross_string_weight
        ),
        candidate_cross_string_interval=(
            arguments.candidate_cross_string_interval
        ),
        candidate_max_supervision_weight=(
            arguments.candidate_max_supervision_weight
        ),
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
    )
    print(
        json.dumps(
            {
                "stage": "precompute_validation_groups",
                "groups": len(validation_dataset),
            }
        ),
        flush=True,
    )
    validation_groups = precompute_candidate_sequence_groups(
        validation_dataset,
        position_model,
        device,
        batch_size=arguments.precompute_batch_size,
    )
    train_windows = CandidateSequenceWindowDataset(
        train_groups,
        sequence_config,
    )
    validation_windows = CandidateSequenceWindowDataset(
        validation_groups,
        sequence_config,
    )
    train_loader = DataLoader(
        train_windows,
        batch_size=arguments.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(arguments.seed),
        num_workers=0,
        collate_fn=collate_candidate_sequence_windows,
    )
    model = CandidateSequenceStringModel(sequence_config).to(device)
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
    )
    append_log(
        arguments.log,
        {
            "epoch": 0,
            "stage": "zero_initialized_baseline",
            "validation": baseline,
        },
    )
    print(
        json.dumps(
            {
                "epoch": 0,
                "validation": baseline,
            }
        ),
        flush=True,
    )

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
    best_metric = float("-inf")
    best_epoch = 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_training_epoch(
            model,
            train_loader,
            device,
            optimizer,
        )
        validation_metrics = evaluate_groups(
            model,
            validation_groups,
            device,
            inference_batch_size=arguments.inference_batch_size,
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        append_log(arguments.log, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        metric = validation_metrics[
            "candidate_oracle_string_accuracy"
        ]
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "sequence_config": sequence_config.to_dict(),
                    "position_config": position_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "name": "stringtrace-candidate-sequence-string",
                        "version": "candidate-sequence-string-v1",
                        "kind": "trained",
                        "created_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "base_checkpoint": base_identity,
                        "base_model_metadata": position_metadata,
                        "training_data": training_data_identity,
                        "validation_data": validation_data_identity,
                        "seed": arguments.seed,
                        "epoch": epoch,
                        "selection_metric": (
                            "candidate_oracle_string_accuracy"
                        ),
                        "selection_metric_value": metric,
                        "trainable_scope": (
                            "candidate-sequence-string-residual"
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
                        "candidate_supervision": {
                            **train_dataset.candidate_supervision,
                            "ambiguity_weight": (
                                arguments.candidate_ambiguity_weight
                            ),
                            "rare_pair_max_weight": (
                                arguments
                                .candidate_rare_pair_max_weight
                            ),
                            "cross_string_weight": (
                                arguments.candidate_cross_string_weight
                            ),
                            "cross_string_interval": (
                                arguments.candidate_cross_string_interval
                            ),
                            "maximum_supervision_weight": (
                                arguments
                                .candidate_max_supervision_weight
                            ),
                        },
                        "feature_cache": cache_report,
                    },
                },
                arguments.output,
            )
        scheduler.step()
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "best_epoch": best_epoch,
                "best_validation_oracle_string_accuracy": best_metric,
                "baseline_validation_oracle_string_accuracy": baseline[
                    "candidate_oracle_string_accuracy"
                ],
                "trainable_parameters": trainable_parameters,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
