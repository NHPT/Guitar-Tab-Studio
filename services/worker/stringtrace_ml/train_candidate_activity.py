from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .candidate_activity import (
    CandidateActivityConfig,
    CandidateActivityDataset,
    CandidateActivityHead,
)
from .features import FeatureConfig
from .onset_position import OnsetPositionConfig, OnsetPositionNet
from .prepare_guitarset_stems import sha256_file
from .train import choose_device


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    return (
        2
        * true_positive
        / max(1, 2 * true_positive + false_positive + false_negative)
    )


def run_epoch(
    position_model: OnsetPositionNet,
    activity_head: CandidateActivityHead,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    activity_positive_weight: float,
    repeat_positive_weight: float,
    activity_loss_weight: float,
    repeat_loss_weight: float,
    progress_label: str,
) -> dict[str, float]:
    training = optimizer is not None
    position_model.eval()
    activity_head.train(training)
    totals = {
        "loss": 0.0,
        "activity_loss": 0.0,
        "repeat_loss": 0.0,
    }
    counts = {
        "activity_true_positive": 0,
        "activity_false_positive": 0,
        "activity_false_negative": 0,
        "repeat_true_positive": 0,
        "repeat_false_positive": 0,
        "repeat_false_negative": 0,
    }
    batches = 0
    for batch in loader:
        features = batch["features"].to(device)
        pitches = batch["pitch_mask"].to(device)
        temporal_context = batch["temporal_context"].to(device)
        activity_targets = batch["activity_target"].to(device)
        repeat_targets = batch["repeat_target"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            encoded = position_model.encode(features, pitches)
        with torch.set_grad_enabled(training):
            logits = activity_head(encoded, temporal_context)
            activity_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0],
                activity_targets,
                pos_weight=torch.tensor(
                    activity_positive_weight,
                    device=device,
                ),
            )
            active = activity_targets > 0
            repeat_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits[active, 1],
                    repeat_targets[active],
                    pos_weight=torch.tensor(
                        repeat_positive_weight,
                        device=device,
                    ),
                )
                if bool(active.any())
                else logits[:, 1].sum() * 0
            )
            loss = (
                activity_loss_weight * activity_loss
                + repeat_loss_weight * repeat_loss
            )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    activity_head.parameters(),
                    5.0,
                )
                optimizer.step()

        activity_predictions = logits[:, 0].detach() >= 0
        activity_references = activity_targets > 0
        counts["activity_true_positive"] += int(
            (activity_predictions & activity_references).sum().cpu()
        )
        counts["activity_false_positive"] += int(
            (activity_predictions & ~activity_references).sum().cpu()
        )
        counts["activity_false_negative"] += int(
            (~activity_predictions & activity_references).sum().cpu()
        )
        repeat_predictions = logits[:, 1].detach() >= 0
        counts["repeat_true_positive"] += int(
            (
                repeat_predictions
                & (repeat_targets > 0)
                & activity_references
            ).sum().cpu()
        )
        counts["repeat_false_positive"] += int(
            (
                repeat_predictions
                & (repeat_targets == 0)
                & activity_references
            ).sum().cpu()
        )
        counts["repeat_false_negative"] += int(
            (
                ~repeat_predictions
                & (repeat_targets > 0)
                & activity_references
            ).sum().cpu()
        )
        totals["loss"] += float(loss.detach().cpu())
        totals["activity_loss"] += float(activity_loss.detach().cpu())
        totals["repeat_loss"] += float(repeat_loss.detach().cpu())
        batches += 1
        progress_interval = max(1, len(loader) // 5)
        if batches % progress_interval == 0 or batches == len(loader):
            print(
                json.dumps(
                    {
                        "stage": progress_label,
                        "batches": batches,
                        "total_batches": len(loader),
                    }
                ),
                flush=True,
            )

    result = {
        key: value / max(1, batches)
        for key, value in totals.items()
    }
    for name in ("activity", "repeat"):
        result[f"{name}_f1"] = _f1(
            counts[f"{name}_true_positive"],
            counts[f"{name}_false_positive"],
            counts[f"{name}_false_negative"],
        )
    return result


def _positive_weight(positive: int, negative: int) -> float:
    return max(1.0, negative / max(1, positive))


def parse_dataset_sampling_weights(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        dataset, separator, raw_weight = value.partition("=")
        dataset = dataset.strip()
        if not separator or not dataset:
            raise ValueError(
                "dataset sampling weights must use DATASET=WEIGHT"
            )
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(
                f"invalid sampling weight for {dataset}"
            ) from error
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"sampling weight for {dataset} must be positive and finite"
            )
        if dataset in result:
            raise ValueError(f"duplicate sampling weight for {dataset}")
        result[dataset] = weight
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a candidate activity and repeated-attack head above a "
            "frozen onset-position encoder."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--activity-loss-weight", type=float, default=0.25)
    parser.add_argument("--repeat-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric",
        choices=("activity_f1", "repeat_f1"),
        default="repeat_f1",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--repeat-interval", type=float, default=0.75)
    parser.add_argument("--temporal-context", action="store_true")
    parser.add_argument("--temporal-max-interval", type=float, default=2.0)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--dataset-sampling-weight",
        action="append",
        default=[],
        metavar="DATASET=WEIGHT",
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--train-candidate-cache", type=Path, required=True)
    parser.add_argument(
        "--validation-candidate-cache",
        type=Path,
        required=True,
    )
    parser.add_argument("--candidate-onset-threshold", type=float, default=0.5)
    parser.add_argument("--candidate-frame-threshold", type=float, default=0.3)
    parser.add_argument(
        "--candidate-minimum-note-length",
        type=float,
        default=127.7,
    )
    parser.add_argument(
        "--positive-only-dataset",
        action="append",
        default=[],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260923)
    arguments = parser.parse_args()

    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if arguments.activity_loss_weight < 0 or arguments.repeat_loss_weight <= 0:
        parser.error("loss weights must be non-negative with repeat enabled")
    if arguments.temporal_max_interval <= 0:
        parser.error("--temporal-max-interval must be positive")
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
    checkpoint = torch.load(
        arguments.position_checkpoint,
        map_location=device,
        weights_only=False,
    )
    feature_config = FeatureConfig.from_dict(
        dict(checkpoint.get("feature_config", {}))
    )
    position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("model_config", {}))
    )
    position_model = OnsetPositionNet(
        feature_config,
        position_config,
    ).to(device)
    position_model.load_state_dict(checkpoint["model_state"])
    position_model.eval()
    for parameter in position_model.parameters():
        parameter.requires_grad = False

    activity_config = CandidateActivityConfig(
        hidden_size=arguments.hidden_size,
        dropout=arguments.dropout,
        repeat_interval=arguments.repeat_interval,
        temporal_feature_count=6 if arguments.temporal_context else 0,
        temporal_max_interval=arguments.temporal_max_interval,
    )
    train_dataset = CandidateActivityDataset(
        arguments.manifest,
        split="train",
        feature_config=feature_config,
        position_config=position_config,
        activity_config=activity_config,
        feature_cache_directory=arguments.feature_cache,
        candidate_cache_directory=arguments.train_candidate_cache,
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=(
            arguments.candidate_minimum_note_length
        ),
        positive_only_datasets=set(arguments.positive_only_dataset),
    )
    validation_dataset = CandidateActivityDataset(
        arguments.validation_manifest,
        split="validation",
        feature_config=feature_config,
        position_config=position_config,
        activity_config=activity_config,
        feature_cache_directory=arguments.feature_cache,
        candidate_cache_directory=arguments.validation_candidate_cache,
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=(
            arguments.candidate_minimum_note_length
        ),
    )
    print(
        json.dumps(
            {
                "stage": "feature_cache",
                "train": train_dataset.prepare_feature_cache(),
                "validation": validation_dataset.prepare_feature_cache(),
                "train_targets": train_dataset.target_counts(),
                "validation_targets": validation_dataset.target_counts(),
                "filtered_negative_groups": (
                    train_dataset.filtered_negative_groups
                ),
            }
        ),
        flush=True,
    )
    train_counts = train_dataset.target_counts()
    activity_positive_weight = _positive_weight(
        train_counts["active"],
        train_counts["inactive"],
    )
    repeat_positive_weight = _positive_weight(
        train_counts["repeated"],
        train_counts["active"] - train_counts["repeated"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=not dataset_sampling_weights,
        sampler=(
            WeightedRandomSampler(
                train_dataset.sampling_weights(dataset_sampling_weights),
                num_samples=len(train_dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(arguments.seed),
            )
            if dataset_sampling_weights
            else None
        ),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    activity_head = CandidateActivityHead(
        position_config,
        activity_config,
    ).to(device)
    initialization: dict[str, str] | None = None
    if arguments.initialize_from is not None:
        initialization_checkpoint = torch.load(
            arguments.initialize_from,
            map_location=device,
            weights_only=False,
        )
        initialized_activity_config = CandidateActivityConfig.from_dict(
            dict(initialization_checkpoint.get("activity_config", {}))
        )
        initialized_position_config = OnsetPositionConfig.from_dict(
            dict(initialization_checkpoint.get("position_config", {}))
        )
        initialized_feature_config = FeatureConfig.from_dict(
            dict(initialization_checkpoint.get("feature_config", {}))
        )
        if initialized_activity_config != activity_config:
            raise ValueError(
                "Initialization activity configuration differs"
            )
        if initialized_position_config != position_config:
            raise ValueError(
                "Initialization position configuration differs"
            )
        if initialized_feature_config != feature_config:
            raise ValueError(
                "Initialization feature configuration differs"
            )
        initialized_metadata = dict(
            initialization_checkpoint.get("metadata", {})
        )
        initialized_base = dict(
            initialized_metadata.get("base_checkpoint", {})
        )
        if initialized_base.get("sha256") != sha256_file(
            arguments.position_checkpoint
        ):
            raise ValueError(
                "Initialization head targets a different base checkpoint"
            )
        activity_head.load_state_dict(
            initialization_checkpoint["model_state"]
        )
        initialization = {
            "name": arguments.initialize_from.name,
            "sha256": sha256_file(arguments.initialize_from),
        }
    optimizer = torch.optim.AdamW(
        activity_head.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, arguments.epochs),
        eta_min=arguments.learning_rate * 0.05,
    )

    history: list[dict[str, object]] = []
    best_metric = float("-inf")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_epoch(
            position_model,
            activity_head,
            train_loader,
            device,
            optimizer,
            activity_positive_weight=activity_positive_weight,
            repeat_positive_weight=repeat_positive_weight,
            activity_loss_weight=arguments.activity_loss_weight,
            repeat_loss_weight=arguments.repeat_loss_weight,
            progress_label=f"train_epoch_{epoch}",
        )
        validation_metrics = run_epoch(
            position_model,
            activity_head,
            validation_loader,
            device,
            None,
            activity_positive_weight=activity_positive_weight,
            repeat_positive_weight=repeat_positive_weight,
            activity_loss_weight=arguments.activity_loss_weight,
            repeat_loss_weight=arguments.repeat_loss_weight,
            progress_label=f"validation_epoch_{epoch}",
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        metric = validation_metrics[arguments.selection_metric]
        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "format_version": 1,
                    "model_state": activity_head.state_dict(),
                    "activity_config": activity_config.to_dict(),
                    "position_config": position_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "name": "stringtrace-candidate-activity-head",
                        "version": "candidate-activity-repeat-v1",
                        "created_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "base_checkpoint": {
                            "name": arguments.position_checkpoint.name,
                            "sha256": sha256_file(
                                arguments.position_checkpoint
                            ),
                        },
                        "manifest_name": arguments.manifest.name,
                        "validation_manifest_name": (
                            arguments.validation_manifest.name
                        ),
                        "positive_only_datasets": sorted(
                            set(arguments.positive_only_dataset)
                        ),
                        "dataset_sampling_weights": dict(
                            sorted(dataset_sampling_weights.items())
                        ),
                        "initialized_from": initialization,
                        "filtered_negative_groups": (
                            train_dataset.filtered_negative_groups
                        ),
                        "train_targets": train_counts,
                        "validation_targets": (
                            validation_dataset.target_counts()
                        ),
                        "selection_metric": arguments.selection_metric,
                        "selection_metric_value": metric,
                        "epoch": epoch,
                        "seed": arguments.seed,
                    },
                    "history": history,
                },
                arguments.output,
            )
        scheduler.step()

    print(
        json.dumps(
            {
                "checkpoint": arguments.output.name,
                "device": str(device),
                "train_groups": len(train_dataset),
                "validation_groups": len(validation_dataset),
                "selection_metric": arguments.selection_metric,
                "best_validation_metric": best_metric,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
