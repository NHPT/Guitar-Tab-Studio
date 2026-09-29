from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import GuitarTabDataset
from .features import FeatureConfig
from .model import (
    OPEN_STRING_PITCHES,
    PITCH_MIN,
    ModelConfig,
    StringFretNet,
    transcription_loss,
)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def configure_trainable_parameters(
    model: StringFretNet,
    *,
    freeze_position_backbone: bool = False,
    onset_head_only: bool = False,
) -> None:
    if freeze_position_backbone and onset_head_only:
        raise ValueError(
            "freeze_position_backbone and onset_head_only are mutually exclusive"
        )
    for name, parameter in model.named_parameters():
        if freeze_position_backbone:
            parameter.requires_grad = (
                name.startswith("pitch_embedding.")
                or name.startswith("position_head.")
            )
        elif onset_head_only:
            parameter.requires_grad = name.startswith("onset_head.")
        else:
            parameter.requires_grad = True


def best_compatible_validation_loss(
    history: list[dict[str, object]],
    metadata: dict[str, object],
    training_config: dict[str, object],
) -> float:
    if metadata.get("training_config") != training_config:
        return float("inf")
    return min(
        (
            float(record["validation"]["loss"])
            for record in history
        ),
        default=float("inf"),
    )


def best_compatible_validation_metric(
    history: list[dict[str, object]],
    metadata: dict[str, object],
    training_config: dict[str, object],
    metric: str,
) -> float:
    if metadata.get("training_config") != training_config:
        return float("inf") if metric == "loss" else float("-inf")
    values = [
        float(record["validation"][metric])
        for record in history
        if metric in record["validation"]
    ]
    if not values:
        return float("inf") if metric == "loss" else float("-inf")
    return min(values) if metric == "loss" else max(values)


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    return (
        2 * true_positive
        / max(1, 2 * true_positive + false_positive + false_negative)
    )


def run_epoch(
    model: StringFretNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    progress_label: str,
    attack_fret_weight: float,
    attack_fret_frames: int,
    fret_distance_weight: float,
    pitch_consistency_weight: float,
    pitch_positive_weight: float,
    position_loss_weight: float,
    string_loss_weights: torch.Tensor,
    freeze_position_backbone: bool = False,
    onset_head_only: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and (freeze_position_backbone or onset_head_only):
        model.encoder.eval()
        model.frequency_projection.eval()
        model.temporal.eval()
        model.dropout.eval()
    if training and onset_head_only:
        model.fret_head.eval()
        if model.pitch_embedding is not None:
            model.pitch_embedding.eval()
        if model.position_head is not None:
            model.position_head.eval()
    totals = {
        "loss": torch.zeros((), device=device),
        "fret_loss": torch.zeros((), device=device),
        "onset_loss": torch.zeros((), device=device),
        "fret_distance_loss": torch.zeros((), device=device),
        "pitch_consistency_loss": torch.zeros((), device=device),
        "position_loss": torch.zeros((), device=device),
    }
    counts = {
        "frame_tablature_true_positive": torch.zeros(
            (),
            device=device,
            dtype=torch.long,
        ),
        "frame_tablature_false_positive": torch.zeros(
            (),
            device=device,
            dtype=torch.long,
        ),
        "frame_tablature_false_negative": torch.zeros(
            (),
            device=device,
            dtype=torch.long,
        ),
        "frame_onset_true_positive": torch.zeros(
            (),
            device=device,
            dtype=torch.long,
        ),
        "frame_onset_false_positive": torch.zeros(
            (),
            device=device,
            dtype=torch.long,
        ),
        "frame_onset_false_negative": torch.zeros(
            (),
            device=device,
            dtype=torch.long,
        ),
        "position_correct": torch.zeros(
            (),
            device=device,
            dtype=torch.long,
        ),
        "position_events": torch.zeros(
            (),
            device=device,
            dtype=torch.long,
        ),
    }
    batches = 0
    for batch in loader:
        features = batch["features"].to(device)
        fret_targets = batch["fret_targets"].to(device)
        onset_targets = batch["onset_targets"].to(device)
        valid_frames = batch["valid_frames"].to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        position_queries = None
        if model.config.pitch_conditioned_position:
            position_event_mask = (
                (onset_targets > 0)
                & (fret_targets > 0)
                & valid_frames.unsqueeze(-1)
            )
            (
                position_batch_indexes,
                position_frame_indexes,
                position_string_indexes,
            ) = torch.nonzero(position_event_mask, as_tuple=True)
            open_pitches = torch.tensor(
                OPEN_STRING_PITCHES,
                device=device,
            )
            position_pitch_indexes = (
                fret_targets[
                    position_batch_indexes,
                    position_frame_indexes,
                    position_string_indexes,
                ]
                - 1
                + open_pitches[position_string_indexes]
                - PITCH_MIN
            )
            position_queries = (
                position_batch_indexes,
                position_frame_indexes,
                position_pitch_indexes,
            )
        with torch.set_grad_enabled(training):
            outputs = model(features, position_queries=position_queries)
            loss, values = transcription_loss(
                outputs,
                fret_targets,
                onset_targets,
                valid_frames,
                attack_fret_weight=attack_fret_weight,
                attack_fret_frames=attack_fret_frames,
                fret_distance_weight=fret_distance_weight,
                pitch_consistency_weight=pitch_consistency_weight,
                pitch_positive_weight=pitch_positive_weight,
                position_loss_weight=position_loss_weight,
                string_loss_weights=string_loss_weights,
            )
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        for key in totals:
            totals[key] += values[key]
        with torch.no_grad():
            valid_strings = valid_frames.unsqueeze(-1)
            fret_predictions = outputs["fret_logits"].argmax(dim=-1)
            fret_targets_active = fret_targets > 0
            fret_predictions_active = fret_predictions > 0
            fret_matches = (
                fret_predictions_active
                & fret_targets_active
                & (fret_predictions == fret_targets)
                & valid_strings
            )
            counts["frame_tablature_true_positive"] += fret_matches.sum()
            counts["frame_tablature_false_positive"] += (
                fret_predictions_active
                & ~fret_matches
                & valid_strings
            ).sum()
            counts["frame_tablature_false_negative"] += (
                fret_targets_active
                & ~fret_matches
                & valid_strings
            ).sum()
            onset_predictions = outputs["onset_logits"] >= 0
            onset_targets_active = onset_targets > 0
            onset_matches = (
                onset_predictions
                & onset_targets_active
                & valid_strings
            )
            counts["frame_onset_true_positive"] += onset_matches.sum()
            counts["frame_onset_false_positive"] += (
                onset_predictions
                & ~onset_targets_active
                & valid_strings
            ).sum()
            counts["frame_onset_false_negative"] += (
                ~onset_predictions
                & onset_targets_active
                & valid_strings
            ).sum()
            if "position_logits" in outputs:
                if position_string_indexes.numel():
                    predictions = outputs["position_logits"].argmax(dim=-1)
                    counts["position_correct"] += (
                        predictions == position_string_indexes
                    ).sum()
                    counts["position_events"] += (
                        position_string_indexes.numel()
                    )
        batches += 1
        progress_interval = max(1, len(loader) // 10)
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
        key: float((value / max(1, batches)).cpu())
        for key, value in totals.items()
    }
    resolved_counts = {
        key: int(value.cpu())
        for key, value in counts.items()
    }
    for name in ("tablature", "onset"):
        true_positive = resolved_counts[f"frame_{name}_true_positive"]
        false_positive = resolved_counts[f"frame_{name}_false_positive"]
        false_negative = resolved_counts[f"frame_{name}_false_negative"]
        result[f"frame_{name}_f1"] = _f1(
            true_positive,
            false_positive,
            false_negative,
        )
    if resolved_counts["position_events"]:
        result["frame_position_accuracy"] = (
            resolved_counts["position_correct"]
            / resolved_counts["position_events"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the StringTrace six-string fret/onset model.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--window-hop-seconds", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--position-learning-rate",
        type=float,
        help="Optional learning rate for the pitch-conditioned position head.",
    )
    parser.add_argument("--attack-fret-weight", type=float, default=2.0)
    parser.add_argument("--attack-fret-frames", type=int, default=6)
    parser.add_argument("--fret-distance-weight", type=float, default=0.0)
    parser.add_argument("--pitch-consistency-weight", type=float, default=0.0)
    parser.add_argument("--pitch-positive-weight", type=float, default=5.0)
    parser.add_argument("--position-loss-weight", type=float, default=0.0)
    parser.add_argument("--enable-position-head", action="store_true")
    parser.add_argument(
        "--freeze-position-backbone",
        action="store_true",
        help="Train only the pitch embedding and conditioned position head.",
    )
    parser.add_argument(
        "--onset-head-only",
        action="store_true",
        help=(
            "Train only the onset head while keeping the feature, fret and "
            "position branches frozen."
        ),
    )
    parser.add_argument("--string-balance-exponent", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "loss",
            "frame_tablature_f1",
            "frame_onset_f1",
            "frame_position_accuracy",
        ),
        default="loss",
    )
    parser.add_argument("--hard-sample-fast-bonus", type=float, default=0.0)
    parser.add_argument("--hard-sample-high-fret-bonus", type=float, default=0.0)
    parser.add_argument("--hard-sample-fast-interval", type=float, default=0.14)
    parser.add_argument("--hard-sample-high-fret", type=int, default=12)
    parser.add_argument(
        "--dataset-sampling-weight",
        action="append",
        default=[],
        metavar="DATASET=WEIGHT",
    )
    parser.add_argument("--pitch-shift-probability", type=float, default=0.0)
    parser.add_argument("--pitch-shift-min", type=int, default=1)
    parser.add_argument("--pitch-shift-max", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model, optimizer, history and epoch from a checkpoint.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Initialize compatible model weights from a checkpoint while "
            "starting a fresh optimizer and history."
        ),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Persistent CQT window cache (default: <manifest-dir>/.feature-cache).",
    )
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Extract each window on demand instead of using a persistent cache.",
    )
    arguments = parser.parse_args()
    if arguments.fret_distance_weight < 0:
        parser.error("--fret-distance-weight must be non-negative")
    if arguments.pitch_consistency_weight < 0:
        parser.error("--pitch-consistency-weight must be non-negative")
    if arguments.pitch_positive_weight < 1:
        parser.error("--pitch-positive-weight must be at least one")
    if arguments.position_loss_weight < 0:
        parser.error("--position-loss-weight must be non-negative")
    if (
        arguments.position_learning_rate is not None
        and arguments.position_learning_rate <= 0
    ):
        parser.error("--position-learning-rate must be positive")
    if arguments.string_balance_exponent < 0:
        parser.error("--string-balance-exponent must be non-negative")
    if arguments.resume is not None and arguments.initialize_from is not None:
        parser.error("--resume and --initialize-from are mutually exclusive")
    if arguments.freeze_position_backbone and arguments.onset_head_only:
        parser.error(
            "--freeze-position-backbone and --onset-head-only are "
            "mutually exclusive"
        )
    if (
        arguments.onset_head_only
        and arguments.resume is None
        and arguments.initialize_from is None
    ):
        parser.error(
            "--onset-head-only requires --resume or --initialize-from"
        )
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
    checkpoint_path = arguments.resume or arguments.initialize_from
    source_checkpoint = (
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint_path is not None
        else None
    )
    feature_config = (
        FeatureConfig.from_dict(source_checkpoint.get("feature_config", {}))
        if source_checkpoint is not None
        else FeatureConfig()
    )
    model_config = (
        ModelConfig.from_dict(source_checkpoint.get("model_config", {}))
        if source_checkpoint is not None
        else ModelConfig()
    )
    if arguments.enable_position_head:
        model_config = replace(
            model_config,
            pitch_conditioned_position=True,
        )
    if arguments.position_loss_weight > 0 and not (
        model_config.pitch_conditioned_position
    ):
        parser.error(
            "--position-loss-weight requires --enable-position-head "
            "or a checkpoint that already contains that head"
        )
    if (
        arguments.selection_metric == "frame_position_accuracy"
        and not model_config.pitch_conditioned_position
    ):
        parser.error(
            "--selection-metric frame_position_accuracy requires a "
            "pitch-conditioned position head"
        )
    if (
        arguments.freeze_position_backbone
        and not model_config.pitch_conditioned_position
    ):
        parser.error(
            "--freeze-position-backbone requires a pitch-conditioned "
            "position head"
        )
    feature_cache_directory = (
        None
        if arguments.no_feature_cache
        else (
            arguments.feature_cache.resolve()
            if arguments.feature_cache is not None
            else (arguments.manifest.resolve().parent / ".feature-cache")
        )
    )
    train_dataset = GuitarTabDataset(
        arguments.manifest,
        split="train",
        feature_config=feature_config,
        window_seconds=arguments.window_seconds,
        window_hop_seconds=arguments.window_hop_seconds,
        feature_cache_directory=feature_cache_directory,
        pitch_shift_probability=arguments.pitch_shift_probability,
        pitch_shift_min=arguments.pitch_shift_min,
        pitch_shift_max=arguments.pitch_shift_max,
    )
    validation_dataset = GuitarTabDataset(
        arguments.manifest,
        split="validation",
        feature_config=feature_config,
        window_seconds=arguments.window_seconds,
        window_hop_seconds=arguments.window_hop_seconds,
        feature_cache_directory=feature_cache_directory,
    )
    if not train_dataset:
        raise RuntimeError("The manifest contains no train windows")
    if feature_cache_directory is not None:
        print(
            json.dumps(
                {
                    "stage": "feature_cache",
                    "status": "preparing",
                    "directory": feature_cache_directory.name,
                }
            ),
            flush=True,
        )
        train_cache = train_dataset.prepare_feature_cache()
        validation_cache = validation_dataset.prepare_feature_cache()
        print(
            json.dumps(
                {
                    "stage": "feature_cache",
                    "status": "ready",
                    "train": train_cache,
                    "validation": validation_cache,
                }
            ),
            flush=True,
        )
    sampling_config: dict[str, object] = {
        "fast_bonus": arguments.hard_sample_fast_bonus,
        "high_fret_bonus": arguments.hard_sample_high_fret_bonus,
        "fast_interval": arguments.hard_sample_fast_interval,
        "high_fret": arguments.hard_sample_high_fret,
    }
    if dataset_sampling_weights:
        sampling_config["dataset_weights"] = dict(
            sorted(dataset_sampling_weights.items())
        )
    use_weighted_sampling = (
        arguments.hard_sample_fast_bonus > 0
        or arguments.hard_sample_high_fret_bonus > 0
        or bool(dataset_sampling_weights)
    )
    difficulty_weights = train_dataset.difficulty_sample_weights(
        fast_interval=arguments.hard_sample_fast_interval,
        high_fret=arguments.hard_sample_high_fret,
        fast_bonus=arguments.hard_sample_fast_bonus,
        high_fret_bonus=arguments.hard_sample_high_fret_bonus,
    )
    dataset_weights = train_dataset.dataset_sample_weights(
        dataset_sampling_weights
    )
    sample_weights = [
        difficulty_weight * dataset_weight
        for difficulty_weight, dataset_weight in zip(
            difficulty_weights,
            dataset_weights,
        )
    ]
    sampler = (
        WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(arguments.seed),
        )
        if use_weighted_sampling
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    if use_weighted_sampling:
        print(
            json.dumps(
                {
                    "stage": "weighted_sampling",
                    "configuration": sampling_config,
                    "minimum_weight": min(sample_weights),
                    "maximum_weight": max(sample_weights),
                    "mean_weight": sum(sample_weights) / len(sample_weights),
                    "weighted_windows": sum(
                        weight > 1 for weight in sample_weights
                    ),
                }
            ),
            flush=True,
        )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if validation_dataset
        else None
    )
    string_loss_weights = torch.tensor(
        train_dataset.active_string_weights(
            exponent=arguments.string_balance_exponent,
        ),
        device=device,
        dtype=torch.float32,
    )
    print(
        json.dumps(
            {
                "stage": "string_balance",
                "exponent": arguments.string_balance_exponent,
                "weights": [
                    round(float(weight), 6)
                    for weight in string_loss_weights.cpu()
                ],
            }
        ),
        flush=True,
    )

    model = StringFretNet(model_config)
    if arguments.resume is not None and source_checkpoint is not None:
        model.load_state_dict(source_checkpoint["model_state"])
    elif arguments.initialize_from is not None and source_checkpoint is not None:
        incompatible = model.load_state_dict(
            source_checkpoint["model_state"],
            strict=False,
        )
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not (
                key.startswith("pitch_embedding.")
                or key.startswith("position_head.")
            )
        ]
        if unexpected or invalid_missing:
            raise ValueError(
                "Initialization checkpoint is incompatible: "
                f"missing={invalid_missing}, unexpected={unexpected}"
            )
    configure_trainable_parameters(
        model,
        freeze_position_backbone=arguments.freeze_position_backbone,
        onset_head_only=arguments.onset_head_only,
    )
    model.to(device)
    position_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            name.startswith("pitch_embedding.")
            or name.startswith("position_head.")
        )
    ]
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not (
            name.startswith("pitch_embedding.")
            or name.startswith("position_head.")
        )
    ]
    optimizer_parameters: list[dict[str, object]] = []
    if backbone_parameters:
        optimizer_parameters.append(
            {
                "params": backbone_parameters,
                "lr": arguments.learning_rate,
            }
        )
    if position_parameters:
        optimizer_parameters.append(
            {
                "params": position_parameters,
                "lr": (
                    arguments.position_learning_rate
                    if arguments.position_learning_rate is not None
                    else arguments.learning_rate
                ),
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        weight_decay=1e-4,
    )
    if (
        arguments.resume is not None
        and source_checkpoint is not None
        and "optimizer_state" in source_checkpoint
    ):
        optimizer.load_state_dict(source_checkpoint["optimizer_state"])
        for index, parameter_group in enumerate(optimizer.param_groups):
            use_position_rate = (
                arguments.freeze_position_backbone
                or (
                    len(optimizer.param_groups) > 1
                    and index == len(optimizer.param_groups) - 1
                )
            )
            parameter_group["lr"] = (
                arguments.position_learning_rate
                if use_position_rate
                and arguments.position_learning_rate is not None
                else arguments.learning_rate
            )
    history: list[dict[str, object]] = (
        list(source_checkpoint.get("history", []))
        if arguments.resume is not None and source_checkpoint is not None
        else []
    )
    start_epoch = (
        int(source_checkpoint.get("metadata", {}).get("epoch", 0)) + 1
        if arguments.resume is not None and source_checkpoint is not None
        else 1
    )
    if start_epoch > arguments.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"--epochs must be at least {start_epoch}"
        )
    loss_config: dict[str, int | float] = {
        "attack_fret_weight": arguments.attack_fret_weight,
        "attack_fret_frames": arguments.attack_fret_frames,
        "fret_distance_weight": arguments.fret_distance_weight,
        "pitch_consistency_weight": arguments.pitch_consistency_weight,
        "pitch_positive_weight": arguments.pitch_positive_weight,
        "position_loss_weight": arguments.position_loss_weight,
        "string_balance_exponent": arguments.string_balance_exponent,
    }
    training_config: dict[str, object] = {
        "loss": loss_config,
        "sampling": sampling_config,
        "augmentation": {
            "pitch_shift_probability": arguments.pitch_shift_probability,
            "pitch_shift_min": arguments.pitch_shift_min,
            "pitch_shift_max": arguments.pitch_shift_max,
        },
        "selection_metric": arguments.selection_metric,
        "freeze_position_backbone": arguments.freeze_position_backbone,
        "onset_head_only": arguments.onset_head_only,
    }
    best_validation_metric = best_compatible_validation_metric(
        history,
        dict(source_checkpoint.get("metadata", {}))
        if arguments.resume is not None and source_checkpoint is not None
        else {},
        training_config,
        arguments.selection_metric,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, arguments.epochs + 1):
        training_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            progress_label=f"train_epoch_{epoch}",
            attack_fret_weight=arguments.attack_fret_weight,
            attack_fret_frames=arguments.attack_fret_frames,
            fret_distance_weight=arguments.fret_distance_weight,
            pitch_consistency_weight=arguments.pitch_consistency_weight,
            pitch_positive_weight=arguments.pitch_positive_weight,
            position_loss_weight=arguments.position_loss_weight,
            string_loss_weights=string_loss_weights,
            freeze_position_backbone=arguments.freeze_position_backbone,
            onset_head_only=arguments.onset_head_only,
        )
        validation_metrics = (
            run_epoch(
                model,
                validation_loader,
                device,
                None,
                progress_label=f"validation_epoch_{epoch}",
                attack_fret_weight=arguments.attack_fret_weight,
                attack_fret_frames=arguments.attack_fret_frames,
                fret_distance_weight=arguments.fret_distance_weight,
                pitch_consistency_weight=(
                    arguments.pitch_consistency_weight
                ),
                pitch_positive_weight=arguments.pitch_positive_weight,
                position_loss_weight=arguments.position_loss_weight,
                string_loss_weights=string_loss_weights,
                freeze_position_backbone=arguments.freeze_position_backbone,
                onset_head_only=arguments.onset_head_only,
            )
            if validation_loader
            else training_metrics
        )
        record = {
            "epoch": epoch,
            "train": training_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        current_validation_metric = validation_metrics[
            arguments.selection_metric
        ]
        improved = (
            current_validation_metric < best_validation_metric
            if arguments.selection_metric == "loss"
            else current_validation_metric > best_validation_metric
        )
        if not improved:
            continue
        best_validation_metric = current_validation_metric
        torch.save(
            {
                "format_version": 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "feature_config": feature_config.to_dict(),
                "metadata": {
                    "name": "stringtrace-string-fret-net",
                    "version": (
                        "tabcnn-gru-v3-pitch-conditioned"
                        if model_config.pitch_conditioned_position
                        else (
                            "tabcnn-gru-v2"
                            if model_config.frequency_encoding == "position-aware"
                            else "tabcnn-gru-v1"
                        )
                    ),
                    "kind": "trained",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dataset": arguments.manifest.resolve().parent.name,
                    "manifest_name": arguments.manifest.name,
                    "seed": arguments.seed,
                    "epoch": epoch,
                    "validation_loss": validation_metrics["loss"],
                    "selection_metric": arguments.selection_metric,
                    "selection_metric_value": current_validation_metric,
                    "batch_size": arguments.batch_size,
                    "window_seconds": arguments.window_seconds,
                    "window_hop_seconds": arguments.window_hop_seconds,
                    "learning_rate": arguments.learning_rate,
                    "position_learning_rate": (
                        arguments.position_learning_rate
                    ),
                    "attack_fret_weight": arguments.attack_fret_weight,
                    "attack_fret_frames": arguments.attack_fret_frames,
                    "fret_distance_weight": (
                        arguments.fret_distance_weight
                    ),
                    "pitch_consistency_weight": (
                        arguments.pitch_consistency_weight
                    ),
                    "pitch_positive_weight": (
                        arguments.pitch_positive_weight
                    ),
                    "position_loss_weight": (
                        arguments.position_loss_weight
                    ),
                    "pitch_conditioned_position": (
                        model_config.pitch_conditioned_position
                    ),
                    "freeze_position_backbone": (
                        arguments.freeze_position_backbone
                    ),
                    "onset_head_only": arguments.onset_head_only,
                    "string_balance_exponent": (
                        arguments.string_balance_exponent
                    ),
                    "string_loss_weights": [
                        round(float(weight), 6)
                        for weight in string_loss_weights.cpu()
                    ],
                    "loss_config": loss_config,
                    "sampling_config": sampling_config,
                    "training_config": training_config,
                    "pitch_shift_probability": (
                        arguments.pitch_shift_probability
                    ),
                    "pitch_shift_min": arguments.pitch_shift_min,
                    "pitch_shift_max": arguments.pitch_shift_max,
                    "resumed_from": (
                        arguments.resume.name
                        if arguments.resume is not None
                        else None
                    ),
                    "initialized_from": (
                        arguments.initialize_from.name
                        if arguments.initialize_from is not None
                        else None
                    ),
                },
                "history": history,
            },
            arguments.output,
        )
    print(
        json.dumps(
            {
                "checkpoint": arguments.output.name,
                "device": str(device),
                "train_windows": len(train_dataset),
                "validation_windows": len(validation_dataset),
                "selection_metric": arguments.selection_metric,
                "best_validation_metric": best_validation_metric,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
