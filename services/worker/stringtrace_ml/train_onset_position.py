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

from .features import FeatureConfig
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    OnsetPositionNet,
    onset_position_loss,
    reference_position_counts,
    tablature_pitch_probabilities,
)
from .train import choose_device


def _f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> float:
    return (
        2 * true_positive
        / max(1, 2 * true_positive + false_positive + false_negative)
    )


def run_epoch(
    model: OnsetPositionNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    silence_weight: float,
    pitch_loss_weight: float,
    progress_label: str,
    adapter_only: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and adapter_only:
        model.eval()
        if model.adapter is None:
            raise ValueError("adapter-only training requires an adapter")
        model.adapter.train()
    totals = {
        "loss": 0.0,
        "tablature_loss": 0.0,
        "pitch_loss": 0.0,
    }
    counts = {
        "tablature_true_positive": 0,
        "tablature_false_positive": 0,
        "tablature_false_negative": 0,
        "pitch_true_positive": 0,
        "pitch_false_positive": 0,
        "pitch_false_negative": 0,
        "position_correct": 0,
        "position_events": 0,
    }
    batches = 0
    for batch in loader:
        features = batch["features"].to(device)
        pitches = batch["pitch_mask"].to(device)
        pitch_targets = batch["pitch_targets"].to(device)
        fret_targets = batch["fret_targets"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features, pitches)
            loss, values = onset_position_loss(
                logits,
                fret_targets,
                pitch_targets,
                silence_weight=silence_weight,
                pitch_loss_weight=pitch_loss_weight,
            )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=-1)
        predictions = probabilities.argmax(dim=-1)
        exact = (predictions == fret_targets) & (fret_targets > 0)
        predicted_active = predictions > 0
        reference_active = fret_targets > 0
        counts["tablature_true_positive"] += int(exact.sum().cpu())
        counts["tablature_false_positive"] += int(
            (predicted_active & ~exact).sum().cpu()
        )
        counts["tablature_false_negative"] += int(
            (reference_active & ~exact).sum().cpu()
        )

        predicted_pitches = tablature_pitch_probabilities(probabilities) >= 0.5
        reference_pitches = pitch_targets > 0
        pitch_exact = predicted_pitches & reference_pitches
        counts["pitch_true_positive"] += int(pitch_exact.sum().cpu())
        counts["pitch_false_positive"] += int(
            (predicted_pitches & ~reference_pitches).sum().cpu()
        )
        counts["pitch_false_negative"] += int(
            (~predicted_pitches & reference_pitches).sum().cpu()
        )
        correct, total = reference_position_counts(
            probabilities.cpu().numpy(),
            batch["event_pitches"].numpy(),
            batch["event_strings"].numpy(),
        )
        counts["position_correct"] += correct
        counts["position_events"] += total
        for key in totals:
            totals[key] += float(values[key].cpu())
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
    for name in ("tablature", "pitch"):
        result[f"{name}_f1"] = _f1(
            counts[f"{name}_true_positive"],
            counts[f"{name}_false_positive"],
            counts[f"{name}_false_negative"],
        )
    result["reference_position_accuracy"] = (
        counts["position_correct"] / max(1, counts["position_events"])
    )
    result["position_events"] = float(counts["position_events"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a pitch-constrained string/fret classifier on short "
            "onset-centered CQT windows."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--silence-weight", type=float, default=0.2)
    parser.add_argument("--pitch-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "reference_position_accuracy",
            "tablature_f1",
            "pitch_f1",
        ),
        default="reference_position_accuracy",
    )
    parser.add_argument("--frames-before", type=int, default=2)
    parser.add_argument("--frames-after", type=int, default=12)
    parser.add_argument("--group-tolerance", type=float, default=0.045)
    parser.add_argument("--onset-jitter-seconds", type=float, default=0.0)
    parser.add_argument("--pitch-drop-probability", type=float, default=0.0)
    parser.add_argument("--pitch-add-probability", type=float, default=0.0)
    parser.add_argument("--repeat-sample-bonus", type=float, default=0.0)
    parser.add_argument("--repeat-sample-interval", type=float, default=0.75)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--train-candidate-cache", type=Path)
    parser.add_argument("--validation-candidate-cache", type=Path)
    parser.add_argument("--candidate-confidence", action="store_true")
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
        help=(
            "For a sparsely annotated training dataset, discard candidate "
            "groups that do not match a reference event."
        ),
    )
    parser.add_argument("--adapter-size", type=int, default=0)
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260923)
    arguments = parser.parse_args()

    if arguments.frames_before < 0 or arguments.frames_after <= 0:
        parser.error("The onset window must contain at least one frame")
    if not 0 < arguments.silence_weight <= 1:
        parser.error("--silence-weight must be in (0, 1]")
    if arguments.pitch_loss_weight < 0:
        parser.error("--pitch-loss-weight must be non-negative")
    if arguments.onset_jitter_seconds < 0:
        parser.error("--onset-jitter-seconds must be non-negative")
    if arguments.repeat_sample_bonus < 0:
        parser.error("--repeat-sample-bonus must be non-negative")
    if arguments.repeat_sample_interval <= 0:
        parser.error("--repeat-sample-interval must be positive")
    if arguments.adapter_size < 0:
        parser.error("--adapter-size must be non-negative")
    if arguments.adapter_only and arguments.initialize_from is None:
        parser.error("--adapter-only requires --initialize-from")
    for name, value in (
        ("--pitch-drop-probability", arguments.pitch_drop_probability),
        ("--pitch-add-probability", arguments.pitch_add_probability),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be in 0..1")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = choose_device(arguments.device)
    source_checkpoint = (
        torch.load(
            arguments.initialize_from,
            map_location=device,
            weights_only=False,
        )
        if arguments.initialize_from is not None
        else None
    )
    feature_config = (
        FeatureConfig.from_dict(
            dict(source_checkpoint.get("feature_config", {}))
        )
        if source_checkpoint is not None
        else FeatureConfig()
    )
    model_config = (
        OnsetPositionConfig.from_dict(
            dict(source_checkpoint.get("model_config", {}))
        )
        if source_checkpoint is not None
        else OnsetPositionConfig(
            frames_before=arguments.frames_before,
            frames_after=arguments.frames_after,
            group_tolerance=arguments.group_tolerance,
        )
    )
    if arguments.candidate_confidence:
        model_config = replace(
            model_config,
            use_candidate_confidence=True,
        )
    if arguments.adapter_size > 0:
        model_config = replace(
            model_config,
            adapter_size=arguments.adapter_size,
        )
    if arguments.adapter_only and model_config.adapter_size <= 0:
        parser.error("--adapter-only requires a positive adapter size")
    feature_cache = (
        arguments.feature_cache.resolve()
        if arguments.feature_cache is not None
        else arguments.manifest.resolve().parent / ".feature-cache"
    )
    validation_manifest = (
        arguments.validation_manifest
        if arguments.validation_manifest is not None
        else arguments.manifest
    )
    train_candidate_cache = (
        arguments.train_candidate_cache
        if arguments.train_candidate_cache is not None
        else arguments.candidate_cache
    )
    validation_candidate_cache = (
        arguments.validation_candidate_cache
        if arguments.validation_candidate_cache is not None
        else arguments.candidate_cache
    )
    train_dataset = OnsetPositionDataset(
        arguments.manifest,
        split="train",
        feature_config=feature_config,
        config=model_config,
        feature_cache_directory=feature_cache,
        onset_jitter_seconds=arguments.onset_jitter_seconds,
        pitch_drop_probability=arguments.pitch_drop_probability,
        pitch_add_probability=arguments.pitch_add_probability,
        candidate_cache_directory=train_candidate_cache,
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
        config=model_config,
        feature_cache_directory=feature_cache,
        candidate_cache_directory=validation_candidate_cache,
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
                "filtered_negative_groups": (
                    train_dataset.filtered_negative_groups
                ),
            }
        ),
        flush=True,
    )
    repeat_sample_weights = train_dataset.repeat_sample_weights(
        interval=arguments.repeat_sample_interval,
        bonus=arguments.repeat_sample_bonus,
    )
    repeat_sampler = (
        WeightedRandomSampler(
            repeat_sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(arguments.seed),
        )
        if arguments.repeat_sample_bonus > 0
        else None
    )
    if repeat_sampler is not None:
        print(
            json.dumps(
                {
                    "stage": "repeat_sampling",
                    "interval": arguments.repeat_sample_interval,
                    "bonus": arguments.repeat_sample_bonus,
                    "minimum_weight": min(repeat_sample_weights),
                    "maximum_weight": max(repeat_sample_weights),
                    "mean_weight": (
                        sum(repeat_sample_weights)
                        / len(repeat_sample_weights)
                    ),
                    "weighted_groups": sum(
                        weight > 1
                        for weight in repeat_sample_weights
                    ),
                }
            ),
            flush=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=repeat_sampler is None,
        sampler=repeat_sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = OnsetPositionNet(feature_config, model_config).to(device)
    if source_checkpoint is not None:
        incompatible = model.load_state_dict(
            source_checkpoint["model_state"],
            strict=False,
        )
        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if not (
                arguments.adapter_only
                and name.startswith("adapter.")
            )
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible initialization checkpoint: "
                f"missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    if arguments.adapter_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if model.adapter is None:
            raise RuntimeError("Adapter configuration was not applied")
        for parameter in model.adapter.parameters():
            parameter.requires_grad = True
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
            model,
            train_loader,
            device,
            optimizer,
            silence_weight=arguments.silence_weight,
            pitch_loss_weight=arguments.pitch_loss_weight,
            progress_label=f"train_epoch_{epoch}",
            adapter_only=arguments.adapter_only,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            None,
            silence_weight=arguments.silence_weight,
            pitch_loss_weight=arguments.pitch_loss_weight,
            progress_label=f"validation_epoch_{epoch}",
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        metric_value = validation_metrics[arguments.selection_metric]
        if metric_value > best_metric:
            best_metric = metric_value
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "name": "stringtrace-onset-position-net",
                        "version": "cqt-flux-pitch-conditioned-v1",
                        "kind": "trained",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "training_datasets": sorted(
                            {
                                str(
                                    annotation.provenance.get(
                                        "dataset",
                                        "unknown",
                                    )
                                )
                                for annotation in train_dataset.annotations
                            }
                        ),
                        "validation_datasets": sorted(
                            {
                                str(
                                    annotation.provenance.get(
                                        "dataset",
                                        "unknown",
                                    )
                                )
                                for annotation in (
                                    validation_dataset.annotations
                                )
                            }
                        ),
                        "manifest_name": arguments.manifest.name,
                        "validation_manifest_name": (
                            validation_manifest.name
                        ),
                        "seed": arguments.seed,
                        "epoch": epoch,
                        "selection_metric": arguments.selection_metric,
                        "selection_metric_value": metric_value,
                        "silence_weight": arguments.silence_weight,
                        "pitch_loss_weight": arguments.pitch_loss_weight,
                        "initialized_from": (
                            arguments.initialize_from.name
                            if arguments.initialize_from is not None
                            else None
                        ),
                        "augmentation": {
                            "onset_jitter_seconds": (
                                arguments.onset_jitter_seconds
                            ),
                            "pitch_drop_probability": (
                                arguments.pitch_drop_probability
                            ),
                            "pitch_add_probability": (
                                arguments.pitch_add_probability
                            ),
                        },
                        "repeat_sampling": {
                            "interval": (
                                arguments.repeat_sample_interval
                            ),
                            "bonus": arguments.repeat_sample_bonus,
                        },
                        "positive_only_datasets": sorted(
                            set(arguments.positive_only_dataset)
                        ),
                        "filtered_negative_groups": (
                            train_dataset.filtered_negative_groups
                        ),
                        "adapter": {
                            "size": model_config.adapter_size,
                            "adapter_only": arguments.adapter_only,
                            "trainable_parameters": sum(
                                parameter.numel()
                                for parameter in trainable_parameters
                            ),
                        },
                        "training_candidate_frontend": (
                            {
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
                                "confidence_input": (
                                    model_config.use_candidate_confidence
                                ),
                            }
                            if train_candidate_cache is not None
                            else None
                        ),
                        "candidate_frontend": (
                            {
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
                                "confidence_input": (
                                    model_config.use_candidate_confidence
                                ),
                            }
                            if validation_candidate_cache is not None
                            else None
                        ),
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
                "adapter_only": arguments.adapter_only,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in trainable_parameters
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
