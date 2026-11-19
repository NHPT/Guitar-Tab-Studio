from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data_sources import manifest_identity
from .features import FeatureConfig
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    OnsetPositionNet,
    candidate_position_loss,
    candidate_position_query_weights,
    candidate_position_queries,
    onset_position_loss,
    reference_position_counts,
    tablature_pitch_probabilities,
)
from .prepare_guitarset_stems import sha256_file
from .schema import load_manifest
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


def _precision(true_positive: int, false_positive: int) -> float:
    return true_positive / max(1, true_positive + false_positive)


def _recall(true_positive: int, false_negative: int) -> float:
    return true_positive / max(1, true_positive + false_negative)


def run_epoch(
    model: OnsetPositionNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    silence_weight: float,
    pitch_loss_weight: float,
    candidate_position_loss_weight: float,
    progress_label: str,
    adapter_only: bool = False,
    candidate_position_only: bool = False,
    candidate_activity_only: bool = False,
    candidate_temporal_string_only: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    categorical_candidate_head = (
        model.config.candidate_position_objective
        in ("categorical", "factorized", "spectral_residual")
    )
    model.train(training)
    if training and adapter_only:
        model.eval()
        if model.adapter is None:
            raise ValueError("adapter-only training requires an adapter")
        model.adapter.train()
    if training and candidate_position_only:
        model.eval()
        if model.candidate_head is None:
            raise ValueError(
                "candidate-position-only training requires its head"
            )
        if (
            model.config.candidate_position_objective
            == "spectral_residual"
        ):
            if model.candidate_residual_head is None:
                raise ValueError(
                    "spectral residual training requires its residual head"
                )
            model.candidate_residual_head.train()
        else:
            if model.candidate_pitch_embedding is None:
                raise ValueError(
                    "candidate-position-only training requires its "
                    "pitch embedding"
                )
            model.candidate_pitch_embedding.train()
            model.candidate_head.train()
    if training and candidate_activity_only:
        model.eval()
        if model.candidate_activity_residual_head is None:
            raise ValueError(
                "candidate-activity-only training requires its residual head"
            )
        model.candidate_activity_residual_head.train()
    if training and candidate_temporal_string_only:
        model.eval()
        if model.candidate_temporal_residual_head is None:
            raise ValueError(
                "candidate-temporal-string-only training requires its "
                "residual head"
            )
        model.candidate_temporal_residual_head.train()
    totals = {
        "loss": 0.0,
        "tablature_loss": 0.0,
        "pitch_loss": 0.0,
        "candidate_position_loss": 0.0,
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
        "candidate_position_correct": 0,
        "candidate_position_events": 0,
        "candidate_position_true_positive": 0,
        "candidate_position_false_positive": 0,
        "candidate_position_false_negative": 0,
        "candidate_rejection_true_positive": 0,
        "candidate_rejection_false_positive": 0,
        "candidate_rejection_false_negative": 0,
        "candidate_categorical_correct": 0,
        "candidate_categorical_events": 0,
        "candidate_oracle_string_correct": 0,
        "candidate_oracle_string_events": 0,
    }
    batches = 0
    for batch in loader:
        features = batch["features"].to(device)
        candidate_pitch_masks = batch["pitch_mask"]
        pitches = candidate_pitch_masks.to(device)
        pitch_targets = batch["pitch_targets"].to(device)
        fret_targets = batch["fret_targets"].to(device)
        event_pitches = batch["event_pitches"]
        event_strings = batch["event_strings"]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if training and (
                candidate_position_only
                or candidate_activity_only
                or candidate_temporal_string_only
            ):
                with torch.no_grad():
                    encoded = model.encode(features, pitches)
                    logits = model.classify(encoded)
            else:
                encoded = model.encode(features, pitches)
                logits = model.classify(encoded)
            loss, values = onset_position_loss(
                logits,
                fret_targets,
                pitch_targets,
                silence_weight=silence_weight,
                pitch_loss_weight=pitch_loss_weight,
            )
            candidate_loss = logits.sum() * 0
            candidate_correct = 0
            candidate_total = 0
            if model.candidate_head is not None:
                (
                    query_batch_indexes,
                    query_pitch_indexes,
                    candidate_targets,
                ) = candidate_position_queries(
                    event_pitches,
                    event_strings,
                    candidate_pitch_masks,
                    categorical=categorical_candidate_head,
                )
                query_batch_indexes = query_batch_indexes.to(device)
                query_pitch_indexes = query_pitch_indexes.to(device)
                candidate_targets = candidate_targets.to(device)
                query_weights = candidate_position_query_weights(
                    event_pitches,
                    batch["candidate_target_weights"],
                    candidate_pitch_masks,
                ).to(device)
                candidate_temporal_context = batch[
                    "candidate_temporal_context"
                ].to(device)
                candidate_logits = model.classify_candidate_positions(
                    encoded[query_batch_indexes],
                    query_pitch_indexes,
                    features=features[query_batch_indexes],
                    candidate_confidences=pitches[
                        query_batch_indexes,
                        query_pitch_indexes,
                    ],
                    temporal_context=(
                        candidate_temporal_context[
                            query_batch_indexes,
                            query_pitch_indexes,
                        ]
                        if model.config.candidate_temporal_string_residual
                        else None
                    ),
                )
                candidate_activity_logits = None
                if model.candidate_activity_residual_head is not None:
                    candidate_activity_logits = (
                        model.classify_candidate_activity(
                            encoded[query_batch_indexes],
                            query_pitch_indexes,
                            features=features[query_batch_indexes],
                            candidate_confidences=pitches[
                                query_batch_indexes,
                                query_pitch_indexes,
                            ],
                            candidate_position_logits=candidate_logits,
                        )
                    )
                    candidate_references_active = (
                        candidate_targets[:, 0] < 0.5
                    )
                    candidate_loss = (
                        functional.binary_cross_entropy_with_logits(
                            candidate_activity_logits,
                            candidate_references_active.to(
                                candidate_activity_logits.dtype
                            ),
                        )
                    )
                    conditional_predictions = (
                        candidate_logits[:, 1:].argmax(dim=1) + 1
                    )
                    conditional_matches = (
                        candidate_targets.gather(
                            1,
                            conditional_predictions.unsqueeze(1),
                        ).squeeze(1)
                        > 0
                    )
                    candidate_predictions_active = (
                        torch.sigmoid(candidate_activity_logits) >= 0.5
                    )
                    candidate_correct = int(
                        (
                            candidate_predictions_active
                            & candidate_references_active
                            & conditional_matches
                        ).sum().detach().cpu()
                    )
                    candidate_total = int(
                        candidate_references_active.sum().detach().cpu()
                    )
                else:
                    (
                        candidate_loss,
                        candidate_correct,
                        candidate_total,
                    ) = candidate_position_loss(
                        candidate_logits,
                        candidate_targets,
                        objective=(
                            model.config.candidate_position_objective
                        ),
                        sample_weights=(
                            query_weights if training else None
                        ),
                    )
                if categorical_candidate_head:
                    if candidate_activity_logits is not None:
                        candidate_predictions = torch.where(
                            candidate_predictions_active,
                            conditional_predictions,
                            torch.zeros_like(conditional_predictions),
                        )
                    else:
                        candidate_predictions = candidate_logits.argmax(dim=1)
                    candidate_matches = (
                        candidate_targets.gather(
                            1,
                            candidate_predictions.unsqueeze(1),
                        ).squeeze(1)
                        > 0
                    )
                    counts["candidate_categorical_correct"] += int(
                        candidate_matches.sum().detach().cpu()
                    )
                    counts["candidate_categorical_events"] += int(
                        candidate_matches.shape[0]
                    )
                    candidate_references_active = candidate_targets[:, 0] < 0.5
                    if candidate_activity_logits is None:
                        candidate_predictions_active = (
                            candidate_predictions > 0
                        )
                        conditional_predictions = (
                            candidate_logits[:, 1:].argmax(dim=1) + 1
                        )
                        conditional_matches = (
                            candidate_targets.gather(
                                1,
                                conditional_predictions.unsqueeze(1),
                            ).squeeze(1)
                            > 0
                        )
                    counts["candidate_oracle_string_correct"] += int(
                        (
                            candidate_references_active
                            & conditional_matches
                        ).sum().detach().cpu()
                    )
                    counts["candidate_oracle_string_events"] += int(
                        candidate_references_active.sum().detach().cpu()
                    )
                    counts["candidate_position_true_positive"] += int(
                        (
                            candidate_predictions_active
                            & candidate_references_active
                        ).sum().detach().cpu()
                    )
                    counts["candidate_position_false_positive"] += int(
                        (
                            candidate_predictions_active
                            & ~candidate_references_active
                        ).sum().detach().cpu()
                    )
                    counts["candidate_position_false_negative"] += int(
                        (
                            ~candidate_predictions_active
                            & candidate_references_active
                        ).sum().detach().cpu()
                    )
                    counts["candidate_rejection_true_positive"] += int(
                        (
                            ~candidate_predictions_active
                            & ~candidate_references_active
                        ).sum().detach().cpu()
                    )
                    counts["candidate_rejection_false_positive"] += int(
                        (
                            ~candidate_predictions_active
                            & candidate_references_active
                        ).sum().detach().cpu()
                    )
                    counts["candidate_rejection_false_negative"] += int(
                        (
                            candidate_predictions_active
                            & ~candidate_references_active
                        ).sum().detach().cpu()
                    )
                else:
                    candidate_predictions = (
                        torch.sigmoid(candidate_logits) >= 0.5
                    )
                    candidate_references = candidate_targets > 0
                    counts["candidate_position_true_positive"] += int(
                        (
                            candidate_predictions & candidate_references
                        ).sum().detach().cpu()
                    )
                    counts["candidate_position_false_positive"] += int(
                        (
                            candidate_predictions & ~candidate_references
                        ).sum().detach().cpu()
                    )
                    counts["candidate_position_false_negative"] += int(
                        (
                            ~candidate_predictions & candidate_references
                        ).sum().detach().cpu()
                    )
                loss = loss + (
                    candidate_position_loss_weight * candidate_loss
                )
            values["candidate_position_loss"] = candidate_loss.detach()
            values["loss"] = loss.detach()
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
        counts["candidate_position_correct"] += candidate_correct
        counts["candidate_position_events"] += candidate_total
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
    result["candidate_position_accuracy"] = (
        counts["candidate_position_correct"]
        / max(1, counts["candidate_position_events"])
    )
    result["candidate_position_events"] = float(
        counts["candidate_position_events"]
    )
    result["candidate_position_f1"] = _f1(
        counts["candidate_position_true_positive"],
        counts["candidate_position_false_positive"],
        counts["candidate_position_false_negative"],
    )
    if categorical_candidate_head:
        result["candidate_categorical_accuracy"] = (
            counts["candidate_categorical_correct"]
            / max(1, counts["candidate_categorical_events"])
        )
        result["candidate_string_accuracy"] = result[
            "candidate_position_accuracy"
        ]
        result["candidate_oracle_string_accuracy"] = (
            counts["candidate_oracle_string_correct"]
            / max(1, counts["candidate_oracle_string_events"])
        )
        result["candidate_acceptance_precision"] = _precision(
            counts["candidate_position_true_positive"],
            counts["candidate_position_false_positive"],
        )
        result["candidate_acceptance_recall"] = _recall(
            counts["candidate_position_true_positive"],
            counts["candidate_position_false_negative"],
        )
        result["candidate_acceptance_f1"] = result["candidate_position_f1"]
        result["candidate_rejection_precision"] = _precision(
            counts["candidate_rejection_true_positive"],
            counts["candidate_rejection_false_positive"],
        )
        result["candidate_rejection_recall"] = _recall(
            counts["candidate_rejection_true_positive"],
            counts["candidate_rejection_false_negative"],
        )
        result["candidate_rejection_f1"] = _f1(
            counts["candidate_rejection_true_positive"],
            counts["candidate_rejection_false_positive"],
            counts["candidate_rejection_false_negative"],
        )
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
        "--candidate-position-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "reference_position_accuracy",
            "candidate_position_accuracy",
            "candidate_position_f1",
            "candidate_string_accuracy",
            "candidate_oracle_string_accuracy",
            "candidate_categorical_accuracy",
            "candidate_acceptance_f1",
            "candidate_rejection_f1",
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
    parser.add_argument("--candidate-position-head", action="store_true")
    parser.add_argument(
        "--candidate-position-objective",
        choices=(
            "binary",
            "categorical",
            "factorized",
            "spectral_residual",
        ),
        default="binary",
    )
    parser.add_argument(
        "--candidate-spectral-evidence",
        action="store_true",
        help=(
            "Use pitch-local harmonic evidence and factorized string scoring."
        ),
    )
    parser.add_argument(
        "--candidate-position-only",
        action="store_true",
    )
    parser.add_argument(
        "--candidate-activity-residual",
        action="store_true",
    )
    parser.add_argument(
        "--candidate-activity-only",
        action="store_true",
    )
    parser.add_argument(
        "--candidate-temporal-string-residual",
        action="store_true",
    )
    parser.add_argument(
        "--candidate-temporal-string-only",
        action="store_true",
    )
    parser.add_argument(
        "--candidate-temporal-feature-count",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--candidate-temporal-hidden-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--candidate-temporal-max-interval",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--candidate-temporal-repeat-interval",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--candidate-ambiguity-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate-rare-pair-max-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate-cross-string-weight",
        type=float,
        default=1.0,
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
    if arguments.candidate_position_loss_weight < 0:
        parser.error(
            "--candidate-position-loss-weight must be non-negative"
        )
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
    if (
        arguments.candidate_position_only
        and arguments.initialize_from is None
    ):
        parser.error(
            "--candidate-position-only requires --initialize-from"
        )
    if (
        arguments.candidate_position_only
        and not arguments.candidate_position_head
    ):
        parser.error(
            "--candidate-position-only requires --candidate-position-head"
        )
    if (
        arguments.candidate_activity_only
        and arguments.initialize_from is None
    ):
        parser.error(
            "--candidate-activity-only requires --initialize-from"
        )
    if (
        arguments.candidate_activity_only
        and not arguments.candidate_activity_residual
    ):
        parser.error(
            "--candidate-activity-only requires "
            "--candidate-activity-residual"
        )
    if (
        arguments.candidate_temporal_string_only
        and arguments.initialize_from is None
    ):
        parser.error(
            "--candidate-temporal-string-only requires --initialize-from"
        )
    if (
        arguments.candidate_temporal_string_only
        and not arguments.candidate_temporal_string_residual
    ):
        parser.error(
            "--candidate-temporal-string-only requires "
            "--candidate-temporal-string-residual"
        )
    if (
        arguments.candidate_temporal_string_residual
        and (
            not arguments.candidate_position_head
            or arguments.candidate_position_objective
            != "spectral_residual"
        )
    ):
        parser.error(
            "--candidate-temporal-string-residual requires the spectral "
            "residual candidate position head"
        )
    if (
        arguments.candidate_activity_residual
        and (
            not arguments.candidate_position_head
            or arguments.candidate_position_objective
            != "spectral_residual"
        )
    ):
        parser.error(
            "--candidate-activity-residual requires the spectral "
            "residual candidate position head"
        )
    if (
        arguments.candidate_position_objective != "binary"
        and not arguments.candidate_position_head
    ):
        parser.error(
            "--candidate-position-objective requires "
            "--candidate-position-head"
        )
    if (
        arguments.candidate_spectral_evidence
        != (
            arguments.candidate_position_objective
            in ("factorized", "spectral_residual")
        )
    ):
        parser.error(
            "--candidate-spectral-evidence requires and is required by "
            "a spectral candidate position objective"
        )
    exclusive_training_modes = sum(
        (
            arguments.adapter_only,
            arguments.candidate_position_only,
            arguments.candidate_activity_only,
            arguments.candidate_temporal_string_only,
        )
    )
    if exclusive_training_modes > 1:
        parser.error(
            "--adapter-only, --candidate-position-only and "
            "--candidate-activity-only and "
            "--candidate-temporal-string-only are exclusive"
        )
    if arguments.candidate_temporal_feature_count != 20:
        parser.error("--candidate-temporal-feature-count must be 20")
    if arguments.candidate_temporal_hidden_size <= 0:
        parser.error("--candidate-temporal-hidden-size must be positive")
    if (
        arguments.candidate_temporal_max_interval <= 0
        or arguments.candidate_temporal_repeat_interval <= 0
        or arguments.candidate_cross_string_interval <= 0
    ):
        parser.error("candidate temporal intervals must be positive")
    for name, value in (
        ("--candidate-ambiguity-weight", arguments.candidate_ambiguity_weight),
        (
            "--candidate-rare-pair-max-weight",
            arguments.candidate_rare_pair_max_weight,
        ),
        (
            "--candidate-cross-string-weight",
            arguments.candidate_cross_string_weight,
        ),
        (
            "--candidate-max-supervision-weight",
            arguments.candidate_max_supervision_weight,
        ),
    ):
        if not np.isfinite(value) or value < 1:
            parser.error(f"{name} must be finite and at least one")
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
    if arguments.candidate_position_head:
        model_config = replace(
            model_config,
            candidate_position_head=True,
            candidate_position_objective=(
                arguments.candidate_position_objective
            ),
            candidate_spectral_evidence=(
                arguments.candidate_spectral_evidence
            ),
        )
    if arguments.candidate_activity_residual:
        model_config = replace(
            model_config,
            candidate_activity_residual=True,
        )
    if arguments.candidate_temporal_string_residual:
        model_config = replace(
            model_config,
            candidate_temporal_string_residual=True,
            candidate_temporal_feature_count=(
                arguments.candidate_temporal_feature_count
            ),
            candidate_temporal_hidden_size=(
                arguments.candidate_temporal_hidden_size
            ),
            candidate_temporal_max_interval=(
                arguments.candidate_temporal_max_interval
            ),
            candidate_temporal_repeat_interval=(
                arguments.candidate_temporal_repeat_interval
            ),
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
                "candidate_supervision": (
                    train_dataset.candidate_supervision
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
            and not (
                arguments.candidate_position_head
                and (
                    name.startswith("candidate_pitch_embedding.")
                    or name.startswith("candidate_head.")
                    or name.startswith("candidate_residual_head.")
                )
            )
            and not (
                arguments.candidate_activity_residual
                and name.startswith("candidate_activity_residual_head.")
            )
            and not (
                arguments.candidate_temporal_string_residual
                and name.startswith("candidate_temporal_residual_head.")
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
    if arguments.candidate_position_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if model.candidate_head is None:
            raise RuntimeError(
                "Candidate position head configuration was not applied"
            )
        if (
            model.config.candidate_position_objective
            == "spectral_residual"
        ):
            if model.candidate_residual_head is None:
                raise RuntimeError(
                    "Candidate residual configuration was not applied"
                )
            for parameter in model.candidate_residual_head.parameters():
                parameter.requires_grad = True
        else:
            if model.candidate_pitch_embedding is None:
                raise RuntimeError(
                    "Candidate pitch embedding was not applied"
                )
            for parameter in model.candidate_pitch_embedding.parameters():
                parameter.requires_grad = True
            for parameter in model.candidate_head.parameters():
                parameter.requires_grad = True
    if arguments.candidate_activity_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if model.candidate_activity_residual_head is None:
            raise RuntimeError(
                "Candidate activity residual configuration was not applied"
            )
        for parameter in (
            model.candidate_activity_residual_head.parameters()
        ):
            parameter.requires_grad = True
    if arguments.candidate_temporal_string_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if model.candidate_temporal_residual_head is None:
            raise RuntimeError(
                "Candidate temporal residual configuration was not applied"
            )
        for parameter in (
            model.candidate_temporal_residual_head.parameters()
        ):
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
    model_version = (
        "candidate-temporal-string-v1"
        if model_config.candidate_temporal_string_residual
        else (
            "candidate-activity-residual-v1"
            if model_config.candidate_activity_residual
            else (
                {
                    "binary": "candidate-position-head-v1",
                    "categorical": "candidate-position-head-v2",
                    "factorized": "candidate-position-head-v3",
                    "spectral_residual": "candidate-position-head-v4",
                }[model_config.candidate_position_objective]
                if model_config.candidate_position_head
                else "cqt-flux-pitch-conditioned-v1"
            )
        )
    )
    initialized_checkpoint = (
        {
            "name": arguments.initialize_from.name,
            "sha256": sha256_file(arguments.initialize_from),
        }
        if arguments.initialize_from is not None
        else None
    )
    inherited_base_checkpoint = (
        dict(
            dict(source_checkpoint.get("metadata", {})).get(
                "base_checkpoint",
                {},
            )
        )
        if source_checkpoint is not None
        else {}
    )
    base_checkpoint = (
        inherited_base_checkpoint
        if inherited_base_checkpoint.get("sha256")
        else initialized_checkpoint
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            silence_weight=arguments.silence_weight,
            pitch_loss_weight=arguments.pitch_loss_weight,
            candidate_position_loss_weight=(
                arguments.candidate_position_loss_weight
            ),
            progress_label=f"train_epoch_{epoch}",
            adapter_only=arguments.adapter_only,
            candidate_position_only=arguments.candidate_position_only,
            candidate_activity_only=arguments.candidate_activity_only,
            candidate_temporal_string_only=(
                arguments.candidate_temporal_string_only
            ),
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            None,
            silence_weight=arguments.silence_weight,
            pitch_loss_weight=arguments.pitch_loss_weight,
            candidate_position_loss_weight=(
                arguments.candidate_position_loss_weight
            ),
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
                        "version": model_version,
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
                        "training_data": training_data_identity,
                        "validation_data": validation_data_identity,
                        "seed": arguments.seed,
                        "epoch": epoch,
                        "selection_metric": arguments.selection_metric,
                        "selection_metric_value": metric_value,
                        "silence_weight": arguments.silence_weight,
                        "pitch_loss_weight": arguments.pitch_loss_weight,
                        "candidate_position_loss_weight": (
                            arguments.candidate_position_loss_weight
                        ),
                        "candidate_position_objective": (
                            model_config.candidate_position_objective
                        ),
                        "trainable_scope": (
                            "candidate-temporal-string-residual"
                            if arguments.candidate_temporal_string_only
                            else (
                                "candidate-activity-residual"
                                if arguments.candidate_activity_only
                                else (
                                    (
                                        "candidate-string-residual"
                                        if (
                                            model_config
                                            .candidate_position_objective
                                            == "spectral_residual"
                                        )
                                        else "candidate-position-head"
                                    )
                                    if arguments.candidate_position_only
                                    else (
                                        "adapter"
                                        if arguments.adapter_only
                                        else "full-model"
                                    )
                                )
                            )
                        ),
                        "initialized_from": (
                            arguments.initialize_from.name
                            if arguments.initialize_from is not None
                            else None
                        ),
                        "initialized_checkpoint": initialized_checkpoint,
                        "base_checkpoint": base_checkpoint,
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
                        "candidate_temporal_context": {
                            "enabled": (
                                model_config
                                .candidate_temporal_string_residual
                            ),
                            "feature_count": (
                                model_config
                                .candidate_temporal_feature_count
                            ),
                            "hidden_size": (
                                model_config
                                .candidate_temporal_hidden_size
                            ),
                            "maximum_interval": (
                                model_config
                                .candidate_temporal_max_interval
                            ),
                            "repeat_interval": (
                                model_config
                                .candidate_temporal_repeat_interval
                            ),
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
                                arguments
                                .candidate_cross_string_interval
                            ),
                            "maximum_supervision_weight": (
                                arguments
                                .candidate_max_supervision_weight
                            ),
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
