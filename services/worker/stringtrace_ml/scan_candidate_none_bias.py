from __future__ import annotations

import argparse
import hashlib
import json
import math
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .evaluate_onset_position import load_model
from .onset_position import (
    OnsetPositionDataset,
    candidate_position_probabilities,
    candidate_position_queries,
)
from .schema import load_manifest, reject_quarantined_references


DEFAULT_BIASES = (-2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / max(1, denominator)


def _prf(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    return {
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": _ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
    }


def candidate_bias_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    none_logit_bias: float,
) -> dict[str, object]:
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("candidate logits and targets must have matching 2D shapes")
    if logits.shape[1] != 7:
        raise ValueError("categorical candidate logits must contain seven classes")
    if logits.shape[0] == 0:
        raise ValueError("candidate logits must not be empty")

    probabilities = candidate_position_probabilities(
        logits,
        objective="categorical",
        none_logit_bias=none_logit_bias,
    )
    predictions = probabilities.argmax(dim=1)
    reference_active = targets[:, 0] < 0.5
    predicted_active = predictions > 0
    valid_prediction = (
        targets.gather(1, predictions.unsqueeze(1)).squeeze(1) > 0
    )
    correct_string = reference_active & predicted_active & valid_prediction
    conditional_predictions = probabilities[:, 1:].argmax(dim=1) + 1
    oracle_string_match = (
        targets.gather(
            1,
            conditional_predictions.unsqueeze(1),
        ).squeeze(1)
        > 0
    )
    oracle_correct_string = reference_active & oracle_string_match

    accepted_true_positive = int((predicted_active & reference_active).sum())
    accepted_false_positive = int((predicted_active & ~reference_active).sum())
    accepted_false_negative = int((~predicted_active & reference_active).sum())
    rejected_true_positive = int((~predicted_active & ~reference_active).sum())
    rejected_false_positive = accepted_false_negative
    rejected_false_negative = accepted_false_positive
    reference_active_count = int(reference_active.sum())
    reference_inactive_count = int((~reference_active).sum())
    predicted_active_count = int(predicted_active.sum())
    predicted_inactive_count = int((~predicted_active).sum())
    correct_string_count = int(correct_string.sum())

    return {
        "none_logit_bias": none_logit_bias,
        "categorical_accuracy": _ratio(
            int(valid_prediction.sum()),
            logits.shape[0],
        ),
        "acceptance": {
            **_prf(
                accepted_true_positive,
                accepted_false_positive,
                accepted_false_negative,
            ),
            "true_positive": accepted_true_positive,
            "false_positive": accepted_false_positive,
            "false_negative": accepted_false_negative,
        },
        "rejection": {
            **_prf(
                rejected_true_positive,
                rejected_false_positive,
                rejected_false_negative,
            ),
            "true_positive": rejected_true_positive,
            "false_positive": rejected_false_positive,
            "false_negative": rejected_false_negative,
        },
        "active_string_accuracy": _ratio(
            correct_string_count,
            reference_active_count,
        ),
        "oracle_active_string_accuracy": _ratio(
            int(oracle_correct_string.sum()),
            reference_active_count,
        ),
        "conditional_accepted_string_accuracy": _ratio(
            correct_string_count,
            accepted_true_positive,
        ),
        "counts": {
            "queries": logits.shape[0],
            "reference_active": reference_active_count,
            "reference_inactive": reference_inactive_count,
            "predicted_active": predicted_active_count,
            "predicted_inactive": predicted_inactive_count,
            "false_accepts": accepted_false_positive,
            "false_rejects": accepted_false_negative,
            "wrong_strings": int(
                (reference_active & predicted_active & ~valid_prediction).sum()
            ),
            "correct_strings": correct_string_count,
            "oracle_wrong_strings": int(
                (reference_active & ~oracle_string_match).sum()
            ),
            "oracle_correct_strings": int(oracle_correct_string.sum()),
        },
    }


def collect_candidate_logits(
    dataset: OnsetPositionDataset,
    model: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits: list[Tensor] = []
    targets: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pitch_masks = batch["pitch_mask"]
            device_pitch_masks = pitch_masks.to(device)
            device_features = batch["features"].to(device)
            encoded = model.encode(
                device_features,
                device_pitch_masks,
            )
            batch_indexes, pitch_indexes, candidate_targets = (
                candidate_position_queries(
                    batch["event_pitches"],
                    batch["event_strings"],
                    pitch_masks,
                    categorical=True,
                )
            )
            candidate_logits = model.classify_candidate_positions(
                encoded[batch_indexes.to(device)],
                pitch_indexes.to(device),
                features=device_features[
                    batch_indexes.to(device)
                ],
                candidate_confidences=device_pitch_masks[
                    batch_indexes.to(device),
                    pitch_indexes.to(device),
                ],
                temporal_context=(
                    batch["candidate_temporal_context"]
                    .to(device)[
                        batch_indexes.to(device),
                        pitch_indexes.to(device),
                    ]
                    if model.config.candidate_temporal_string_residual
                    else None
                ),
            )
            logits.append(candidate_logits.detach().cpu())
            targets.append(candidate_targets)
    if not logits:
        raise RuntimeError("The validation dataset produced no candidate queries")
    return torch.cat(logits), torch.cat(targets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan categorical candidate-head none-logit offsets without "
            "rerunning model inference."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--biases",
        type=float,
        nargs="+",
        default=list(DEFAULT_BIASES),
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--candidate-onset-threshold", type=float, default=0.35)
    parser.add_argument("--candidate-frame-threshold", type=float, default=0.25)
    parser.add_argument(
        "--candidate-minimum-note-length",
        type=float,
        default=70.0,
    )
    parser.add_argument("--candidate-match-tolerance", type=float, default=0.05)
    return parser


def parse_arguments(arguments: list[str] | None = None) -> Namespace:
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    if parsed.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not parsed.biases:
        parser.error("--biases must not be empty")
    if any(not math.isfinite(bias) for bias in parsed.biases):
        parser.error("--biases must contain only finite values")
    if not 0 <= parsed.candidate_onset_threshold <= 1:
        parser.error("--candidate-onset-threshold must be in 0..1")
    if not 0 <= parsed.candidate_frame_threshold <= 1:
        parser.error("--candidate-frame-threshold must be in 0..1")
    if parsed.candidate_minimum_note_length <= 0:
        parser.error("--candidate-minimum-note-length must be positive")
    if parsed.candidate_match_tolerance <= 0:
        parser.error("--candidate-match-tolerance must be positive")
    return parsed


def main() -> None:
    arguments = parse_arguments()
    annotations = load_manifest(arguments.manifest, arguments.split)
    if not annotations:
        raise RuntimeError(f"The manifest contains no {arguments.split} tracks")
    reject_quarantined_references(annotations)

    device = torch.device(arguments.device)
    model, feature_config, model_config, metadata = load_model(
        arguments.checkpoint,
        device,
    )
    if (
        not model_config.candidate_position_head
        or model_config.candidate_position_objective
        not in ("categorical", "factorized", "spectral_residual")
    ):
        raise ValueError(
            "none-logit scanning requires a categorical candidate-position head"
        )

    manifest_directory = arguments.manifest.resolve().parent
    dataset = OnsetPositionDataset(
        arguments.manifest,
        split=arguments.split,
        feature_config=feature_config,
        config=model_config,
        feature_cache_directory=(
            arguments.feature_cache
            if arguments.feature_cache is not None
            else manifest_directory / ".feature-cache"
        ),
        candidate_cache_directory=(
            arguments.candidate_cache
            if arguments.candidate_cache is not None
            else manifest_directory / ".basic-pitch-cache"
        ),
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=arguments.candidate_minimum_note_length,
        candidate_match_tolerance=arguments.candidate_match_tolerance,
    )
    dataset.prepare_feature_cache()
    logits, targets = collect_candidate_logits(
        dataset,
        model,
        device,
        batch_size=arguments.batch_size,
    )
    unique_biases = list(dict.fromkeys(float(bias) for bias in arguments.biases))
    report = {
        "kind": "candidate-none-logit-bias-scan",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": arguments.split,
        "track_count": len(annotations),
        "group_count": len(dataset),
        "candidate_query_count": logits.shape[0],
        "manifest": {
            "name": arguments.manifest.name,
            "sha256": _sha256_file(arguments.manifest),
        },
        "checkpoint": {
            "name": arguments.checkpoint.name,
            "sha256": _sha256_file(arguments.checkpoint),
            "metadata": metadata,
        },
        "candidate_frontend": {
            "onset_threshold": arguments.candidate_onset_threshold,
            "frame_threshold": arguments.candidate_frame_threshold,
            "minimum_note_length": arguments.candidate_minimum_note_length,
            "match_tolerance": arguments.candidate_match_tolerance,
        },
        "filtered_negative_groups": dataset.filtered_negative_groups,
        "biases": [
            candidate_bias_metrics(
                logits,
                targets,
                none_logit_bias=bias,
            )
            for bias in unique_biases
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
