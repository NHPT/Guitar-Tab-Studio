from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .candidate_sequence import (
    CandidateSequenceGroup,
    infer_candidate_sequence_confidence_logits,
    load_candidate_sequence_confidence_model,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import manifest_identity, sha256_file
from .onset_position import OnsetPositionDataset
from .schema import load_manifest
from .train import choose_device
from .train_candidate_sequence import load_position_model
from .train_candidate_sequence_activity import (
    refine_groups_with_string_sequence,
)


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("distribution values must be a non-empty vector")
    if not bool(np.isfinite(values).all()):
        raise ValueError("distribution values must be finite")
    quantiles = np.quantile(
        values,
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    )
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "minimum": float(values.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "maximum": float(values.max()),
        "fraction_below_zero": float((values < 0).mean()),
        "fraction_at_or_above_zero": float((values >= 0).mean()),
    }


def separation_metrics(
    correctness_targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    targets = np.asarray(correctness_targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("separation targets and scores must align")
    if targets.size == 0 or not bool(np.isfinite(scores).all()):
        raise ValueError("separation scores must be non-empty and finite")
    if not bool(np.isin(targets, [0, 1]).all()):
        raise ValueError("separation targets must be binary")
    if np.unique(targets).size != 2:
        raise ValueError("separation metrics require both target classes")
    positive = targets == 1
    negative = ~positive
    accepted = scores >= 0
    return {
        "count": int(targets.size),
        "correct": int(positive.sum()),
        "wrong": int(negative.sum()),
        "correct_rate": float(positive.mean()),
        "auroc": float(roc_auc_score(targets, scores)),
        "average_precision": float(
            average_precision_score(targets, scores)
        ),
        "mean_score_gap_correct_minus_wrong": float(
            scores[positive].mean() - scores[negative].mean()
        ),
        "zero_threshold_correct_retention": float(
            accepted[positive].mean()
        ),
        "zero_threshold_wrong_rejection": float(
            (~accepted[negative]).mean()
        ),
        "zero_threshold_accepted_correct": int(
            (accepted & positive).sum()
        ),
        "zero_threshold_accepted_wrong": int(
            (accepted & negative).sum()
        ),
    }


def collect_confidence_scores(
    groups: list[CandidateSequenceGroup],
    model: torch.nn.Module,
    device: torch.device,
    *,
    inference_batch_size: int,
) -> dict[str, np.ndarray]:
    groups_by_annotation: dict[int, list[CandidateSequenceGroup]] = (
        defaultdict(list)
    )
    for group in groups:
        groups_by_annotation[group.annotation_index].append(group)

    targets: list[bool] = []
    confidence_logits: list[float] = []
    string_margins: list[float] = []
    repeat_targets: list[bool] = []
    for annotation_groups in groups_by_annotation.values():
        annotation_logits = infer_candidate_sequence_confidence_logits(
            model,
            annotation_groups,
            device,
            batch_size=inference_batch_size,
        )
        for group, group_logits in zip(
            annotation_groups,
            annotation_logits,
        ):
            active = group.activity_targets > 0
            if not bool(active.any()):
                continue
            masked_logits = group.base_string_logits.masked_fill(
                ~group.valid_strings,
                -torch.inf,
            )
            predictions = masked_logits.argmax(dim=1)
            matches = torch.gather(
                group.string_targets > 0,
                1,
                predictions.unsqueeze(1),
            ).squeeze(1)
            probabilities = torch.softmax(masked_logits, dim=1)
            top_two = probabilities.topk(k=2, dim=1).values
            margins = top_two[:, 0] - top_two[:, 1]

            targets.extend(matches[active].tolist())
            confidence_logits.extend(group_logits[active].tolist())
            string_margins.extend(margins[active].tolist())
            repeat_targets.extend(
                (group.repeat_targets[active] > 0).tolist()
            )

    return {
        "correctness_targets": np.asarray(targets, dtype=np.int64),
        "confidence_logits": np.asarray(
            confidence_logits,
            dtype=np.float64,
        ),
        "string_probability_margins": np.asarray(
            string_margins,
            dtype=np.float64,
        ),
        "repeat_targets": np.asarray(repeat_targets, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a candidate sequence confidence checkpoint "
            "separates correct and wrong frozen top-string predictions."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--string-sequence-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--confidence-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--repeat-interval", type=float, default=0.75)
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
    confidence_model, confidence_metadata = (
        load_candidate_sequence_confidence_model(
            arguments.confidence_checkpoint,
            arguments.position_checkpoint,
            arguments.string_sequence_checkpoint,
            position_config,
            feature_config,
            device,
        )
    )
    dataset = OnsetPositionDataset(
        arguments.manifest,
        split=arguments.split,
        feature_config=feature_config,
        config=position_config,
        feature_cache_directory=arguments.feature_cache,
        candidate_cache_directory=arguments.candidate_cache,
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=(
            arguments.candidate_minimum_note_length
        ),
    )
    cache_report = dataset.prepare_feature_cache()
    print(
        json.dumps(
            {
                "stage": "precompute_groups",
                "groups": len(dataset),
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
        json.dumps({"stage": "apply_frozen_string_sequence"}),
        flush=True,
    )
    groups = refine_groups_with_string_sequence(
        groups,
        string_model,
        device,
        batch_size=arguments.inference_batch_size,
    )
    values = collect_confidence_scores(
        groups,
        confidence_model,
        device,
        inference_batch_size=arguments.inference_batch_size,
    )
    targets = values["correctness_targets"]
    correct = targets == 1
    wrong = ~correct
    repeat = values["repeat_targets"] == 1
    confidence_logits = values["confidence_logits"]
    string_margins = values["string_probability_margins"]
    if targets.size == 0 or not bool(correct.any()) or not bool(wrong.any()):
        raise RuntimeError(
            "confidence audit requires active correct and wrong candidates"
        )

    annotations = load_manifest(arguments.manifest.resolve())
    report = {
        "schema_version": 1,
        "kind": "candidate-sequence-confidence-separability-audit",
        "diagnostic_only": True,
        "threshold_selection_authorized": False,
        "manifest": manifest_identity(
            arguments.manifest.resolve(),
            annotations,
            purpose="training",
        ),
        "split": arguments.split,
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
        "confidence_checkpoint": {
            "name": arguments.confidence_checkpoint.name,
            "sha256": sha256_file(arguments.confidence_checkpoint),
            "selected_epoch": confidence_metadata.get("epoch"),
        },
        "candidate_frontend": {
            "kind": "spotify-basic-pitch-0.4",
            "onset_threshold": arguments.candidate_onset_threshold,
            "frame_threshold": arguments.candidate_frame_threshold,
            "minimum_note_length": (
                arguments.candidate_minimum_note_length
            ),
        },
        "cache": cache_report,
        "groups": len(groups),
        "active_candidates": int(targets.size),
        "correct_top_string": int(correct.sum()),
        "wrong_top_string": int(wrong.sum()),
        "repeated_active_candidates": int(repeat.sum()),
        "confidence_logit": {
            "separation": separation_metrics(
                targets,
                confidence_logits,
            ),
            "distribution": {
                "all": distribution_summary(confidence_logits),
                "correct": distribution_summary(
                    confidence_logits[correct]
                ),
                "wrong": distribution_summary(
                    confidence_logits[wrong]
                ),
                "repeated": distribution_summary(
                    confidence_logits[repeat]
                ),
            },
        },
        "frozen_string_probability_margin": {
            "separation": separation_metrics(targets, string_margins),
            "distribution": {
                "all": distribution_summary(string_margins),
                "correct": distribution_summary(string_margins[correct]),
                "wrong": distribution_summary(string_margins[wrong]),
                "repeated": distribution_summary(string_margins[repeat]),
            },
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
                "active_candidates": int(targets.size),
                "confidence_auroc": report["confidence_logit"][
                    "separation"
                ]["auroc"],
                "margin_auroc": report[
                    "frozen_string_probability_margin"
                ]["separation"]["auroc"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
