from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .candidate_sequence import CandidateSequenceGroup
from .data_sources import sha256_file
from .model import PITCH_MIN
from .schema import TrackAnnotation


CALIBRATION_ASSIGNMENT_ORDERS = {
    "candidate-sequence-margin-v1": "pre_assignment",
    "candidate-sequence-margin-v2": "post_assignment",
}


def candidate_string_probability_margins(
    string_logits: Tensor,
    valid_strings: Tensor,
) -> Tensor:
    if string_logits.ndim != 2 or string_logits.shape[1] != 6:
        raise ValueError("candidate string logits must have shape [items, 6]")
    if valid_strings.shape != string_logits.shape:
        raise ValueError("candidate valid-string masks must align")
    if bool((~valid_strings.any(dim=1)).any()):
        raise ValueError("every candidate must have a playable string")
    probabilities = torch.softmax(
        string_logits.masked_fill(~valid_strings, -torch.inf),
        dim=1,
    )
    top_two = probabilities.topk(k=2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def calibration_partition_keys(
    annotations: list[TrackAnnotation],
    *,
    fraction: float,
    seed: int,
) -> tuple[set[tuple[str, str]], dict[str, dict[str, int]]]:
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("calibration fraction must be between zero and one")
    groups_by_dataset: dict[str, set[str]] = {}
    for annotation in annotations:
        dataset = str(annotation.provenance.get("dataset", "unknown"))
        groups_by_dataset.setdefault(dataset, set()).add(annotation.group_id)
    if not groups_by_dataset:
        raise ValueError("calibration partition requires annotations")

    selected: set[tuple[str, str]] = set()
    counts: dict[str, dict[str, int]] = {}
    for dataset, group_ids in sorted(groups_by_dataset.items()):
        ranked = sorted(
            group_ids,
            key=lambda group_id: hashlib.sha256(
                f"{seed}:{dataset}:{group_id}".encode()
            ).digest(),
        )
        selected_count = max(
            1,
            min(
                len(ranked),
                math.floor(len(ranked) * fraction + 0.5),
            ),
        )
        selected.update(
            (dataset, group_id)
            for group_id in ranked[:selected_count]
        )
        counts[dataset] = {
            "available_groups": len(ranked),
            "selected_groups": selected_count,
        }
    return selected, counts


def calibration_partition_digest(
    keys: set[tuple[str, ...]],
) -> str:
    payload = json.dumps(
        sorted([list(key) for key in keys]),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def stratified_calibration_partition(
    annotations: list[TrackAnnotation],
    *,
    fraction: float,
    seed: int,
    stratify_by_group_datasets: set[str],
) -> tuple[set[int], set[tuple[str, ...]], dict[str, object]]:
    selected_group_keys, base_counts = calibration_partition_keys(
        annotations,
        fraction=fraction,
        seed=seed,
    )
    available_datasets = {
        str(annotation.provenance.get("dataset", "unknown"))
        for annotation in annotations
    }
    missing_datasets = stratify_by_group_datasets - available_datasets
    if missing_datasets:
        raise ValueError(
            "calibration stratification datasets are absent: "
            + ", ".join(sorted(missing_datasets))
        )

    selected_indexes: set[int] = set()
    selected_keys: set[tuple[str, ...]] = {
        ("group", dataset, group_id)
        for dataset, group_id in selected_group_keys
        if dataset not in stratify_by_group_datasets
    }
    counts: dict[str, object] = {
        dataset: {
            "mode": "dataset_group",
            **dataset_counts,
        }
        for dataset, dataset_counts in base_counts.items()
        if dataset not in stratify_by_group_datasets
    }
    tracks_by_stratum: dict[tuple[str, str], set[str]] = {}
    for annotation in annotations:
        dataset = str(annotation.provenance.get("dataset", "unknown"))
        if dataset in stratify_by_group_datasets:
            tracks_by_stratum.setdefault(
                (dataset, annotation.group_id),
                set(),
            ).add(annotation.track_id)

    selected_tracks: set[tuple[str, str, str]] = set()
    for (dataset, group_id), track_ids in sorted(
        tracks_by_stratum.items()
    ):
        ranked = sorted(
            track_ids,
            key=lambda track_id: hashlib.sha256(
                f"{seed}:{dataset}:{group_id}:{track_id}".encode()
            ).digest(),
        )
        selected_count = max(
            1,
            min(
                len(ranked),
                math.floor(len(ranked) * fraction + 0.5),
            ),
        )
        chosen = ranked[:selected_count]
        selected_tracks.update(
            (dataset, group_id, track_id) for track_id in chosen
        )
        dataset_counts = counts.setdefault(
            dataset,
            {
                "mode": "track_within_group",
                "available_groups": 0,
                "selected_groups": 0,
                "available_tracks": 0,
                "selected_tracks": 0,
                "groups": {},
            },
        )
        assert isinstance(dataset_counts, dict)
        dataset_counts["available_groups"] += 1
        dataset_counts["selected_groups"] += 1
        dataset_counts["available_tracks"] += len(ranked)
        dataset_counts["selected_tracks"] += selected_count
        groups = dataset_counts["groups"]
        assert isinstance(groups, dict)
        groups[group_id] = {
            "available_tracks": len(ranked),
            "selected_tracks": selected_count,
        }
        selected_keys.update(
            ("track", dataset, group_id, track_id)
            for track_id in chosen
        )

    for index, annotation in enumerate(annotations):
        dataset = str(annotation.provenance.get("dataset", "unknown"))
        if dataset in stratify_by_group_datasets:
            if (
                dataset,
                annotation.group_id,
                annotation.track_id,
            ) in selected_tracks:
                selected_indexes.add(index)
        elif (dataset, annotation.group_id) in selected_group_keys:
            selected_indexes.add(index)
    if not selected_indexes:
        raise ValueError("calibration partition selected no annotations")
    return selected_indexes, selected_keys, counts


def collect_margin_calibration_values(
    groups: list[CandidateSequenceGroup],
    *,
    assignment_order: str = "independent_top1",
) -> dict[str, np.ndarray]:
    if assignment_order not in ("independent_top1", "post_assignment"):
        raise ValueError("unsupported margin calibration assignment order")
    base_activity_logits: list[float] = []
    margins: list[float] = []
    reference_active: list[bool] = []
    top_string_matches: list[bool] = []
    repeated: list[bool] = []
    for group in groups:
        masked_logits = group.base_string_logits.masked_fill(
            ~group.valid_strings,
            -torch.inf,
        )
        if assignment_order == "independent_top1":
            predictions = masked_logits.argmax(dim=1)
            matches = torch.gather(
                group.string_targets > 0,
                1,
                predictions.unsqueeze(1),
            ).squeeze(1)
        else:
            from .onset_position import (
                assign_hierarchical_candidate_pitch_group,
            )

            probabilities = torch.softmax(masked_logits, dim=1)
            assignments = assign_hierarchical_candidate_pitch_group(
                torch.sigmoid(group.base_activity_logits)
                .detach()
                .cpu()
                .numpy(),
                probabilities.detach().cpu().numpy(),
                [
                    int(pitch) + PITCH_MIN
                    for pitch in group.pitches.tolist()
                ],
                minimum_activity_probability=0.5,
                minimum_string_margin=0.0,
                string_margin_assignment_order="post_assignment",
            )
            matches = torch.tensor(
                [
                    (
                        position is not None
                        and bool(
                            group.string_targets[
                                index,
                                position[0],
                            ]
                            > 0
                        )
                    )
                    for index, position in enumerate(assignments)
                ],
                dtype=torch.bool,
            )
        group_margins = candidate_string_probability_margins(
            group.base_string_logits,
            group.valid_strings,
        )
        base_activity_logits.extend(group.base_activity_logits.tolist())
        margins.extend(group_margins.tolist())
        reference_active.extend((group.activity_targets > 0).tolist())
        top_string_matches.extend(matches.tolist())
        repeated.extend((group.repeat_targets > 0).tolist())
    values = {
        "base_activity_logits": np.asarray(
            base_activity_logits,
            dtype=np.float64,
        ),
        "string_margins": np.asarray(margins, dtype=np.float64),
        "reference_active": np.asarray(reference_active, dtype=np.bool_),
        "top_string_matches": np.asarray(
            top_string_matches,
            dtype=np.bool_,
        ),
        "repeated": np.asarray(repeated, dtype=np.bool_),
    }
    shapes = {value.shape for value in values.values()}
    shape = next(iter(shapes)) if len(shapes) == 1 else ()
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError("margin calibration values must align and be non-empty")
    return values


def _f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> float:
    return (
        2
        * true_positive
        / max(1, 2 * true_positive + false_positive + false_negative)
    )


def _decision_metrics(
    *,
    threshold: float,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    wrong_string: int,
    repeat_true_positive: int,
    repeat_total: int,
    accepted: int,
) -> dict[str, float | int]:
    return {
        "threshold": threshold,
        "joint_f1": _f1(
            true_positive,
            false_positive,
            false_negative,
        ),
        "joint_true_positive": true_positive,
        "joint_false_positive": false_positive,
        "joint_false_negative": false_negative,
        "wrong_string": wrong_string,
        "repeat_position_recall": (
            repeat_true_positive / max(1, repeat_total)
        ),
        "repeat_true_positive": repeat_true_positive,
        "repeat_false_negative": repeat_total - repeat_true_positive,
        "accepted": accepted,
    }


def select_margin_threshold(
    values: dict[str, np.ndarray],
) -> dict[str, object]:
    required = {
        "base_activity_logits",
        "string_margins",
        "reference_active",
        "top_string_matches",
        "repeated",
    }
    if set(values) != required:
        raise ValueError("margin calibration values are incomplete")
    shapes = {np.asarray(value).shape for value in values.values()}
    shape = next(iter(shapes)) if len(shapes) == 1 else ()
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError("margin calibration values must align and be non-empty")
    activity_logits = np.asarray(
        values["base_activity_logits"],
        dtype=np.float64,
    )
    margins = np.asarray(values["string_margins"], dtype=np.float64)
    active = np.asarray(values["reference_active"], dtype=np.bool_)
    matches = np.asarray(values["top_string_matches"], dtype=np.bool_)
    repeated = np.asarray(values["repeated"], dtype=np.bool_)
    if (
        not bool(np.isfinite(activity_logits).all())
        or not bool(np.isfinite(margins).all())
        or bool(((margins < 0) | (margins > 1)).any())
        or bool((repeated & ~active).any())
    ):
        raise ValueError("margin calibration values are invalid")

    base_accepted = activity_logits >= 0
    correct_position = active & matches
    wrong_position = active & ~matches
    false_positive = ~active | ~matches
    repeat_correct = repeated & correct_position
    repeat_total = int(repeated.sum())

    base_true_positive = int((base_accepted & correct_position).sum())
    base_false_positive = int((base_accepted & false_positive).sum())
    base_false_negative = int(
        (active & (~base_accepted | ~matches)).sum()
    )
    base_wrong_string = int((base_accepted & wrong_position).sum())
    base_repeat_true_positive = int(
        (base_accepted & repeat_correct).sum()
    )
    base_accepted_count = int(base_accepted.sum())
    baseline = _decision_metrics(
        threshold=0.0,
        true_positive=base_true_positive,
        false_positive=base_false_positive,
        false_negative=base_false_negative,
        wrong_string=base_wrong_string,
        repeat_true_positive=base_repeat_true_positive,
        repeat_total=repeat_total,
        accepted=base_accepted_count,
    )

    accepted_indexes = np.flatnonzero(base_accepted)
    order = accepted_indexes[
        np.argsort(margins[accepted_indexes], kind="stable")
    ]
    ordered_margins = margins[order]
    unique_margins, start_indexes, group_counts = np.unique(
        ordered_margins,
        return_index=True,
        return_counts=True,
    )
    rejected_true_positive = np.cumsum(correct_position[order])
    rejected_false_positive = np.cumsum(false_positive[order])
    rejected_wrong_string = np.cumsum(wrong_position[order])
    rejected_repeat_true_positive = np.cumsum(repeat_correct[order])

    selected: dict[str, float | int] | None = None
    eligible_thresholds = 0
    evaluated_thresholds = 1
    for index, margin in enumerate(unique_margins):
        if margin >= 1:
            continue
        end = int(start_indexes[index] + group_counts[index] - 1)
        next_margin = (
            float(unique_margins[index + 1])
            if index + 1 < unique_margins.size
            else 1.0
        )
        threshold = float((float(margin) + next_margin) / 2)
        true_positive = (
            base_true_positive - int(rejected_true_positive[end])
        )
        current_false_positive = (
            base_false_positive - int(rejected_false_positive[end])
        )
        false_negative = (
            base_false_negative + int(rejected_true_positive[end])
        )
        wrong_string = (
            base_wrong_string - int(rejected_wrong_string[end])
        )
        repeat_true_positive = (
            base_repeat_true_positive
            - int(rejected_repeat_true_positive[end])
        )
        accepted = base_accepted_count - end - 1
        metrics = _decision_metrics(
            threshold=threshold,
            true_positive=true_positive,
            false_positive=current_false_positive,
            false_negative=false_negative,
            wrong_string=wrong_string,
            repeat_true_positive=repeat_true_positive,
            repeat_total=repeat_total,
            accepted=accepted,
        )
        evaluated_thresholds += 1
        if (
            float(metrics["joint_f1"]) <= float(baseline["joint_f1"])
            or wrong_string >= base_wrong_string
            or repeat_true_positive < base_repeat_true_positive
        ):
            continue
        eligible_thresholds += 1
        selection_key = (
            float(metrics["joint_f1"]),
            -wrong_string,
            true_positive,
            -threshold,
        )
        selected_key = (
            (
                float(selected["joint_f1"]),
                -int(selected["wrong_string"]),
                int(selected["joint_true_positive"]),
                -float(selected["threshold"]),
            )
            if selected is not None
            else None
        )
        if selected_key is None or selection_key > selected_key:
            selected = metrics

    calibration_gate_passed = selected is not None
    if selected is None:
        selected = dict(baseline)
    return {
        "selection_rule": (
            "maximize_joint_f1_then_minimize_wrong_string_then_"
            "maximize_true_positive_then_minimize_threshold"
        ),
        "constraints": {
            "joint_f1": "strict_increase",
            "wrong_string": "strict_decrease",
            "repeat_true_positive": "nondecrease",
        },
        "evaluated_thresholds": evaluated_thresholds,
        "eligible_thresholds": eligible_thresholds,
        "calibration_gate_passed": calibration_gate_passed,
        "baseline": baseline,
        "selected": selected,
    }


def load_candidate_margin_calibration(
    path: Path,
    *,
    position_checkpoint_path: Path,
    string_sequence_checkpoint_path: Path,
    candidate_onset_threshold: float,
    candidate_frame_threshold: float,
    candidate_minimum_note_length: float,
) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Unsupported candidate string margin calibration")
    version = str(report.get("version", ""))
    assignment_order = CALIBRATION_ASSIGNMENT_ORDERS.get(version)
    expected_schema = 1 if version == "candidate-sequence-margin-v1" else 2
    if (
        assignment_order is None
        or int(report.get("schema_version", 0)) != expected_schema
        or report.get("kind") != "candidate-string-margin-calibration"
    ):
        raise ValueError("Unsupported candidate string margin calibration")
    artifact_assignment_order = report.get(
        "assignment_order",
        "pre_assignment" if version == "candidate-sequence-margin-v1" else None,
    )
    if artifact_assignment_order != assignment_order:
        raise ValueError(
            "Candidate string margin calibration assignment order is invalid"
        )
    if (
        version == "candidate-sequence-margin-v2"
        and report.get("calibration_match_mode")
        != "activity_then_full_group_assignment"
    ):
        raise ValueError(
            "Candidate string margin calibration match mode is invalid"
        )
    if report.get("calibration_gate_passed") is not True:
        raise ValueError("Candidate string margin calibration did not pass")
    if (
        report.get("training_only") is not True
        or report.get("governed_validation_used") is not False
        or report.get("cross_source_used") is not False
    ):
        raise ValueError(
            "Candidate string margin calibration provenance is invalid"
        )
    expected_position = dict(report.get("position_checkpoint", {}))
    if expected_position.get("sha256") != sha256_file(
        position_checkpoint_path
    ):
        raise ValueError(
            "Candidate string margin calibration targets a different "
            "position checkpoint"
        )
    expected_string = dict(
        report.get("string_sequence_checkpoint", {})
    )
    if expected_string.get("sha256") != sha256_file(
        string_sequence_checkpoint_path
    ):
        raise ValueError(
            "Candidate string margin calibration targets a different "
            "string sequence checkpoint"
        )
    frontend = dict(report.get("candidate_frontend", {}))
    if frontend.get("kind") != "spotify-basic-pitch-0.4":
        raise ValueError(
            "Candidate string margin calibration frontend is unsupported"
        )
    expected_frontend = {
        "onset_threshold": candidate_onset_threshold,
        "frame_threshold": candidate_frame_threshold,
        "minimum_note_length": candidate_minimum_note_length,
    }
    for name, expected_value in expected_frontend.items():
        try:
            calibrated_value = float(frontend[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Candidate string margin calibration frontend is invalid"
            ) from error
        if not math.isclose(
            calibrated_value,
            expected_value,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Candidate string margin calibration frontend differs "
                f"for {name}"
            )
    threshold = float(report.get("selected_threshold", math.nan))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Candidate string margin threshold is invalid")
    selected = dict(report.get("selected", {}))
    try:
        selected_threshold = float(selected["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Candidate string margin selection is invalid"
        ) from error
    if not math.isclose(
        selected_threshold,
        threshold,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Candidate string margin selection threshold differs"
        )
    report["assignment_order"] = assignment_order
    return report
