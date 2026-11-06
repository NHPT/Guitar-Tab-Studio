from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reference_review import load_project, sha256_file
from .schema import STANDARD_TUNING_LIST


def priority_strings_from_coverage(report: dict[str, Any]) -> list[int]:
    datasets = report.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        return []
    missing_by_dataset: list[set[int]] = []
    for summary in datasets.values():
        if not isinstance(summary, dict):
            raise ValueError("Coverage dataset summaries must be objects")
        missing = summary.get("missing_strings")
        if not isinstance(missing, list):
            raise ValueError("Coverage summaries must contain missing_strings")
        strings = {int(value) for value in missing}
        if any(not 1 <= string <= 6 for string in strings):
            raise ValueError("Coverage missing strings must be in 1..6")
        missing_by_dataset.append(strings)
    return sorted(set.intersection(*missing_by_dataset))


def _position_confidence(note: dict[str, Any]) -> float:
    value = float(note.get("positionConfidence", note.get("confidence", 0)))
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _window_notes(
    measures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        note
        for measure in measures
        for beat in measure.get("beats", [])
        for note in beat.get("notes", [])
    ]


def parse_model_predictions(
    value: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported model prediction schema")
    raw_models = value.get("models")
    if not isinstance(raw_models, list) or len(raw_models) < 2:
        raise ValueError("At least two model predictions are required")
    result: dict[str, list[dict[str, Any]]] = {}
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise ValueError("Model prediction entries must be objects")
        model_id = str(raw_model.get("model_id", "")).strip()
        if not model_id or model_id in result:
            raise ValueError("Model prediction IDs must be unique")
        events = raw_model.get("events")
        if not isinstance(events, list):
            raise ValueError(f"Model {model_id} events must be a list")
        result[model_id] = [
            dict(event)
            for event in events
            if isinstance(event, dict)
        ]
        if len(result[model_id]) != len(events):
            raise ValueError(f"Model {model_id} events must be objects")
    return result


def _event_onset(event: dict[str, Any]) -> float:
    return float(event.get("onset", event.get("at", -1)))


def _event_pitch(event: dict[str, Any]) -> int | None:
    if event.get("pitch") is not None:
        return int(event["pitch"])
    if event.get("string") is None or event.get("fret") is None:
        return None
    string = int(event["string"])
    if not 1 <= string <= 6:
        return None
    return STANDARD_TUNING_LIST[string - 1] + int(event["fret"])


def _event_position(event: dict[str, Any]) -> tuple[int, int] | None:
    if event.get("string") is None or event.get("fret") is None:
        return None
    string = int(event["string"])
    fret = int(event["fret"])
    if not 1 <= string <= 6 or not 0 <= fret <= 24:
        return None
    pitch = event.get("pitch")
    if (
        pitch is not None
        and STANDARD_TUNING_LIST[string - 1] + fret != int(pitch)
    ):
        return None
    return string, fret


def _note_pitch(note: dict[str, Any]) -> int | None:
    return _event_pitch(note)


def analyze_model_disagreements(
    measures: list[dict[str, Any]],
    model_predictions: dict[str, list[dict[str, Any]]],
    *,
    matching_onset_tolerance: float = 0.06,
    agreement_onset_tolerance: float = 0.03,
    agreement_confidence: float = 0.85,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if len(model_predictions) < 2:
        raise ValueError("At least two models are required for disagreement")
    if matching_onset_tolerance <= 0 or agreement_onset_tolerance <= 0:
        raise ValueError("Model onset tolerances must be positive")
    if not 0 <= agreement_confidence <= 1:
        raise ValueError("Model agreement confidence must be in 0..1")

    model_ids = sorted(model_predictions)
    used: dict[str, set[int]] = {
        model_id: set()
        for model_id in model_ids
    }
    by_measure: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "model_ids": model_ids,
            "events_compared": 0,
            "auto_candidates": 0,
            "review_only_events": 0,
            "position_disagreements": 0,
            "missing_position_labels": 0,
            "presence_disagreements": 0,
            "timing_disagreements": 0,
            "low_confidence_events": 0,
            "unmatched_predictions": 0,
            "maximum_onset_spread_seconds": 0.0,
            "event_decisions": [],
        }
    )

    for measure in measures:
        measure_number = int(measure["number"])
        notes = _window_notes([measure])
        for note in notes:
            note_id = str(note.get("id", "")).strip()
            note_onset = float(note["at"])
            note_pitch = _note_pitch(note)
            matched: dict[str, dict[str, Any]] = {}
            for model_id in model_ids:
                events = model_predictions[model_id]
                exact = [
                    (index, event)
                    for index, event in enumerate(events)
                    if index not in used[model_id]
                    and note_id
                    and str(event.get("note_id", "")) == note_id
                ]
                candidates = exact or [
                    (index, event)
                    for index, event in enumerate(events)
                    if index not in used[model_id]
                    and note_pitch is not None
                    and _event_pitch(event) == note_pitch
                    and abs(_event_onset(event) - note_onset)
                    <= matching_onset_tolerance
                ]
                if not candidates:
                    continue
                index, event = min(
                    candidates,
                    key=lambda item: (
                        abs(_event_onset(item[1]) - note_onset),
                        item[0],
                    ),
                )
                used[model_id].add(index)
                matched[model_id] = event

            positions_by_model = {
                model_id: position
                for model_id, event in matched.items()
                if (position := _event_position(event)) is not None
            }
            positions = set(positions_by_model.values())
            onsets = [
                _event_onset(event)
                for event in matched.values()
            ]
            confidences = [
                _position_confidence(event)
                for event in matched.values()
            ]
            onset_spread = (
                max(onsets) - min(onsets)
                if len(onsets) > 1
                else 0.0
            )
            presence_disagreement = len(matched) != len(model_ids)
            missing_position_label = len(positions_by_model) != len(matched)
            position_disagreement = len(positions) > 1
            timing_disagreement = (
                len(onsets) > 1
                and onset_spread > agreement_onset_tolerance
            )
            low_confidence = (
                not confidences
                or min(confidences) < agreement_confidence
            )
            auto_candidate = (
                len(matched) == len(model_ids)
                and len(positions) == 1
                and not missing_position_label
                and not timing_disagreement
                and not low_confidence
            )
            summary = by_measure[measure_number]
            summary["events_compared"] += 1
            summary["auto_candidates"] += int(auto_candidate)
            summary["review_only_events"] += int(not auto_candidate)
            summary["position_disagreements"] += int(
                position_disagreement
            )
            summary["missing_position_labels"] += int(
                missing_position_label
            )
            summary["presence_disagreements"] += int(
                presence_disagreement
            )
            summary["timing_disagreements"] += int(timing_disagreement)
            summary["low_confidence_events"] += int(low_confidence)
            summary["maximum_onset_spread_seconds"] = max(
                float(summary["maximum_onset_spread_seconds"]),
                onset_spread,
            )
            reasons = [
                name
                for name, active in (
                    ("missing-model-event", presence_disagreement),
                    ("missing-position-label", missing_position_label),
                    ("position-disagreement", position_disagreement),
                    ("timing-disagreement", timing_disagreement),
                    ("low-confidence", low_confidence),
                )
                if active
            ]
            summary["event_decisions"].append(
                {
                    "note_id": note_id or None,
                    "decision": (
                        "auto-candidate"
                        if auto_candidate
                        else "review-only"
                    ),
                    "reasons": reasons or ["high-confidence-agreement"],
                    "models_present": sorted(matched),
                    "positions": {
                        model_id: {
                            "string": int(event["string"]),
                            "fret": int(event["fret"]),
                            "confidence": round(
                                _position_confidence(event),
                                6,
                            ),
                            "onset_delta_seconds": round(
                                _event_onset(event) - note_onset,
                                6,
                            ),
                        }
                        for model_id, event in sorted(matched.items())
                        if model_id in positions_by_model
                    },
                }
            )

    unmatched_events: list[tuple[str, dict[str, Any]]] = []
    for model_id, events in model_predictions.items():
        for index, event in enumerate(events):
            if index in used[model_id]:
                continue
            unmatched_events.append((model_id, event))

    clusters: list[dict[str, Any]] = []
    for model_id, event in sorted(
        unmatched_events,
        key=lambda item: (_event_onset(item[1]), item[0]),
    ):
        onset = _event_onset(event)
        pitch = _event_pitch(event)
        note_id = str(event.get("note_id", "")).strip() or None
        candidates = [
            cluster
            for cluster in clusters
            if model_id not in cluster["events"]
            and (
                (
                    note_id is not None
                    and cluster["note_id"] == note_id
                )
                or (
                    not (
                        note_id is not None
                        and cluster["note_id"] is not None
                        and cluster["note_id"] != note_id
                    )
                    and pitch is not None
                    and cluster["pitch"] == pitch
                    and abs(
                        onset
                        - sum(cluster["onsets"]) / len(cluster["onsets"])
                    )
                    <= matching_onset_tolerance
                )
            )
        ]
        if candidates:
            cluster = min(
                candidates,
                key=lambda item: abs(
                    onset - sum(item["onsets"]) / len(item["onsets"])
                ),
            )
        else:
            cluster = {
                "note_id": note_id,
                "pitch": pitch,
                "events": {},
                "onsets": [],
            }
            clusters.append(cluster)
        if cluster["note_id"] is None:
            cluster["note_id"] = note_id
        cluster["events"][model_id] = event
        cluster["onsets"].append(onset)

    for cluster in clusters:
        cluster_events = cluster["events"]
        onsets = [
            _event_onset(event)
            for event in cluster_events.values()
        ]
        onset = sum(onsets) / len(onsets)
        target = next(
                (
                    measure
                    for measure in measures
                    if float(measure["start"])
                    <= onset
                    < float(measure["start"])
                    + float(measure["duration"])
                ),
                None,
            )
        if target is None:
            continue
        positions_by_model = {
            model_id: position
            for model_id, event in cluster_events.items()
            if (position := _event_position(event)) is not None
        }
        positions = set(positions_by_model.values())
        confidences = [
            _position_confidence(event)
            for event in cluster_events.values()
        ]
        onset_spread = max(onsets) - min(onsets) if len(onsets) > 1 else 0.0
        presence_disagreement = len(cluster_events) != len(model_ids)
        missing_position_label = (
            len(positions_by_model) != len(cluster_events)
        )
        position_disagreement = len(positions) > 1
        timing_disagreement = (
            len(onsets) > 1
            and onset_spread > agreement_onset_tolerance
        )
        low_confidence = (
            not confidences
            or min(confidences) < agreement_confidence
        )
        reasons = [
            name
            for name, active in (
                ("missing-model-event", presence_disagreement),
                ("missing-position-label", missing_position_label),
                ("position-disagreement", position_disagreement),
                ("timing-disagreement", timing_disagreement),
                ("low-confidence", low_confidence),
            )
            if active
        ]
        summary = by_measure[int(target["number"])]
        summary["unmatched_predictions"] += 1
        summary["presence_disagreements"] += int(presence_disagreement)
        summary["position_disagreements"] += int(position_disagreement)
        summary["missing_position_labels"] += int(missing_position_label)
        summary["timing_disagreements"] += int(timing_disagreement)
        summary["low_confidence_events"] += int(low_confidence)
        summary["review_only_events"] += 1
        summary["maximum_onset_spread_seconds"] = max(
            float(summary["maximum_onset_spread_seconds"]),
            onset_spread,
        )
        summary["event_decisions"].append(
            {
                "note_id": cluster["note_id"],
                "decision": "review-only",
                "reasons": reasons or ["unmatched-project-note"],
                "models_present": sorted(cluster_events),
                "positions": {
                    model_id: {
                        "string": position[0],
                        "fret": position[1],
                        "confidence": round(
                            _position_confidence(cluster_events[model_id]),
                            6,
                        ),
                        "onset_seconds": round(
                            _event_onset(cluster_events[model_id]),
                            6,
                        ),
                    }
                    for model_id, position in sorted(
                        positions_by_model.items()
                    )
                },
            }
        )

    aggregate = {
        "model_ids": model_ids,
        "events_compared": sum(
            int(summary["events_compared"])
            for summary in by_measure.values()
        ),
        "auto_candidates": sum(
            int(summary["auto_candidates"])
            for summary in by_measure.values()
        ),
        "review_only_events": sum(
            int(summary["review_only_events"])
            for summary in by_measure.values()
        ),
        "position_disagreements": sum(
            int(summary["position_disagreements"])
            for summary in by_measure.values()
        ),
        "missing_position_labels": sum(
            int(summary["missing_position_labels"])
            for summary in by_measure.values()
        ),
        "presence_disagreements": sum(
            int(summary["presence_disagreements"])
            for summary in by_measure.values()
        ),
        "timing_disagreements": sum(
            int(summary["timing_disagreements"])
            for summary in by_measure.values()
        ),
        "low_confidence_events": sum(
            int(summary["low_confidence_events"])
            for summary in by_measure.values()
        ),
        "unmatched_predictions": sum(
            int(summary["unmatched_predictions"])
            for summary in by_measure.values()
        ),
    }
    return dict(by_measure), aggregate


def _excerpt_disagreement(
    measure_numbers: list[int],
    disagreement_by_measure: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    summaries = [
        disagreement_by_measure[number]
        for number in measure_numbers
        if disagreement_by_measure is not None
        and number in disagreement_by_measure
    ]
    events = [
        event
        for summary in summaries
        for event in summary["event_decisions"]
    ]
    return {
        "model_ids": sorted(
            {
                model_id
                for summary in summaries
                for model_id in summary["model_ids"]
            }
        ),
        "events_compared": sum(
            int(summary["events_compared"])
            for summary in summaries
        ),
        "auto_candidates": sum(
            int(summary["auto_candidates"])
            for summary in summaries
        ),
        "review_only_events": sum(
            int(summary["review_only_events"])
            for summary in summaries
        ),
        "position_disagreements": sum(
            int(summary["position_disagreements"])
            for summary in summaries
        ),
        "missing_position_labels": sum(
            int(summary["missing_position_labels"])
            for summary in summaries
        ),
        "presence_disagreements": sum(
            int(summary["presence_disagreements"])
            for summary in summaries
        ),
        "timing_disagreements": sum(
            int(summary["timing_disagreements"])
            for summary in summaries
        ),
        "low_confidence_events": sum(
            int(summary["low_confidence_events"])
            for summary in summaries
        ),
        "unmatched_predictions": sum(
            int(summary["unmatched_predictions"])
            for summary in summaries
        ),
        "maximum_onset_spread_seconds": round(
            max(
                (
                    float(summary["maximum_onset_spread_seconds"])
                    for summary in summaries
                ),
                default=0.0,
            ),
            6,
        ),
        "event_decisions": events,
        "disposition": (
            "auto-candidate"
            if events
            and all(
                event["decision"] == "auto-candidate"
                for event in events
            )
            else "review-only"
        ),
    }


def score_excerpt(
    measures: list[dict[str, Any]],
    *,
    priority_strings: set[int],
    fast_interval_seconds: float,
    high_fret: int,
    low_confidence_threshold: float,
    disagreement_by_measure: dict[int, dict[str, Any]] | None = None,
) -> dict[str, object] | None:
    notes = _window_notes(measures)
    if not notes:
        return None
    if any(
        str(note.get("positionSource", "")).startswith("source-score")
        for note in notes
    ):
        return None

    ordered_notes = sorted(
        notes,
        key=lambda note: (float(note["at"]), int(note["string"])),
    )
    previous_by_string: dict[int, float] = {}
    fast_intervals = 0
    high_fret_events = 0
    priority_string_events = 0
    low_confidence_events = 0
    confidence_sum = 0.0
    uncertainty = 0.0
    for note in ordered_notes:
        string = int(note["string"])
        fret = int(note["fret"])
        onset = float(note["at"])
        confidence = _position_confidence(note)
        confidence_sum += confidence
        uncertainty += 1.0 - confidence
        if confidence < low_confidence_threshold:
            low_confidence_events += 1
        if fret >= high_fret:
            high_fret_events += 1
        if string in priority_strings:
            priority_string_events += 1
        previous = previous_by_string.get(string)
        if (
            previous is not None
            and 0 < onset - previous <= fast_interval_seconds
        ):
            fast_intervals += 1
        previous_by_string[string] = onset

    measure_numbers = [
        int(measure["number"])
        for measure in measures
    ]
    model_disagreement = _excerpt_disagreement(
        measure_numbers,
        disagreement_by_measure,
    )
    score = (
        fast_intervals * 3.0
        + high_fret_events * 2.0
        + priority_string_events * 2.0
        + uncertainty
        + model_disagreement["position_disagreements"] * 5.0
        + model_disagreement["missing_position_labels"] * 5.0
        + model_disagreement["presence_disagreements"] * 4.0
        + model_disagreement["timing_disagreements"] * 3.0
        + model_disagreement["low_confidence_events"] * 2.0
    )
    first_measure = measures[0]
    last_measure = measures[-1]
    start = float(first_measure["start"])
    end = float(last_measure["start"]) + float(last_measure["duration"])
    return {
        "measure_numbers": measure_numbers,
        "start_seconds": round(start, 6),
        "end_seconds": round(end, 6),
        "duration_seconds": round(end - start, 6),
        "events": len(ordered_notes),
        "score": round(score, 6),
        "candidate_disposition": model_disagreement["disposition"],
        "model_disagreement": model_disagreement,
        "signals": {
            "fast_same_string_intervals": fast_intervals,
            "high_fret_events": high_fret_events,
            "priority_string_events": priority_string_events,
            "low_confidence_events": low_confidence_events,
            "mean_position_confidence": round(
                confidence_sum / len(ordered_notes),
                6,
            ),
            "position_uncertainty": round(uncertainty, 6),
        },
    }


def build_review_queue(
    project_path: Path,
    *,
    coverage_report: dict[str, Any] | None = None,
    priority_strings: list[int] | None = None,
    excerpt_measures: int = 2,
    limit: int = 12,
    fast_interval_seconds: float = 0.12,
    high_fret: int = 12,
    low_confidence_threshold: float = 0.6,
    model_predictions: dict[str, list[dict[str, Any]]] | None = None,
    model_matching_onset_tolerance: float = 0.06,
    model_agreement_onset_tolerance: float = 0.03,
    model_agreement_confidence: float = 0.85,
) -> dict[str, object]:
    if excerpt_measures < 1:
        raise ValueError("excerpt_measures must be positive")
    if limit < 1:
        raise ValueError("limit must be positive")
    if fast_interval_seconds <= 0:
        raise ValueError("fast_interval_seconds must be positive")
    if not 1 <= high_fret <= 24:
        raise ValueError("high_fret must be in 1..24")
    if not 0 <= low_confidence_threshold <= 1:
        raise ValueError("low_confidence_threshold must be in 0..1")

    project = load_project(project_path)
    measures = list(project["tab"].get("measures", []))
    inferred_priority = (
        priority_strings_from_coverage(coverage_report)
        if coverage_report is not None
        else []
    )
    resolved_priority = sorted(
        set(inferred_priority) | set(priority_strings or [])
    )
    if any(not 1 <= string <= 6 for string in resolved_priority):
        raise ValueError("priority strings must be in 1..6")
    disagreement_by_measure: dict[int, dict[str, Any]] | None = None
    disagreement_summary: dict[str, Any] | None = None
    if model_predictions is not None:
        (
            disagreement_by_measure,
            disagreement_summary,
        ) = analyze_model_disagreements(
            measures,
            model_predictions,
            matching_onset_tolerance=model_matching_onset_tolerance,
            agreement_onset_tolerance=(
                model_agreement_onset_tolerance
            ),
            agreement_confidence=model_agreement_confidence,
        )

    candidates: list[dict[str, object]] = []
    final_start = len(measures) - excerpt_measures + 1
    for start_index in range(max(0, final_start)):
        excerpt = measures[start_index : start_index + excerpt_measures]
        candidate = score_excerpt(
            excerpt,
            priority_strings=set(resolved_priority),
            fast_interval_seconds=fast_interval_seconds,
            high_fret=high_fret,
            low_confidence_threshold=low_confidence_threshold,
            disagreement_by_measure=disagreement_by_measure,
        )
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            item["measure_numbers"],
        )
    )

    selected: list[dict[str, object]] = []
    occupied_measures: set[int] = set()
    for candidate in candidates:
        numbers = {
            int(number)
            for number in candidate["measure_numbers"]
        }
        if numbers & occupied_measures:
            continue
        selected.append(
            {
                "rank": len(selected) + 1,
                **candidate,
            }
        )
        occupied_measures.update(numbers)
        if len(selected) >= limit:
            break

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_id": str(project["id"]),
        "project_sha256": sha256_file(project_path),
        "criteria": {
            "excerpt_measures": excerpt_measures,
            "limit": limit,
            "fast_interval_seconds": fast_interval_seconds,
            "high_fret": high_fret,
            "low_confidence_threshold": low_confidence_threshold,
            "priority_strings": resolved_priority,
            "score_weights": {
                "fast_same_string_interval": 3.0,
                "high_fret_event": 2.0,
                "priority_string_event": 2.0,
                "position_uncertainty": 1.0,
                "model_position_disagreement": 5.0,
                "model_missing_position_label": 5.0,
                "model_presence_disagreement": 4.0,
                "model_timing_disagreement": 3.0,
                "model_low_confidence": 2.0,
            },
            "model_matching_onset_tolerance": (
                model_matching_onset_tolerance
            ),
            "model_agreement_onset_tolerance": (
                model_agreement_onset_tolerance
            ),
            "model_agreement_confidence": model_agreement_confidence,
        },
        "model_disagreement_summary": disagreement_summary,
        "promotion_policy": {
            "auto_candidate_is_gold": False,
            "review_only_requires_human_review": True,
            "gold_requires_approved_human_review": True,
        },
        "candidates": selected,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prioritize non-overlapping project excerpts for gold review.",
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excerpt-measures", type=int, default=2)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--fast-interval", type=float, default=0.12)
    parser.add_argument("--high-fret", type=int, default=12)
    parser.add_argument("--low-confidence", type=float, default=0.6)
    parser.add_argument("--priority-string", type=int, action="append")
    parser.add_argument("--model-predictions", type=Path)
    parser.add_argument(
        "--model-matching-onset-tolerance",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--model-agreement-onset-tolerance",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--model-agreement-confidence",
        type=float,
        default=0.85,
    )
    arguments = parser.parse_args()
    coverage_report = (
        json.loads(arguments.coverage.read_text(encoding="utf-8"))
        if arguments.coverage is not None
        else None
    )
    model_predictions_document = (
        json.loads(
            arguments.model_predictions.read_text(encoding="utf-8")
        )
        if arguments.model_predictions is not None
        else None
    )
    report = build_review_queue(
        arguments.project,
        coverage_report=coverage_report,
        priority_strings=arguments.priority_string,
        excerpt_measures=arguments.excerpt_measures,
        limit=arguments.limit,
        fast_interval_seconds=arguments.fast_interval,
        high_fret=arguments.high_fret,
        low_confidence_threshold=arguments.low_confidence,
        model_predictions=(
            parse_model_predictions(model_predictions_document)
            if model_predictions_document is not None
            else None
        ),
        model_matching_onset_tolerance=(
            arguments.model_matching_onset_tolerance
        ),
        model_agreement_onset_tolerance=(
            arguments.model_agreement_onset_tolerance
        ),
        model_agreement_confidence=(
            arguments.model_agreement_confidence
        ),
    )
    if arguments.model_predictions is not None:
        report["model_predictions"] = {
            "sha256": sha256_file(arguments.model_predictions),
            "models": sorted(
                parse_model_predictions(
                    model_predictions_document
                )
            ),
        }
    _write_report(arguments.output, report)
    print(
        json.dumps(
            {
                "project_id": report["project_id"],
                "candidates": len(report["candidates"]),
                "output": arguments.output.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
