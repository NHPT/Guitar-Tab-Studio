from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reference_review import load_project, sha256_file


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


def score_excerpt(
    measures: list[dict[str, Any]],
    *,
    priority_strings: set[int],
    fast_interval_seconds: float,
    high_fret: int,
    low_confidence_threshold: float,
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

    score = (
        fast_intervals * 3.0
        + high_fret_events * 2.0
        + priority_string_events * 2.0
        + uncertainty
    )
    first_measure = measures[0]
    last_measure = measures[-1]
    start = float(first_measure["start"])
    end = float(last_measure["start"]) + float(last_measure["duration"])
    return {
        "measure_numbers": [
            int(measure["number"])
            for measure in measures
        ],
        "start_seconds": round(start, 6),
        "end_seconds": round(end, 6),
        "duration_seconds": round(end - start, 6),
        "events": len(ordered_notes),
        "score": round(score, 6),
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
            },
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
    arguments = parser.parse_args()
    coverage_report = (
        json.loads(arguments.coverage.read_text(encoding="utf-8"))
        if arguments.coverage is not None
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
    )
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
