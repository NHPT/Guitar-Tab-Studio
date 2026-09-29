from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .schema import TrackAnnotation, load_manifest


LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _distribution(
    counts: Counter[int] | Counter[str],
    total: int,
) -> dict[str, dict[str, int | float]]:
    return {
        str(key): {
            "events": count,
            "ratio": round(count / max(1, total), 6),
        }
        for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def summarize_annotations(
    annotations: list[TrackAnnotation],
    *,
    fast_interval_seconds: float = 0.12,
    high_fret: int = 12,
) -> dict[str, object]:
    event_count = sum(len(annotation.events) for annotation in annotations)
    eligible_annotations = [
        annotation
        for annotation in annotations
        if annotation.provenance.get("verification_status") != "quarantined"
    ]
    eligible_events = [
        event
        for annotation in eligible_annotations
        for event in annotation.events
        if not any(
            excluded_range.start <= event.onset < excluded_range.end
            for excluded_range in annotation.excluded_ranges
        )
    ]
    eligible_string_counts = Counter(event.string for event in eligible_events)
    excluded_event_count = (
        sum(len(annotation.events) for annotation in eligible_annotations)
        - len(eligible_events)
    )
    excluded_range_count = sum(
        len(annotation.excluded_ranges)
        for annotation in eligible_annotations
    )
    excluded_duration = sum(
        excluded_range.end - excluded_range.start
        for annotation in eligible_annotations
        for excluded_range in annotation.excluded_ranges
    )
    string_counts: Counter[int] = Counter()
    technique_counts: Counter[str] = Counter()
    fret_buckets: Counter[str] = Counter()
    high_fret_events = 0
    fast_same_string_intervals = 0
    eligible_same_string_intervals = 0
    technique_reviewed_events = 0
    for annotation in annotations:
        previous_by_string: dict[int, float] = {}
        technique_reviewed = bool(
            annotation.provenance.get(
                "technique_reviewed",
                str(annotation.provenance.get("position_source", "")).startswith(
                    "source-score"
                ),
            )
        )
        for event in annotation.events:
            string_counts[event.string] += 1
            technique_counts[event.technique] += 1
            if technique_reviewed:
                technique_reviewed_events += 1
            if event.fret >= high_fret:
                high_fret_events += 1
            if event.fret <= 4:
                fret_buckets["0-4"] += 1
            elif event.fret < high_fret:
                fret_buckets[f"5-{high_fret - 1}"] += 1
            elif event.fret <= 17:
                fret_buckets[f"{high_fret}-17"] += 1
            else:
                fret_buckets["18-24"] += 1
            previous = previous_by_string.get(event.string)
            if previous is not None and event.onset > previous:
                eligible_same_string_intervals += 1
                if event.onset - previous <= fast_interval_seconds:
                    fast_same_string_intervals += 1
            previous_by_string[event.string] = event.onset

    capos = sorted({annotation.capo for annotation in annotations})
    bpms = [
        annotation.bpm
        for annotation in annotations
        if annotation.bpm is not None
    ]
    source_kinds = sorted(
        {
            str(
                annotation.provenance.get(
                    "source_kind",
                    (
                        annotation.provenance.get("source", {})
                        if isinstance(annotation.provenance.get("source"), dict)
                        else {}
                    ).get("kind", "unknown"),
                )
            )
            for annotation in annotations
        }
    )
    annotation_sources = sorted(
        {
            str(
                annotation.provenance.get(
                    "annotation_source",
                    annotation.provenance.get("position_source", "unknown"),
                )
            )
            for annotation in annotations
        }
    )
    verification_counts = Counter(
        str(annotation.provenance.get("verification_status", "unspecified"))
        for annotation in annotations
    )
    alignment_status_counts = Counter(
        str(
            (
                annotation.provenance.get("alignment_audit", {})
                if isinstance(
                    annotation.provenance.get("alignment_audit"),
                    dict,
                )
                else {}
            ).get("status", "unspecified")
        )
        for annotation in annotations
    )
    return {
        "tracks": len(annotations),
        "groups": len({annotation.group_id for annotation in annotations}),
        "evaluation_eligible_tracks": len(eligible_annotations),
        "evaluation_eligible_events": len(eligible_events),
        "evaluation_excluded_events": excluded_event_count,
        "evaluation_excluded_range_count": excluded_range_count,
        "evaluation_excluded_duration_seconds": round(excluded_duration, 6),
        "evaluation_eligible_missing_strings": [
            string
            for string in range(1, 7)
            if eligible_string_counts[string] == 0
        ],
        "evaluation_eligible_high_fret_events": sum(
            event.fret >= high_fret
            for event in eligible_events
        ),
        "quarantined_tracks": verification_counts["quarantined"],
        "duration_seconds": round(
            sum(annotation.duration for annotation in annotations),
            6,
        ),
        "events": event_count,
        "strings": _distribution(string_counts, event_count),
        "missing_strings": [
            string
            for string in range(1, 7)
            if string_counts[string] == 0
        ],
        "frets": {
            "buckets": _distribution(fret_buckets, event_count),
            "high_fret_threshold": high_fret,
            "high_fret_events": high_fret_events,
            "high_fret_ratio": round(
                high_fret_events / max(1, event_count),
                6,
            ),
        },
        "same_string_intervals": {
            "fast_threshold_seconds": fast_interval_seconds,
            "eligible_intervals": eligible_same_string_intervals,
            "fast_intervals": fast_same_string_intervals,
            "fast_ratio": round(
                fast_same_string_intervals
                / max(1, eligible_same_string_intervals),
                6,
            ),
        },
        "techniques": {
            "distribution": _distribution(technique_counts, event_count),
            "reviewed_events": technique_reviewed_events,
            "reviewed_ratio": round(
                technique_reviewed_events / max(1, event_count),
                6,
            ),
        },
        "capos": capos,
        "bpm": {
            "minimum": min(bpms) if bpms else None,
            "maximum": max(bpms) if bpms else None,
        },
        "source_kinds": source_kinds,
        "annotation_sources": annotation_sources,
        "verification_statuses": dict(sorted(verification_counts.items())),
        "alignment_audit_statuses": dict(
            sorted(alignment_status_counts.items())
        ),
    }


def parse_manifest_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not LABEL_PATTERN.fullmatch(label):
        raise argparse.ArgumentTypeError(
            "Manifest must use a safe LABEL=PATH value"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("Manifest path is required")
    return label, Path(raw_path)


def build_coverage_report(
    manifests: list[tuple[str, Path]],
    *,
    fast_interval_seconds: float = 0.12,
    high_fret: int = 12,
) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for label, path in manifests:
        if label in datasets:
            raise ValueError(f"Duplicate manifest label: {label}")
        annotations = load_manifest(path)
        non_test = sorted(
            {
                annotation.split
                for annotation in annotations
                if annotation.split != "test"
            }
        )
        if non_test:
            raise ValueError(
                f"Real reference manifest {label!r} contains non-test splits: "
                f"{non_test}"
            )
        datasets[label] = summarize_annotations(
            annotations,
            fast_interval_seconds=fast_interval_seconds,
            high_fret=high_fret,
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "fast_interval_seconds": fast_interval_seconds,
            "high_fret": high_fret,
        },
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report held-out real-reference coverage by audio condition.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=parse_manifest_spec,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--fast-interval", type=float, default=0.12)
    parser.add_argument("--high-fret", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.fast_interval <= 0:
        parser.error("--fast-interval must be positive")
    if not 1 <= arguments.high_fret <= 24:
        parser.error("--high-fret must be in 1..24")
    report = build_coverage_report(
        arguments.manifest,
        fast_interval_seconds=arguments.fast_interval,
        high_fret=arguments.high_fret,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "datasets": list(report["datasets"]),
                "output": arguments.output.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
