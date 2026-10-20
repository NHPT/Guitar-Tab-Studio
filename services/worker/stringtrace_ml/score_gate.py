from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_score_gate(
    report: dict[str, object],
    *,
    minimum_reference_events: int = 100,
    minimum_beat_f1: float = 0.95,
    maximum_median_beat_error_ms: float = 20.0,
    minimum_onset_f1: float = 0.90,
    minimum_pitch_f1: float = 0.90,
    minimum_tablature_f1: float = 0.90,
    minimum_repeated_recall: float = 0.90,
    minimum_chord_accuracy: float = 0.90,
    minimum_technique_precision: float = 0.90,
) -> dict[str, object]:
    metrics = dict(report.get("metrics") or {})
    beat_metrics = dict(report.get("beat_metrics") or {})
    chord_metrics = dict(report.get("chord_metrics") or {})
    onset = dict(metrics.get("onset") or {})
    pitch = dict(metrics.get("pitch") or {})
    tablature = dict(metrics.get("tablature") or {})
    repeated = dict(metrics.get("repeated_tablature") or {})
    technique = dict(metrics.get("technique") or {})
    technique_reference_count = int(technique.get("true_positive", 0)) + int(
        technique.get("false_negative", 0)
    )
    median_beat_error = beat_metrics.get("median_error_ms")
    checks = {
        "reference_coverage": (
            int(report.get("reference_events", 0)) >= minimum_reference_events
        ),
        "beat_f1": float(beat_metrics.get("f1", 0)) >= minimum_beat_f1,
        "beat_timing": (
            median_beat_error is not None
            and float(median_beat_error) <= maximum_median_beat_error_ms
        ),
        "onset_f1": float(onset.get("f1", 0)) >= minimum_onset_f1,
        "pitch_f1": float(pitch.get("f1", 0)) >= minimum_pitch_f1,
        "tablature_f1": (
            float(tablature.get("f1", 0)) >= minimum_tablature_f1
        ),
        "repeated_attack_recall": (
            float(repeated.get("recall", 0)) >= minimum_repeated_recall
        ),
        "chord_accuracy": (
            float(chord_metrics.get("exact_accuracy", 0))
            >= minimum_chord_accuracy
        ),
        "technique_precision": (
            technique_reference_count == 0
            or float(technique.get("precision", 0))
            >= minimum_technique_precision
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "values": {
            "reference_events": int(report.get("reference_events", 0)),
            "beat_f1": float(beat_metrics.get("f1", 0)),
            "median_beat_error_ms": median_beat_error,
            "onset_f1": float(onset.get("f1", 0)),
            "pitch_f1": float(pitch.get("f1", 0)),
            "tablature_f1": float(tablature.get("f1", 0)),
            "repeated_attack_recall": float(repeated.get("recall", 0)),
            "chord_accuracy": float(
                chord_metrics.get("exact_accuracy", 0)
            ),
            "technique_precision": (
                float(technique.get("precision", 0))
                if technique_reference_count > 0
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gate a complete score against reviewed beat, note, TAB, chord, "
            "and technique references."
        ),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-reference-events", type=int, default=100)
    parser.add_argument("--minimum-beat-f1", type=float, default=0.95)
    parser.add_argument(
        "--maximum-median-beat-error-ms",
        type=float,
        default=20.0,
    )
    parser.add_argument("--minimum-onset-f1", type=float, default=0.90)
    parser.add_argument("--minimum-pitch-f1", type=float, default=0.90)
    parser.add_argument("--minimum-tablature-f1", type=float, default=0.90)
    parser.add_argument("--minimum-repeated-recall", type=float, default=0.90)
    parser.add_argument("--minimum-chord-accuracy", type=float, default=0.90)
    parser.add_argument(
        "--minimum-technique-precision",
        type=float,
        default=0.90,
    )
    arguments = parser.parse_args()
    result = evaluate_score_gate(
        json.loads(arguments.report.read_text(encoding="utf-8")),
        minimum_reference_events=arguments.minimum_reference_events,
        minimum_beat_f1=arguments.minimum_beat_f1,
        maximum_median_beat_error_ms=(
            arguments.maximum_median_beat_error_ms
        ),
        minimum_onset_f1=arguments.minimum_onset_f1,
        minimum_pitch_f1=arguments.minimum_pitch_f1,
        minimum_tablature_f1=arguments.minimum_tablature_f1,
        minimum_repeated_recall=arguments.minimum_repeated_recall,
        minimum_chord_accuracy=arguments.minimum_chord_accuracy,
        minimum_technique_precision=arguments.minimum_technique_precision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
