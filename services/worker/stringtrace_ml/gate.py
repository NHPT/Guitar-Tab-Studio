from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric_root(report: dict[str, object]) -> dict[str, object]:
    return dict(report.get("aggregate") or report.get("metrics") or {})


def evaluate_gate(
    report: dict[str, object],
    baseline: dict[str, object],
    *,
    minimum_tracks: int,
    minimum_onset_f1: float,
    minimum_tablature_f1: float,
    minimum_improvement: float,
    minimum_repeated_recall: float,
) -> dict[str, object]:
    metrics = metric_root(report)
    baseline_metrics = metric_root(baseline)
    track_count = int(report.get("track_count", 1))
    onset_f1 = float(metrics.get("onset", {}).get("f1", 0))
    tablature_f1 = float(metrics.get("tablature", {}).get("f1", 0))
    repeated_recall = float(
        metrics.get("repeated_tablature", {}).get("recall", 0)
    )
    baseline_tablature_f1 = float(
        baseline_metrics.get("tablature", {}).get("f1", 0)
    )
    checks = {
        "minimum_tracks": track_count >= minimum_tracks,
        "onset_f1": onset_f1 >= minimum_onset_f1,
        "tablature_f1": tablature_f1 >= minimum_tablature_f1,
        "baseline_improvement": (
            tablature_f1 - baseline_tablature_f1 >= minimum_improvement
        ),
        "repeated_attack_recall": repeated_recall >= minimum_repeated_recall,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "values": {
            "track_count": track_count,
            "onset_f1": onset_f1,
            "tablature_f1": tablature_f1,
            "baseline_tablature_f1": baseline_tablature_f1,
            "tablature_improvement": tablature_f1 - baseline_tablature_f1,
            "repeated_attack_recall": repeated_recall,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block deployment unless a transcription checkpoint passes gates.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--minimum-tracks", type=int, default=100)
    parser.add_argument("--minimum-onset-f1", type=float, default=0.80)
    parser.add_argument("--minimum-tablature-f1", type=float, default=0.75)
    parser.add_argument("--minimum-improvement", type=float, default=0.15)
    parser.add_argument("--minimum-repeated-recall", type=float, default=0.85)
    arguments = parser.parse_args()

    result = evaluate_gate(
        json.loads(arguments.report.read_text(encoding="utf-8")),
        json.loads(arguments.baseline.read_text(encoding="utf-8")),
        minimum_tracks=arguments.minimum_tracks,
        minimum_onset_f1=arguments.minimum_onset_f1,
        minimum_tablature_f1=arguments.minimum_tablature_f1,
        minimum_improvement=arguments.minimum_improvement,
        minimum_repeated_recall=arguments.minimum_repeated_recall,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
