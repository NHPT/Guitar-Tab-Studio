#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "worker"))

from stringtrace_ml.metrics import evaluate_beats  # noqa: E402
from worker import estimate_tempo_with_diagnostics  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    if value is None or isinstance(value, (bool, str)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Review content contains an invalid number")
        if number == 0:
            return "0"
        return f"{number:.12f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json.dumps(str(key), ensure_ascii=False)
                + ":"
                + canonical_json(value[key])
                for key in sorted(value)
            )
            + "}"
        )
    raise ValueError("Review content contains an unsupported value")


def structure_content_digest(review: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in review.items()
        if key not in {"status", "submission", "submission_history"}
    }
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def effective_reference_beats(
    analysis: dict[str, Any],
    review: dict[str, Any],
) -> list[float]:
    if review.get("status") not in {"submitted", "adjudicated"}:
        raise ValueError(
            f"{review.get('recording_id', 'unknown')} review is not submitted",
        )
    excluded = {
        round(float(value), 4) for value in review.get("excluded_candidate_beats", [])
    }
    retained = [
        float(value)
        for value in analysis.get("beats", [])
        if round(float(value), 4) not in excluded
    ]
    added = [float(value) for value in review.get("added_candidate_beats", [])]
    return sorted([*retained, *added])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate tempo-multiplier changes against submitted full-song "
            "development reviews without modifying the review corpus."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / ".cache" / "beat-gold" / "sources",
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=ROOT / ".cache" / "beat-gold" / "pilot-v2",
    )
    parser.add_argument("--minimum-f1", type=float, default=0.95)
    parser.add_argument("--tolerance", type=float, default=0.07)
    parser.add_argument(
        "--recording",
        action="append",
        help="Evaluate only the named recording; may be repeated.",
    )
    arguments = parser.parse_args()

    source_dir = arguments.source_dir.resolve()
    pilot_dir = arguments.pilot_dir.resolve()
    analysis_dir = pilot_dir / "analysis"
    review_dir = pilot_dir / "structure-reviews" / "reviewer-a"
    recordings = sorted(
        path.stem
        for path in review_dir.glob("*.json")
        if not arguments.recording or path.stem in arguments.recording
    )
    if not recordings:
        raise ValueError("No submitted structure reviews matched the request")

    failed: list[str] = []
    aggregate_reference = 0
    aggregate_predicted = 0
    multiplier_errors = 0
    print(
        "recording\tbpm\tgrid\tconfidence\treference\tpredicted\t"
        "precision\trecall\tf1\tp95_ms\tstatus",
    )
    for recording in recordings:
        audio_path = source_dir / f"{recording}.media"
        analysis_path = analysis_dir / f"{recording}.json"
        review_path = review_dir / f"{recording}.json"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if not analysis_path.is_file():
            raise FileNotFoundError(analysis_path)

        analysis = read_json(analysis_path)
        review = read_json(review_path)
        audio_sha256 = sha256_file(audio_path)
        if (
            review.get("recording_id") != recording
            or analysis.get("source_audio_sha256") != audio_sha256
            or review.get("source_audio_sha256") != audio_sha256
        ):
            raise ValueError(
                f"{recording} review, analysis and audio identities differ"
            )
        candidate_source = review.get("candidate_source")
        if (
            not isinstance(candidate_source, dict)
            or candidate_source.get("analysis_sha256")
            != sha256_file(analysis_path)
        ):
            raise ValueError(
                f"{recording} review does not match the current analysis"
            )
        submission = review.get("submission")
        if (
            not isinstance(submission, dict)
            or submission.get("content_sha256")
            != structure_content_digest(review)
        ):
            raise ValueError(
                f"{recording} review has an invalid submission digest"
            )
        reference = effective_reference_beats(analysis, review)
        bpm, predicted, _onsets, _tempo_map, diagnostics = (
            estimate_tempo_with_diagnostics(audio_path)
        )
        metrics = evaluate_beats(
            reference,
            predicted,
            tolerance=arguments.tolerance,
        )
        count_ratio = len(predicted) / max(1, len(reference))
        multiplier_error = not 0.8 <= count_ratio <= 1.2
        passed = float(metrics["f1"]) >= arguments.minimum_f1 and not multiplier_error
        if multiplier_error:
            multiplier_errors += 1
        if not passed:
            failed.append(recording)
        aggregate_reference += len(reference)
        aggregate_predicted += len(predicted)
        p95 = metrics["p95_error_ms"]
        print(
            "\t".join(
                (
                    recording,
                    f"{bpm:.2f}",
                    str(diagnostics["selected_grid"]),
                    f"{float(diagnostics['confidence']):.3f}",
                    str(len(reference)),
                    str(len(predicted)),
                    f"{float(metrics['precision']):.4f}",
                    f"{float(metrics['recall']):.4f}",
                    f"{float(metrics['f1']):.4f}",
                    "n/a" if p95 is None else f"{float(p95):.2f}",
                    "PASS" if passed else "FAIL",
                )
            )
        )

    print(
        json.dumps(
            {
                "reference_kind": "submitted-development-review",
                "independent_gold": False,
                "recordings": len(recordings),
                "passed": len(recordings) - len(failed),
                "failed": failed,
                "multiplier_errors": multiplier_errors,
                "reference_beats": aggregate_reference,
                "predicted_beats": aggregate_predicted,
                "minimum_f1": arguments.minimum_f1,
                "tolerance_ms": arguments.tolerance * 1000,
            },
            ensure_ascii=False,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
