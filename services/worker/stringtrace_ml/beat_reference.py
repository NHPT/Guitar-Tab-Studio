from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import soundfile

from .metrics import evaluate_beats


SCHEMA_VERSION = 2
REVIEW_KIND = "beat-grid"
REQUIRED_CHECKS = ("beat_times", "downbeats", "meter", "completeness")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_number(value: int | float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Review content contains a non-finite number")
    if number == 0:
        return "0"
    return f"{number:.12f}".rstrip("0").rstrip(".")


def _canonical_json(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _canonical_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json.dumps(str(key), ensure_ascii=False)
                + ":"
                + _canonical_json(value[key])
                for key in sorted(value)
            )
            + "}"
        )
    raise ValueError("Review content contains an unsupported value")


def review_content_digest(review: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in review.items()
        if key not in {"status", "submission", "adjudication"}
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def beat_markers_from_analysis(
    analysis: dict[str, Any],
    *,
    source_start: float,
    source_end: float,
) -> list[dict[str, int | float | bool]]:
    numerator, _denominator = [
        int(value)
        for value in analysis.get("time_signature", [4, 4])
    ]
    pickup_beats = int(analysis.get("pickup_beats", 0))
    markers: list[dict[str, int | float | bool]] = []
    for index, raw_at in enumerate(analysis.get("beats", [])):
        at = float(raw_at)
        if index < pickup_beats:
            beat_number = numerator - pickup_beats + index + 1
            measure = 0
        else:
            beat_number = (index - pickup_beats) % numerator + 1
            measure = (index - pickup_beats) // numerator + 1
        if source_start <= at < source_end:
            markers.append(
                {
                    "at": round(at - source_start, 6),
                    "beat": beat_number,
                    "measure": measure,
                    "downbeat": beat_number == 1,
                }
            )
    return markers


def evaluation_markers(review: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation_start = float(review["evaluation_start"])
    evaluation_end = float(review["evaluation_end"])
    return [
        marker
        for marker in review["beats"]
        if evaluation_start <= float(marker["at"]) < evaluation_end
    ]


def validate_review(
    review: dict[str, Any],
    *,
    require_submitted: bool = False,
    require_adjudicated: bool = False,
) -> None:
    if review.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported beat review schema version")
    if review.get("review_kind") != REVIEW_KIND:
        raise ValueError("Review kind must be beat-grid")
    for name in ("clip_id", "group_id"):
        if not ID_PATTERN.fullmatch(str(review.get(name, ""))):
            raise ValueError(f"{name} contains unsafe characters")
    if not ALIAS_PATTERN.fullmatch(str(review.get("annotator_alias", ""))):
        raise ValueError("annotator_alias contains unsafe characters")
    audio_name = Path(str(review.get("audio", "")))
    if (
        not audio_name.name
        or audio_name.is_absolute()
        or audio_name.name != str(audio_name)
    ):
        raise ValueError("audio must be a local filename")
    for name in ("source_audio_sha256", "audio_sha256"):
        if not SHA256_PATTERN.fullmatch(str(review.get(name, ""))):
            raise ValueError(f"{name} must be a SHA-256 digest")
    corpus_version = str(review.get("corpus_version", ""))
    if not ID_PATTERN.fullmatch(corpus_version):
        raise ValueError("corpus_version contains unsafe characters")
    audio_start = float(review.get("audio_source_start", math.nan))
    audio_end = float(review.get("audio_source_end", math.nan))
    audio_duration = float(review.get("audio_duration", math.nan))
    evaluation_start = float(review.get("evaluation_start", math.nan))
    evaluation_end = float(review.get("evaluation_end", math.nan))
    evaluation_duration = float(
        review.get("evaluation_duration", math.nan)
    )
    if (
        not all(
            math.isfinite(value)
            for value in (
                audio_start,
                audio_end,
                audio_duration,
                evaluation_start,
                evaluation_end,
                evaluation_duration,
            )
        )
        or audio_start < 0
        or audio_duration <= 0
        or abs(audio_end - audio_start - audio_duration) > 1e-5
        or evaluation_start < 0
        or evaluation_duration <= 0
        or evaluation_end > audio_duration + 1e-5
        or abs(evaluation_end - evaluation_start - evaluation_duration) > 1e-5
    ):
        raise ValueError("Invalid beat review audio or evaluation boundaries")
    signature = review.get("time_signature")
    if (
        not isinstance(signature, list)
        or len(signature) != 2
        or any(not isinstance(value, int) or value <= 0 for value in signature)
    ):
        raise ValueError("time_signature must contain two positive integers")
    numerator = int(signature[0])
    pickup_beats = int(review.get("pickup_beats", 0))
    if not 0 <= pickup_beats < numerator:
        raise ValueError("pickup_beats must be smaller than the meter numerator")
    beats = review.get("beats")
    if not isinstance(beats, list):
        raise ValueError("beats must be a list")
    previous: dict[str, Any] | None = None
    for index, marker in enumerate(beats):
        if not isinstance(marker, dict):
            raise ValueError(f"Beat {index + 1} must be an object")
        at = float(marker.get("at", math.nan))
        beat = int(marker.get("beat", 0))
        measure = int(marker.get("measure", -1))
        downbeat = marker.get("downbeat")
        if (
            not math.isfinite(at)
            or not 0 <= at < audio_duration
            or not 1 <= beat <= numerator
            or measure < 0
            or not isinstance(downbeat, bool)
            or downbeat != (beat == 1)
        ):
            raise ValueError(f"Beat {index + 1} is invalid")
        if previous is not None:
            expected_beat = int(previous["beat"]) % numerator + 1
            expected_measure = int(previous["measure"]) + (
                1 if expected_beat == 1 else 0
            )
            if (
                at <= float(previous["at"])
                or beat != expected_beat
                or measure != expected_measure
            ):
                raise ValueError("Beat markers must be sequential")
        previous = marker
    checks = review.get("checks")
    if (
        not isinstance(checks, dict)
        or any(not isinstance(checks.get(name), bool) for name in REQUIRED_CHECKS)
    ):
        raise ValueError("Beat review checks must be boolean values")
    status = review.get("status")
    if status not in {"draft", "submitted", "adjudicated"}:
        raise ValueError("Unknown beat review status")
    if require_submitted and status not in {"submitted", "adjudicated"}:
        raise ValueError("Beat review is not submitted")
    if status == "submitted":
        if (
            len(evaluation_markers(review)) < 2
            or not all(checks[name] for name in REQUIRED_CHECKS)
        ):
            raise ValueError("Submitted beat reviews require complete checks")
        submission = review.get("submission")
        if (
            not isinstance(submission, dict)
            or submission.get("content_sha256") != review_content_digest(review)
        ):
            raise ValueError("Beat review changed after submission")
    if require_adjudicated and status != "adjudicated":
        raise ValueError("Beat review is not adjudicated")
    if status == "adjudicated":
        if review.get("verification_status") != "adjudicated_gold":
            raise ValueError("Adjudicated review must be marked as gold")
        if review.get("benchmark_eligible") is not True:
            raise ValueError("Adjudicated review must be benchmark eligible")
        adjudication = review.get("adjudication")
        if (
            not isinstance(adjudication, dict)
            or not ALIAS_PATTERN.fullmatch(
                str(adjudication.get("adjudicator_alias", ""))
            )
            or adjudication.get("content_sha256")
            != review_content_digest(review)
        ):
            raise ValueError("Adjudicated beat review has an invalid digest")


def create_review(
    *,
    audio_path: Path,
    review_path: Path,
    clip_id: str,
    group_id: str,
    annotator_alias: str,
    source_start: float,
    duration: float,
    corpus_version: str = "beat-pilot-v2",
    pre_roll: float = 4.0,
    post_roll: float = 4.0,
    analysis_path: Path | None = None,
    time_signature: tuple[int, int] = (4, 4),
    pickup_beats: int = 0,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    if review_path.exists():
        raise FileExistsError(f"Review already exists: {review_path.name}")
    if (
        source_start < 0
        or duration <= 0
        or pre_roll < 0
        or post_roll < 0
    ):
        raise ValueError("Clip boundaries and context must be non-negative")
    source_duration = float(soundfile.info(audio_path).duration)
    requested_evaluation_end = source_start + duration
    if requested_evaluation_end > source_duration + 1e-5:
        raise ValueError("Evaluation window exceeds the source audio")
    audio_source_start = max(0.0, source_start - pre_roll)
    audio_source_end = min(
        source_duration,
        requested_evaluation_end + post_roll,
    )
    audio, _sample_rate = librosa.load(
        audio_path,
        sr=22050,
        mono=True,
        offset=audio_source_start,
        duration=audio_source_end - audio_source_start,
    )
    if audio.size == 0:
        raise ValueError("Beat review clip contains no audio")
    actual_duration = len(audio) / 22050
    clip_audio = review_path.with_suffix(".wav")
    clip_audio.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(clip_audio, audio, 22050, subtype="PCM_16")
    analysis = (
        json.loads(analysis_path.read_text(encoding="utf-8"))
        if analysis_path is not None
        else None
    )
    if analysis is not None:
        raw_signature = analysis.get("time_signature", time_signature)
        time_signature = (int(raw_signature[0]), int(raw_signature[1]))
    audio_source_end = audio_source_start + actual_duration
    evaluation_start = source_start - audio_source_start
    evaluation_end = evaluation_start + duration
    if evaluation_end > actual_duration + 1e-5:
        raise ValueError("Decoded audio does not contain the evaluation window")
    candidate_beats = (
        beat_markers_from_analysis(
            analysis,
            source_start=audio_source_start,
            source_end=audio_source_end,
        )
        if analysis is not None
        else []
    )
    if analysis is not None:
        pickup_beats = int(analysis.get("pickup_beats", pickup_beats))
    review: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review_kind": REVIEW_KIND,
        "status": "draft",
        "corpus_version": corpus_version,
        "clip_id": clip_id,
        "group_id": group_id,
        "annotator_alias": annotator_alias,
        "audio": clip_audio.name,
        "source_audio_sha256": sha256_file(audio_path),
        "audio_sha256": sha256_file(clip_audio),
        "audio_source_start": round(audio_source_start, 6),
        "audio_source_end": round(audio_source_end, 6),
        "audio_duration": round(actual_duration, 6),
        "evaluation_start": round(evaluation_start, 6),
        "evaluation_end": round(evaluation_end, 6),
        "evaluation_duration": round(duration, 6),
        "time_signature": list(time_signature),
        "pickup_beats": pickup_beats,
        "beats": candidate_beats,
        "tags": sorted(set(tags or [])),
        "checks": {name: False for name in REQUIRED_CHECKS},
        "candidate_source": (
            {
                "kind": "worker-analysis",
                "analysis_sha256": sha256_file(analysis_path),
            }
            if analysis_path is not None
            else {"kind": "manual"}
        ),
        "submission": None,
        "adjudication": None,
        "verification_status": "draft",
        "benchmark_eligible": False,
    }
    validate_review(review)
    _write_json(review_path, review)
    return review


def submit_review(review_path: Path) -> dict[str, Any]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    validate_review(review)
    review["checks"] = {name: True for name in REQUIRED_CHECKS}
    review["status"] = "submitted"
    review["verification_status"] = "independent_annotation"
    review["submission"] = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": review_content_digest(review),
    }
    validate_review(review, require_submitted=True)
    _write_json(review_path, review)
    return review


def _assert_same_clip(left: dict[str, Any], right: dict[str, Any]) -> None:
    for name in (
        "corpus_version",
        "clip_id",
        "group_id",
        "source_audio_sha256",
        "audio_source_start",
        "audio_source_end",
        "audio_duration",
        "evaluation_start",
        "evaluation_end",
        "evaluation_duration",
    ):
        if left.get(name) != right.get(name):
            raise ValueError(f"Beat reviews differ in {name}")


def compare_reviews(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    validate_review(left, require_submitted=True)
    validate_review(right, require_submitted=True)
    _assert_same_clip(left, right)
    if left["annotator_alias"] == right["annotator_alias"]:
        raise ValueError("Independent beat reviews require different annotators")
    left_window = evaluation_markers(left)
    right_window = evaluation_markers(right)
    left_beats = [float(marker["at"]) for marker in left_window]
    right_beats = [float(marker["at"]) for marker in right_window]
    left_downbeats = [
        float(marker["at"])
        for marker in left_window
        if marker["downbeat"]
    ]
    right_downbeats = [
        float(marker["at"])
        for marker in right_window
        if marker["downbeat"]
    ]
    return {
        "clip_id": left["clip_id"],
        "annotators": [
            left["annotator_alias"],
            right["annotator_alias"],
        ],
        "beat_metrics": evaluate_beats(left_beats, right_beats),
        "downbeat_metrics": evaluate_beats(
            left_downbeats,
            right_downbeats,
        ),
        "meter_agreement": (
            left["time_signature"] == right["time_signature"]
        ),
        "pickup_agreement": left["pickup_beats"] == right["pickup_beats"],
    }


def adjudicate_reviews(
    *,
    left_path: Path,
    right_path: Path,
    resolved_path: Path,
    output_path: Path,
    adjudicator_alias: str,
) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    agreement = compare_reviews(left, right)
    validate_review(resolved, require_submitted=True)
    _assert_same_clip(left, resolved)
    if adjudicator_alias in {
        left["annotator_alias"],
        right["annotator_alias"],
    }:
        raise ValueError("Adjudicator must differ from both annotators")
    if resolved["annotator_alias"] != adjudicator_alias:
        raise ValueError("Resolved review must be submitted by the adjudicator")
    source_audio = resolved_path.parent / str(resolved["audio"])
    output_audio = output_path.with_suffix(".wav")
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    if source_audio.resolve() != output_audio.resolve():
        shutil.copy2(source_audio, output_audio)
    final = {
        **resolved,
        "status": "adjudicated",
        "audio": output_audio.name,
        "audio_sha256": sha256_file(output_audio),
        "submission": None,
        "source_reviews": [
            {
                "annotator_alias": review["annotator_alias"],
                "content_sha256": review["submission"]["content_sha256"],
            }
            for review in (left, right)
        ],
        "agreement": agreement,
        "verification_status": "adjudicated_gold",
        "benchmark_eligible": True,
        "adjudication": None,
    }
    final["adjudication"] = {
        "adjudicator_alias": adjudicator_alias,
        "adjudicated_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": review_content_digest(final),
    }
    validate_review(final, require_adjudicated=True)
    _write_json(output_path, final)
    return final


def evaluate_review(
    review: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    validate_review(review, require_adjudicated=True)
    predicted = beat_markers_from_analysis(
        analysis,
        source_start=float(review["audio_source_start"]),
        source_end=float(review["audio_source_end"]),
    )
    evaluation_start = float(review["evaluation_start"])
    evaluation_end = float(review["evaluation_end"])
    reference_window = evaluation_markers(review)
    predicted_window = [
        marker
        for marker in predicted
        if evaluation_start <= float(marker["at"]) < evaluation_end
    ]
    reference_beats = [
        float(marker["at"]) for marker in reference_window
    ]
    predicted_beats = [
        float(marker["at"]) for marker in predicted_window
    ]
    reference_downbeats = [
        float(marker["at"])
        for marker in reference_window
        if marker["downbeat"]
    ]
    predicted_downbeats = [
        float(marker["at"])
        for marker in predicted_window
        if marker["downbeat"]
    ]
    return {
        "clip_id": review["clip_id"],
        "reference_beats": len(reference_beats),
        "predicted_beats": len(predicted_beats),
        "beat_metrics": evaluate_beats(reference_beats, predicted_beats),
        "downbeat_metrics": evaluate_beats(
            reference_downbeats,
            predicted_downbeats,
        ),
        "meter_correct": (
            list(analysis.get("time_signature", [4, 4]))
            == review["time_signature"]
        ),
        "pickup_correct": (
            int(analysis.get("pickup_beats", 0))
            == int(review["pickup_beats"])
        ),
    }


def evaluate_gate(
    report: dict[str, Any],
    *,
    minimum_beat_f1: float = 0.95,
    maximum_median_error_ms: float = 20.0,
    maximum_p95_error_ms: float = 50.0,
    minimum_downbeat_f1: float = 0.95,
) -> dict[str, Any]:
    beat_metrics = dict(report.get("beat_metrics") or {})
    downbeat_metrics = dict(report.get("downbeat_metrics") or {})
    median_error = beat_metrics.get("median_error_ms")
    p95_error = beat_metrics.get("p95_error_ms")
    checks = {
        "beat_f1": float(beat_metrics.get("f1", 0)) >= minimum_beat_f1,
        "beat_median_error": (
            median_error is not None
            and float(median_error) <= maximum_median_error_ms
        ),
        "beat_p95_error": (
            p95_error is not None
            and float(p95_error) <= maximum_p95_error_ms
        ),
        "downbeat_f1": (
            float(downbeat_metrics.get("f1", 0)) >= minimum_downbeat_f1
        ),
        "meter": report.get("meter_correct") is True,
        "pickup": report.get("pickup_correct") is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "values": {
            "beat_f1": float(beat_metrics.get("f1", 0)),
            "median_error_ms": median_error,
            "p95_error_ms": p95_error,
            "downbeat_f1": float(downbeat_metrics.get("f1", 0)),
            "meter_correct": report.get("meter_correct") is True,
            "pickup_correct": report.get("pickup_correct") is True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create, submit, adjudicate and evaluate beat gold reviews.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--audio", type=Path, required=True)
    initialize.add_argument("--analysis", type=Path)
    initialize.add_argument("--review", type=Path, required=True)
    initialize.add_argument("--clip-id", required=True)
    initialize.add_argument("--group-id", required=True)
    initialize.add_argument("--annotator-alias", required=True)
    initialize.add_argument("--start", type=float, required=True)
    initialize.add_argument("--duration", type=float, default=24.0)
    initialize.add_argument("--corpus-version", default="beat-pilot-v2")
    initialize.add_argument("--pre-roll", type=float, default=4.0)
    initialize.add_argument("--post-roll", type=float, default=4.0)
    initialize.add_argument("--time-signature", default="4/4")
    initialize.add_argument("--pickup-beats", type=int, default=0)
    initialize.add_argument("--tag", action="append")

    submit = commands.add_parser("submit")
    submit.add_argument("--review", type=Path, required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path)

    adjudicate = commands.add_parser("adjudicate")
    adjudicate.add_argument("--left", type=Path, required=True)
    adjudicate.add_argument("--right", type=Path, required=True)
    adjudicate.add_argument("--resolved", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    adjudicate.add_argument("--adjudicator-alias", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--review", type=Path, required=True)
    evaluate.add_argument("--analysis", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)

    gate = commands.add_parser("gate")
    gate.add_argument("--report", type=Path, required=True)
    gate.add_argument("--minimum-beat-f1", type=float, default=0.95)
    gate.add_argument("--maximum-median-error-ms", type=float, default=20.0)
    gate.add_argument("--maximum-p95-error-ms", type=float, default=50.0)
    gate.add_argument("--minimum-downbeat-f1", type=float, default=0.95)
    arguments = parser.parse_args()

    if arguments.command == "init":
        numerator, separator, denominator = arguments.time_signature.partition("/")
        if not separator:
            parser.error("--time-signature must use NUMERATOR/DENOMINATOR")
        result = create_review(
            audio_path=arguments.audio,
            analysis_path=arguments.analysis,
            review_path=arguments.review,
            clip_id=arguments.clip_id,
            group_id=arguments.group_id,
            annotator_alias=arguments.annotator_alias,
            source_start=arguments.start,
            duration=arguments.duration,
            corpus_version=arguments.corpus_version,
            pre_roll=arguments.pre_roll,
            post_roll=arguments.post_roll,
            time_signature=(int(numerator), int(denominator)),
            pickup_beats=arguments.pickup_beats,
            tags=arguments.tag,
        )
    elif arguments.command == "submit":
        result = submit_review(arguments.review)
    elif arguments.command == "compare":
        result = compare_reviews(
            json.loads(arguments.left.read_text(encoding="utf-8")),
            json.loads(arguments.right.read_text(encoding="utf-8")),
        )
        if arguments.output:
            _write_json(arguments.output, result)
    elif arguments.command == "adjudicate":
        result = adjudicate_reviews(
            left_path=arguments.left,
            right_path=arguments.right,
            resolved_path=arguments.resolved,
            output_path=arguments.output,
            adjudicator_alias=arguments.adjudicator_alias,
        )
    elif arguments.command == "evaluate":
        result = evaluate_review(
            json.loads(arguments.review.read_text(encoding="utf-8")),
            json.loads(arguments.analysis.read_text(encoding="utf-8")),
        )
        if arguments.output:
            _write_json(arguments.output, result)
    else:
        result = evaluate_gate(
            json.loads(arguments.report.read_text(encoding="utf-8")),
            minimum_beat_f1=arguments.minimum_beat_f1,
            maximum_median_error_ms=arguments.maximum_median_error_ms,
            maximum_p95_error_ms=arguments.maximum_p95_error_ms,
            minimum_downbeat_f1=arguments.minimum_downbeat_f1,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if arguments.command == "gate" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
