from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import soundfile

from .schema import ExcludedRange, LabeledEvent, TrackAnnotation


REVIEW_SCHEMA_VERSION = 1
REQUIRED_REVIEW_CHECKS = ("timing", "string_fret", "completeness")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REVIEWER_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def required_review_checks(review: dict[str, Any]) -> tuple[str, ...]:
    if review.get("review_kind") == "residual-onset-training":
        return ("timing", "completeness")
    return REQUIRED_REVIEW_CHECKS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_project(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("tab"), dict):
        raise ValueError("Project must contain a tab document")
    return value


def select_project_measures(
    project: dict[str, Any],
    measure_numbers: list[int],
) -> list[dict[str, Any]]:
    selected_numbers = sorted(set(measure_numbers))
    if not selected_numbers:
        raise ValueError("At least one measure is required")
    measures = project["tab"].get("measures", [])
    indexes = [
        index
        for index, measure in enumerate(measures)
        if int(measure["number"]) in selected_numbers
    ]
    if len(indexes) != len(selected_numbers):
        available = {int(measure["number"]) for measure in measures}
        missing = sorted(set(selected_numbers) - available)
        raise ValueError(f"Unknown measure numbers: {missing}")
    if indexes != list(range(indexes[0], indexes[-1] + 1)):
        raise ValueError("Reviewed measures must form one contiguous excerpt")
    return [measures[index] for index in indexes]


def measure_clip(
    selected_measures: list[dict[str, Any]],
) -> tuple[float, float]:
    clip_start = min(float(measure["start"]) for measure in selected_measures)
    clip_end = max(
        float(measure["start"]) + float(measure["duration"])
        for measure in selected_measures
    )
    return clip_start, clip_end


def relative_beats(
    selected_measures: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> list[float]:
    return sorted(
        {
            round(float(beat["at"]) - clip_start, 6)
            for measure in selected_measures
            for beat in measure["beats"]
            if clip_start <= float(beat["at"]) < clip_end
        }
    )


def _event_interval(
    note: dict[str, Any],
    clip_start: float,
    clip_end: float,
) -> tuple[float, float]:
    source_onset = float(note["at"])
    source_offset = source_onset + float(note["duration"])
    onset = max(0.0, source_onset - clip_start)
    offset = min(clip_end - clip_start, source_offset - clip_start)
    return round(onset, 6), round(max(onset + 0.01, offset), 6)


def candidate_events(
    selected_measures: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for measure in selected_measures:
        for beat in measure["beats"]:
            for note in beat["notes"]:
                onset, offset = _event_interval(note, clip_start, clip_end)
                events.append(
                    {
                        "onset": onset,
                        "offset": offset,
                        "string": int(note["string"]),
                        "fret": int(note["fret"]),
                        "technique": "unknown",
                        "confidence": 1.0,
                        "candidate_provenance": {
                            "project_note_id": str(note.get("id", "")),
                            "position_source": str(
                                note.get("positionSource", "unknown")
                            ),
                            "technique_source": str(
                                note.get("techniqueSource", "unknown")
                            ),
                        },
                    }
                )
    events.sort(key=lambda event: (float(event["onset"]), int(event["string"])))
    return events


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def review_content_digest(review: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in review.items()
        if key not in {"status", "approval"}
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_review(
    review: dict[str, Any],
    *,
    require_approved: bool,
) -> None:
    if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported review schema version {review.get('schema_version')}"
        )
    if not ID_PATTERN.fullmatch(str(review.get("review_id", ""))):
        raise ValueError("Review ID contains unsafe characters")
    if not ID_PATTERN.fullmatch(str(review.get("project_id", ""))):
        raise ValueError("Project ID contains unsafe characters")
    review_audio = Path(str(review.get("review_audio", "")))
    if (
        not review_audio.name
        or review_audio.is_absolute()
        or review_audio.name != str(review_audio)
    ):
        raise ValueError("Review audio must be a local filename")
    if review.get("review_kind") == "residual-onset-training":
        training_audio = Path(str(review.get("training_audio", "")))
        if (
            not training_audio.name
            or training_audio.is_absolute()
            or training_audio.name != str(training_audio)
        ):
            raise ValueError("Training audio must be a local filename")
        if training_audio == review_audio:
            raise ValueError(
                "Review and training audio must be separate files"
            )
    measure_numbers = review.get("measure_numbers")
    if (
        not isinstance(measure_numbers, list)
        or not measure_numbers
        or any(not isinstance(number, int) for number in measure_numbers)
    ):
        raise ValueError("measure_numbers must contain integer measure numbers")
    duration = float(review.get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Review duration must be positive")
    source_start = float(review.get("source_start", 0))
    source_end = float(review.get("source_end", 0))
    source_duration = source_end - source_start
    if not all(math.isfinite(value) for value in (source_start, source_end)):
        raise ValueError("Review clip boundaries must be finite")
    if abs(source_duration - duration) > 1e-5:
        raise ValueError("Review clip boundaries do not match its duration")
    capo = int(review.get("capo", -1))
    if not 0 <= capo <= 12:
        raise ValueError("Review capo must be in 0..12")
    bpm = float(review.get("bpm", 0))
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("Review BPM must be positive")
    beats = review.get("beats")
    if (
        not isinstance(beats, list)
        or beats != sorted(beats)
        or any(
            not math.isfinite(float(beat))
            or not 0 <= float(beat) < duration
            for beat in beats
        )
    ):
        raise ValueError("Review beats must be ordered within the excerpt")
    time_signature = review.get("time_signature")
    if (
        not isinstance(time_signature, list)
        or len(time_signature) != 2
        or any(int(value) <= 0 for value in time_signature)
    ):
        raise ValueError("Review time signature must contain two positive values")
    events_value = review.get("events")
    if not isinstance(events_value, list):
        raise ValueError("Review events must be a list")
    events = [LabeledEvent.from_dict(event) for event in events_value]
    excluded_ranges_value = review.get("excluded_ranges", [])
    if not isinstance(excluded_ranges_value, list) or any(
        not isinstance(value, dict)
        for value in excluded_ranges_value
    ):
        raise ValueError("Review excluded_ranges must be a list of objects")
    excluded_ranges = [
        ExcludedRange.from_dict(value)
        for value in excluded_ranges_value
    ]
    TrackAnnotation(
        track_id=str(review["review_id"]),
        group_id=str(review["project_id"]),
        audio="review.wav",
        split="test",
        duration=duration,
        events=events,
        excluded_ranges=excluded_ranges,
    ).validate()
    checks = review.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("Review checks are required")
    check_names = (*REQUIRED_REVIEW_CHECKS, "technique")
    if any(not isinstance(checks.get(name), bool) for name in check_names):
        raise ValueError("Review checks must be boolean values")
    hash_names = [
        "project_sha256",
        "source_audio_sha256",
        "review_audio_sha256",
    ]
    if review.get("review_kind") == "residual-onset-training":
        hash_names.append("training_audio_sha256")
    for name in hash_names:
        if not SHA256_PATTERN.fullmatch(str(review.get(name, ""))):
            raise ValueError(f"{name} must be a SHA-256 digest")
    if review.get("review_kind") == "residual-onset-training":
        provenance = review.get("training_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("Residual-onset review provenance is required")
        for name in (
            "review_source_track_id",
            "training_source_audio_sha256",
            "audio_alignment",
            "review_audio_processing",
        ):
            if not isinstance(provenance.get(name), str):
                raise ValueError(
                    f"Residual-onset provenance field {name} is required"
                )
        if not ID_PATTERN.fullmatch(provenance["review_source_track_id"]):
            raise ValueError("Review source track ID contains unsafe characters")
        if not SHA256_PATTERN.fullmatch(
            provenance["training_source_audio_sha256"]
        ):
            raise ValueError(
                "training_source_audio_sha256 must be a SHA-256 digest"
            )
        if provenance["audio_alignment"] != "sample-aligned-no-offset":
            raise ValueError("Review and training audio must be sample-aligned")
        if (
            provenance["review_audio_processing"]
            != "rms-normalized--20-dbfs-soft-limited--1-dbfs"
        ):
            raise ValueError("Review audio processing is unsupported")
        review_gain_db = float(
            provenance.get("review_audio_gain_db", math.nan)
        )
        if not math.isfinite(review_gain_db) or not -60 <= review_gain_db <= 60:
            raise ValueError("Review audio gain is invalid")
    if not require_approved:
        return
    if review.get("status") != "approved":
        raise ValueError("Review is not approved")
    missing = [
        name
        for name in required_review_checks(review)
        if not checks[name]
    ]
    if missing:
        raise ValueError(f"Review is missing required checks: {missing}")
    approval = review.get("approval")
    if (
        not isinstance(approval, dict)
        or not REVIEWER_ALIAS_PATTERN.fullmatch(
            str(approval.get("reviewer_alias", ""))
        )
    ):
        raise ValueError("Approved review must contain a safe reviewer alias")
    expected_digest = review_content_digest(review)
    if approval.get("content_sha256") != expected_digest:
        raise ValueError("Review content changed after approval")


def create_review(
    project_path: Path,
    audio_path: Path,
    review_path: Path,
    measure_numbers: list[int],
    *,
    capo: int | None = None,
) -> dict[str, Any]:
    if review_path.exists():
        raise FileExistsError(f"Review already exists: {review_path.name}")
    project = load_project(project_path)
    selected_measures = select_project_measures(project, measure_numbers)
    selected_numbers = [
        int(measure["number"])
        for measure in selected_measures
    ]
    clip_start, clip_end = measure_clip(selected_measures)
    duration = round(clip_end - clip_start, 6)
    review_audio_path = review_path.with_suffix(".wav")
    audio, _sample_rate = librosa.load(
        audio_path,
        sr=22050,
        mono=True,
        offset=clip_start,
        duration=duration,
    )
    review_audio_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(review_audio_path, audio, 22050, subtype="PCM_16")
    measure_suffix = "-".join(str(number) for number in selected_numbers)
    source = project.get("source", {})
    review: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "status": "draft",
        "review_id": f"{project['id']}-m{measure_suffix}",
        "project_id": str(project["id"]),
        "project_sha256": sha256_file(project_path),
        "source_audio_sha256": sha256_file(audio_path),
        "review_audio": review_audio_path.name,
        "review_audio_sha256": sha256_file(review_audio_path),
        "source_kind": str(source.get("kind", "unknown")),
        "measure_numbers": selected_numbers,
        "source_start": round(clip_start, 6),
        "source_end": round(clip_end, 6),
        "duration": duration,
        "capo": (
            int(capo)
            if capo is not None
            else int(project["tab"].get("capo", 0))
        ),
        "bpm": float(project["bpm"]),
        "beats": relative_beats(selected_measures, clip_start, clip_end),
        "time_signature": [4, 4],
        "events": candidate_events(selected_measures, clip_start, clip_end),
        "excluded_ranges": [],
        "checks": {
            "timing": False,
            "string_fret": False,
            "completeness": False,
            "technique": False,
        },
        "approval": None,
    }
    validate_review(review, require_approved=False)
    _write_json(review_path, review)
    return review


def approve_review(
    review_path: Path,
    *,
    reviewer_alias: str,
    timing_reviewed: bool,
    string_fret_reviewed: bool,
    completeness_reviewed: bool,
    technique_reviewed: bool = False,
) -> dict[str, Any]:
    reviewer_alias = reviewer_alias.strip()
    if not REVIEWER_ALIAS_PATTERN.fullmatch(reviewer_alias):
        raise ValueError("Reviewer alias contains unsafe characters")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    validate_review(review, require_approved=False)
    checks = {
        "timing": timing_reviewed,
        "string_fret": string_fret_reviewed,
        "completeness": completeness_reviewed,
        "technique": technique_reviewed,
    }
    missing = [
        name
        for name in required_review_checks(review)
        if not checks[name]
    ]
    if missing:
        raise ValueError(f"Cannot approve without checks: {missing}")
    review["checks"] = checks
    review["status"] = "approved"
    review["approval"] = {
        "reviewer_alias": reviewer_alias,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": review_content_digest(review),
    }
    validate_review(review, require_approved=True)
    _write_json(review_path, review)
    return review


def load_approved_review(
    review_path: Path,
    project_path: Path,
    audio_path: Path,
) -> dict[str, Any]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    validate_review(review, require_approved=True)
    if sha256_file(project_path) != review["project_sha256"]:
        raise ValueError("Project snapshot changed after review creation")
    if sha256_file(audio_path) != review["source_audio_sha256"]:
        raise ValueError("Source audio changed after review creation")
    review_audio_path = review_path.parent / str(review["review_audio"])
    if not review_audio_path.is_file():
        raise ValueError("Reviewed audio excerpt is missing")
    if sha256_file(review_audio_path) != review["review_audio_sha256"]:
        raise ValueError("Reviewed audio excerpt changed after review creation")
    project = load_project(project_path)
    if str(project.get("id")) != review["project_id"]:
        raise ValueError("Review project ID does not match the project snapshot")
    return review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and approve complete held-out reference reviews.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--project", type=Path, required=True)
    initialize.add_argument("--audio", type=Path, required=True)
    initialize.add_argument("--review", type=Path, required=True)
    initialize.add_argument("--measure", type=int, action="append", required=True)
    initialize.add_argument("--capo", type=int)

    approve = commands.add_parser("approve")
    approve.add_argument("--review", type=Path, required=True)
    approve.add_argument("--reviewer-alias", required=True)
    approve.add_argument("--timing-reviewed", action="store_true")
    approve.add_argument("--string-fret-reviewed", action="store_true")
    approve.add_argument("--completeness-reviewed", action="store_true")
    approve.add_argument("--technique-reviewed", action="store_true")
    arguments = parser.parse_args()

    if arguments.command == "init":
        review = create_review(
            arguments.project,
            arguments.audio,
            arguments.review,
            arguments.measure,
            capo=arguments.capo,
        )
    else:
        review = approve_review(
            arguments.review,
            reviewer_alias=arguments.reviewer_alias,
            timing_reviewed=arguments.timing_reviewed,
            string_fret_reviewed=arguments.string_fret_reviewed,
            completeness_reviewed=arguments.completeness_reviewed,
            technique_reviewed=arguments.technique_reviewed,
        )
    print(
        json.dumps(
            {
                "review_id": review["review_id"],
                "status": review["status"],
                "events": len(review["events"]),
                "measure_numbers": review["measure_numbers"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
