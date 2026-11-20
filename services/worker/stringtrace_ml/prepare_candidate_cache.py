from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

from .data_sources import sha256_file
from .evaluate_onset_position import BASIC_PITCH_CACHE_VERSION, basic_pitch_events
from .onset_position import basic_pitch_cache_path
from .schema import load_manifest


ProgressCallback = Callable[[dict[str, object]], None]


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def collect_candidate_audio(
    manifest_paths: Iterable[Path],
    *,
    splits: frozenset[str] = frozenset(),
    allowed_root: Path | None = None,
) -> tuple[list[Path], list[dict[str, object]]]:
    root = allowed_root.resolve() if allowed_root is not None else None
    audio_by_path: dict[Path, None] = {}
    manifest_reports: list[dict[str, object]] = []
    for source_path in manifest_paths:
        manifest_path = source_path.resolve()
        annotations = load_manifest(manifest_path)
        selected = [
            annotation
            for annotation in annotations
            if not splits or annotation.split in splits
        ]
        for annotation in selected:
            audio_path = annotation.resolve_audio(manifest_path).resolve()
            if root is not None and not _inside_root(audio_path, root):
                raise ValueError(
                    f"Audio resolves outside allowed root: {annotation.track_id}"
                )
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Missing audio for track {annotation.track_id}: "
                    f"{audio_path.name}"
                )
            audio_by_path[audio_path] = None
        manifest_reports.append(
            {
                "name": manifest_path.name,
                "sha256": sha256_file(manifest_path),
                "records": len(annotations),
                "selected_records": len(selected),
            }
        )
    return sorted(audio_by_path), manifest_reports


def validate_candidate_cache(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid candidate cache {path.name}: {error}") from error
    if int(payload.get("version", 0)) != BASIC_PITCH_CACHE_VERSION:
        raise ValueError(
            f"Unsupported candidate cache version in {path.name}: "
            f"{payload.get('version')!r}"
        )
    events = payload.get("events")
    dropped = payload.get("dropped")
    if not isinstance(events, list):
        raise ValueError(f"Candidate cache events are not a list: {path.name}")
    if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped < 0:
        raise ValueError(f"Candidate cache dropped count is invalid: {path.name}")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(
                f"Candidate cache event {index} is not an object: {path.name}"
            )
        try:
            onset = float(event["onset"])
            offset = float(event["offset"])
            pitch = int(event["pitch"])
            confidence = float(event["confidence"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Candidate cache event {index} is malformed: {path.name}"
            ) from error
        if (
            not math.isfinite(onset)
            or not math.isfinite(offset)
            or onset < 0
            or offset <= onset
            or not 0 <= pitch <= 127
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError(
                f"Candidate cache event {index} is out of range: {path.name}"
            )
    return {"events": len(events), "dropped": dropped}


def prepare_candidate_caches(
    audio_paths: Iterable[Path],
    cache_directory: Path,
    *,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length: float,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    cache_directory = cache_directory.resolve()
    paths = list(audio_paths)
    generated = 0
    existing = 0
    event_total = 0
    dropped_total = 0
    cache_identities: list[tuple[str, str]] = []
    cache_directory.mkdir(parents=True, exist_ok=True)
    for index, audio_path in enumerate(paths, start=1):
        cache_path = basic_pitch_cache_path(
            audio_path,
            cache_directory,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
        )
        if cache_path.is_file():
            status = "existing"
            existing += 1
        else:
            basic_pitch_events(
                audio_path,
                cache_directory,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=minimum_note_length,
            )
            status = "generated"
            generated += 1
        cache_counts = validate_candidate_cache(cache_path)
        event_total += cache_counts["events"]
        dropped_total += cache_counts["dropped"]
        cache_identities.append(
            (cache_path.name, sha256_file(cache_path))
        )
        if progress is not None:
            progress(
                {
                    "stage": "candidate_cache",
                    "processed": index,
                    "total": len(paths),
                    "status": status,
                    "cache": cache_path.name,
                    **cache_counts,
                }
            )
    cache_set_digest = hashlib.sha256()
    for name, digest in sorted(cache_identities):
        cache_set_digest.update(f"{name}:{digest}\n".encode("utf-8"))
    return {
        "audio_files": len(paths),
        "existing": existing,
        "generated": generated,
        "events": event_total,
        "dropped": dropped_total,
        "cache_set_sha256": cache_set_digest.hexdigest(),
    }


def _threshold(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("threshold must be in 0..1")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute and validate Basic Pitch candidate caches for one or "
            "more governed manifests."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        default=[],
    )
    parser.add_argument("--allowed-root", type=Path)
    parser.add_argument(
        "--onset-threshold",
        type=_threshold,
        default=0.35,
    )
    parser.add_argument(
        "--frame-threshold",
        type=_threshold,
        default=0.25,
    )
    parser.add_argument(
        "--minimum-note-length",
        type=_positive_float,
        default=70.0,
    )
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()

    audio_paths, manifest_reports = collect_candidate_audio(
        arguments.manifest,
        splits=frozenset(arguments.split),
        allowed_root=arguments.allowed_root,
    )

    def emit_progress(record: dict[str, object]) -> None:
        if (
            record["status"] == "generated"
            or int(record["processed"]) == len(audio_paths)
        ):
            print(json.dumps(record), flush=True)

    summary = prepare_candidate_caches(
        audio_paths,
        arguments.cache_directory,
        onset_threshold=arguments.onset_threshold,
        frame_threshold=arguments.frame_threshold,
        minimum_note_length=arguments.minimum_note_length,
        progress=emit_progress,
    )
    report = {
        "schema_version": 1,
        "kind": "stringtrace-candidate-cache-preparation",
        "manifests": manifest_reports,
        "splits": sorted(set(arguments.split)),
        "frontend": {
            "kind": "spotify-basic-pitch-0.4",
            "onset_threshold": arguments.onset_threshold,
            "frame_threshold": arguments.frame_threshold,
            "minimum_note_length": arguments.minimum_note_length,
        },
        "summary": summary,
    }
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
