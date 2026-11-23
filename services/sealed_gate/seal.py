from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import soundfile

from stringtrace_ml.data_sources import (
    enforce_records,
    load_source_ledger,
    sha256_file,
)
from stringtrace_ml.schema import (
    TrackAnnotation,
    load_manifest,
    reject_quarantined_references,
)


MINIMUM_TRACKS = 100
MINIMUM_WORKS = 100
MINIMUM_RECORDINGS = 100
MINIMUM_LINEAGES = 100
MINIMUM_PERFORMERS = 5
MINIMUM_SESSIONS = 5
MINIMUM_SOURCE_GROUPS = 5
MINIMUM_GUITARS = 3
MINIMUM_TRACK_DURATION_SECONDS = 30.0
MINIMUM_KNOWN_TECHNIQUES = 3
REQUIRED_CAPTURE_MODES = {"microphone", "pickup"}
IDENTITY_FIELDS = (
    "track_id",
    "group_id",
    "performer_id",
    "work_id",
    "recording_id",
    "session_id",
    "lineage_id",
    "audio_sha256",
)
SEALED_IDENTITY_FIELDS = IDENTITY_FIELDS[2:]
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+ -]{0,199}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def identity_digest(field: str, value: str) -> str:
    return hashlib.sha256(f"{field}\0{value}".encode("utf-8")).hexdigest()


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        records.append(value)
    return records


def _provenance(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("provenance")
    return value if isinstance(value, dict) else {}


def _first_nonempty(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def exclusion_identity(
    record: dict[str, Any],
    field: str,
) -> str | None:
    provenance = _provenance(record)
    if field == "track_id":
        return _first_nonempty(record.get("track_id"))
    if field == "group_id":
        return _first_nonempty(record.get("group_id"))
    if field == "performer_id":
        return _first_nonempty(
            provenance.get("performer_id"),
            provenance.get("player_id"),
            provenance.get("source_group_id"),
        )
    if field == "work_id":
        return _first_nonempty(
            provenance.get("work_id"),
            provenance.get("review_source_track_id"),
            provenance.get("source_label"),
        )
    if field == "recording_id":
        return _first_nonempty(
            provenance.get("recording_id"),
            provenance.get("source_track_id"),
        )
    if field == "session_id":
        return _first_nonempty(
            provenance.get("session_id"),
            provenance.get("project_id"),
        )
    if field == "lineage_id":
        return _first_nonempty(provenance.get("lineage_id"))
    if field == "audio_sha256":
        return _first_nonempty(
            provenance.get("audio_sha256"),
            provenance.get("source_audio_sha256"),
            provenance.get("training_audio_sha256"),
        )
    raise ValueError(f"Unknown identity field: {field}")


def build_exclusion_index(
    manifest_paths: Iterable[Path],
) -> dict[str, Any]:
    paths = sorted({path.resolve() for path in manifest_paths})
    if not paths:
        raise ValueError("At least one exclusion manifest is required")
    identities: dict[str, set[str]] = {
        field: set() for field in IDENTITY_FIELDS
    }
    manifests: list[dict[str, object]] = []
    for path in paths:
        records = _jsonl_records(path)
        manifests.append(
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "records": len(records),
            }
        )
        for record in records:
            for field in IDENTITY_FIELDS:
                value = exclusion_identity(record, field)
                if value is not None:
                    identities[field].add(identity_digest(field, value))
    return {
        "schema_version": 1,
        "kind": "sealed-evaluation-exclusion-index",
        "manifests": manifests,
        "identity_counts": {
            field: len(values)
            for field, values in identities.items()
        },
        "identity_digests": {
            field: sorted(values)
            for field, values in identities.items()
        },
    }


def write_exclusion_index(
    manifest_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, Any]:
    index = build_exclusion_index(manifest_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json(index) + b"\n")
    return index


def load_exclusion_index(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("kind") != "sealed-evaluation-exclusion-index"
    ):
        raise ValueError("Unsupported sealed exclusion index")
    identities = value.get("identity_digests")
    if not isinstance(identities, dict):
        raise ValueError("Sealed exclusion index has no identity digests")
    for field in IDENTITY_FIELDS:
        digests = identities.get(field)
        if not isinstance(digests, list):
            raise ValueError(
                f"Sealed exclusion index is missing {field} digests"
            )
        if (
            digests != sorted(set(digests))
            or any(
                not isinstance(digest, str)
                or not SHA256_PATTERN.fullmatch(digest)
                for digest in digests
            )
        ):
            raise ValueError(
                f"Sealed exclusion index has invalid {field} digests"
            )
    return value


def _sealed_identity(
    annotation: TrackAnnotation,
    field: str,
) -> str:
    value = annotation.provenance.get(field)
    if (
        not isinstance(value, str)
        or not SAFE_ID_PATTERN.fullmatch(value.strip())
    ):
        raise ValueError(
            f"Sealed track {annotation.track_id} requires a valid {field}"
        )
    return value.strip()


def _declared_audio_sha256(annotation: TrackAnnotation) -> str:
    value = annotation.provenance.get("audio_sha256")
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(
            f"Sealed track {annotation.track_id} requires audio_sha256"
        )
    return value


def _tempo_bucket(bpm: float) -> str:
    if bpm < 90:
        return "slow"
    if bpm < 130:
        return "medium"
    return "fast"


def _validate_assets(
    manifest_path: Path,
    annotations: Iterable[TrackAnnotation],
) -> list[str]:
    root = manifest_path.resolve().parent
    audio_hashes: list[str] = []
    for annotation in annotations:
        audio_path = annotation.resolve_audio(manifest_path)
        try:
            audio_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Sealed audio escapes the sealed root: {annotation.track_id}"
            ) from error
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        declared = _declared_audio_sha256(annotation)
        if sha256_file(audio_path) != declared:
            raise ValueError(
                f"Sealed audio SHA-256 mismatch: {annotation.track_id}"
            )
        try:
            audio_info = soundfile.info(audio_path)
        except RuntimeError as error:
            raise ValueError(
                f"Sealed audio cannot be decoded: {annotation.track_id}"
            ) from error
        if audio_info.samplerate < 8_000:
            raise ValueError(
                f"Sealed audio sample rate is too low: {annotation.track_id}"
            )
        if audio_info.channels not in {1, 2}:
            raise ValueError(
                f"Sealed audio must be mono or stereo: {annotation.track_id}"
            )
        if (
            abs(float(audio_info.duration) - annotation.duration)
            > max(0.05, annotation.duration * 1e-4)
        ):
            raise ValueError(
                f"Sealed audio duration differs: {annotation.track_id}"
            )
        audio_hashes.append(declared)
    return audio_hashes


def _identity_sets(
    annotations: Iterable[TrackAnnotation],
) -> dict[str, set[str]]:
    result = {
        field: set() for field in SEALED_IDENTITY_FIELDS
    }
    for annotation in annotations:
        for field in SEALED_IDENTITY_FIELDS:
            value = (
                _declared_audio_sha256(annotation)
                if field == "audio_sha256"
                else _sealed_identity(annotation, field)
            )
            result[field].add(value)
    return result


def _validate_isolation(
    annotations: Iterable[TrackAnnotation],
    exclusion_index: dict[str, Any],
) -> None:
    excluded = {
        field: set(exclusion_index["identity_digests"][field])
        for field in IDENTITY_FIELDS
    }
    collisions: Counter[str] = Counter()
    for annotation in annotations:
        values = {
            "track_id": annotation.track_id,
            "group_id": annotation.group_id,
            **{
                field: (
                    _declared_audio_sha256(annotation)
                    if field == "audio_sha256"
                    else _sealed_identity(annotation, field)
                )
                for field in SEALED_IDENTITY_FIELDS
            },
        }
        for field, value in values.items():
            if identity_digest(field, value) in excluded[field]:
                collisions[field] += 1
    if collisions:
        detail = ", ".join(
            f"{field}={count}"
            for field, count in sorted(collisions.items())
        )
        raise ValueError(f"Sealed evaluation overlaps prior data: {detail}")


def validate_sealed_manifest(
    manifest_path: Path,
    exclusion_index: dict[str, Any],
) -> dict[str, Any]:
    annotations = load_manifest(manifest_path)
    if not annotations:
        raise ValueError("Sealed manifest is empty")
    reject_quarantined_references(annotations)
    source_audit = enforce_records(
        annotations,
        purpose="evaluation",
        require_source_id=True,
        require_portable_paths=True,
    )
    if any(annotation.split != "test" for annotation in annotations):
        raise ValueError("Every sealed track must use the test split")
    if len({annotation.track_id for annotation in annotations}) != len(
        annotations
    ):
        raise ValueError("Sealed track IDs must be unique")
    if any(
        annotation.duration < MINIMUM_TRACK_DURATION_SECONDS
        for annotation in annotations
    ):
        raise ValueError(
            "Every sealed work must contain at least 30 seconds of audio"
        )
    if any(not annotation.events for annotation in annotations):
        raise ValueError("Every sealed work requires reference events")

    identities = _identity_sets(annotations)
    audio_hashes = _validate_assets(manifest_path, annotations)
    _validate_isolation(annotations, exclusion_index)
    counts = {
        "tracks": len(annotations),
        "groups": len({annotation.group_id for annotation in annotations}),
        **{
            field.removesuffix("_id") + "s": len(values)
            for field, values in identities.items()
            if field != "audio_sha256"
        },
        "audio_assets": len(set(audio_hashes)),
    }
    minimums = {
        "tracks": MINIMUM_TRACKS,
        "groups": MINIMUM_SOURCE_GROUPS,
        "performers": MINIMUM_PERFORMERS,
        "works": MINIMUM_WORKS,
        "recordings": MINIMUM_RECORDINGS,
        "sessions": MINIMUM_SESSIONS,
        "lineages": MINIMUM_LINEAGES,
        "audio_assets": MINIMUM_TRACKS,
    }
    failures = {
        key: {"actual": counts[key], "minimum": minimum}
        for key, minimum in minimums.items()
        if counts[key] < minimum
    }
    if failures:
        detail = ", ".join(
            f"{key}={value['actual']}<{value['minimum']}"
            for key, value in failures.items()
        )
        raise ValueError(f"Sealed diversity requirements failed: {detail}")

    capture_modes: Counter[str] = Counter()
    guitars: set[str] = set()
    pickup_ids: set[str] = set()
    microphone_ids: set[str] = set()
    tempo_buckets: Counter[str] = Counter()
    techniques: Counter[str] = Counter()
    capo_values: Counter[int] = Counter()
    for annotation in annotations:
        provenance = annotation.provenance
        capture_mode = str(provenance.get("capture_mode", ""))
        if capture_mode not in {"microphone", "pickup", "mixed"}:
            raise ValueError(
                f"Sealed track {annotation.track_id} has invalid capture_mode"
            )
        capture_modes[capture_mode] += 1
        guitar_id = provenance.get("guitar_id")
        if (
            not isinstance(guitar_id, str)
            or not SAFE_ID_PATTERN.fullmatch(guitar_id)
        ):
            raise ValueError(
                f"Sealed track {annotation.track_id} requires guitar_id"
            )
        guitars.add(guitar_id)
        if capture_mode == "pickup":
            pickup_id = provenance.get("pickup_id")
            if not isinstance(pickup_id, str) or not pickup_id.strip():
                raise ValueError(
                    f"Pickup track {annotation.track_id} requires pickup_id"
                )
            pickup_ids.add(pickup_id.strip())
        if capture_mode == "microphone":
            microphone_id = provenance.get("microphone_id")
            if not isinstance(microphone_id, str) or not microphone_id.strip():
                raise ValueError(
                    f"Microphone track {annotation.track_id} requires "
                    "microphone_id"
                )
            microphone_ids.add(microphone_id.strip())
        if annotation.bpm is None:
            raise ValueError(
                f"Sealed track {annotation.track_id} requires BPM metadata"
            )
        tempo_buckets[_tempo_bucket(annotation.bpm)] += 1
        capo_values[annotation.capo] += 1
        for event in annotation.events:
            if event.technique not in {"", "unknown"}:
                techniques[event.technique] += 1

    missing_modes = REQUIRED_CAPTURE_MODES - set(capture_modes)
    if missing_modes:
        raise ValueError(
            "Sealed evaluation lacks capture modes: "
            + ", ".join(sorted(missing_modes))
        )
    if len(guitars) < MINIMUM_GUITARS:
        raise ValueError("Sealed evaluation requires at least three guitars")
    if len(pickup_ids) < 2:
        raise ValueError("Sealed evaluation requires at least two pickups")
    if not microphone_ids:
        raise ValueError("Sealed evaluation requires microphone recordings")
    if set(tempo_buckets) != {"slow", "medium", "fast"}:
        raise ValueError("Sealed evaluation requires slow, medium and fast works")
    if 0 not in capo_values or not any(capo > 0 for capo in capo_values):
        raise ValueError(
            "Sealed evaluation requires open and nonzero capo positions"
        )
    if len(techniques) < MINIMUM_KNOWN_TECHNIQUES:
        raise ValueError(
            "Sealed evaluation requires at least three known techniques"
        )

    return {
        **counts,
        "duration_seconds": sum(
            annotation.duration for annotation in annotations
        ),
        "events": sum(len(annotation.events) for annotation in annotations),
        "sources": source_audit["sources"],
        "capture_modes": dict(sorted(capture_modes.items())),
        "guitars": len(guitars),
        "pickups": len(pickup_ids),
        "microphones": len(microphone_ids),
        "tempo_buckets": dict(sorted(tempo_buckets.items())),
        "capo_positions": len(capo_values),
        "known_techniques": len(techniques),
    }


def seal_manifest(
    manifest_path: Path,
    exclusion_index_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    exclusion_index = load_exclusion_index(exclusion_index_path)
    summary = validate_sealed_manifest(manifest_path, exclusion_index)
    ledger = load_source_ledger()
    annotations = load_manifest(manifest_path)
    identity_rows = [
        {
            field: identity_digest(
                field,
                (
                    annotation.track_id
                    if field == "track_id"
                    else annotation.group_id
                    if field == "group_id"
                    else _declared_audio_sha256(annotation)
                    if field == "audio_sha256"
                    else _sealed_identity(annotation, field)
                ),
            )
            for field in IDENTITY_FIELDS
        }
        for annotation in annotations
    ]
    commitment = {
        "schema_version": 1,
        "kind": "sealed-evaluation-commitment",
        "manifest": {
            "name": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
        "exclusion_index": {
            "name": exclusion_index_path.name,
            "sha256": sha256_file(exclusion_index_path),
            "manifest_count": len(exclusion_index["manifests"]),
        },
        "source_ledger": {
            "version": ledger.ledger_version,
            "sha256": ledger.sha256,
        },
        "identity_commitment_sha256": hashlib.sha256(
            canonical_json(sorted(identity_rows, key=canonical_json))
        ).hexdigest(),
        "summary": summary,
        "disclosure": "aggregate-only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json(commitment) + b"\n")
    return commitment


def validate_intake_plan(plan_path: Path) -> dict[str, int]:
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("kind") != "sealed-evaluation-intake-plan"
    ):
        raise ValueError("Unsupported sealed intake plan")
    target = value.get("target_works")
    if (
        not isinstance(target, int)
        or isinstance(target, bool)
        or target < MINIMUM_WORKS
    ):
        raise ValueError("Sealed intake plan requires at least 100 works")
    batches = value.get("performer_batches")
    if not isinstance(batches, list) or len(batches) < MINIMUM_PERFORMERS:
        raise ValueError("Sealed intake plan requires at least five batches")
    batch_ids: set[str] = set()
    work_slots = 0
    microphone_slots = 0
    pickup_slots = 0
    nonzero_capo_minimum = 0
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError("Sealed intake batches must be objects")
        batch_id = batch.get("batch_id")
        if (
            not isinstance(batch_id, str)
            or not SAFE_ID_PATTERN.fullmatch(batch_id)
            or batch_id in batch_ids
        ):
            raise ValueError("Sealed intake batch IDs must be unique")
        batch_ids.add(batch_id)
        slots = batch.get("work_slots")
        capture = batch.get("capture_quota")
        if (
            not isinstance(slots, int)
            or isinstance(slots, bool)
            or slots <= 0
            or not isinstance(capture, dict)
        ):
            raise ValueError(f"Invalid intake quota for {batch_id}")
        microphone = capture.get("microphone")
        pickup = capture.get("pickup")
        if (
            not isinstance(microphone, int)
            or isinstance(microphone, bool)
            or not isinstance(pickup, int)
            or isinstance(pickup, bool)
            or microphone < 0
            or pickup < 0
            or microphone + pickup != slots
        ):
            raise ValueError(f"Capture quota does not fill {batch_id}")
        capo = batch.get("nonzero_capo_minimum")
        if (
            not isinstance(capo, int)
            or isinstance(capo, bool)
            or not 0 <= capo <= slots
        ):
            raise ValueError(f"Invalid capo quota for {batch_id}")
        work_slots += slots
        microphone_slots += microphone
        pickup_slots += pickup
        nonzero_capo_minimum += capo
    if work_slots != target:
        raise ValueError("Performer work slots do not equal target works")

    global_quota = value.get("global_quota")
    if not isinstance(global_quota, dict):
        raise ValueError("Sealed intake plan requires a global quota")
    tempo_buckets = global_quota.get("tempo_buckets")
    if (
        not isinstance(tempo_buckets, dict)
        or set(tempo_buckets)
        != {
            "slow_below_90_bpm",
            "medium_90_to_129_bpm",
            "fast_130_bpm_or_above",
        }
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            for count in tempo_buckets.values()
        )
        or sum(tempo_buckets.values()) != target
    ):
        raise ValueError("Tempo quotas must positively partition target works")
    open_minimum = global_quota.get("open_position_works_minimum")
    capo_minimum = global_quota.get("nonzero_capo_works_minimum")
    if (
        open_minimum != target - nonzero_capo_minimum
        or capo_minimum != nonzero_capo_minimum
    ):
        raise ValueError("Global capo quotas differ from performer batches")
    if microphone_slots <= 0 or pickup_slots <= 0:
        raise ValueError("Intake plan requires microphone and pickup works")
    if global_quota.get("guitars_minimum", 0) < MINIMUM_GUITARS:
        raise ValueError("Intake plan requires at least three guitars")
    if global_quota.get("pickup_identities_minimum", 0) < 2:
        raise ValueError("Intake plan requires at least two pickups")
    if (
        global_quota.get("known_techniques_minimum", 0)
        < MINIMUM_KNOWN_TECHNIQUES
    ):
        raise ValueError("Intake plan requires at least three techniques")
    return {
        "target_works": target,
        "performer_batches": len(batches),
        "microphone_slots": microphone_slots,
        "pickup_slots": pickup_slots,
        "nonzero_capo_minimum": nonzero_capo_minimum,
    }


def _read_manifest_list(path: Path) -> list[Path]:
    result: list[Path] = []
    root = path.resolve().parent
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = (root / line).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        result.append(candidate)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build prior-data exclusion indexes and aggregate-only sealed "
            "evaluation commitments."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("build-exclusion-index")
    index_parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
    )
    index_parser.add_argument("--manifest-list", type=Path)
    index_parser.add_argument("--output", type=Path, required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--manifest", type=Path, required=True)
    seal_parser.add_argument("--exclusion-index", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    plan_parser = subparsers.add_parser("validate-intake-plan")
    plan_parser.add_argument("--plan", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "build-exclusion-index":
        manifests = list(arguments.manifest)
        if arguments.manifest_list is not None:
            manifests.extend(_read_manifest_list(arguments.manifest_list))
        result = write_exclusion_index(manifests, arguments.output)
        rendered = {
            "output": arguments.output.name,
            "sha256": sha256_file(arguments.output),
            "manifests": len(result["manifests"]),
            "identity_counts": result["identity_counts"],
        }
    elif arguments.command == "seal":
        result = seal_manifest(
            arguments.manifest.resolve(),
            arguments.exclusion_index.resolve(),
            arguments.output.resolve(),
        )
        rendered = {
            "output": arguments.output.name,
            "sha256": sha256_file(arguments.output),
            "summary": result["summary"],
        }
    else:
        rendered = {
            "plan": arguments.plan.name,
            "sha256": sha256_file(arguments.plan),
            **validate_intake_plan(arguments.plan.resolve()),
        }
    print(json.dumps(rendered, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
