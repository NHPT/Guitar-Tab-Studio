from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .data_sources import (
    SourceLedger,
    load_source_ledger,
    source_provenance,
)
from .schema import (
    STANDARD_TUNING_LIST,
    LabeledEvent,
    TrackAnnotation,
    write_manifest,
)
from .technique_schema import TechniqueTrack, load_technique_manifest


SOURCE_LEDGER = load_source_ledger()
TRAINING_DATASET_POLICIES: dict[str, dict[str, str]] = {
    alias: {
        "source_id": policy.source_id,
        "license": policy.license_spdx,
        "source": policy.source_url,
    }
    for policy in SOURCE_LEDGER.policies.values()
    if "training" in policy.allowed_purposes
    and policy.verification_status in {"verified", "internal"}
    for alias in policy.aliases
}
KNOWN_DATASET_LICENSES = {
    alias: policy.license_spdx
    for policy in SOURCE_LEDGER.policies.values()
    for alias in policy.aliases
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counter_dict(values: Counter[str]) -> dict[str, int]:
    return {
        key: values[key]
        for key in sorted(values)
    }


def _track_rejection_reason(
    track: TechniqueTrack,
    *,
    source_split: str,
    purpose: str,
    ledger: SourceLedger = SOURCE_LEDGER,
) -> str | None:
    if track.split != source_split:
        return "non_selected_split"
    try:
        policy = ledger.resolve(track.provenance)
    except ValueError:
        return "source_not_registered"
    if track.provenance.get("license") != policy.license_spdx:
        return "license_mismatch"
    if policy.verification_status not in {"verified", "internal"}:
        return f"source_{policy.verification_status}"
    if purpose not in policy.allowed_purposes:
        return f"{purpose}_not_allowed"
    return None


def convert_technique_positions(
    source_path: Path,
    output_path: Path,
    audit_path: Path,
    *,
    source_split: str = "train",
    maximum_fret: int = 24,
) -> tuple[list[TrackAnnotation], dict[str, object]]:
    if source_split != "train":
        raise ValueError("Only the source training split may be converted")
    if maximum_fret < 0:
        raise ValueError("maximum_fret must be non-negative")

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    audit_path = audit_path.resolve()
    tracks = load_technique_manifest(source_path)
    ledger = load_source_ledger()

    source_tracks: Counter[str] = Counter()
    source_events: Counter[str] = Counter()
    included_tracks: Counter[str] = Counter()
    included_events: Counter[str] = Counter()
    rejected_tracks: Counter[str] = Counter()
    rejected_events: Counter[str] = Counter()
    invalid_frets: Counter[str] = Counter()
    declared_license_mismatches: Counter[str] = Counter()
    converted: list[TrackAnnotation] = []

    for track in tracks:
        dataset = str(track.provenance.get("dataset", "unknown"))
        source_tracks[dataset] += 1
        source_events[dataset] += len(track.events)
        known_license = KNOWN_DATASET_LICENSES.get(dataset)
        if (
            known_license is not None
            and track.provenance.get("license") != known_license
        ):
            declared_license_mismatches[dataset] += 1

        rejection_reason = _track_rejection_reason(
            track,
            source_split=source_split,
            purpose="training",
            ledger=ledger,
        )
        if rejection_reason is not None:
            key = f"{dataset}:{rejection_reason}"
            rejected_tracks[key] += 1
            rejected_events[key] += len(track.events)
            continue

        audio_path = track.resolve_audio(source_path)
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Missing audio for technique track {track.track_id!r}"
            )

        events: list[LabeledEvent] = []
        for event in track.events:
            fret = event.pitch - STANDARD_TUNING_LIST[event.string - 1]
            if not 0 <= fret <= maximum_fret:
                invalid_frets[dataset] += 1
                continue
            events.append(
                LabeledEvent(
                    onset=event.onset,
                    offset=event.offset,
                    string=event.string,
                    fret=fret,
                    technique=event.technique,
                    confidence=1.0,
                )
            )
        if not events:
            key = f"{dataset}:no_valid_position_events"
            rejected_tracks[key] += 1
            rejected_events[key] += len(track.events)
            continue

        policy = ledger.resolve(track.provenance)
        provenance = source_provenance(track.provenance, policy)
        provenance["position_conversion"] = {
            "method": "standard-tuning-pitch-minus-open-v1",
            "source_manifest": source_path.name,
            "source_split": source_split,
            "maximum_fret": maximum_fret,
            "filtered_invalid_fret_events": (
                len(track.events) - len(events)
            ),
        }
        converted.append(
            TrackAnnotation(
                track_id=track.track_id,
                group_id=track.group_id,
                audio=os.path.relpath(audio_path, output_path.parent),
                split="train",
                duration=track.duration,
                events=events,
                provenance=provenance,
            )
        )
        included_tracks[dataset] += 1
        included_events[dataset] += len(events)

    if not converted:
        raise ValueError("No approved technique position tracks were produced")

    write_manifest(output_path, converted)
    audit: dict[str, object] = {
        "schema_version": 1,
        "conversion": "technique-events-to-standard-tuning-positions-v1",
        "source_ledger": {
            "version": ledger.ledger_version,
            "sha256": ledger.sha256,
        },
        "source_manifest": {
            "name": source_path.name,
            "sha256": sha256_file(source_path),
            "selected_split": source_split,
        },
        "output_manifest": {
            "name": output_path.name,
            "sha256": sha256_file(output_path),
        },
        "position_constraints": {
            "tuning_midi_string_1_to_6": STANDARD_TUNING_LIST,
            "minimum_fret": 0,
            "maximum_fret": maximum_fret,
        },
        "approved_dataset_policies": TRAINING_DATASET_POLICIES,
        "known_dataset_licenses": KNOWN_DATASET_LICENSES,
        "counts": {
            "source_tracks_by_dataset": _counter_dict(source_tracks),
            "source_events_by_dataset": _counter_dict(source_events),
            "included_tracks_by_dataset": _counter_dict(included_tracks),
            "included_events_by_dataset": _counter_dict(included_events),
            "included_groups": len(
                {track.group_id for track in converted}
            ),
            "filtered_invalid_fret_events_by_dataset": _counter_dict(
                invalid_frets
            ),
            "rejected_tracks_by_dataset_and_reason": _counter_dict(
                rejected_tracks
            ),
            "rejected_events_by_dataset_and_reason": _counter_dict(
                rejected_events
            ),
            "declared_license_mismatch_tracks_by_dataset": _counter_dict(
                declared_license_mismatches
            ),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return converted, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert licensed technique annotations into standard-tuning "
            "string/fret supervision."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--source-split", choices=("train",), default="train")
    parser.add_argument("--maximum-fret", type=int, default=24)
    arguments = parser.parse_args()

    tracks, audit = convert_technique_positions(
        arguments.manifest,
        arguments.output,
        arguments.audit_output,
        source_split=arguments.source_split,
        maximum_fret=arguments.maximum_fret,
    )
    print(
        json.dumps(
            {
                "tracks": len(tracks),
                "events": sum(len(track.events) for track in tracks),
                "groups": audit["counts"]["included_groups"],
                "output": arguments.output.name,
                "audit": arguments.audit_output.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
