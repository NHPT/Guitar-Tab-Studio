from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from .convert_technique_positions import (
    KNOWN_DATASET_LICENSES,
    TRAINING_DATASET_POLICIES,
    sha256_file,
)
from .schema import (
    STANDARD_TUNING_LIST,
    LabeledEvent,
    TrackAnnotation,
    write_manifest,
)
from .technique_schema import load_technique_manifest


def _counter_dict(values: Counter[str]) -> dict[str, int]:
    return {key: values[key] for key in sorted(values)}


def convert_technique_position_evaluation(
    source_path: Path,
    output_path: Path,
    audit_path: Path,
    *,
    source_split: str,
    maximum_fret: int = 24,
) -> tuple[list[TrackAnnotation], dict[str, object]]:
    if source_split not in {"validation", "test"}:
        raise ValueError(
            "Evaluation conversion only accepts validation or test splits"
        )
    if maximum_fret < 0:
        raise ValueError("maximum_fret must be non-negative")

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    tracks = load_technique_manifest(source_path)
    included_tracks: Counter[str] = Counter()
    included_events: Counter[str] = Counter()
    rejected_tracks: Counter[str] = Counter()
    filtered_events: Counter[str] = Counter()
    converted: list[TrackAnnotation] = []

    for track in tracks:
        if track.split != source_split:
            continue
        dataset = str(track.provenance.get("dataset", "unknown"))
        policy = TRAINING_DATASET_POLICIES.get(dataset)
        if policy is None:
            rejected_tracks[f"{dataset}:dataset_not_approved"] += 1
            continue
        if track.provenance.get("license") != policy["license"]:
            rejected_tracks[f"{dataset}:license_mismatch"] += 1
            continue
        audio_path = track.resolve_audio(source_path)
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Missing audio for evaluation track {track.track_id!r}"
            )
        events: list[LabeledEvent] = []
        for event in track.events:
            fret = event.pitch - STANDARD_TUNING_LIST[event.string - 1]
            if not 0 <= fret <= maximum_fret:
                filtered_events[dataset] += 1
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
            rejected_tracks[f"{dataset}:no_valid_position_events"] += 1
            continue
        provenance = dict(track.provenance)
        provenance["position_conversion"] = {
            "method": "standard-tuning-pitch-minus-open-v1",
            "source_manifest": source_path.name,
            "source_split": source_split,
            "maximum_fret": maximum_fret,
            "filtered_invalid_fret_events": (
                len(track.events) - len(events)
            ),
        }
        provenance["evaluation_only"] = True
        converted.append(
            TrackAnnotation(
                track_id=track.track_id,
                group_id=track.group_id,
                audio=os.path.relpath(audio_path, output_path.parent),
                split=source_split,
                duration=track.duration,
                events=events,
                provenance=provenance,
            )
        )
        included_tracks[dataset] += 1
        included_events[dataset] += len(events)

    if not converted:
        raise ValueError("No approved evaluation tracks were produced")
    write_manifest(output_path, converted)
    audit: dict[str, object] = {
        "schema_version": 1,
        "conversion": "evaluation-technique-to-position-v1",
        "evaluation_only": True,
        "training_prohibited": True,
        "source_manifest": {
            "name": source_path.name,
            "sha256": sha256_file(source_path),
            "selected_split": source_split,
        },
        "output_manifest": {
            "name": output_path.name,
            "sha256": sha256_file(output_path),
        },
        "approved_dataset_policies": TRAINING_DATASET_POLICIES,
        "known_dataset_licenses": KNOWN_DATASET_LICENSES,
        "counts": {
            "included_tracks_by_dataset": _counter_dict(included_tracks),
            "included_events_by_dataset": _counter_dict(included_events),
            "included_groups": len(
                {annotation.group_id for annotation in converted}
            ),
            "filtered_invalid_fret_events_by_dataset": _counter_dict(
                filtered_events
            ),
            "rejected_tracks_by_dataset_and_reason": _counter_dict(
                rejected_tracks
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
            "Convert an approved held-out technique split into an "
            "evaluation-only string/fret manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument(
        "--source-split",
        choices=("validation", "test"),
        required=True,
    )
    parser.add_argument("--maximum-fret", type=int, default=24)
    arguments = parser.parse_args()

    tracks, audit = convert_technique_position_evaluation(
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
                "split": arguments.source_split,
                "output": arguments.output.name,
            }
        )
    )


if __name__ == "__main__":
    main()
