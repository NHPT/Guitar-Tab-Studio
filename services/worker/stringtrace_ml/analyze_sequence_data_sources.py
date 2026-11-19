from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data_sources import load_source_ledger, sha256_file
from .schema import TrackAnnotation, load_manifest


def event_pitch(annotation: TrackAnnotation, event_index: int) -> int:
    event = annotation.events[event_index]
    return (
        annotation.tuning[event.string - 1]
        + annotation.capo
        + event.fret
    )


def source_sequence_summary(
    manifest_path: Path,
    *,
    maximum_transition_interval: float,
) -> dict[str, object]:
    annotations = load_manifest(manifest_path)
    split_counts = Counter(annotation.split for annotation in annotations)
    dataset_counts = Counter(
        str(annotation.provenance.get("dataset", "unknown"))
        for annotation in annotations
    )
    source_counts = Counter(
        str(annotation.provenance.get("source_id", "unknown"))
        for annotation in annotations
    )
    string_counts: Counter[int] = Counter()
    fret_counts: Counter[int] = Counter()
    transition_pairs: Counter[str] = Counter()
    transition_count = 0
    tracks_with_transition = 0

    for annotation in annotations:
        events_by_pitch: dict[int, list[int]] = defaultdict(list)
        for event_index, event in enumerate(annotation.events):
            string_counts[event.string] += 1
            fret_counts[event.fret] += 1
            events_by_pitch[event_pitch(annotation, event_index)].append(
                event_index
            )

        track_has_transition = False
        for pitch_events in events_by_pitch.values():
            ordered = sorted(
                pitch_events,
                key=lambda index: annotation.events[index].onset,
            )
            for previous_index, current_index in zip(
                ordered,
                ordered[1:],
            ):
                previous = annotation.events[previous_index]
                current = annotation.events[current_index]
                interval = current.onset - previous.onset
                if (
                    0 < interval <= maximum_transition_interval
                    and previous.string != current.string
                ):
                    transition_count += 1
                    track_has_transition = True
                    transition_pairs[
                        f"{previous.string}->{current.string}"
                    ] += 1
        tracks_with_transition += int(track_has_transition)

    event_count = sum(len(annotation.events) for annotation in annotations)
    return {
        "manifest": {
            "name": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
        "tracks": len(annotations),
        "splits": dict(sorted(split_counts.items())),
        "datasets": dict(sorted(dataset_counts.items())),
        "source_ids": dict(sorted(source_counts.items())),
        "events": event_count,
        "multi_event_tracks": sum(
            len(annotation.events) > 1 for annotation in annotations
        ),
        "string_events": {
            str(string): string_counts[string] for string in range(1, 7)
        },
        "fret_9_12_events": sum(
            count
            for fret, count in fret_counts.items()
            if 9 <= fret <= 12
        ),
        "same_pitch_cross_string_transitions": transition_count,
        "tracks_with_same_pitch_cross_string_transition": (
            tracks_with_transition
        ),
        "transition_pairs": dict(
            sorted(
                transition_pairs.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit governed real string/fret manifests for explicit "
            "same-pitch cross-string sequence supervision."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-transition-interval",
        type=float,
        default=2.0,
    )
    arguments = parser.parse_args()
    if arguments.maximum_transition_interval <= 0:
        parser.error("--maximum-transition-interval must be positive")

    ledger = load_source_ledger()
    summaries = [
        source_sequence_summary(
            manifest.resolve(),
            maximum_transition_interval=(
                arguments.maximum_transition_interval
            ),
        )
        for manifest in arguments.manifest
    ]
    source_ids = {
        source_id
        for summary in summaries
        for source_id in summary["source_ids"]
    }
    report = {
        "schema_version": 1,
        "kind": "candidate-sequence-data-source-audit",
        "maximum_transition_interval": (
            arguments.maximum_transition_interval
        ),
        "source_ledger": {
            "version": ledger.ledger_version,
            "sha256": ledger.sha256,
        },
        "source_policies": {
            source_id: {
                "name": ledger.policies[source_id].name,
                "version": ledger.policies[source_id].version,
                "license": ledger.policies[source_id].license_spdx,
                "verification_status": (
                    ledger.policies[source_id].verification_status
                ),
                "training_allowed": (
                    "training"
                    in ledger.policies[source_id].allowed_purposes
                ),
                "model_release_allowed": (
                    "model-release"
                    in ledger.policies[source_id].allowed_purposes
                ),
            }
            for source_id in sorted(source_ids)
        },
        "manifests": summaries,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "manifests": len(summaries),
                "transitions": {
                    summary["manifest"]["name"]: summary[
                        "same_pitch_cross_string_transitions"
                    ]
                    for summary in summaries
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
