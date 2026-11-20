from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

from .data_sources import audit_records, sha256_file
from .import_guitar_techs import (
    midi_events,
    sha256_file as asset_sha256_file,
    verified_p3_archive,
)
from .schema import STANDARD_TUNING_LIST, load_manifest
from .technique_schema import load_technique_manifest


EXPECTED_EXCERPTS = tuple(f"{index:02d}" for index in range(1, 13))
EXPECTED_MODALITIES = ("directinput", "micamp")


def counter_dict(counter: Counter[int | str]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter)
    }


def percentile(values: list[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = proportion * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def raw_midi_coverage(root: Path) -> dict[str, object]:
    string_counts: Counter[int] = Counter()
    fret_counts: Counter[int] = Counter()
    invalid_frets: list[dict[str, object]] = []
    intervals: list[float] = []
    excerpts: list[dict[str, object]] = []
    total_events = 0
    total_valid_events = 0
    total_polyphonic_onsets = 0
    total_polyphonic_events = 0
    maximum_polyphony = 0

    for excerpt in EXPECTED_EXCERPTS:
        path = root / "P3_music" / "midi" / f"midi_{excerpt}.mid"
        events = midi_events(path, "pick")
        onsets = Counter(round(event.onset, 6) for event in events)
        unique_onsets = sorted(onsets)
        excerpt_intervals = [
            current - previous
            for previous, current in zip(unique_onsets, unique_onsets[1:])
        ]
        intervals.extend(excerpt_intervals)
        polyphonic_sizes = [size for size in onsets.values() if size > 1]
        valid_events = 0
        for event in events:
            fret = event.pitch - STANDARD_TUNING_LIST[event.string - 1]
            string_counts[event.string] += 1
            fret_counts[fret] += 1
            if 0 <= fret <= 24:
                valid_events += 1
            else:
                invalid_frets.append(
                    {
                        "excerpt": excerpt,
                        "onset_seconds": round(event.onset, 6),
                        "string": event.string,
                        "pitch": event.pitch,
                        "derived_fret": fret,
                    }
                )
        total_events += len(events)
        total_valid_events += valid_events
        total_polyphonic_onsets += len(polyphonic_sizes)
        total_polyphonic_events += sum(polyphonic_sizes)
        maximum_polyphony = max(
            maximum_polyphony,
            max(onsets.values(), default=0),
        )
        excerpts.append(
            {
                "excerpt": excerpt,
                "events": len(events),
                "valid_position_events": valid_events,
                "active_strings": sorted(
                    {event.string for event in events}
                ),
                "unique_onsets": len(onsets),
                "polyphonic_onsets": len(polyphonic_sizes),
                "maximum_polyphony": max(onsets.values(), default=0),
                "onset_span_seconds": round(
                    max(event.offset for event in events)
                    - min(event.onset for event in events),
                    6,
                ),
                "midi_sha256": asset_sha256_file(path),
            }
        )

    return {
        "events": total_events,
        "valid_position_events": total_valid_events,
        "invalid_position_events": len(invalid_frets),
        "string_counts": counter_dict(string_counts),
        "fret_counts": counter_dict(fret_counts),
        "invalid_positions": invalid_frets,
        "polyphony": {
            "polyphonic_onsets": total_polyphonic_onsets,
            "events_at_polyphonic_onsets": total_polyphonic_events,
            "maximum_simultaneous_events": maximum_polyphony,
        },
        "inter_onset_seconds": {
            "count": len(intervals),
            "minimum": round(min(intervals), 6),
            "median": round(statistics.median(intervals), 6),
            "p90": round(percentile(intervals, 0.9), 6),
            "maximum": round(max(intervals), 6),
        },
        "excerpts": excerpts,
    }


def asset_hash_mismatches(
    manifest_path: Path,
    records: Iterable[object],
    midi_root: Path,
) -> dict[str, list[str]]:
    audio_mismatches: list[str] = []
    midi_mismatches: list[str] = []
    for record in records:
        track_id = str(record.track_id)
        provenance = record.provenance
        audio_path = record.resolve_audio(manifest_path)
        if (
            asset_sha256_file(audio_path)
            != provenance.get("source_audio_sha256")
        ):
            audio_mismatches.append(track_id)
        excerpt = str(provenance.get("source_label", ""))
        midi_path = midi_root / f"midi_{excerpt}.mid"
        if (
            asset_sha256_file(midi_path)
            != provenance.get("source_midi_sha256")
        ):
            midi_mismatches.append(track_id)
    return {
        "audio": audio_mismatches,
        "midi": midi_mismatches,
    }


def build_audit(
    root: Path,
    technique_manifest_path: Path,
    position_manifest_path: Path,
    formal_manifest_path: Path,
) -> dict[str, object]:
    root = root.resolve()
    technique_manifest_path = technique_manifest_path.resolve()
    position_manifest_path = position_manifest_path.resolve()
    formal_manifest_path = formal_manifest_path.resolve()
    technique_tracks = load_technique_manifest(technique_manifest_path)
    position_tracks = load_manifest(position_manifest_path)
    formal_tracks = load_manifest(formal_manifest_path)
    raw = raw_midi_coverage(root)
    archive = verified_p3_archive(root)

    modalities = Counter(
        str(track.provenance.get("audio_modality", ""))
        for track in technique_tracks
    )
    excerpts = Counter(
        str(track.provenance.get("source_label", ""))
        for track in technique_tracks
    )
    groups = {track.group_id for track in technique_tracks}
    splits = Counter(track.split for track in technique_tracks)
    alignments = [
        float(track.provenance["alignment"]["offset_seconds"])
        for track in technique_tracks
    ]
    technique_events = sum(len(track.events) for track in technique_tracks)
    position_events = [
        event
        for track in position_tracks
        for event in track.events
    ]
    technique_polyphonic_onsets = sum(
        sum(
            count > 1
            for count in Counter(
                round(event.onset, 6)
                for event in track.events
            ).values()
        )
        for track in technique_tracks
    )
    source_policy = {
        "technique": audit_records(
            technique_tracks,
            purpose="training",
        ),
        "position": audit_records(
            position_tracks,
            purpose="training",
        ),
    }
    hash_mismatches = asset_hash_mismatches(
        technique_manifest_path,
        technique_tracks,
        root / "P3_music" / "midi",
    )
    candidate_track_ids = {track.track_id for track in position_tracks}
    formal_track_ids = {track.track_id for track in formal_tracks}
    formal_overlap = sorted(candidate_track_ids & formal_track_ids)

    checks = {
        "expected_excerpts": (
            set(excerpts) == set(EXPECTED_EXCERPTS)
            and all(excerpts[item] == 2 for item in EXPECTED_EXCERPTS)
        ),
        "paired_modalities": (
            modalities
            == Counter({"directinput": 12, "micamp": 12})
        ),
        "single_performer_group": groups == {"guitar-techs-P3"},
        "training_candidate_only": splits == Counter({"train": 24}),
        "raw_polyphony_present": (
            int(raw["polyphony"]["polyphonic_onsets"]) > 0
        ),
        "polyphony_preserved_per_modality": (
            technique_polyphonic_onsets
            == 2 * int(raw["polyphony"]["polyphonic_onsets"])
        ),
        "technique_event_count": (
            technique_events == 2 * int(raw["events"])
        ),
        "position_event_count": (
            len(position_events)
            == 2 * int(raw["valid_position_events"])
        ),
        "position_constraints": all(
            1 <= event.string <= 6 and 0 <= event.fret <= 24
            for event in position_events
        ),
        "alignment_within_search_range": all(
            abs(offset) <= 0.15 for offset in alignments
        ),
        "asset_hashes": not any(hash_mismatches.values()),
        "technique_source_policy": bool(
            source_policy["technique"]["passed"]
        ),
        "position_source_policy": bool(
            source_policy["position"]["passed"]
        ),
        "formal_training_overlap_zero": not formal_overlap,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "audit": "guitar-techs-p3-music-position-candidate-v1",
        "source": {
            "record": "https://zenodo.org/records/14963133",
            "license": "CC-BY-4.0",
            "archive": archive,
        },
        "manifests": {
            "technique": {
                "name": technique_manifest_path.name,
                "sha256": sha256_file(technique_manifest_path),
                "tracks": len(technique_tracks),
                "events": technique_events,
            },
            "position_candidate": {
                "name": position_manifest_path.name,
                "sha256": sha256_file(position_manifest_path),
                "tracks": len(position_tracks),
                "events": len(position_events),
            },
            "formal_training_reference": {
                "name": formal_manifest_path.name,
                "sha256": sha256_file(formal_manifest_path),
                "overlapping_track_ids": formal_overlap,
            },
        },
        "partition": {
            "player": "P3",
            "groups": sorted(groups),
            "splits": counter_dict(splits),
            "modalities": counter_dict(modalities),
            "excerpt_occurrences": counter_dict(excerpts),
            "limitation": (
                "All excerpts share one performer; track-level holdout "
                "would not be performer-independent."
            ),
        },
        "alignment": {
            "minimum_offset_seconds": min(alignments),
            "maximum_offset_seconds": max(alignments),
            "median_offset_seconds": statistics.median(alignments),
            "absolute_maximum_seconds": max(map(abs, alignments)),
        },
        "raw_midi_coverage": raw,
        "position_coverage": {
            "string_counts": counter_dict(
                Counter(event.string for event in position_events)
            ),
            "fret_counts": counter_dict(
                Counter(event.fret for event in position_events)
            ),
        },
        "asset_hash_mismatches": hash_mismatches,
        "source_policy": source_policy,
        "checks": checks,
        "passed": passed,
        "decision": {
            "eligible_as_training_candidate": passed,
            "authorized_for_formal_training": False,
            "next_gate": (
                "Freeze a preregistered experiment before composing this "
                "candidate into a new formal training manifest."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Guitar-TECHS P3 music as position supervision.",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--technique-manifest", type=Path, required=True)
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_audit(
        arguments.root,
        arguments.technique_manifest,
        arguments.position_manifest,
        arguments.formal_manifest,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "raw_events": report["raw_midi_coverage"]["events"],
                "position_events": report["manifests"][
                    "position_candidate"
                ]["events"],
                "output": arguments.output.name,
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
