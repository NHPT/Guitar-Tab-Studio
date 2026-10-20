#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "worker"))

from worker import (  # noqa: E402
    NoteEvent,
    align_beats_to_downbeats,
    assign_fingering_sequence,
    collapse_inaudible_rearticulations,
    estimate_tempo,
    extract_note_features,
    infer_chords,
    prepend_pickup_beats,
    tempo_map_from_beats,
    transcribe_basic_pitch,
)


def note_event(value: dict[str, object]) -> NoteEvent:
    return NoteEvent(
        start=float(value["start"]),
        end=float(value["end"]),
        pitch=int(value["pitch"]),
        velocity=int(value.get("velocity", 80)),
        string=int(value["string"]),
        fret=int(value["fret"]),
        technique=str(value.get("technique", "unknown")),
        confidence=float(value.get("confidence", 0.5)),
        position_confidence=float(
            value.get("position_confidence", value.get("confidence", 0.5))
        ),
        position_source=str(
            value.get("position_source", "legacy-playability")
        ),
    )


def note_groups(
    notes: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    for note in sorted(notes, key=lambda item: (item["start"], item["pitch"])):
        if (
            not groups
            or float(note["start"]) - float(groups[-1][0]["start"]) > 0.045
        ):
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def movement_metrics(notes: list[dict[str, object]]) -> dict[str, float | int]:
    groups = note_groups(notes)
    average_frets = [
        sum(float(note["fret"]) for note in group) / len(group)
        for group in groups
        if group
    ]
    movement = sum(
        abs(current - previous)
        for previous, current in zip(average_frets, average_frets[1:])
    )
    collisions = sum(
        len(group) - len({int(note["string"]) for note in group})
        for group in groups
    )
    return {
        "groups": len(groups),
        "total_fret_movement": round(movement, 4),
        "mean_fret_movement": round(
            movement / max(1, len(average_frets) - 1),
            6,
        ),
        "high_fret_notes": sum(int(note["fret"]) > 12 for note in notes),
        "same_group_string_collisions": collisions,
    }


def optimize_positions(
    notes: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups = note_groups(notes)
    assignments = assign_fingering_sequence(
        [
            [int(note["pitch"]) for note in group]
            for group in groups
        ]
    )
    for group, (positions, confidence) in zip(
        groups,
        assignments,
        strict=True,
    ):
        for note, (string, fret) in zip(group, positions, strict=True):
            note["string"] = string
            note["fret"] = fret
            note["position_confidence"] = confidence
            note["position_source"] = "sequence-optimizer-v3"
    return notes


def collapse_acoustic_fragments(
    notes: list[dict[str, object]],
    audio_path: Path,
) -> list[dict[str, object]]:
    if not notes or not audio_path.is_file():
        return notes
    events = [note_event(note) for note in notes]
    features = extract_note_features(audio_path, events)
    retained, _retained_features = collapse_inaudible_rearticulations(
        events,
        features,
    )
    source_by_event = {
        (round(float(note["start"]), 6), int(note["pitch"])): note
        for note in notes
    }
    collapsed: list[dict[str, object]] = []
    for event in retained:
        source = dict(
            source_by_event[(round(event.start, 6), event.pitch)]
        )
        source["end"] = event.end
        source["velocity"] = event.velocity
        source["confidence"] = event.confidence
        source["position_confidence"] = event.position_confidence
        collapsed.append(source)
    return collapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the production timing and fingering post-processors to a "
            "bundled analysis without treating its model output as gold data."
        )
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=ROOT / "apps" / "api" / "assets" / "internationale-analysis.json",
    )
    parser.add_argument(
        "--timing-source",
        type=Path,
        default=ROOT / "apps" / "api" / "assets" / "internationale-drums.m4a",
    )
    parser.add_argument(
        "--note-source",
        type=Path,
        default=ROOT / "apps" / "api" / "assets" / "internationale-guitar.m4a",
    )
    parser.add_argument(
        "--refresh-notes",
        action="store_true",
        help="Re-run Basic Pitch before applying the post-processors.",
    )
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()

    analysis = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    source_notes = list(analysis.get("notes", []))
    if arguments.refresh_notes:
        refreshed, _dropped = transcribe_basic_pitch(arguments.note_source)
        source_notes = [asdict(note) for note in refreshed]
        for note in source_notes:
            note["technique"] = "unknown"
            note["technique_confidence"] = 0.0
            note["technique_source"] = "unreviewed-heuristic"
            note["technique_evidence"] = ["heuristic-candidate-withheld"]
            note["related_note_index"] = None
            note["harmonic_type"] = None
            note["harmonic_touch_fret"] = None
    before = movement_metrics(source_notes)
    source_note_count = len(source_notes)
    primary_notes = optimize_positions(
        collapse_acoustic_fragments(
            source_notes,
            arguments.note_source,
        )
    )
    analysis["notes"] = primary_notes
    timing_source = "existing-analysis"
    downbeat_shift = 0
    downbeat_confidence = 0.0
    pickup_beats = int(analysis.get("pickup_beats", 0))
    beats = [float(beat) for beat in analysis.get("beats", [])]
    if arguments.timing_source.is_file():
        _bpm, beats, _onsets, _tempo_map = estimate_tempo(
            arguments.timing_source
        )
        timing_source = arguments.timing_source.name
        beats, downbeat_shift, downbeat_confidence = (
            align_beats_to_downbeats(
                beats,
                [note_event(note) for note in primary_notes],
            )
        )
        beats, pickup_beats = prepend_pickup_beats(
            beats,
            [note_event(note) for note in primary_notes],
        )
        analysis["beats"] = beats
        analysis["pickup_beats"] = pickup_beats
    tempo_map = tempo_map_from_beats(
        beats,
        float(analysis.get("bpm", 120)),
    )
    analysis["tempo_map"] = tempo_map
    analysis["bpm"] = round(float(tempo_map["median_bpm"]), 2)

    analysis["chords"] = infer_chords(
        [note_event(note) for note in primary_notes],
        float(analysis["bpm"]),
        beats,
        pickup_beats=pickup_beats,
    )
    guitar_parts = list(analysis.get("guitar_parts", []))
    for part in guitar_parts:
        part_notes = (
            [dict(note) for note in primary_notes]
            if arguments.refresh_notes and len(guitar_parts) == 1
            else optimize_positions(
                collapse_acoustic_fragments(
                    list(part.get("notes", [])),
                    arguments.note_source,
                )
            )
        )
        part["notes"] = part_notes
        part["chords"] = infer_chords(
            [note_event(note) for note in part_notes],
            float(analysis["bpm"]),
            beats,
            pickup_beats=pickup_beats,
        )
    model = dict(analysis.get("transcription_model") or {})
    model["timing_model"] = "percussion-downbeat-grid-v2"
    model["rearticulation_filter"] = "acoustic-onset-gate-v1"
    model["position_optimizer"] = "sequence-optimizer-v3"
    analysis["transcription_model"] = model
    warning = (
        "Performed timestamps are preserved; notation uses a percussion "
        "downbeat grid, pickup measure, local tempo map, and sequence-level "
        "fingering. Human review is still required."
    )
    warnings = [
        str(item)
        for item in analysis.get("warnings", [])
        if not str(item).startswith(
            "Performed timestamps are preserved; notation uses"
        )
    ]
    warnings.append(warning)
    analysis["warnings"] = warnings
    after = movement_metrics(primary_notes)
    report = {
        "analysis": arguments.analysis.name,
        "tempo": {
            "source": timing_source,
            "downbeat_shift": downbeat_shift,
            "downbeat_confidence": round(downbeat_confidence, 6),
            "pickup_beats": pickup_beats,
            "mode": tempo_map["mode"],
            "median_bpm": tempo_map["median_bpm"],
            "variation": tempo_map["variation"],
            "points": len(tempo_map["points"]),
        },
        "notes": len(primary_notes),
        "merged_fragments": source_note_count - len(primary_notes),
        "chords": len(analysis["chords"]),
        "before": before,
        "after": after,
        "written": arguments.write,
    }
    if arguments.write:
        arguments.analysis.write_text(
            json.dumps(
                analysis,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
