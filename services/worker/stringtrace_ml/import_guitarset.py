from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import (
    STANDARD_TUNING_LIST,
    LabeledEvent,
    TrackAnnotation,
    write_manifest,
)


def split_for_player(player_id: str) -> str:
    if player_id == "05":
        return "test"
    if player_id == "04":
        return "validation"
    return "train"


def midi_value(value: object) -> float:
    if isinstance(value, dict):
        for key in ("midi", "pitch", "note"):
            if key in value:
                return float(value[key])
    return float(value)


def find_audio(root: Path, track_id: str, audio_kind: str) -> Path:
    suffix = "mic" if audio_kind == "mic" else "mix"
    matches = list(root.rglob(f"{track_id}_{suffix}.wav"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {track_id}_{suffix}.wav below {root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple audio files found for {track_id}: {matches}"
        )
    return matches[0].resolve()


def parse_jams(path: Path, root: Path, audio_kind: str) -> TrackAnnotation:
    document = json.loads(path.read_text(encoding="utf-8"))
    track_id = path.stem
    note_annotations = [
        annotation
        for annotation in document.get("annotations", [])
        if annotation.get("namespace") in {"note_midi", "midi_note"}
    ]
    if len(note_annotations) != 6:
        raise ValueError(
            f"{path} contains {len(note_annotations)} MIDI-note annotations; expected 6"
        )
    annotations_by_source: dict[int, dict[str, object]] = {}
    for annotation in note_annotations:
        metadata = annotation.get("annotation_metadata", {})
        if not isinstance(metadata, dict) or "data_source" not in metadata:
            raise ValueError(f"{path} has a MIDI-note annotation without data_source")
        source_index = int(metadata["data_source"])
        if not 0 <= source_index < 6:
            raise ValueError(
                f"{path} has an invalid string data_source {source_index}"
            )
        if source_index in annotations_by_source:
            raise ValueError(
                f"{path} repeats MIDI-note data_source {source_index}"
            )
        annotations_by_source[source_index] = annotation
    if set(annotations_by_source) != set(range(6)):
        raise ValueError(f"{path} does not contain data_source values 0..5")
    duration = float(document.get("file_metadata", {}).get("duration", 0))
    if duration <= 0:
        duration = max(
            float(observation["time"]) + float(observation["duration"])
            for annotation in annotations_by_source.values()
            for observation in annotation.get("data", [])
        )
    events: list[LabeledEvent] = []
    dropped = 0
    for source_string_index, annotation in sorted(annotations_by_source.items()):
        string = 6 - source_string_index
        open_pitch = STANDARD_TUNING_LIST[string - 1]
        for observation in annotation.get("data", []):
            pitch = round(midi_value(observation["value"]))
            fret = pitch - open_pitch
            if not 0 <= fret <= 24:
                dropped += 1
                continue
            onset = float(observation["time"])
            offset = onset + float(observation["duration"])
            confidence_value = observation.get("confidence")
            events.append(
                LabeledEvent(
                    onset=round(onset, 6),
                    offset=round(offset, 6),
                    string=string,
                    fret=fret,
                    confidence=(
                        1.0
                        if confidence_value is None
                        else float(confidence_value)
                    ),
                )
            )
    events.sort(key=lambda event: (event.onset, event.string))

    beat_annotations = [
        annotation
        for annotation in document.get("annotations", [])
        if annotation.get("namespace") == "beat_position"
    ]
    beats = (
        [
            round(float(observation["time"]), 6)
            for observation in beat_annotations[0].get("data", [])
        ]
        if beat_annotations
        else []
    )
    tempo_annotations = [
        annotation
        for annotation in document.get("annotations", [])
        if annotation.get("namespace") == "tempo"
    ]
    bpm = None
    if tempo_annotations and tempo_annotations[0].get("data"):
        bpm = float(tempo_annotations[0]["data"][0]["value"])

    player_id = track_id.split("_", 1)[0]
    return TrackAnnotation(
        track_id=track_id,
        group_id=f"guitarset-player-{player_id}",
        audio=str(find_audio(root, track_id, audio_kind)),
        split=split_for_player(player_id),
        duration=duration,
        events=events,
        bpm=bpm,
        beats=beats,
        provenance={
            "dataset": "GuitarSet",
            "annotation": str(path.resolve()),
            "audio_kind": audio_kind,
            "player_id": player_id,
            "dropped_out_of_range_events": dropped,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GuitarSet JAMS annotations to StringTrace JSONL.",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-kind", choices=("mic", "mix"), default="mic")
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    jams_files = sorted(root.rglob("*.jams"))
    if not jams_files:
        raise FileNotFoundError(f"No .jams files found below {root}")
    annotations = [
        parse_jams(path, root, arguments.audio_kind)
        for path in jams_files
    ]
    write_manifest(arguments.output.resolve(), annotations)
    split_counts = {
        split: sum(annotation.split == split for annotation in annotations)
        for split in ("train", "validation", "test")
    }
    print(
        json.dumps(
            {
                "manifest": str(arguments.output.resolve()),
                "tracks": len(annotations),
                "events": sum(len(annotation.events) for annotation in annotations),
                "splits": split_counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
