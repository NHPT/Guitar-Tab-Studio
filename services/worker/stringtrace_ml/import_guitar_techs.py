from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile

from .technique_schema import (
    TechniqueEvent,
    TechniqueTrack,
    write_technique_manifest,
)


TECHNIQUE_BY_STEM = {
    "allsinglenotes": "pick",
    "Bendings": "bend",
    "Harmonics": "harmonic",
    "PalmMute": "palm-mute",
    "PinchHarmonics": "pinch-harmonic",
    "Vibrato": "vibrato",
}
STRING_BY_INSTRUMENT = {
    "e": 1,
    "B": 2,
    "G": 3,
    "D": 4,
    "A": 5,
    "E": 6,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def midi_events(path: Path, technique: str) -> list[TechniqueEvent]:
    midi = pretty_midi.PrettyMIDI(str(path))
    events: list[TechniqueEvent] = []
    for instrument in midi.instruments:
        string = STRING_BY_INSTRUMENT.get(instrument.name)
        if string is None:
            raise ValueError(
                f"Unknown Guitar-TECHS MIDI instrument: {instrument.name!r}"
            )
        events.extend(
            TechniqueEvent(
                onset=float(note.start),
                offset=float(note.end),
                pitch=int(note.pitch),
                string=string,
                technique=technique,
            )
            for note in instrument.notes
        )
    return sorted(events, key=lambda event: (event.onset, event.string))


def estimate_alignment_offset(
    audio_path: Path,
    events: list[TechniqueEvent],
    *,
    maximum_offset: float = 0.15,
    step: float = 0.005,
) -> tuple[float, float]:
    audio, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
    hop_length = 256
    onset_strength = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )
    onset_times = np.asarray([event.onset for event in events])
    candidates = np.arange(
        -maximum_offset,
        maximum_offset + step / 2,
        step,
    )
    best_offset = 0.0
    best_score = float("-inf")
    for offset in candidates:
        frames = np.rint(
            (onset_times + offset) * sample_rate / hop_length
        ).astype(int)
        scores = []
        for frame in frames:
            start = max(0, frame - 1)
            end = min(len(onset_strength), frame + 2)
            if start < end:
                scores.append(float(onset_strength[start:end].max()))
        score = float(np.mean(scores)) if scores else float("-inf")
        if score > best_score:
            best_score = score
            best_offset = float(offset)
    return round(best_offset, 6), round(best_score, 6)


def shifted_events(
    events: list[TechniqueEvent],
    offset: float,
    duration: float,
) -> list[TechniqueEvent]:
    shifted = []
    for event in events:
        onset = event.onset + offset
        event_offset = event.offset + offset
        if onset < 0 or onset >= duration:
            continue
        shifted.append(
            TechniqueEvent(
                onset=round(onset, 6),
                offset=round(
                    min(duration, max(onset + 0.01, event_offset)),
                    6,
                ),
                pitch=event.pitch,
                string=event.string,
                technique=event.technique,
            )
        )
    return shifted


def merge_gesture_fragments(
    events: list[TechniqueEvent],
    *,
    maximum_gap: float = 0.03,
) -> list[TechniqueEvent]:
    if not events:
        return []
    groups: list[list[TechniqueEvent]] = []
    group_end = float("-inf")
    for event in sorted(events, key=lambda item: (item.onset, item.string)):
        if groups and event.onset > group_end + maximum_gap:
            groups.append([])
        elif not groups:
            groups.append([])
        groups[-1].append(event)
        group_end = max(group_end, event.offset)

    merged: list[TechniqueEvent] = []
    for group in groups:
        duration_by_string = {
            string: sum(
                event.offset - event.onset
                for event in group
                if event.string == string
            )
            for string in {event.string for event in group}
        }
        representative_string = max(
            duration_by_string,
            key=lambda string: (duration_by_string[string], -string),
        )
        string_events = [
            event
            for event in group
            if event.string == representative_string
        ]
        representative = min(
            string_events,
            key=lambda event: (event.onset, -(event.offset - event.onset)),
        )
        merged.append(
            TechniqueEvent(
                onset=min(event.onset for event in group),
                offset=max(event.offset for event in group),
                pitch=representative.pitch,
                string=representative_string,
                technique=representative.technique,
            )
        )
    return merged


def discover_recordings(
    root: Path,
) -> list[tuple[str, str, Path, Path]]:
    recordings: list[tuple[str, str, Path, Path]] = []
    for player in ("P1", "P2"):
        directories = (
            root / f"{player}_singlenotes",
            root / f"{player}_techniques",
        )
        for directory in directories:
            midi_directory = directory / "midi"
            for midi_path in sorted(midi_directory.glob("midi_*.mid")):
                source_label = midi_path.stem.removeprefix("midi_")
                technique = TECHNIQUE_BY_STEM.get(source_label)
                if technique is None:
                    continue
                for modality in ("directinput", "micamp"):
                    audio_path = (
                        directory
                        / "audio"
                        / modality
                        / f"{modality}_{source_label}.wav"
                    )
                    if not audio_path.is_file():
                        raise FileNotFoundError(audio_path)
                    recordings.append(
                        (player, technique, midi_path, audio_path)
                    )
    return recordings


def import_guitar_techs(root: Path, output: Path) -> list[TechniqueTrack]:
    output = output.resolve()
    tracks: list[TechniqueTrack] = []
    for player, technique, midi_path, audio_path in discover_recordings(root):
        source_events = merge_gesture_fragments(
            midi_events(midi_path, technique)
        )
        duration = float(soundfile.info(audio_path).duration)
        alignment_offset, alignment_score = estimate_alignment_offset(
            audio_path,
            source_events,
        )
        events = shifted_events(
            source_events,
            alignment_offset,
            duration,
        )
        modality = audio_path.parent.name
        tracks.append(
            TechniqueTrack(
                track_id=f"guitar-techs-{player}-{technique}-{modality}",
                group_id=f"guitar-techs-{player}-{technique}",
                audio=os.path.relpath(audio_path.resolve(), output.parent),
                split="train" if player == "P1" else "validation",
                duration=round(duration, 6),
                events=events,
                provenance={
                    "dataset": "Guitar-TECHS",
                    "source_type": "public-research-dataset",
                    "license": "CC-BY-4.0",
                    "player": player,
                    "audio_modality": modality,
                    "source_audio_sha256": sha256_file(audio_path),
                    "source_midi_sha256": sha256_file(midi_path),
                    "event_normalization": {
                        "method": "connected-midi-gesture-v1",
                        "maximum_gap_seconds": 0.03,
                    },
                    "alignment": {
                        "method": "onset-strength-grid-search-v1",
                        "offset_seconds": alignment_offset,
                        "score": alignment_score,
                        "search_range_seconds": 0.15,
                        "step_seconds": 0.005,
                    },
                },
            )
        )
    write_technique_manifest(output, tracks)
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Guitar-TECHS audio and MIDI technique labels.",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    tracks = import_guitar_techs(
        arguments.root.resolve(),
        arguments.output,
    )
    print(
        json.dumps(
            {
                "tracks": len(tracks),
                "events": sum(len(track.events) for track in tracks),
                "labels": sorted(
                    {
                        event.technique
                        for track in tracks
                        for event in track.events
                    }
                ),
                "splits": {
                    split: sum(track.split == split for track in tracks)
                    for split in ("train", "validation", "test")
                },
                "output": arguments.output.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
