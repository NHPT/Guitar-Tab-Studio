from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile

from .data_sources import load_source_ledger, source_provenance
from .technique_schema import (
    TechniqueEvent,
    TechniqueTrack,
    write_technique_manifest,
)

GUITAR_TECHS_SOURCE_ID = "guitar-techs-zenodo-14963133"
GUITAR_TECHS_PLAYERS = ("P1", "P2", "P3")
P3_ARCHIVE_NAME = "P3_music.zip"
P3_ARCHIVE_SIZE = 129_505_089
P3_ARCHIVE_MD5 = "071ba80aecf00f4a31fbd167b3f22198"


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


@dataclass(frozen=True)
class GuitarTechsRecording:
    player: str
    content_type: str
    source_label: str
    technique: str
    midi_path: Path
    audio_path: Path
    merge_gestures: bool


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return file_digest(path, "sha256")


def verified_p3_archive(root: Path) -> dict[str, object]:
    archive_path = root / P3_ARCHIVE_NAME
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    size = archive_path.stat().st_size
    if size != P3_ARCHIVE_SIZE:
        raise ValueError(
            f"{P3_ARCHIVE_NAME} size mismatch: "
            f"expected {P3_ARCHIVE_SIZE}, got {size}"
        )
    md5 = file_digest(archive_path, "md5")
    if md5 != P3_ARCHIVE_MD5:
        raise ValueError(
            f"{P3_ARCHIVE_NAME} MD5 mismatch: "
            f"expected {P3_ARCHIVE_MD5}, got {md5}"
        )
    return {
        "name": P3_ARCHIVE_NAME,
        "size_bytes": size,
        "published_md5": md5,
        "local_sha256": sha256_file(archive_path),
    }


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
        shifted_onset = event.onset + offset
        shifted_offset = event.offset + offset
        if shifted_offset <= 0 or shifted_onset >= duration:
            continue
        onset = max(0.0, shifted_onset)
        shifted.append(
            TechniqueEvent(
                onset=round(onset, 6),
                offset=round(
                    min(duration, max(onset + 0.01, shifted_offset)),
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


def recording_events(
    recording: GuitarTechsRecording,
) -> list[TechniqueEvent]:
    events = midi_events(recording.midi_path, recording.technique)
    return (
        merge_gesture_fragments(events)
        if recording.merge_gestures
        else events
    )


def discover_recordings(
    root: Path,
    players: tuple[str, ...] = GUITAR_TECHS_PLAYERS,
) -> list[GuitarTechsRecording]:
    unknown_players = set(players) - set(GUITAR_TECHS_PLAYERS)
    if unknown_players:
        raise ValueError(
            "Unknown Guitar-TECHS players: "
            + ", ".join(sorted(unknown_players))
        )

    recordings: list[GuitarTechsRecording] = []
    for player in players:
        if player == "P3":
            directory = root / "P3_music"
            midi_directory = directory / "midi"
            for midi_path in sorted(midi_directory.glob("midi_*.mid")):
                source_label = midi_path.stem.removeprefix("midi_")
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
                        GuitarTechsRecording(
                            player=player,
                            content_type="music",
                            source_label=source_label,
                            technique="pick",
                            midi_path=midi_path,
                            audio_path=audio_path,
                            merge_gestures=False,
                        )
                    )
            continue

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
                        GuitarTechsRecording(
                            player=player,
                            content_type=(
                                "single-notes"
                                if directory.name.endswith("_singlenotes")
                                else "techniques"
                            ),
                            source_label=source_label,
                            technique=technique,
                            midi_path=midi_path,
                            audio_path=audio_path,
                            merge_gestures=True,
                        )
                    )
    return recordings


def import_guitar_techs(
    root: Path,
    output: Path,
    *,
    players: tuple[str, ...] = GUITAR_TECHS_PLAYERS,
) -> list[TechniqueTrack]:
    output = output.resolve()
    policy = load_source_ledger().policies[GUITAR_TECHS_SOURCE_ID]
    archive = verified_p3_archive(root) if "P3" in players else None
    tracks: list[TechniqueTrack] = []
    for recording in discover_recordings(root, players):
        source_events = recording_events(recording)
        duration = float(soundfile.info(recording.audio_path).duration)
        alignment_offset, alignment_score = estimate_alignment_offset(
            recording.audio_path,
            source_events,
        )
        events = shifted_events(
            source_events,
            alignment_offset,
            duration,
        )
        modality = recording.audio_path.parent.name
        is_music = recording.content_type == "music"
        identity = (
            f"music-{recording.source_label}"
            if is_music
            else recording.technique
        )
        event_normalization: dict[str, object] = (
            {
                "method": "preserve-per-string-midi-notes-v1",
                "polyphony": "preserved",
            }
            if is_music
            else {
                "method": "connected-midi-gesture-v1",
                "maximum_gap_seconds": 0.03,
            }
        )
        provenance: dict[str, object] = {
            "dataset": "Guitar-TECHS",
            "source_type": "public-research-dataset",
            "player": recording.player,
            "content_type": recording.content_type,
            "source_label": recording.source_label,
            "audio_modality": modality,
            "source_audio_sha256": sha256_file(recording.audio_path),
            "source_midi_sha256": sha256_file(recording.midi_path),
            "event_normalization": event_normalization,
            "alignment": {
                "method": "onset-strength-grid-search-v1",
                "offset_seconds": alignment_offset,
                "score": alignment_score,
                "search_range_seconds": 0.15,
                "step_seconds": 0.005,
            },
        }
        if is_music:
            provenance["source_archive"] = archive
        tracks.append(
            TechniqueTrack(
                track_id=(
                    f"guitar-techs-{recording.player}-{identity}-{modality}"
                ),
                group_id=(
                    f"guitar-techs-{recording.player}"
                    if is_music
                    else f"guitar-techs-{recording.player}-{identity}"
                ),
                audio=os.path.relpath(
                    recording.audio_path.resolve(),
                    output.parent,
                ),
                split=(
                    "validation"
                    if recording.player == "P2"
                    else "train"
                ),
                duration=round(duration, 6),
                events=events,
                provenance=source_provenance(provenance, policy),
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
    parser.add_argument(
        "--players",
        nargs="+",
        choices=GUITAR_TECHS_PLAYERS,
        default=GUITAR_TECHS_PLAYERS,
        help="Import selected performer partitions (default: P1 P2 P3).",
    )
    arguments = parser.parse_args()
    tracks = import_guitar_techs(
        arguments.root.resolve(),
        arguments.output,
        players=tuple(arguments.players),
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
