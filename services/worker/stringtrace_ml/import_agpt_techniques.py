from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from .technique_schema import (
    TechniqueEvent,
    TechniqueTrack,
    write_technique_manifest,
)


AGPT_DATASET_DOI = "10.5281/zenodo.10159492"
AGPT_ARCHIVE_MD5 = "1dff8103f9ad6e1a86cee2e5e39cbe87"
TECHNIQUE_BY_ID = {
    "4": "harmonic",
    "5": "palm-mute",
    "6": "pick",
    "7": "pick",
}
SPLIT_BY_PLAYER = {
    "0": "train",
    "1": "train",
    "2": "train",
    "3": "train",
    "4": "validation",
    "8": "test",
    "10": "test",
}
KNOWN_QUARANTINED_FILES = {
    "acoustic_guitar_pitched_allstring1_soundholepick_mf_MicRos_20230202.wav",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_pitch(value: str) -> int:
    parsed = float(value)
    if parsed > 127:
        parsed = 69 + 12 * math.log2(parsed / 440)
    return round(parsed)


def import_agpt_techniques(
    root: Path,
    output: Path,
) -> list[TechniqueTrack]:
    metadata_directory = root / "metadata"
    audio_directory = root / "data" / "audio"
    with (metadata_directory / "files.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        files = {
            row["filename"]: row
            for row in csv.DictReader(stream)
        }
    events_by_audio: dict[str, list[TechniqueEvent]] = defaultdict(list)
    with (metadata_directory / "note_labels.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        for row in csv.DictReader(stream):
            technique = TECHNIQUE_BY_ID.get(
                row["expressive_technique_id"]
            )
            if technique is None:
                continue
            if row["pitch_midi"] == "None":
                continue
            onset = float(row["onset_label_seconds"])
            filename = row["audio_file_path"]
            duration = float(files[filename]["duration"])
            events_by_audio[filename].append(
                TechniqueEvent(
                    onset=round(onset, 6),
                    offset=round(min(duration, onset + 0.4), 6),
                    pitch=normalized_pitch(row["pitch_midi"]),
                    string=int(row["string_number"]),
                    technique=technique,
                )
            )

    tracks = []
    for filename, events in sorted(events_by_audio.items()):
        if filename in KNOWN_QUARANTINED_FILES:
            continue
        metadata = files[filename]
        audio_path = audio_directory / filename
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        actual_sha256 = sha256_file(audio_path)
        if actual_sha256 != metadata["sha256"]:
            raise ValueError(
                f"AG-PT audio hash mismatch: {filename}"
            )
        player_id = metadata["player_id"]
        split = SPLIT_BY_PLAYER.get(player_id)
        if split is None:
            raise ValueError(f"Unassigned AG-PT player: {player_id}")
        tracks.append(
            TechniqueTrack(
                track_id=f"agpt-{Path(filename).stem}",
                group_id=f"agpt-player-{player_id}",
                audio=os.path.relpath(audio_path.resolve(), output.parent),
                split=split,
                duration=round(float(metadata["duration"]), 6),
                events=sorted(
                    events,
                    key=lambda event: (event.onset, event.string),
                ),
                provenance={
                    "dataset": "AG-PT-set",
                    "dataset_doi": AGPT_DATASET_DOI,
                    "source_type": "public-research-dataset",
                    "license": "CC-BY-4.0",
                    "archive_md5": AGPT_ARCHIVE_MD5,
                    "player_id": player_id,
                    "guitar_id": metadata["guitar_id"],
                    "playing_intensity": metadata[
                        "playing_dynamics_or_intensity"
                    ],
                    "source_audio_sha256": actual_sha256,
                    "label_mapping": "agpt-note-label-v1",
                    "event_duration_seconds": 0.4,
                },
            )
        )
    write_technique_manifest(output, tracks)
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the labeled AG-PT-set pitched techniques.",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    tracks = import_agpt_techniques(
        arguments.root.resolve(),
        arguments.output.resolve(),
    )
    label_counts = {
        label: sum(
            event.technique == label
            for track in tracks
            for event in track.events
        )
        for label in sorted(
            {
                event.technique
                for track in tracks
                for event in track.events
            }
        )
    }
    print(
        json.dumps(
            {
                "tracks": len(tracks),
                "events": sum(len(track.events) for track in tracks),
                "labels": label_counts,
                "splits": {
                    split: sum(track.split == split for track in tracks)
                    for split in ("train", "validation", "test")
                },
                "quarantined_files": sorted(KNOWN_QUARANTINED_FILES),
                "output": arguments.output.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
