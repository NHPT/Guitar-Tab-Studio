from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


TECHNIQUE_LABELS = (
    "pick",
    "bend",
    "harmonic",
    "palm-mute",
    "pinch-harmonic",
    "vibrato",
)
VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class TechniqueEvent:
    onset: float
    offset: float
    pitch: int
    string: int
    technique: str

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TechniqueEvent":
        return cls(
            onset=float(value["onset"]),
            offset=float(value["offset"]),
            pitch=int(value["pitch"]),
            string=int(value["string"]),
            technique=str(value["technique"]),
        )

    def validate(self, duration: float) -> None:
        if not 0 <= self.onset < self.offset <= duration + 0.1:
            raise ValueError("Technique event is outside its audio track")
        if not 0 <= self.pitch <= 127:
            raise ValueError("Technique event pitch must be a MIDI value")
        if not 1 <= self.string <= 6:
            raise ValueError("Technique event string must be in 1..6")
        if self.technique not in TECHNIQUE_LABELS:
            raise ValueError(f"Unknown technique label: {self.technique}")


@dataclass
class TechniqueTrack:
    track_id: str
    group_id: str
    audio: str
    split: str
    duration: float
    events: list[TechniqueEvent]
    provenance: dict[str, object] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TechniqueTrack":
        events = value.get("events", [])
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        track = cls(
            track_id=str(value["track_id"]),
            group_id=str(value["group_id"]),
            audio=str(value["audio"]),
            split=str(value["split"]),
            duration=float(value["duration"]),
            events=[TechniqueEvent.from_dict(event) for event in events],
            provenance=dict(value.get("provenance", {})),
            schema_version=int(value.get("schema_version", 1)),
        )
        track.validate()
        return track

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported technique manifest schema")
        if not self.track_id or not self.group_id:
            raise ValueError("track_id and group_id are required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {self.split}")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        for event in self.events:
            event.validate(self.duration)
        for previous, current in zip(self.events, self.events[1:]):
            if current.onset < previous.onset:
                raise ValueError("events must be ordered by onset")

    def resolve_audio(self, manifest_path: Path) -> Path:
        path = Path(self.audio)
        return (
            path
            if path.is_absolute()
            else (manifest_path.parent / path).resolve()
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_technique_manifest(
    path: Path,
    split: str | None = None,
) -> list[TechniqueTrack]:
    tracks = [
        TechniqueTrack.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_by_group: dict[str, str] = {}
    for track in tracks:
        existing = split_by_group.setdefault(track.group_id, track.split)
        if existing != track.split:
            raise ValueError(
                f"Data leakage: group {track.group_id!r} appears in "
                f"{existing!r} and {track.split!r}"
            )
    return [
        track
        for track in tracks
        if split is None or track.split == split
    ]


def write_technique_manifest(
    path: Path,
    tracks: Iterable[TechniqueTrack],
) -> None:
    values = list(tracks)
    for track in values:
        track.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(track.to_dict(), ensure_ascii=False) + "\n"
            for track in values
        ),
        encoding="utf-8",
    )


def combine_technique_manifests(
    paths: Iterable[Path],
    output: Path,
) -> list[TechniqueTrack]:
    output = output.resolve()
    combined: list[TechniqueTrack] = []
    track_ids: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        for track in load_technique_manifest(resolved):
            if track.track_id in track_ids:
                raise ValueError(f"Duplicate technique track: {track.track_id}")
            track_ids.add(track.track_id)
            combined.append(
                TechniqueTrack(
                    track_id=track.track_id,
                    group_id=track.group_id,
                    audio=os.path.relpath(
                        track.resolve_audio(resolved),
                        output.parent,
                    ),
                    split=track.split,
                    duration=track.duration,
                    events=track.events,
                    provenance=track.provenance,
                    schema_version=track.schema_version,
                )
            )
    write_technique_manifest(output, combined)
    load_technique_manifest(output)
    return combined
