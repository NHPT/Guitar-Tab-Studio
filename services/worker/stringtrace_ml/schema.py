from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


CURRENT_SCHEMA_VERSION = 1
STANDARD_TUNING_LIST = [64, 59, 55, 50, 45, 40]
VALID_SPLITS = {"train", "validation", "test"}
QUARANTINED_VERIFICATION_STATUS = "quarantined"


@dataclass(frozen=True)
class LabeledEvent:
    onset: float
    offset: float
    string: int
    fret: int
    technique: str = "unknown"
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LabeledEvent":
        return cls(
            onset=float(value["onset"]),
            offset=float(value["offset"]),
            string=int(value["string"]),
            fret=int(value["fret"]),
            technique=str(value.get("technique", "unknown")),
            confidence=float(value.get("confidence", 1.0)),
        )

    def validate(self, duration: float) -> None:
        if not 0 <= self.onset < self.offset <= duration + 0.05:
            raise ValueError(
                f"Invalid event interval {self.onset:.4f}-{self.offset:.4f} "
                f"for duration {duration:.4f}"
            )
        if not 1 <= self.string <= 6:
            raise ValueError(f"String must be in 1..6, got {self.string}")
        if not 0 <= self.fret <= 24:
            raise ValueError(f"Fret must be in 0..24, got {self.fret}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in 0..1, got {self.confidence}")


@dataclass(frozen=True)
class ExcludedRange:
    start: float
    end: float
    reason: str

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ExcludedRange":
        return cls(
            start=float(value["start"]),
            end=float(value["end"]),
            reason=str(value["reason"]).strip(),
        )

    def validate(self, duration: float) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or not 0 <= self.start < self.end <= duration
        ):
            raise ValueError(
                f"Invalid excluded range {self.start:.4f}-{self.end:.4f} "
                f"for duration {duration:.4f}"
            )
        if not 1 <= len(self.reason) <= 200:
            raise ValueError("Excluded range reason must contain 1..200 characters")


@dataclass
class TrackAnnotation:
    track_id: str
    group_id: str
    audio: str
    split: str
    duration: float
    events: list[LabeledEvent]
    excluded_ranges: list[ExcludedRange] = field(default_factory=list)
    tuning: list[int] = field(default_factory=lambda: list(STANDARD_TUNING_LIST))
    capo: int = 0
    bpm: float | None = None
    beats: list[float] = field(default_factory=list)
    time_signature: tuple[int, int] = (4, 4)
    provenance: dict[str, object] = field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrackAnnotation":
        time_signature_value = value.get("time_signature", [4, 4])
        if not isinstance(time_signature_value, (list, tuple)):
            raise ValueError("time_signature must be a two-element list")
        events_value = value.get("events", [])
        if not isinstance(events_value, list):
            raise ValueError("events must be a list")
        excluded_ranges_value = value.get("excluded_ranges", [])
        if not isinstance(excluded_ranges_value, list):
            raise ValueError("excluded_ranges must be a list")
        annotation = cls(
            track_id=str(value["track_id"]),
            group_id=str(value["group_id"]),
            audio=str(value["audio"]),
            split=str(value["split"]),
            duration=float(value["duration"]),
            events=[LabeledEvent.from_dict(event) for event in events_value],
            excluded_ranges=[
                ExcludedRange.from_dict(excluded_range)
                for excluded_range in excluded_ranges_value
            ],
            tuning=[int(pitch) for pitch in value.get("tuning", STANDARD_TUNING_LIST)],
            capo=int(value.get("capo", 0)),
            bpm=float(value["bpm"]) if value.get("bpm") is not None else None,
            beats=[float(beat) for beat in value.get("beats", [])],
            time_signature=(
                int(time_signature_value[0]),
                int(time_signature_value[1]),
            ),
            provenance=dict(value.get("provenance", {})),
            schema_version=int(value.get("schema_version", CURRENT_SCHEMA_VERSION)),
        )
        annotation.validate()
        return annotation

    def validate(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version {self.schema_version}; "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )
        if not self.track_id or not self.group_id:
            raise ValueError("track_id and group_id are required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {self.split}")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if len(self.tuning) != 6:
            raise ValueError("tuning must contain six MIDI pitches ordered string 1..6")
        if not 0 <= self.capo <= 12:
            raise ValueError("capo must be in 0..12")
        if len(self.time_signature) != 2 or min(self.time_signature) <= 0:
            raise ValueError("time_signature must contain positive numerator/denominator")
        for event in self.events:
            event.validate(self.duration)
        for previous, current in zip(self.events, self.events[1:]):
            if current.onset < previous.onset:
                raise ValueError("events must be ordered by onset")
        for excluded_range in self.excluded_ranges:
            excluded_range.validate(self.duration)
        for previous, current in zip(
            self.excluded_ranges,
            self.excluded_ranges[1:],
        ):
            if current.start < previous.start:
                raise ValueError("excluded_ranges must be ordered by start")
            if current.start < previous.end:
                raise ValueError("excluded_ranges must not overlap")

    def resolve_audio(self, manifest_path: Path) -> Path:
        path = Path(self.audio)
        return path if path.is_absolute() else (manifest_path.parent / path).resolve()

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["time_signature"] = list(self.time_signature)
        return result


def validate_group_splits(annotations: Iterable[TrackAnnotation]) -> None:
    split_by_group: dict[str, str] = {}
    for annotation in annotations:
        existing = split_by_group.setdefault(annotation.group_id, annotation.split)
        if existing != annotation.split:
            raise ValueError(
                f"Data leakage: group {annotation.group_id!r} appears in both "
                f"{existing!r} and {annotation.split!r}"
            )


def load_manifest(path: Path, split: str | None = None) -> list[TrackAnnotation]:
    annotations: list[TrackAnnotation] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            annotation = TrackAnnotation.from_dict(json.loads(line))
        except Exception as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        annotations.append(annotation)
    validate_group_splits(annotations)
    return [
        annotation
        for annotation in annotations
        if split is None or annotation.split == split
    ]


def reject_quarantined_references(
    annotations: Iterable[TrackAnnotation],
) -> None:
    quarantined = [
        annotation.track_id
        for annotation in annotations
        if annotation.provenance.get("verification_status")
        == QUARANTINED_VERIFICATION_STATUS
    ]
    if quarantined:
        identifiers = ", ".join(sorted(quarantined))
        raise ValueError(
            "Evaluation is blocked for quarantined references: "
            f"{identifiers}. Export an approved human review instead."
        )


def write_manifest(path: Path, annotations: Iterable[TrackAnnotation]) -> None:
    values = list(annotations)
    validate_group_splits(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(annotation.to_dict(), ensure_ascii=False) + "\n"
            for annotation in values
        ),
        encoding="utf-8",
    )
