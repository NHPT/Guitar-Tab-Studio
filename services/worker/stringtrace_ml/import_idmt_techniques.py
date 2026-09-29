from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import soundfile

from .technique_schema import (
    TechniqueEvent,
    TechniqueTrack,
    write_technique_manifest,
)


IDMT_DATASET_DOI = "10.5281/zenodo.7544110"
IDMT_ARCHIVE_MD5 = "06796e08731bccffaed6ae59361486e4"
IDMT_INSTRUMENT_SPLITS = {
    "AR": "train",
    "FS": "validation",
    "LP": "test",
}
IDMT_LOCKED_TEST_SPLITS = {
    "AR": "train",
    "FS": "train",
    "LP": "test",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_technique(
    excitation_style: str,
    expression_style: str,
) -> str | None:
    expression_mapping = {
        "BE": "bend",
        "HA": "harmonic",
        "VI": "vibrato",
    }
    if expression_style in expression_mapping:
        return expression_mapping[expression_style]
    if expression_style != "NO":
        return None
    if excitation_style == "MU":
        return "palm-mute"
    if excitation_style in {"FS", "PK"}:
        return "pick"
    return None


def resolve_audio_path(
    annotation_path: Path,
    audio_directory: Path,
    declared_name: str,
) -> Path:
    candidates = (
        audio_directory / declared_name,
        audio_directory / f"{annotation_path.stem}.wav",
        audio_directory
        / f"{annotation_path.stem.replace('_fret_0-20', '_fret_1-20')}.wav",
        audio_directory
        / f"{annotation_path.stem.replace('_fret_1-20', '_fret_0-20')}.wav",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    prefix_matches = sorted(
        (
            candidate
            for candidate in audio_directory.glob("*.wav")
            if annotation_path.stem.startswith(candidate.stem)
        ),
        key=lambda candidate: len(candidate.stem),
        reverse=True,
    )
    if prefix_matches:
        return prefix_matches[0]
    raise FileNotFoundError(audio_directory / declared_name)


def parse_idmt_annotation(
    annotation_path: Path,
    audio_directory: Path,
    output_path: Path,
    *,
    split: str = "test",
) -> TechniqueTrack | None:
    root = ET.parse(annotation_path).getroot()
    global_parameters = root.find("./globalParameter")
    if global_parameters is None:
        raise ValueError(f"Missing globalParameter: {annotation_path.name}")
    audio_name = (
        global_parameters.findtext("audioFileName", "")
        .replace("\\", "")
        .replace("/", "")
    )
    audio_path = resolve_audio_path(
        annotation_path,
        audio_directory,
        audio_name,
    )
    duration = float(soundfile.info(audio_path).duration)
    events = []
    for value in root.findall("./transcription/event"):
        technique = mapped_technique(
            value.findtext("excitationStyle", ""),
            value.findtext("expressionStyle", ""),
        )
        if technique is None:
            continue
        onset = float(value.findtext("onsetSec", "0"))
        if not 0 <= onset < duration:
            continue
        offset = min(
            duration,
            max(onset + 0.01, float(value.findtext("offsetSec", "0"))),
        )
        source_string = int(value.findtext("stringNumber", "0"))
        events.append(
            TechniqueEvent(
                onset=round(onset, 6),
                offset=round(offset, 6),
                pitch=int(value.findtext("pitch", "0")),
                string=7 - source_string,
                technique=technique,
            )
        )
    if not events:
        return None
    instrument_code = annotation_path.stem.split("_", 1)[0]
    return TechniqueTrack(
        track_id=f"idmt-{annotation_path.stem}",
        group_id=f"idmt-instrument-{instrument_code}",
        audio=os.path.relpath(audio_path.resolve(), output_path.parent),
        split=split,
        duration=round(duration, 6),
        events=sorted(events, key=lambda event: (event.onset, event.string)),
        provenance={
            "dataset": "IDMT-SMT-GUITAR_V2",
            "dataset_doi": IDMT_DATASET_DOI,
            "source_type": "public-research-dataset",
            "license": "CC-BY-NC-ND-4.0",
            "archive_md5": IDMT_ARCHIVE_MD5,
            "instrument_code": instrument_code,
            "instrument_model": global_parameters.findtext(
                "instrumentModel",
                "unknown",
            ),
            "recording_artist": global_parameters.findtext(
                "recordingArtist",
                "unknown",
            ),
            "source_audio_sha256": sha256_file(audio_path),
            "source_annotation_sha256": sha256_file(annotation_path),
            "label_mapping": "idmt-expression-style-v1",
        },
    )


def import_idmt_techniques(
    root: Path,
    output: Path,
    *,
    split_by_instrument: bool = False,
    train_ar_fs: bool = False,
) -> list[TechniqueTrack]:
    annotation_directory = root / "annotation"
    audio_directory = root / "audio"
    tracks = [
        track
        for annotation_path in sorted(annotation_directory.glob("*.xml"))
        if (
            track := parse_idmt_annotation(
                annotation_path,
                audio_directory,
                output.resolve(),
                split=(
                    (
                        IDMT_LOCKED_TEST_SPLITS
                        if train_ar_fs
                        else IDMT_INSTRUMENT_SPLITS
                    )[annotation_path.stem.split("_", 1)[0]]
                    if split_by_instrument or train_ar_fs
                    else "test"
                ),
            )
        )
        is not None
    ]
    write_technique_manifest(output, tracks)
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import IDMT-SMT-Guitar expression labels for testing.",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-by-instrument",
        action="store_true",
        help="Use AR for train, FS for validation, and LP for locked test.",
    )
    parser.add_argument(
        "--train-ar-fs",
        action="store_true",
        help="Use AR and FS for train, and retain LP as the locked test.",
    )
    arguments = parser.parse_args()
    if arguments.split_by_instrument and arguments.train_ar_fs:
        parser.error(
            "--split-by-instrument and --train-ar-fs are mutually exclusive"
        )
    tracks = import_idmt_techniques(
        arguments.root.resolve(),
        arguments.output.resolve(),
        split_by_instrument=arguments.split_by_instrument,
        train_ar_fs=arguments.train_ar_fs,
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
                "output": arguments.output.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
