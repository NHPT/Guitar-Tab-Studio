from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pretty_midi

try:
    from .acquire import ACQUISITION_KIND
    from .blind_set import sha256_file
    from .catalog import canonical_json
except ImportError:
    from acquire import ACQUISITION_KIND
    from blind_set import sha256_file
    from catalog import canonical_json


SCHEMA_VERSION = 1
LATTICE_KIND = "sealed-evaluation-playable-position-lattice"
LATTICE_COMMITMENT_KIND = (
    "sealed-evaluation-playable-position-lattice-commitment"
)
STANDARD_TUNING = (64, 59, 55, 50, 45, 40)
MAXIMUM_PHYSICAL_FRET = 24
MAXIMUM_ZIP_MEMBERS = 32
MAXIMUM_MIDI_BYTES = 20 * 1024 * 1024


def playable_positions(
    pitch: int,
    *,
    tuning: tuple[int, ...] = STANDARD_TUNING,
    capo: int = 0,
    maximum_fret: int = MAXIMUM_PHYSICAL_FRET,
) -> list[dict[str, int]]:
    positions = []
    for string, open_pitch in enumerate(tuning, start=1):
        physical_fret = pitch - open_pitch
        if capo <= physical_fret <= maximum_fret:
            positions.append(
                {
                    "string": string,
                    "fret": physical_fret,
                }
            )
    return positions


def _load_acquisition(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != ACQUISITION_KIND
    ):
        raise ValueError("Unsupported blind acquisition manifest")
    if not value.get("complete"):
        raise ValueError("Blind acquisition must be complete")
    return value


def _midi_documents(path: Path) -> list[tuple[str, pretty_midi.PrettyMIDI]]:
    suffix = path.suffix.casefold()
    if suffix in {".mid", ".midi"}:
        return [(path.name, pretty_midi.PrettyMIDI(str(path)))]
    if suffix != ".zip":
        raise ValueError(f"Unsupported score asset: {path.name}")
    documents = []
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            (
                info
                for info in archive.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.casefold()
                in {".mid", ".midi"}
            ),
            key=lambda info: info.filename,
        )
        if not members:
            raise ValueError(f"Score archive has no MIDI: {path.name}")
        if len(members) > MAXIMUM_ZIP_MEMBERS:
            raise ValueError(f"Score archive has too many MIDI files: {path.name}")
        for member in members:
            if member.file_size > MAXIMUM_MIDI_BYTES:
                raise ValueError(
                    f"MIDI member exceeds size limit: {path.name}"
                )
            payload = archive.read(member)
            documents.append(
                (
                    member.filename,
                    pretty_midi.PrettyMIDI(io.BytesIO(payload)),
                )
            )
    return documents


def _member_lattice(
    name: str,
    midi: pretty_midi.PrettyMIDI,
) -> dict[str, Any]:
    events = []
    unplayable = 0
    for instrument_index, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            positions = playable_positions(note.pitch)
            if not positions:
                unplayable += 1
            events.append(
                {
                    "onset": round(float(note.start), 6),
                    "offset": round(float(note.end), 6),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "instrument_index": instrument_index,
                    "positions": positions,
                }
            )
    events.sort(
        key=lambda value: (
            value["onset"],
            value["pitch"],
            value["offset"],
            value["instrument_index"],
        )
    )
    member_id = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return {
        "member_id": member_id,
        "duration_seconds": round(float(midi.get_end_time()), 6),
        "events": events,
        "event_count": len(events),
        "unplayable_event_count": unplayable,
    }


def build_lattice(
    acquisition_path: Path,
    asset_root: Path,
) -> dict[str, Any]:
    acquisition = _load_acquisition(acquisition_path)
    scores = []
    failures = []
    for record in acquisition["layer_b"]["records"]:
        path = asset_root / "layer-b" / record["path"]
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"Score SHA-256 mismatch: {path.name}")
            if record["asset_format"] != "midi":
                raise ValueError(
                    f"Score is not MIDI: {record['blind_id']}"
                )
            members = [
                _member_lattice(name, midi)
                for name, midi in _midi_documents(path)
            ]
            scores.append(
                {
                    "blind_id": record["blind_id"],
                    "source_asset_sha256": record["sha256"],
                    "tuning": list(STANDARD_TUNING),
                    "capo": 0,
                    "maximum_physical_fret": MAXIMUM_PHYSICAL_FRET,
                    "members": members,
                    "review_status": "machine-generated-pending-review",
                }
            )
        except Exception as error:
            failures.append(
                {
                    "blind_id": record["blind_id"],
                    "error_type": type(error).__name__,
                    "error": str(error)[:300],
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": LATTICE_KIND,
        "acquisition": {
            "name": acquisition_path.name,
            "sha256": sha256_file(acquisition_path),
        },
        "assumptions": {
            "tuning": list(STANDARD_TUNING),
            "capo": 0,
            "maximum_physical_fret": MAXIMUM_PHYSICAL_FRET,
            "position_semantics": "set-valued-playability-not-observed-fingering",
        },
        "scores": sorted(scores, key=lambda value: value["blind_id"]),
        "failures": sorted(failures, key=lambda value: value["blind_id"]),
        "complete": not failures and len(scores) == acquisition["layer_b"]["target"],
        "reviewed_scores": 0,
    }


def build_public_commitment(
    lattice: dict[str, Any],
    *,
    lattice_name: str,
    lattice_sha256: str,
) -> dict[str, Any]:
    scores = lattice["scores"]
    members = [
        member
        for score in scores
        for member in score["members"]
    ]
    source_hashes = sorted(
        score["source_asset_sha256"] for score in scores
    )
    commitment = {
        "schema_version": SCHEMA_VERSION,
        "kind": LATTICE_COMMITMENT_KIND,
        "acquisition": lattice["acquisition"],
        "lattice": {
            "name": lattice_name,
            "sha256": lattice_sha256,
        },
        "source_asset_commitment_sha256": hashlib.sha256(
            canonical_json(source_hashes)
        ).hexdigest(),
        "scores": len(scores),
        "members": len(members),
        "events": sum(member["event_count"] for member in members),
        "unplayable_events": sum(
            member["unplayable_event_count"] for member in members
        ),
        "failures": len(lattice["failures"]),
        "complete": lattice["complete"],
        "reviewed_scores": lattice["reviewed_scores"],
        "status": "machine-generated-pending-review",
        "disclosure": "aggregate-only",
    }
    rendered = canonical_json(commitment)
    if any(
        score["blind_id"].encode("utf-8") in rendered
        for score in scores
    ):
        raise ValueError("Lattice commitment discloses private identities")
    return commitment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert acquired blind MIDI scores into set-valued playable "
            "string/fret lattices."
        ),
    )
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    acquisition_path = arguments.acquisition.resolve()
    output_path = arguments.output.resolve()
    lattice = build_lattice(
        acquisition_path,
        arguments.asset_root.resolve(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json(lattice) + b"\n")
    commitment = build_public_commitment(
        lattice,
        lattice_name=output_path.name,
        lattice_sha256=sha256_file(output_path),
    )
    commitment_path = arguments.commitment.resolve()
    commitment_path.parent.mkdir(parents=True, exist_ok=True)
    commitment_path.write_bytes(canonical_json(commitment) + b"\n")
    print(
        json.dumps(
            {
                "output": output_path.name,
                "output_sha256": sha256_file(output_path),
                "commitment": commitment_path.name,
                "commitment_sha256": sha256_file(commitment_path),
                "scores": commitment["scores"],
                "members": commitment["members"],
                "events": commitment["events"],
                "failures": commitment["failures"],
                "complete": commitment["complete"],
                "reviewed_scores": 0,
            },
            sort_keys=True,
        )
    )
    if arguments.require_complete and not lattice["complete"]:
        raise RuntimeError("Playable-position lattice is incomplete")


if __name__ == "__main__":
    main()
