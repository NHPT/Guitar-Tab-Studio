from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile

from .data_sources import (
    load_source_ledger,
    manifest_identity,
    sha256_file,
    source_provenance,
)
from .hybrid import candidate_positions
from .schema import (
    STANDARD_TUNING_LIST,
    VALID_SPLITS,
    LabeledEvent,
    TrackAnnotation,
    validate_group_splits,
    write_manifest,
)


SYNTHETIC_SOURCE_ID = "stringtrace-synthetic-v1"
RIGHTS_BASES = {
    "original",
    "public-domain",
    "licensed-derivative",
}
GENERATOR_VERSION = "symbolic-decaying-harmonics-v2"


@dataclass(frozen=True)
class SymbolicTrackSpec:
    source_sha256: str
    content_sha256: str
    source_format: str
    split: str
    bpm: float
    tuning: tuple[int, ...]
    capo: int
    time_signature: tuple[int, int]
    events: tuple[LabeledEvent, ...]
    velocities: tuple[int, ...]
    rights_basis: str


def midi_to_hz(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rights_basis(value: object) -> str:
    result = str(value or "").strip()
    if result not in RIGHTS_BASES:
        allowed = ", ".join(sorted(RIGHTS_BASES))
        raise ValueError(
            f"Symbolic input requires rights_basis in: {allowed}"
        )
    return result


def plucked_note(
    frequency: float,
    duration: float,
    sample_rate: int,
    *,
    string: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_count = max(1, round(duration * sample_rate))
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    thickness = (string - 1) / 5
    brightness = 0.72 - thickness * 0.24
    inharmonicity = 0.00004 + thickness * 0.00012
    signal = np.zeros(sample_count, dtype=np.float32)
    for harmonic in range(1, 9):
        stretched_frequency = frequency * harmonic * (
            1 + inharmonicity * harmonic * harmonic
        )
        phase = rng.uniform(0, 2 * math.pi)
        harmonic_decay = np.exp(
            -time * (2.4 + harmonic * (0.55 + thickness * 0.25))
        )
        signal += (
            brightness ** (harmonic - 1)
            / harmonic
            * np.sin(2 * math.pi * stretched_frequency * time + phase)
            * harmonic_decay
        )
    attack_samples = min(sample_count, round(0.018 * sample_rate))
    attack = np.zeros(sample_count, dtype=np.float32)
    attack[:attack_samples] = (
        rng.normal(0, 1, attack_samples)
        * np.linspace(1, 0, attack_samples, dtype=np.float32)
        * (0.12 + thickness * 0.06)
    )
    envelope = np.minimum(1, time / 0.004) * np.exp(
        -time * (1.8 + thickness)
    )
    result = signal * envelope + attack
    peak = float(np.max(np.abs(result)))
    return result / max(1.0, peak)


def _validate_symbolic_events(
    events: tuple[LabeledEvent, ...],
    *,
    capo: int,
) -> None:
    if not events:
        raise ValueError("Symbolic input must contain at least one event")
    previous_by_string: dict[int, LabeledEvent] = {}
    for event in events:
        if capo and event.fret < capo:
            raise ValueError(
                "Absolute physical frets cannot be below the capo position"
            )
        previous = previous_by_string.get(event.string)
        if previous is not None and event.onset < previous.offset - 1e-6:
            raise ValueError(
                "Symbolic events on the same string must not overlap"
            )
        previous_by_string[event.string] = event


def load_json_symbolic(
    path: Path,
    *,
    default_split: str,
    rights_basis: str | None,
) -> SymbolicTrackSpec:
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported symbolic JSON schema")
    bpm = float(document.get("bpm", 120))
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("Symbolic BPM must be positive and finite")
    split = str(document.get("split", default_split))
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown symbolic split: {split}")
    tuning = tuple(
        int(value)
        for value in document.get("tuning", STANDARD_TUNING_LIST)
    )
    if len(tuning) != 6:
        raise ValueError("Symbolic tuning must contain six MIDI pitches")
    capo = int(document.get("capo", 0))
    if not 0 <= capo <= 12:
        raise ValueError("Symbolic capo must be in 0..12")
    raw_time_signature = document.get("time_signature", [4, 4])
    if (
        not isinstance(raw_time_signature, list)
        or len(raw_time_signature) != 2
    ):
        raise ValueError("Symbolic time_signature must have two values")
    time_signature = (
        int(raw_time_signature[0]),
        int(raw_time_signature[1]),
    )
    if min(time_signature) <= 0:
        raise ValueError("Symbolic time signature values must be positive")

    raw_events = document.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("Symbolic JSON events must be a list")
    seconds_per_beat = 60 / bpm
    values: list[tuple[LabeledEvent, int]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            raise ValueError("Symbolic events must be objects")
        onset = float(raw_event["onset_beats"]) * seconds_per_beat
        duration = float(raw_event["duration_beats"]) * seconds_per_beat
        if not math.isfinite(onset) or not math.isfinite(duration):
            raise ValueError("Symbolic event timing must be finite")
        if onset < 0 or duration <= 0:
            raise ValueError("Symbolic event timing must be positive")
        event = LabeledEvent(
            onset=round(onset, 6),
            offset=round(onset + duration, 6),
            string=int(raw_event["string"]),
            fret=int(raw_event["fret"]),
            technique=str(raw_event.get("technique", "pick")),
            confidence=1.0,
        )
        event.validate(event.offset)
        velocity = int(raw_event.get("velocity", 96))
        if not 1 <= velocity <= 127:
            raise ValueError("Symbolic velocity must be in 1..127")
        values.append((event, velocity))
    values.sort(key=lambda item: (item[0].onset, item[0].string))
    events = tuple(event for event, _velocity in values)
    _validate_symbolic_events(events, capo=capo)
    return SymbolicTrackSpec(
        source_sha256=sha256_file(path),
        content_sha256=_json_sha256(
            {
                "bpm": bpm,
                "tuning": tuning,
                "capo": capo,
                "time_signature": time_signature,
                "events": [
                    {
                        "onset": event.onset,
                        "offset": event.offset,
                        "string": event.string,
                        "fret": event.fret,
                        "technique": event.technique,
                        "velocity": velocity,
                    }
                    for event, velocity in values
                ],
            }
        ),
        source_format="stringtrace-symbolic-json-v1",
        split=split,
        bpm=bpm,
        tuning=tuning,
        capo=capo,
        time_signature=time_signature,
        events=events,
        velocities=tuple(velocity for _event, velocity in values),
        rights_basis=_rights_basis(
            rights_basis or document.get("rights_basis")
        ),
    )


def _position_cost(
    pitches: list[int],
    positions: tuple[tuple[int, int], ...],
    previous_positions: tuple[tuple[int, int], ...],
) -> float:
    frets = [fret for _string, fret in positions]
    cost = max(frets) - min(frets) if frets else 0.0
    cost += sum(frets) * 0.03
    ordered = sorted(zip(pitches, positions), key=lambda item: item[0])
    cost += sum(
        right[0] >= left[0]
        for (_left_pitch, left), (_right_pitch, right)
        in zip(ordered, ordered[1:])
    )
    if previous_positions:
        previous_mean = sum(
            fret for _string, fret in previous_positions
        ) / len(previous_positions)
        current_mean = sum(frets) / len(frets)
        cost += abs(current_mean - previous_mean) * 0.25
    return cost


def assign_symbolic_pitch_groups(
    groups: list[list[int]],
) -> list[tuple[tuple[int, int], ...]]:
    assignments: list[tuple[tuple[int, int], ...]] = []
    previous: tuple[tuple[int, int], ...] = ()
    for pitches in groups:
        if len(pitches) > 6:
            raise ValueError("A guitar chord cannot contain more than six notes")
        candidate_sets = [candidate_positions(pitch) for pitch in pitches]
        if any(not candidates for candidates in candidate_sets):
            raise ValueError("MIDI contains a pitch outside the guitar range")
        options = [
            positions
            for positions in itertools.product(*candidate_sets)
            if len({string for string, _fret in positions}) == len(positions)
        ]
        if not options:
            raise ValueError(
                "MIDI chord cannot be assigned to distinct guitar strings"
            )
        selected = min(
            options,
            key=lambda positions: (
                _position_cost(pitches, positions, previous),
                positions,
            ),
        )
        assignments.append(selected)
        previous = selected
    return assignments


def _clip_midi_string_overlaps(
    events: list[tuple[LabeledEvent, int]],
) -> list[tuple[LabeledEvent, int]]:
    indexes_by_string: dict[int, list[int]] = {}
    for index, (event, _velocity) in enumerate(events):
        indexes_by_string.setdefault(event.string, []).append(index)
    result = list(events)
    for indexes in indexes_by_string.values():
        for previous_index, current_index in zip(indexes, indexes[1:]):
            previous, velocity = result[previous_index]
            current, _current_velocity = result[current_index]
            if previous.offset > current.onset:
                result[previous_index] = (
                    replace(
                        previous,
                        offset=max(
                            previous.onset + 0.01,
                            current.onset,
                        ),
                    ),
                    velocity,
                )
    return result


def load_midi_symbolic(
    path: Path,
    *,
    default_split: str,
    rights_basis: str | None,
    onset_tolerance: float = 0.01,
) -> SymbolicTrackSpec:
    resolved_rights = _rights_basis(rights_basis)
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(
        (
            note
            for instrument in midi.instruments
            if not instrument.is_drum
            for note in instrument.notes
        ),
        key=lambda note: (note.start, note.pitch),
    )
    if not notes:
        raise ValueError("MIDI input contains no pitched notes")
    groups: list[list[pretty_midi.Note]] = []
    for note in notes:
        if not groups or note.start - groups[-1][0].start > onset_tolerance:
            groups.append([note])
        else:
            groups[-1].append(note)
    positions = assign_symbolic_pitch_groups(
        [[int(note.pitch) for note in group] for group in groups]
    )
    values = [
        (
            LabeledEvent(
                onset=round(float(note.start), 6),
                offset=round(
                    max(float(note.start) + 0.01, float(note.end)),
                    6,
                ),
                string=string,
                fret=fret,
                technique="pick",
                confidence=1.0,
            ),
            int(note.velocity),
        )
        for group, group_positions in zip(groups, positions, strict=True)
        for note, (string, fret) in zip(
            group,
            group_positions,
            strict=True,
        )
    ]
    values.sort(key=lambda item: (item[0].onset, item[0].string))
    values = _clip_midi_string_overlaps(values)
    events = tuple(event for event, _velocity in values)
    _validate_symbolic_events(events, capo=0)
    tempo_times, tempos = midi.get_tempo_changes()
    bpm = (
        float(tempos[0])
        if len(tempo_times) and len(tempos)
        else float(midi.estimate_tempo())
    )
    if not math.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    split = default_split
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown symbolic split: {split}")
    return SymbolicTrackSpec(
        source_sha256=sha256_file(path),
        content_sha256=_json_sha256(
            {
                "bpm": bpm,
                "tuning": STANDARD_TUNING_LIST,
                "capo": 0,
                "time_signature": [4, 4],
                "events": [
                    {
                        "onset": event.onset,
                        "offset": event.offset,
                        "string": event.string,
                        "fret": event.fret,
                        "velocity": velocity,
                    }
                    for event, velocity in values
                ],
            }
        ),
        source_format="midi",
        split=split,
        bpm=bpm,
        tuning=tuple(STANDARD_TUNING_LIST),
        capo=0,
        time_signature=(4, 4),
        events=events,
        velocities=tuple(velocity for _event, velocity in values),
        rights_basis=resolved_rights,
    )


def load_symbolic_spec(
    path: Path,
    *,
    default_split: str,
    rights_basis: str | None,
) -> SymbolicTrackSpec:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_symbolic(
            path,
            default_split=default_split,
            rights_basis=rights_basis,
        )
    if suffix in {".mid", ".midi"}:
        return load_midi_symbolic(
            path,
            default_split=default_split,
            rights_basis=rights_basis,
        )
    raise ValueError(f"Unsupported symbolic input format: {suffix}")


def _event_digest(events: tuple[LabeledEvent, ...]) -> str:
    return _json_sha256(
        [
            {
                "onset": event.onset,
                "offset": event.offset,
                "string": event.string,
                "fret": event.fret,
                "technique": event.technique,
            }
            for event in events
        ]
    )


def render_symbolic_track(
    spec: SymbolicTrackSpec,
    output: Path,
    *,
    seed: int,
    sample_rate: int,
) -> TrackAnnotation:
    track_seed = int(
        hashlib.sha256(
            f"{seed}:{spec.content_sha256}".encode("ascii")
        ).hexdigest()[:16],
        16,
    ) % (2**32)
    rng = np.random.default_rng(track_seed)
    duration = max(event.offset for event in spec.events) + 0.5
    audio = np.zeros(round(duration * sample_rate), dtype=np.float32)
    for event, velocity in zip(spec.events, spec.velocities, strict=True):
        pitch = spec.tuning[event.string - 1] + event.fret
        note = plucked_note(
            midi_to_hz(pitch),
            event.offset - event.onset,
            sample_rate,
            string=event.string,
            rng=rng,
        )
        start_sample = round(event.onset * sample_rate)
        end_sample = min(audio.size, start_sample + note.size)
        gain = 0.22 + 0.28 * velocity / 127
        audio[start_sample:end_sample] += (
            note[: end_sample - start_sample] * gain
        )
    audio += rng.normal(0, 0.0015, audio.size).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    audio /= max(1.0, peak / 0.95)

    track_id = f"symbolic-{spec.content_sha256[:16]}-{track_seed:08x}"
    audio_path = output / "audio" / f"{track_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(audio_path, audio, sample_rate, subtype="PCM_16")
    recipe = {
        "format_version": 1,
        "generator": GENERATOR_VERSION,
        "source_sha256": spec.source_sha256,
        "content_sha256": spec.content_sha256,
        "source_format": spec.source_format,
        "event_sha256": _event_digest(spec.events),
        "seed": seed,
        "track_seed": track_seed,
        "sample_rate": sample_rate,
        "fret_semantics": "absolute-physical-fret",
        "capo_semantics": "independent-metadata-no-fret-remapping",
    }
    policy = load_source_ledger().policies[SYNTHETIC_SOURCE_ID]
    seconds_per_beat = 60 / spec.bpm
    beat_count = max(1, math.floor(duration / seconds_per_beat) + 1)
    return TrackAnnotation(
        track_id=track_id,
        group_id=f"symbolic-content-{spec.content_sha256[:24]}",
        audio=str(audio_path.relative_to(output)),
        split=spec.split,
        duration=round(duration, 6),
        events=list(spec.events),
        tuning=list(spec.tuning),
        capo=spec.capo,
        bpm=spec.bpm,
        beats=[
            round(index * seconds_per_beat, 6)
            for index in range(beat_count)
        ],
        time_signature=spec.time_signature,
        provenance=source_provenance(
            {
                "dataset": "stringtrace-symbolic-synthetic",
                "rights_basis": spec.rights_basis,
                "symbolic_source_sha256": spec.source_sha256,
                "symbolic_content_sha256": spec.content_sha256,
                "symbolic_source_format": spec.source_format,
                "generator": GENERATOR_VERSION,
                "generator_recipe": recipe,
                "generator_recipe_sha256": _json_sha256(recipe),
                "generated_audio_sha256": sha256_file(audio_path),
                "benchmark_eligible": False,
            },
            policy,
        ),
    )


def generate_symbolic_dataset(
    inputs: list[Path],
    output: Path,
    *,
    seed: int,
    default_split: str = "train",
    rights_basis: str | None = None,
    sample_rate: int = 22050,
) -> tuple[list[TrackAnnotation], dict[str, object]]:
    if not inputs:
        raise ValueError("At least one symbolic input is required")
    if sample_rate < 8000:
        raise ValueError("Synthetic sample rate must be at least 8000 Hz")
    output = output.resolve()
    specs = [
        load_symbolic_spec(
            path.resolve(),
            default_split=default_split,
            rights_basis=rights_basis,
        )
        for path in inputs
    ]
    content_hashes = [spec.content_sha256 for spec in specs]
    if len(content_hashes) != len(set(content_hashes)):
        raise ValueError(
            "Duplicate symbolic content cannot cross dataset records or splits"
        )
    annotations = [
        render_symbolic_track(
            spec,
            output,
            seed=seed,
            sample_rate=sample_rate,
        )
        for spec in specs
    ]
    validate_group_splits(annotations)
    if len({item.track_id for item in annotations}) != len(annotations):
        raise ValueError("Duplicate deterministic synthetic track_id")
    manifest = output / "manifest.jsonl"
    write_manifest(manifest, annotations)
    identity = manifest_identity(
        manifest,
        annotations,
        purpose="training",
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "symbolic-synthesis-report",
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "sample_rate": sample_rate,
        "manifest": identity,
        "tracks": len(annotations),
        "splits": dict(
            sorted(Counter(item.split for item in annotations).items())
        ),
        "sources": [
            {
                "sha256": spec.source_sha256,
                "content_sha256": spec.content_sha256,
                "format": spec.source_format,
                "rights_basis": spec.rights_basis,
            }
            for spec in specs
        ],
        "fret_semantics": "absolute-physical-fret",
        "capo_semantics": "independent-metadata-no-fret-remapping",
        "benchmark_eligible": False,
    }
    report_path = output / "synthesis-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return annotations, report


def build_track(
    index: int,
    output: Path,
    sample_rate: int,
    rng: np.random.Generator,
) -> TrackAnnotation:
    duration = 3.0
    audio = np.zeros(round(duration * sample_rate), dtype=np.float32)
    events: list[LabeledEvent] = []
    repeated_fret = 2 + index % 4
    event_specs = [
        (0.20, 5, 0),
        (0.48, 4, 2),
        (0.76, 2, repeated_fret),
        (1.02, 2, repeated_fret),
        (1.30, 3, 2 + (index % 3)),
        (1.58, 1, 0),
        (1.86, 2, repeated_fret),
        (2.14, 4, 2),
        (2.42, 5, 0),
    ]
    for onset, string, fret in event_specs:
        note_duration = min(0.42, duration - onset)
        pitch = STANDARD_TUNING_LIST[string - 1] + fret
        note = plucked_note(
            midi_to_hz(pitch),
            note_duration,
            sample_rate,
            string=string,
            rng=rng,
        )
        start_sample = round(onset * sample_rate)
        end_sample = min(audio.size, start_sample + note.size)
        audio[start_sample:end_sample] += (
            note[: end_sample - start_sample] * 0.42
        )
        events.append(
            LabeledEvent(
                onset=onset,
                offset=onset + note_duration,
                string=string,
                fret=fret,
            )
        )
    audio += rng.normal(0, 0.002, audio.size).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    audio /= max(1.0, peak / 0.95)
    track_id = f"synthetic-{index:02d}"
    audio_path = output / "audio" / f"{track_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(audio_path, audio, sample_rate, subtype="PCM_16")
    recipe = {
        "format_version": 1,
        "generator": "decaying-harmonics-smoke-v2",
        "index": index,
        "sample_rate": sample_rate,
    }
    policy = load_source_ledger().policies[SYNTHETIC_SOURCE_ID]
    return TrackAnnotation(
        track_id=track_id,
        group_id=f"synthetic-performer-{index:02d}",
        audio=str(audio_path.relative_to(output)),
        split="validation" if index >= 6 else "train",
        duration=duration,
        events=events,
        bpm=120,
        beats=[0, 0.5, 1, 1.5, 2, 2.5],
        provenance=source_provenance(
            {
                "dataset": "stringtrace-synthetic-smoke",
                "rights_basis": "original",
                "generator": "decaying-harmonics-smoke-v2",
                "generator_recipe": recipe,
                "generator_recipe_sha256": _json_sha256(recipe),
                "generated_audio_sha256": sha256_file(audio_path),
                "benchmark_eligible": False,
            },
            policy,
        ),
    )


def build_training_track(
    index: int,
    output: Path,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    split: str = "train",
) -> TrackAnnotation:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown synthetic split: {split}")
    duration = 4.0
    audio = np.zeros(round(duration * sample_rate), dtype=np.float32)
    repeated_string = 1 + index % 6
    repeated_fret = (index * 7 + 5) % 25
    event_specs: list[tuple[float, int, int]] = []
    for step in range(14):
        onset = 0.18 + step * 0.25
        if step in {6, 7, 12}:
            string = repeated_string
            fret = repeated_fret
        elif step < 6:
            string = step + 1
            fret = (index * 11 + step * 7) % 25
        else:
            string = 1 + (index + step * 5) % 6
            fret = (index * 11 + step * 7) % 25
        event_specs.append((onset, string, fret))
        if step in {2, 8}:
            chord_string = 1 + string % 6
            chord_fret = (fret + 5 + index) % 25
            event_specs.append((onset, chord_string, chord_fret))
    event_specs.sort(key=lambda value: (value[0], value[1]))

    events: list[LabeledEvent] = []
    for onset, string, fret in event_specs:
        note_duration = min(0.2, duration - onset)
        pitch = STANDARD_TUNING_LIST[string - 1] + fret
        note = plucked_note(
            midi_to_hz(pitch),
            note_duration,
            sample_rate,
            string=string,
            rng=rng,
        )
        start_sample = round(onset * sample_rate)
        end_sample = min(audio.size, start_sample + note.size)
        audio[start_sample:end_sample] += (
            note[: end_sample - start_sample] * 0.34
        )
        events.append(
            LabeledEvent(
                onset=round(onset, 6),
                offset=round(onset + note_duration, 6),
                string=string,
                fret=fret,
            )
        )

    audio += rng.normal(0, 0.0015, audio.size).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    audio /= max(1.0, peak / 0.95)
    track_id = f"symbolic-training-{index:04d}"
    audio_path = output / "audio" / f"{track_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(audio_path, audio, sample_rate, subtype="PCM_16")
    recipe = {
        "format_version": 1,
        "generator": "physical-fret-training-v1",
        "index": index,
        "sample_rate": sample_rate,
        "event_specs": event_specs,
    }
    policy = load_source_ledger().policies[SYNTHETIC_SOURCE_ID]
    return TrackAnnotation(
        track_id=track_id,
        group_id=f"symbolic-training-group-{index:04d}",
        audio=str(audio_path.relative_to(output)),
        split=split,
        duration=duration,
        events=events,
        bpm=120,
        beats=[round(index * 0.5, 6) for index in range(8)],
        provenance=source_provenance(
            {
                "dataset": "stringtrace-symbolic-synthetic",
                "rights_basis": "original",
                "generator": "physical-fret-training-v1",
                "generator_recipe": recipe,
                "generator_recipe_sha256": _json_sha256(recipe),
                "generated_audio_sha256": sha256_file(audio_path),
                "benchmark_eligible": False,
            },
            policy,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic synthetic guitar data from structured "
            "symbolic input or the built-in pipeline smoke recipe."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--profile",
        choices=("smoke", "training"),
        default="smoke",
    )
    parser.add_argument(
        "--symbolic-input",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--split",
        choices=tuple(sorted(VALID_SPLITS)),
        default="train",
    )
    parser.add_argument(
        "--rights-basis",
        choices=tuple(sorted(RIGHTS_BASES)),
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.symbolic_input:
        annotations, report = generate_symbolic_dataset(
            arguments.symbolic_input,
            output,
            seed=arguments.seed,
            default_split=arguments.split,
            rights_basis=arguments.rights_basis,
            sample_rate=arguments.sample_rate,
        )
        print(
            json.dumps(
                {
                    "manifest": report["manifest"]["name"],
                    "tracks": len(annotations),
                    "splits": report["splits"],
                    "report": "synthesis-report.json",
                    "benchmark_eligible": False,
                },
                ensure_ascii=False,
            )
        )
        return

    if arguments.profile == "smoke" and arguments.tracks < 8:
        raise ValueError(
            "At least 8 tracks are required for train/validation splits"
        )
    if arguments.profile == "training" and arguments.tracks < 1:
        raise ValueError("At least one training track is required")
    rng = np.random.default_rng(arguments.seed)
    if arguments.profile == "training":
        annotations = [
            build_training_track(
                index,
                output,
                arguments.sample_rate,
                rng,
                split=arguments.split,
            )
            for index in range(arguments.tracks)
        ]
    else:
        annotations = [
            build_track(index, output, arguments.sample_rate, rng)
            for index in range(arguments.tracks)
        ]
    manifest = output / "manifest.jsonl"
    write_manifest(manifest, annotations)
    identity = manifest_identity(
        manifest,
        annotations,
        purpose="training",
    )
    print(
        json.dumps(
            {
                "manifest": identity["name"],
                "manifest_sha256": identity["sha256"],
                "profile": arguments.profile,
                "tracks": len(annotations),
                "train": sum(item.split == "train" for item in annotations),
                "validation": sum(
                    item.split == "validation"
                    for item in annotations
                ),
                "benchmark_eligible": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
