from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile

from .schema import STANDARD_TUNING_LIST, LabeledEvent, TrackAnnotation, write_manifest


def midi_to_hz(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


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
    envelope = np.minimum(1, time / 0.004) * np.exp(-time * (1.8 + thickness))
    result = signal * envelope + attack
    peak = float(np.max(np.abs(result)))
    return result / max(1.0, peak)


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
        audio[start_sample:end_sample] += note[: end_sample - start_sample] * 0.42
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
    return TrackAnnotation(
        track_id=track_id,
        group_id=f"synthetic-performer-{index:02d}",
        audio=str(audio_path.relative_to(output)),
        split="validation" if index >= 6 else "train",
        duration=duration,
        events=events,
        bpm=120,
        beats=[0, 0.5, 1, 1.5, 2, 2.5],
        provenance={
            "dataset": "stringtrace-synthetic-smoke",
            "generator": "decaying-harmonics-v1",
            "benchmark_eligible": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a tiny synthetic dataset for pipeline smoke tests.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260921)
    arguments = parser.parse_args()
    if arguments.tracks < 8:
        raise ValueError("At least 8 tracks are required for train/validation splits")
    output = arguments.output.resolve()
    rng = np.random.default_rng(arguments.seed)
    annotations = [
        build_track(index, output, 22050, rng)
        for index in range(arguments.tracks)
    ]
    manifest = output / "manifest.jsonl"
    write_manifest(manifest, annotations)
    print(
        json.dumps(
            {
                "manifest": str(manifest),
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
