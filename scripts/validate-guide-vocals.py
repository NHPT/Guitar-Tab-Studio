#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import librosa
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads(
    (ROOT / "apps/api/assets/wayfaring-stranger-beginner.json").read_text(
        encoding="utf-8"
    )
)
VOCALS = ROOT / "apps/api/assets" / MANIFEST["audio"]["vocalsFilename"]
SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_hz(name: str) -> float:
    match = re.fullmatch(r"([A-G])(#?)(\d)", name)
    if not match:
        raise RuntimeError(f"unsupported guide-vocal pitch {name}")
    midi = (
        (int(match.group(3)) + 1) * 12
        + SEMITONES[match.group(1)]
        + bool(match.group(2))
    )
    return 440 * 2 ** ((midi - 69) / 12)


def main() -> None:
    audio, sample_rate = librosa.load(VOCALS, sr=None, mono=True)
    beat_duration = 60 / MANIFEST["bpm"]
    ticks_per_beat = MANIFEST["ticksPerBeat"]
    ticks_per_measure = ticks_per_beat * MANIFEST["timeSignature"][0]
    errors: list[float] = []
    minimum_rms = float("inf")

    for measure_index, measure in enumerate(MANIFEST["measures"]):
        for event in measure["events"]:
            if not event.get("vocalSyllable") or event.get("tieFromPrevious"):
                continue
            start = (
                (measure_index * ticks_per_measure + event["tick"])
                / ticks_per_beat
                * beat_duration
            )
            duration = event["durationTicks"] / ticks_per_beat * beat_duration
            segment = audio[
                round((start + 0.08) * sample_rate) :
                round((start + max(0.18, duration - 0.05)) * sample_rate)
            ]
            rms = float(np.sqrt(np.mean(segment**2)))
            minimum_rms = min(minimum_rms, rms)
            f0, voiced, _ = librosa.pyin(
                segment,
                fmin=110,
                fmax=380,
                sr=sample_rate,
                frame_length=2048,
                hop_length=256,
            )
            pitches = f0[np.isfinite(f0) & voiced]
            if pitches.size == 0:
                raise RuntimeError(
                    f"no pitch detected at measure {measure_index + 1}"
                )
            detected = float(np.median(pitches))
            errors.append(
                abs(
                    1200
                    * math.log2(
                        detected / pitch_hz(event["sourcePitch"])
                    )
                )
            )

    if len(errors) != 156:
        raise RuntimeError(f"expected 156 vocal attacks, measured {len(errors)}")
    maximum_error = max(errors)
    if maximum_error > 100:
        raise RuntimeError(
            f"guide-vocal pitch error {maximum_error:.1f} cents exceeds limit"
        )
    if minimum_rms < 0.01:
        raise RuntimeError("guide-vocal syllable is effectively silent")
    print(
        "guide vocal:"
        f" {len(errors)} attacks,"
        f" median {np.median(errors):.1f} cents,"
        f" max {maximum_error:.1f} cents,"
        f" minimum RMS {minimum_rms:.4f}"
    )


if __name__ == "__main__":
    main()
