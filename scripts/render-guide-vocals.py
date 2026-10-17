#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def voiced_pitch(samples: np.ndarray, sample_rate: int) -> float:
    f0, voiced, _ = librosa.pyin(
        samples,
        fmin=110,
        fmax=350,
        sr=sample_rate,
        frame_length=2048,
        hop_length=256,
    )
    pitches = f0[np.isfinite(f0) & voiced]
    if pitches.size == 0:
        raise RuntimeError("guide-vocal syllable has no detectable pitch")
    return float(np.median(pitches))


def fit_event(
    source: np.ndarray,
    source_pitch: float,
    target_pitch: float,
    target_frames: int,
    sample_rate: int,
    vowel: str,
) -> np.ndarray:
    semitones = 12 * math.log2(target_pitch / source_pitch)
    pitched = librosa.effects.pitch_shift(
        source,
        sr=sample_rate,
        n_steps=semitones,
        bins_per_octave=12,
        res_type="soxr_hq",
    )
    rate = len(pitched) / max(1, target_frames)
    fitted = librosa.effects.time_stretch(pitched, rate=max(0.05, rate))
    output = np.zeros(target_frames, dtype=np.float64)
    length = min(target_frames, len(fitted))
    output[:length] = fitted[:length] * 0.38
    formants = {
        "a": ((800, 100), (1150, 130), (2900, 180)),
        "e": ((500, 90), (1700, 140), (2500, 180)),
        "i": ((300, 80), (2300, 150), (3000, 200)),
        "o": ((500, 90), (900, 110), (2400, 180)),
        "u": ((350, 80), (700, 100), (2200, 180)),
    }.get(vowel, ((500, 100), (1500, 150), (2500, 180)))
    time = np.arange(target_frames, dtype=np.float64) / sample_rate
    vibrato = 2 ** ((0.12 * np.sin(2 * np.pi * 5.1 * time)) / 12)
    phase = np.cumsum(2 * np.pi * target_pitch * vibrato / sample_rate)
    tone = np.zeros(target_frames, dtype=np.float64)
    for harmonic in range(1, 19):
        frequency = target_pitch * harmonic
        formant_gain = 0.04 + sum(
            math.exp(-0.5 * ((frequency - center) / width) ** 2)
            for center, width in formants
        )
        tone += (
            np.sin(phase * harmonic)
            * formant_gain
            / (harmonic**0.82)
        )
    tone_peak = float(np.max(np.abs(tone)))
    if tone_peak > 0:
        tone /= tone_peak
    output += tone * 0.62
    attack = min(length, round(sample_rate * 0.018))
    release = min(length, round(sample_rate * 0.045))
    if attack > 0:
        output[:attack] *= np.linspace(0, 1, attack, endpoint=True)
    if release > 0:
        output[length - release : length] *= np.linspace(
            1, 0, release, endpoint=True
        )
    return output


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: render-guide-vocals.py TASKS_JSON SOURCE_DIR OUTPUT_WAV"
        )
    tasks_path = Path(sys.argv[1])
    source_root = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    specification = json.loads(tasks_path.read_text(encoding="utf-8"))
    sample_rate = int(specification["sampleRate"])
    frame_count = int(specification["frameCount"])
    mix = np.zeros(frame_count, dtype=np.float64)
    sources: dict[str, tuple[np.ndarray, float]] = {}
    fitted_cache: dict[tuple[str, float, int], np.ndarray] = {}

    for task in specification["events"]:
        source_key = task["sourceKey"]
        if source_key not in sources:
            source, source_rate = librosa.load(
                source_root / f"{source_key}.wav",
                sr=None,
                mono=True,
            )
            if source_rate != sample_rate:
                raise RuntimeError("guide-vocal source sample rate mismatch")
            source, _ = librosa.effects.trim(source, top_db=36)
            if source.size < 1024:
                raise RuntimeError("guide-vocal syllable is empty")
            sources[source_key] = (
                source.astype(np.float64),
                voiced_pitch(source, sample_rate),
            )

        target_pitch = float(task["targetPitch"])
        target_frames = int(task["targetFrames"])
        vowel = task["vowel"]
        cache_key = (
            source_key,
            round(target_pitch, 6),
            target_frames,
            vowel,
        )
        fitted = fitted_cache.get(cache_key)
        if fitted is None:
            source, source_pitch = sources[source_key]
            fitted = fit_event(
                source,
                source_pitch,
                target_pitch,
                target_frames,
                sample_rate,
                vowel,
            )
            fitted_cache[cache_key] = fitted

        start = int(task["startFrame"])
        end = min(frame_count, start + len(fitted))
        mix[start:end] += fitted[: end - start] * 0.72

    dry = mix.copy()
    for delay_seconds, gain in ((0.055, 0.08), (0.089, 0.05)):
        delay = round(delay_seconds * sample_rate)
        mix[delay:] += dry[:-delay] * gain
    peak = float(np.max(np.abs(mix)))
    if peak <= 0:
        raise RuntimeError("guide-vocal output is silent")
    mix = np.tanh(mix * (0.82 / peak))
    sf.write(output_path, mix, sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    main()
