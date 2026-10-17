"""Measure guitar pitch activation and attacks directly from a separated recording.

Produces inspectable intermediate evidence. It does not approve a transcription.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter, maximum_filter1d
from scipy.signal import find_peaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(args.audio, sr=22050, mono=True)
    hop = 256
    n_fft = 8192
    magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)).astype(np.float32)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    keep = (frequencies >= 65) & (frequencies <= 5000)
    frequencies = frequencies[keep]
    magnitude = magnitude[keep]
    pitches = np.arange(43, 85)
    tuning = float(librosa.estimate_tuning(y=y[:sr * 30], sr=sr))
    # Fit complete harmonic stacks, rather than counting every harmonic as a note.
    columns = []
    for midi in pitches:
        f0 = librosa.midi_to_hz(float(midi) + tuning)
        template = np.zeros(len(frequencies), dtype=np.float32)
        for harmonic in range(1, 35):
            center = f0 * harmonic
            if center > 5000:
                break
            width = max(sr / n_fft * 0.9, center * 0.0015)
            template += np.exp(-0.5 * ((frequencies - center) / width) ** 2) / harmonic
        columns.append(template)
    dictionary = np.stack(columns, axis=1)
    whitening = 1 / np.maximum(np.mean(magnitude, axis=1), 0.06) ** 0.55
    dictionary *= whitening[:, None]
    target = magnitude * whitening[:, None]
    norms = np.maximum(np.linalg.norm(dictionary, axis=0), 1e-8)
    dictionary /= norms
    gram = dictionary.T @ dictionary
    correlation = dictionary.T @ target
    # Nonnegative least squares with a small sparsity cost; FISTA converges without
    # imposing any chord sequence or a fixed number of events.
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    activation = np.maximum(correlation, 0)
    extrapolated = activation.copy()
    alpha = 1.0
    penalty = np.percentile(correlation, 95) * 0.012
    for _ in range(110):
        next_activation = np.maximum(
            extrapolated + (correlation - gram @ extrapolated - penalty) / lipschitz,
            0,
        )
        next_alpha = (1 + np.sqrt(1 + 4 * alpha * alpha)) / 2
        extrapolated = next_activation + ((alpha - 1) / next_alpha) * (
            next_activation - activation
        )
        activation, alpha = next_activation, next_alpha
    activation = median_filter(activation, size=(1, 5))
    times = librosa.frames_to_time(np.arange(activation.shape[1]), sr=sr, hop_length=hop)
    # Broadband attacks provide independent timing anchors.
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    attacks = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop,
        units="time",
        backtrack=False,
    )
    events = []
    global_peak = float(np.percentile(activation.max(axis=0), 95))
    for row, pitch in zip(activation, pitches):
        amplitude = median_filter(row, size=5)
        # Onset novelty separated from sustained energy, which avoids repeatedly
        # emitting a held note at each metronome beat.
        prior = np.concatenate([np.zeros(5), amplitude[:-5]])
        novelty = np.maximum(amplitude - prior, 0)
        threshold = max(global_peak * 0.045, float(np.max(amplitude)) * 0.06)
        peaks, _ = find_peaks(
            novelty, height=threshold,
            prominence=threshold * 0.65, distance=8,
        )
        for peak in peaks:
            if amplitude[peak] < global_peak * 0.065:
                continue
            # Local half-height crossing estimates the attack before its novelty peak.
            begin = peak
            baseline = amplitude[max(0, peak - 12):max(1, peak - 5)].min()
            crossing = baseline + (amplitude[peak] - baseline) * 0.25
            while begin > max(0, peak - 12) and amplitude[begin - 1] > crossing:
                begin -= 1
            end = peak + 1
            release_level = max(global_peak * 0.025, amplitude[peak] * 0.22)
            while end < len(amplitude) and amplitude[end] > release_level:
                end += 1
            start_time = float(times[begin])
            end_time = min(len(y) / sr, float(end * hop / sr))
            if end_time - start_time < 0.085:
                continue
            anchor_distance = float(np.min(abs(attacks - start_time))) if len(attacks) else 1
            events.append({
                "pitch": int(pitch),
                "start": round(start_time, 4),
                "end": round(end_time, 4),
                "strength": round(float(amplitude[peak] / global_peak), 4),
                "attackDistance": round(anchor_distance, 4),
            })
    events.sort(key=lambda e: (e["start"], e["pitch"]))
    by_pitch: dict[int, list[dict]] = {}
    for event in events:
        by_pitch.setdefault(event["pitch"], []).append(event)
    for same_pitch in by_pitch.values():
        for left, right in zip(same_pitch, same_pitch[1:]):
            left["end"] = min(left["end"], right["start"])
    np.savez_compressed(
        args.output / "spectrum.npz",
        activation=activation, times=times, pitches=pitches,
        magnitude=magnitude, frequencies=frequencies,
        attacks=attacks, tuning=tuning,
    )
    (args.output / "candidates.json").write_text(json.dumps({
        "method": "harmonic-nnls-measurement-v1",
        "tuningSemitones": tuning,
        "duration": len(y) / sr,
        "hopSeconds": hop / sr,
        "events": events,
        "attacks": attacks.tolist(),
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "tuning": tuning, "events": len(events),
        "attacks": len(attacks), "first": events[:25],
    }))


if __name__ == "__main__":
    main()
