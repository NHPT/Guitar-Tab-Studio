#!/usr/bin/env python3
"""Offline media separation and guitar-note transcription worker."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable


STANDARD_TUNING = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
GUITAR_PITCH_RANGE = (STANDARD_TUNING[6], STANDARD_TUNING[1] + 24)
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
TECHNIQUE_LABELS = (
    "unknown",
    "pick",
    "hammer-on",
    "pull-off",
    "slide",
    "harmonic",
    "palm-mute",
    "dead-note",
    "slap",
    "body-tap",
    "tremolo",
    "bend",
    "pinch-harmonic",
    "vibrato",
)


@dataclass
class TechniqueCandidate:
    technique: str
    confidence: float
    evidence: list[str]


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int
    velocity: int
    string: int
    fret: int
    technique: str
    confidence: float
    technique_confidence: float = 0.0
    technique_source: str = "acoustic-heuristic-v2"
    technique_evidence: list[str] = field(default_factory=list)
    technique_candidates: list[TechniqueCandidate] = field(default_factory=list)
    related_note_index: int | None = None
    harmonic_type: str | None = None
    harmonic_touch_fret: int | None = None
    position_confidence: float = 0.0
    position_source: str = "playable-optimizer-v2"


@dataclass
class AudioFeature:
    onset: float
    centroid: float
    flatness: float
    zero_crossing: float


def run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def require_command(command: str) -> str:
    resolved = shutil.which(command)
    if not resolved:
        raise RuntimeError(f"Required executable is unavailable: {command}")
    return resolved


def ffmpeg_command() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    if configured and Path(configured).is_file():
        return configured
    return require_command("ffmpeg")


def acquire_url(url: str, destination: Path) -> tuple[Path, dict[str, object]]:
    ffmpeg = ffmpeg_command()
    output = destination / "source.%(ext)s"
    run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--write-info-json",
            "--format",
            "bestaudio/best",
            "--ffmpeg-location",
            ffmpeg,
            "--output",
            str(output),
            url,
        ]
    )
    matches = sorted(
        path
        for path in destination.glob("source.*")
        if path.suffix not in {".json", ".part", ".ytdl"}
    )
    if not matches:
        raise RuntimeError("The media adapter did not produce an audio file")
    info_path = destination / "source.info.json"
    metadata: dict[str, object] = {"title": matches[0].stem, "artist": "在线媒体"}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        metadata = {
            "title": info.get("title") or matches[0].stem,
            "artist": info.get("uploader") or info.get("artist") or "在线媒体",
            "duration": info.get("duration"),
            "webpage_url": info.get("webpage_url") or url,
        }
    return matches[0], metadata


def normalize_audio(source: Path, destination: Path) -> Path:
    ffmpeg = ffmpeg_command()
    normalized = destination / "normalized.wav"
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(normalized),
        ]
    )
    return normalized


def separate_audio(source: Path, destination: Path) -> dict[str, Path]:
    stems_root = destination / "stems"
    run(
        [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs_6s",
            "--out",
            str(stems_root),
            str(source),
        ]
    )
    result_root = stems_root / "htdemucs_6s" / source.stem
    expected = ("vocals", "guitar", "bass", "drums", "piano", "other")
    stems = {name: result_root / f"{name}.wav" for name in expected}
    missing = [name for name, path in stems.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Demucs output is incomplete: {', '.join(missing)}")
    return stems


def stabilize_beat_times(
    beat_times: Iterable[float],
    *,
    maximum_correction: float = 0.025,
    passes: int = 2,
) -> list[float]:
    stabilized = [float(time) for time in beat_times]
    if len(stabilized) < 3:
        return stabilized
    for _pass in range(passes):
        previous_pass = stabilized
        stabilized = list(previous_pass)
        for index in range(1, len(previous_pass) - 1):
            midpoint = (previous_pass[index - 1] + previous_pass[index + 1]) / 2
            correction = midpoint - previous_pass[index]
            if abs(correction) <= maximum_correction:
                stabilized[index] += correction * 0.5
    return stabilized


def estimate_tempo(path: Path) -> tuple[float, list[float], list[float]]:
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    onset_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        trim=False,
    )
    value = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
    bpm = round(value if math.isfinite(value) and value > 0 else 120.0, 2)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        units="frames",
        backtrack=False,
    )
    beat_times = stabilize_beat_times(
        librosa.frames_to_time(beat_frames, sr=sample_rate),
    )
    if onset_frames.size:
        first_onset = float(librosa.frames_to_time(onset_frames[0], sr=sample_rate))
        beat_times = [
            time for time in beat_times if time >= first_onset - 0.08
        ]
    usable_beats = [
        round(float(time), 4)
        for time in beat_times
        if np.isfinite(time) and time >= 0
    ]
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    usable_onsets = [
        round(float(time), 4)
        for time in onset_times
        if np.isfinite(time) and time >= 0
    ]
    return bpm, usable_beats, usable_onsets


def waveform(path: Path, bar_count: int = 120) -> list[float]:
    import librosa
    import numpy as np

    audio, _sample_rate = librosa.load(path, sr=11025, mono=True)
    if audio.size == 0:
        return [0.0] * bar_count
    chunks = np.array_split(np.abs(audio), bar_count)
    peak = max(float(chunk.max(initial=0.0)) for chunk in chunks) or 1.0
    return [round(float(chunk.max(initial=0.0)) / peak, 4) for chunk in chunks]


def visualization_timeline(path: Path, point_count: int = 720) -> list[dict[str, float]]:
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    if audio.size == 0:
        return [
            {
                "energy": 0.0,
                "pitch": 0.5,
                "low": 0.0,
                "mid": 0.0,
                "high": 0.0,
                "impact": 0.0,
            }
            for _ in range(point_count)
        ]

    hop_length = 512
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=hop_length,
        n_mels=96,
        fmin=40,
        fmax=8000,
        power=2.0,
    )
    mel_amplitude = np.sqrt(np.maximum(mel, 0))
    mel_frequencies = librosa.mel_frequencies(
        n_mels=mel.shape[0],
        fmin=40,
        fmax=8000,
    )
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )
    frame_count = min(mel.shape[1], len(rms), len(onset))
    mel_amplitude = mel_amplitude[:, :frame_count]
    rms = rms[:frame_count]
    onset = onset[:frame_count]
    energy_scale = max(float(np.percentile(rms, 95)), 1e-6)
    onset_scale = max(float(np.percentile(onset, 95)), 1e-6)
    band_scale = max(float(np.percentile(mel_amplitude, 95)), 1e-6)
    low_mask = mel_frequencies < 250
    mid_mask = (mel_frequencies >= 250) & (mel_frequencies < 2000)
    high_mask = mel_frequencies >= 2000

    timeline: list[dict[str, float]] = []
    for point_index in range(point_count):
        start = min(frame_count - 1, int(point_index * frame_count / point_count))
        end = min(
            frame_count,
            max(start + 1, int((point_index + 1) * frame_count / point_count)),
        )
        frame_slice = mel_amplitude[:, start:end]
        frame_energy = min(1.0, float(np.max(rms[start:end])) / energy_scale)
        frame_impact = min(1.0, float(np.max(onset[start:end])) / onset_scale)
        mean_spectrum = np.mean(frame_slice, axis=1)
        dominant_frequency = float(mel_frequencies[int(np.argmax(mean_spectrum))])
        pitch = min(
            1.0,
            max(0.0, math.log2(max(70.0, dominant_frequency) / 70.0) / math.log2(2200 / 70)),
        )

        def band_level(mask: object) -> float:
            selected = frame_slice[mask]
            return min(1.0, float(np.max(selected)) / band_scale) if selected.size else 0.0

        timeline.append(
            {
                "energy": round(frame_energy, 4),
                "pitch": round(pitch, 4),
                "low": round(band_level(low_mask), 4),
                "mid": round(band_level(mid_mask), 4),
                "high": round(band_level(high_mask), 4),
                "impact": round(frame_impact, 4),
            }
        )
    return timeline


def audio_rms(path: Path) -> float:
    import librosa
    import numpy as np

    audio, _sample_rate = librosa.load(path, sr=11025, mono=True)
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def transcription_source(
    normalized: Path,
    stems: dict[str, Path],
) -> tuple[Path, str | None]:
    original_level = audio_rms(normalized)
    guitar_level = audio_rms(stems["guitar"])
    if guitar_level >= max(0.0001, original_level * 0.04):
        return stems["guitar"], None

    other_level = audio_rms(stems["other"])
    if other_level >= original_level * 0.08:
        return stems["other"], "吉他分轨能量过低，已使用伴奏残余轨进行音符识别。"
    return normalized, "吉他分轨能量过低，已使用原始混音进行音符识别。"


def compress_stems(stems: dict[str, Path], destination: Path) -> dict[str, Path]:
    ffmpeg = ffmpeg_command()
    media_root = destination / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    compressed: dict[str, Path] = {}
    for name, source in stems.items():
        target = media_root / f"{name}.m4a"
        run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                str(target),
            ]
        )
        compressed[name] = target
        source.unlink(missing_ok=True)
    return compressed


def candidate_positions(pitch: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for string, open_pitch in STANDARD_TUNING.items():
        fret = pitch - open_pitch
        if 0 <= fret <= 24:
            candidates.append((string, fret))
    return candidates


def select_fingering(
    pitch: int,
    previous: tuple[int, int] | None,
) -> tuple[int, int]:
    candidates = candidate_positions(pitch)
    if not candidates:
        clamped_pitch = min(max(pitch, STANDARD_TUNING[6]), STANDARD_TUNING[1] + 24)
        candidates = candidate_positions(clamped_pitch)
    if not candidates:
        return (1, 0)
    if previous is None:
        return min(candidates, key=lambda item: (item[1] > 12, item[1], item[0]))

    previous_string, previous_fret = previous
    return min(
        candidates,
        key=lambda item: (
            abs(item[1] - previous_fret) * 1.6
            + abs(item[0] - previous_string) * 0.7
            + max(0, item[1] - 12) * 0.4
        ),
    )


def fingering_group_cost(
    pitches: list[int],
    positions: list[tuple[int, int]],
    previous_positions: list[tuple[int, int]],
) -> float:
    cost = 0.0
    fretted = [fret for _string, fret in positions if fret > 0]
    if fretted:
        fret_span = max(fretted) - min(fretted)
        cost += fret_span * 0.18 + max(0, fret_span - 4) * 1.4

    for _string, fret in positions:
        cost += fret * 0.045 + max(0, fret - 12) * 0.9
        if fret == 0:
            cost -= 0.35

    ordered = sorted(zip(pitches, positions), key=lambda item: item[0])
    for (_left_pitch, left), (_right_pitch, right) in zip(
        ordered,
        ordered[1:],
    ):
        if right[0] >= left[0]:
            cost += 2.4

    if previous_positions:
        previous_fret = sum(fret for _string, fret in previous_positions) / len(
            previous_positions
        )
        previous_string = sum(
            string for string, _fret in previous_positions
        ) / len(previous_positions)
        current_fret = sum(fret for _string, fret in positions) / len(positions)
        current_string = sum(string for string, _fret in positions) / len(positions)
        cost += abs(current_fret - previous_fret) * 0.42
        cost += abs(current_string - previous_string) * 0.16
    return cost


def assign_fingering_group(
    pitches: list[int],
    previous_positions: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], float]:
    if not pitches:
        return [], 0.0
    previous_positions = previous_positions or []
    if len(pitches) > 6:
        fallback: list[tuple[int, int]] = []
        previous = previous_positions[-1] if previous_positions else None
        for pitch in pitches:
            position = select_fingering(pitch, previous)
            fallback.append(position)
            previous = position
        return fallback, 0.32
    candidate_sets = [candidate_positions(pitch) for pitch in pitches]
    if any(not candidates for candidates in candidate_sets):
        fallback: list[tuple[int, int]] = []
        previous = previous_positions[-1] if previous_positions else None
        for pitch in pitches:
            position = select_fingering(pitch, previous)
            fallback.append(position)
            previous = position
        return fallback, 0.42

    scored: list[tuple[float, list[tuple[int, int]]]] = []

    def visit(
        index: int,
        used_strings: set[int],
        positions: list[tuple[int, int]],
    ) -> None:
        if index == len(candidate_sets):
            scored.append(
                (
                    fingering_group_cost(
                        pitches,
                        positions,
                        previous_positions,
                    ),
                    list(positions),
                )
            )
            return

        available = [
            position
            for position in candidate_sets[index]
            if position[0] not in used_strings
        ]
        if not available:
            available = candidate_sets[index]
        for position in available:
            duplicate_penalty = 9.0 if position[0] in used_strings else 0.0
            if duplicate_penalty and len(pitches) <= 6:
                continue
            positions.append(position)
            visit(index + 1, used_strings | {position[0]}, positions)
            positions.pop()

    visit(0, set(), [])
    if not scored:
        return [
            select_fingering(
                pitch,
                previous_positions[-1] if previous_positions else None,
            )
            for pitch in pitches
        ], 0.4

    scored.sort(key=lambda item: item[0])
    best_score, best_positions = scored[0]
    if len(scored) == 1:
        confidence = 0.94
    else:
        margin = max(0.0, scored[1][0] - best_score)
        confidence = min(0.92, 0.56 + margin * 0.1)
    return best_positions, round(confidence, 4)


NATURAL_HARMONIC_INTERVALS = {
    12: 12,
    19: 7,
    24: 5,
}


def natural_harmonic_positions(pitch: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for string, open_pitch in STANDARD_TUNING.items():
        interval = pitch - open_pitch
        if interval in NATURAL_HARMONIC_INTERVALS:
            positions.append((string, NATURAL_HARMONIC_INTERVALS[interval]))
    return positions


def has_natural_harmonic_candidate(pitch: int) -> bool:
    return bool(natural_harmonic_positions(pitch))


def select_harmonic_position(
    pitch: int,
    previous: tuple[int, int] | None,
) -> tuple[int, int, str, int | None, float] | None:
    natural = natural_harmonic_positions(pitch)
    if natural:
        selected = min(
            natural,
            key=lambda position: (
                abs(position[1] - previous[1]) * 0.5
                + abs(position[0] - previous[0]) * 0.3
                if previous
                else position[1]
            ),
        )
        confidence = 0.92 if len(natural) == 1 else 0.84
        return selected[0], selected[1], "natural", None, confidence

    artificial: list[tuple[int, int, int]] = []
    for string, open_pitch in STANDARD_TUNING.items():
        base_fret = pitch - open_pitch - 12
        if 0 <= base_fret <= 12:
            artificial.append((string, base_fret, base_fret + 12))
    if not artificial:
        return None
    selected = min(
        artificial,
        key=lambda position: (
            abs(position[1] - previous[1]) * 0.6
            + abs(position[0] - previous[0]) * 0.3
            if previous
            else position[1]
        ),
    )
    confidence = 0.8 if len(artificial) == 1 else 0.68
    return selected[0], selected[1], "artificial", selected[2], confidence


def collapse_harmonic_stacks(
    events: list[NoteEvent],
    onset_tolerance: float = 0.025,
) -> tuple[list[NoteEvent], set[int]]:
    groups: list[list[NoteEvent]] = []
    for event in sorted(events, key=lambda item: (item.start, item.pitch)):
        if (
            not groups
            or event.start - groups[-1][0].start > onset_tolerance
        ):
            groups.append([event])
        else:
            groups[-1].append(event)

    retained: list[NoteEvent] = []
    forced_event_ids: set[int] = set()
    for group in groups:
        removed_ids: set[int] = set()
        by_pitch = {event.pitch: event for event in group}
        for base in sorted(group, key=lambda event: event.pitch):
            if id(base) in removed_ids or not has_natural_harmonic_candidate(base.pitch):
                continue
            partials = [
                by_pitch.get(base.pitch + 12),
                by_pitch.get(base.pitch + 24),
            ]
            if any(partial is None for partial in partials):
                continue
            cluster = [base, *[partial for partial in partials if partial]]
            durations = [event.end - event.start for event in cluster]
            if max(durations) - min(durations) > max(0.12, durations[0] * 0.12):
                continue
            forced_event_ids.add(id(base))
            removed_ids.update(id(event) for event in cluster[1:])

        retained.extend(event for event in group if id(event) not in removed_ids)

    retained.sort(key=lambda event: (event.start, event.pitch))
    forced_indices = {
        index
        for index, event in enumerate(retained)
        if id(event) in forced_event_ids
    }
    return retained, forced_indices


def collapse_note_fragments(
    events: list[NoteEvent],
    maximum_gap: float = 0.03,
) -> list[NoteEvent]:
    retained: list[NoteEvent] = []
    latest_by_pitch: dict[int, NoteEvent] = {}
    for event in sorted(events, key=lambda item: (item.start, item.pitch)):
        previous = latest_by_pitch.get(event.pitch)
        if previous:
            gap = event.start - previous.end
            onset_gap = event.start - previous.start
            previous_duration = previous.end - previous.start
            event_duration = event.end - event.start
            if (
                0 <= gap <= maximum_gap
                and onset_gap <= 0.25
                and previous_duration <= 0.18
                and previous.velocity <= 45
                and event_duration >= 0.18
                and event.velocity >= previous.velocity + 12
            ):
                retained.remove(previous)
        latest_by_pitch[event.pitch] = event
        retained.append(event)
    return retained


def extract_note_features(path: Path, notes: list[NoteEvent]) -> list[AudioFeature]:
    import librosa
    import numpy as np

    if not notes:
        return []
    audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    hop_length = 256
    onset = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        hop_length=hop_length,
    )[0]
    zero_crossing = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=hop_length,
    )[0]
    onset_scale = max(float(np.percentile(onset, 95)), 1e-6)
    frame_count = min(
        len(onset),
        len(rms),
        len(centroid),
        len(flatness),
        len(zero_crossing),
    )

    features: list[AudioFeature] = []
    for note in notes:
        frame = int(
            librosa.time_to_frames(
                note.start,
                sr=sample_rate,
                hop_length=hop_length,
            )
        )
        start = max(0, min(frame_count - 1, frame - 1))
        end = max(start + 1, min(frame_count, frame + 3))
        features.append(
            AudioFeature(
                onset=min(1.0, float(np.max(onset[start:end])) / onset_scale),
                centroid=float(np.mean(centroid[start:end])),
                flatness=float(np.mean(flatness[start:end])),
                zero_crossing=float(np.mean(zero_crossing[start:end])),
            )
        )
    return features


def confidence_score(value: float) -> float:
    return round(min(0.98, max(0.01, value)), 4)


def infer_technique_candidates(
    previous: NoteEvent | None,
    current: NoteEvent,
    feature: AudioFeature,
) -> list[TechniqueCandidate]:
    duration = current.end - current.start
    natural_harmonic_position = has_natural_harmonic_candidate(current.pitch)
    candidates: list[TechniqueCandidate] = []

    pick_evidence = ["stable-pitched-event"]
    if feature.onset >= 0.38:
        pick_evidence.append("audible-onset")
    candidates.append(
        TechniqueCandidate(
            "pick",
            confidence_score(
                0.38
                + feature.onset * 0.38
                + (0.12 if duration >= 0.2 else 0)
                + min(1.0, current.velocity / 127) * 0.08
            ),
            pick_evidence,
        )
    )

    harmonic_evidence: list[str] = []
    harmonic_score = 0.08
    if natural_harmonic_position:
        harmonic_score += 0.2
        harmonic_evidence.append("natural-harmonic-fret")
    if feature.centroid >= 1800:
        harmonic_score += min(0.28, (feature.centroid - 1800) / 4000)
        harmonic_evidence.append("bright-spectrum")
    if feature.flatness <= 0.12:
        harmonic_score += 0.2
        harmonic_evidence.append("tonal-spectrum")
    if feature.onset <= 0.58:
        harmonic_score += 0.14
        harmonic_evidence.append("soft-onset")
    if duration >= 0.34:
        harmonic_score += 0.16
        harmonic_evidence.append("sustained-note")
    if (
        duration >= 0.34
        and current.velocity <= 96
        and feature.flatness <= 0.12
        and feature.onset <= 0.58
        and (
            (natural_harmonic_position and feature.centroid >= 1800)
            or feature.centroid >= 2600
        )
    ):
        candidates.append(
            TechniqueCandidate(
                "harmonic",
                confidence_score(harmonic_score),
                harmonic_evidence,
            )
        )

    if (
        duration <= 0.16
        and feature.onset >= 0.82
        and feature.flatness >= 0.24
        and feature.zero_crossing >= 0.12
    ):
        candidates.append(
            TechniqueCandidate(
                "body-tap",
                confidence_score(
                    0.7
                    + feature.onset * 0.2
                    + feature.flatness * 0.18
                    + feature.zero_crossing * 0.18
                ),
                ["hard-transient", "noise-dominant", "very-short-event"],
            )
        )

    if (
        duration <= 0.2
        and feature.onset >= 0.72
        and (feature.zero_crossing >= 0.08 or feature.centroid >= 2800)
    ):
        candidates.append(
            TechniqueCandidate(
                "slap",
                confidence_score(
                    0.58
                    + feature.onset * 0.22
                    + min(1.0, feature.centroid / 4000) * 0.12
                    + min(1.0, feature.zero_crossing / 0.2) * 0.08
                ),
                ["hard-onset", "bright-or-noisy-transient", "short-event"],
            )
        )

    if (
        current.string >= 5
        and duration <= 0.22
        and feature.onset >= 0.34
        and feature.centroid <= 2400
    ):
        candidates.append(
            TechniqueCandidate(
                "palm-mute",
                confidence_score(
                    0.5
                    + feature.onset * 0.16
                    + (1 - min(1.0, feature.centroid / 2400)) * 0.2
                    + (0.12 if current.string >= 5 else 0)
                ),
                ["short-low-string-note", "damped-spectrum"],
            )
        )

    if (
        duration < 0.13
        and current.velocity < 58
        and feature.flatness >= 0.18
    ):
        candidates.append(
            TechniqueCandidate(
                "dead-note",
                confidence_score(
                    0.54
                    + feature.flatness * 0.22
                    + (1 - current.velocity / 127) * 0.16
                ),
                ["very-short-event", "weak-pitched-content", "noise-component"],
            )
        )

    if previous is None:
        return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)[:3]

    gap = current.start - previous.end
    onset_gap = current.start - previous.start
    fret_delta = current.fret - previous.fret
    same_string = current.string == previous.string
    connected = -0.08 <= gap <= 0.16 and 0.06 <= onset_gap <= 0.55
    weak_onset = feature.onset <= 0.34

    if same_string and onset_gap < 0.18 and abs(fret_delta) <= 1 and feature.onset >= 0.42:
        candidates.append(
            TechniqueCandidate(
                "tremolo",
                confidence_score(0.76 + feature.onset * 0.18),
                ["same-string-repeat", "short-onset-gap", "rearticulated-onset"],
            )
        )
    if same_string and connected and weak_onset and 0 < fret_delta <= 4:
        candidates.append(
            TechniqueCandidate(
                "hammer-on",
                confidence_score(0.7 + (0.34 - feature.onset) * 0.45 + 0.12),
                ["same-string", "connected-notes", "weak-second-onset", "ascending-fret"],
            )
        )
    if same_string and connected and weak_onset and -4 <= fret_delta < 0:
        candidates.append(
            TechniqueCandidate(
                "pull-off",
                confidence_score(0.7 + (0.34 - feature.onset) * 0.45 + 0.12),
                ["same-string", "connected-notes", "weak-second-onset", "descending-fret"],
            )
        )
    if same_string and connected and weak_onset and 3 <= abs(fret_delta) <= 7:
        candidates.append(
            TechniqueCandidate(
                "slide",
                confidence_score(
                    0.72
                    + min(0.16, abs(fret_delta) * 0.025)
                    + (0.34 - feature.onset) * 0.28
                    + (
                        0.1
                        if feature.onset <= 0.22
                        and gap <= 0.08
                        and abs(fret_delta) <= 7
                        else 0
                    )
                ),
                ["same-string", "connected-notes", "weak-second-onset", "large-fret-change"],
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)[:3]


def infer_technique(
    previous: NoteEvent | None,
    current: NoteEvent,
    feature: AudioFeature,
) -> str:
    return infer_technique_candidates(previous, current, feature)[0].technique


def select_legato_predecessor(
    events: list[NoteEvent],
    current_index: int,
    feature: AudioFeature,
) -> tuple[int, NoteEvent, int] | None:
    if feature.onset > 0.34:
        return None
    current = events[current_index]
    candidates: list[tuple[float, int, NoteEvent, int]] = []
    for previous_index in range(current_index - 1, max(-1, current_index - 16), -1):
        previous = events[previous_index]
        onset_gap = current.start - previous.start
        if onset_gap > 0.5:
            break
        if onset_gap < 0.06:
            continue
        gap = current.start - previous.end
        if not -0.08 <= gap <= 0.16:
            continue
        target_fret = current.pitch - STANDARD_TUNING[previous.string]
        if not 0 <= target_fret <= 24:
            continue
        fret_delta = target_fret - previous.fret
        if fret_delta == 0:
            continue
        hammer_or_pull = abs(fret_delta) <= 4
        slide = (
            3 <= abs(fret_delta) <= 7
            and feature.onset <= 0.22
            and gap <= 0.08
        )
        if not hammer_or_pull and not slide:
            continue
        score = (
            abs(gap) * 2
            + onset_gap * 0.15
            + abs(fret_delta) * 0.04
            + (0 if previous.string == current.string else 0.18)
        )
        candidates.append((score, previous_index, previous, target_fret))
    if not candidates:
        return None
    _score, previous_index, previous, target_fret = min(
        candidates,
        key=lambda candidate: candidate[0],
    )
    return previous_index, previous, target_fret


def annotate_techniques(
    events: list[NoteEvent],
    features: list[AudioFeature],
    forced_harmonic_indices: set[int] | None = None,
    preserve_positions: bool = False,
) -> list[NoteEvent]:
    forced_harmonic_indices = forced_harmonic_indices or set()
    previous_by_string: dict[int, tuple[int, NoteEvent]] = {}
    relational_techniques = {"hammer-on", "pull-off", "slide"}
    transition_techniques = relational_techniques | {"tremolo"}
    for event_index, (event, feature) in enumerate(zip(events, features, strict=True)):
        original_position = (event.string, event.fret)
        previous_entry = previous_by_string.get(event.string)
        legato_entry = (
            None
            if preserve_positions
            else select_legato_predecessor(events, event_index, feature)
        )
        if legato_entry:
            previous_index, previous_event, target_fret = legato_entry
            event.string = previous_event.string
            event.fret = target_fret
            previous_entry = (previous_index, previous_event)
        previous_event = previous_entry[1] if previous_entry else None
        candidates = infer_technique_candidates(previous_event, event, feature)
        if event_index in forced_harmonic_indices:
            candidates = [
                TechniqueCandidate(
                    "harmonic",
                    0.94,
                    [
                        "harmonic-overtone-stack",
                        "coincident-partials",
                        "sustained-note",
                    ],
                ),
                *[
                    candidate
                    for candidate in candidates
                    if candidate.technique != "harmonic"
                ],
            ][:3]
        primary = candidates[0]
        if legato_entry and primary.technique in relational_techniques:
            event.position_confidence = max(event.position_confidence, 0.74)
            event.position_source = "legato-optimizer-v1"
        elif legato_entry:
            event.string, event.fret = original_position
            previous_entry = previous_by_string.get(event.string)
        event.technique = primary.technique
        event.technique_confidence = primary.confidence
        event.technique_candidates = candidates
        event.technique_evidence = primary.evidence
        event.technique_source = (
            "transition-heuristic-v2"
            if primary.technique in transition_techniques
            else "acoustic-heuristic-v2"
        )
        if primary.technique in relational_techniques and previous_entry:
            event.related_note_index = previous_entry[0]
        if primary.technique == "harmonic" and not preserve_positions:
            harmonic_position = select_harmonic_position(
                event.pitch,
                (event.string, event.fret),
            )
            if harmonic_position:
                (
                    event.string,
                    event.fret,
                    event.harmonic_type,
                    event.harmonic_touch_fret,
                    event.position_confidence,
                ) = harmonic_position
                event.position_source = "harmonic-optimizer-v1"
        previous_by_string[event.string] = (event_index, event)
    return events


def transcribe_basic_pitch(
    path: Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.7,
) -> tuple[list[NoteEvent], int]:
    from basic_pitch.inference import predict

    _model_output, midi_data, _raw_events = predict(
        str(path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
    )
    all_midi_notes = sorted(
        (note for instrument in midi_data.instruments for note in instrument.notes),
        key=lambda note: (note.start, note.pitch),
    )
    midi_notes = [
        note
        for note in all_midi_notes
        if GUITAR_PITCH_RANGE[0] <= int(note.pitch) <= GUITAR_PITCH_RANGE[1]
    ]
    dropped_note_count = len(all_midi_notes) - len(midi_notes)

    note_groups: list[list[object]] = []
    for midi_note in midi_notes:
        if (
            not note_groups
            or float(midi_note.start) - float(note_groups[-1][0].start) > 0.045
        ):
            note_groups.append([midi_note])
        else:
            note_groups[-1].append(midi_note)

    events: list[NoteEvent] = []
    previous_positions: list[tuple[int, int]] = []
    for group in note_groups:
        positions, position_confidence = assign_fingering_group(
            [int(note.pitch) for note in group],
            previous_positions,
        )
        for midi_note, (string, fret) in zip(group, positions, strict=True):
            events.append(
                NoteEvent(
                    start=round(float(midi_note.start), 4),
                    end=round(float(midi_note.end), 4),
                    pitch=int(midi_note.pitch),
                    velocity=int(midi_note.velocity),
                    string=string,
                    fret=fret,
                    technique="pick",
                    confidence=min(0.98, 0.54 + midi_note.velocity / 220),
                    position_confidence=position_confidence,
                )
            )
        previous_positions = positions
    events = collapse_note_fragments(events)
    events, forced_harmonic_indices = collapse_harmonic_stacks(events)
    features = extract_note_features(path, events)
    return (
        annotate_techniques(events, features, forced_harmonic_indices),
        dropped_note_count,
    )


def transcribe_string_model(
    path: Path,
    checkpoint_path: Path,
) -> tuple[list[NoteEvent], dict[str, object]]:
    from stringtrace_ml.inference import transcribe_audio

    predictions, metadata = transcribe_audio(
        path,
        checkpoint_path,
        device_name=os.environ.get("STRINGTRACE_MODEL_DEVICE", "cpu"),
    )
    events = [
        NoteEvent(
            start=prediction.onset,
            end=prediction.offset,
            pitch=STANDARD_TUNING[prediction.string] + prediction.fret,
            velocity=round(40 + prediction.confidence * 80),
            string=prediction.string,
            fret=prediction.fret,
            technique="pick",
            confidence=prediction.confidence,
            position_confidence=prediction.confidence,
            position_source="string-fret-model-v1",
        )
        for prediction in predictions
    ]
    features = extract_note_features(path, events)
    return (
        annotate_techniques(
            events,
            features,
            preserve_positions=True,
        ),
        {
            "name": metadata.get("name", "stringtrace-string-fret-net"),
            "version": metadata.get("version", "unknown"),
            "kind": "trained",
            "checkpoint": checkpoint_path.name,
        },
    )


def transcribe_hybrid_model(
    path: Path,
    checkpoint_path: Path,
) -> tuple[list[NoteEvent], int, dict[str, object]]:
    import torch

    from stringtrace_ml.features import extract_file_cqt
    from stringtrace_ml.hybrid import (
        PitchEvent,
        PositionAssignmentConfig,
        assign_pitch_events,
    )
    from stringtrace_ml.inference import (
        load_checkpoint,
        predict_probabilities_with_position,
    )

    notes, dropped_note_count = transcribe_basic_pitch(path)
    pitch_events = [
        PitchEvent(
            onset=note.start,
            offset=note.end,
            pitch=note.pitch,
            confidence=note.confidence,
        )
        for note in notes
    ]
    device = torch.device(os.environ.get("STRINGTRACE_MODEL_DEVICE", "cpu"))
    loaded = load_checkpoint(checkpoint_path, device)
    if not loaded.model.config.pitch_conditioned_position:
        raise ValueError(
            "Hybrid transcription requires a pitch-conditioned "
            "StringTrace checkpoint"
        )
    features = extract_file_cqt(path, loaded.feature_config)
    (
        fret_probabilities,
        onset_probabilities,
        position_probabilities,
    ) = predict_probabilities_with_position(
        loaded,
        features,
        device=device,
    )
    assigned = assign_pitch_events(
        pitch_events,
        fret_probabilities,
        onset_probabilities,
        loaded.feature_config,
        position_probabilities=position_probabilities,
        config=PositionAssignmentConfig(
            position_head_weight=4.0,
            continuity_weight=0.75,
        ),
    )
    assigned_by_key: dict[tuple[float, int], list[object]] = {}
    for event in assigned:
        key = (
            event.onset,
            STANDARD_TUNING[event.string] + event.fret,
        )
        assigned_by_key.setdefault(key, []).append(event)

    positioned: list[NoteEvent] = []
    for note in notes:
        key = (note.start, note.pitch)
        candidates = assigned_by_key.get(key, [])
        if not candidates:
            continue
        assignment = candidates.pop(0)
        positioned.append(
            replace(
                note,
                string=assignment.string,
                fret=assignment.fret,
                technique="pick",
                technique_confidence=0.0,
                technique_source="acoustic-heuristic-v4",
                technique_evidence=[],
                technique_candidates=[],
                related_note_index=None,
                harmonic_type=None,
                harmonic_touch_fret=None,
                position_confidence=assignment.confidence,
                position_source="hybrid-pitch-conditioned-v1",
            )
        )
    positioned_features = extract_note_features(path, positioned)
    return (
        annotate_techniques(
            positioned,
            positioned_features,
            preserve_positions=True,
        ),
        dropped_note_count,
        {
            "name": "basic-pitch-stringtrace-hybrid",
            "version": loaded.metadata.get(
                "version",
                "tabcnn-gru-v3-pitch-conditioned",
            ),
            "kind": "hybrid",
            "pitch_model": "spotify-basic-pitch-0.4",
            "position_model": loaded.metadata.get(
                "name",
                "stringtrace-string-fret-net",
            ),
            "checkpoint": checkpoint_path.name,
        },
    )


def transcribe_guitar(
    path: Path,
    checkpoint_path: Path | None = None,
    model_mode: str = "direct",
) -> tuple[list[NoteEvent], int, dict[str, object]]:
    if checkpoint_path:
        if model_mode == "hybrid":
            return transcribe_hybrid_model(path, checkpoint_path)
        if model_mode != "direct":
            raise ValueError(f"Unknown trained model mode: {model_mode}")
        notes, model_info = transcribe_string_model(path, checkpoint_path)
        return notes, 0, model_info
    notes, dropped_note_count = transcribe_basic_pitch(path)
    return (
        notes,
        dropped_note_count,
        {
            "name": "spotify-basic-pitch",
            "version": "0.4",
            "kind": "baseline",
        },
    )


def annotate_with_technique_model(
    path: Path,
    events: list[NoteEvent],
    checkpoint_path: Path,
    gate_path: Path,
) -> tuple[list[NoteEvent], dict[str, object]]:
    import torch

    from stringtrace_ml.technique_inference import predict_techniques

    device = torch.device(
        os.environ.get("STRINGTRACE_TECHNIQUE_DEVICE", "cpu")
    )
    decisions, model_info = predict_techniques(
        path,
        events,
        checkpoint_path,
        gate_path,
        device=device,
    )
    for event, decision in zip(events, decisions):
        event.technique = decision.technique
        event.technique_confidence = decision.confidence
        event.technique_source = decision.source
        event.technique_evidence = decision.evidence
        event.technique_candidates = [
            TechniqueCandidate(
                technique=label,
                confidence=confidence,
                evidence=["model-candidate"],
            )
            for label, confidence in decision.candidates
        ]
        event.related_note_index = None
        event.harmonic_type = None
        event.harmonic_touch_fret = None
    return events, model_info


def annotate_with_technique_bundle(
    path: Path,
    events: list[NoteEvent],
    bundle_path: Path,
) -> tuple[list[NoteEvent], dict[str, object]]:
    import torch

    from stringtrace_ml.technique_inference import predict_technique_bundle

    device = torch.device(
        os.environ.get("STRINGTRACE_TECHNIQUE_DEVICE", "cpu")
    )
    decisions, model_info = predict_technique_bundle(
        path,
        events,
        bundle_path,
        device=device,
    )
    for event, decision in zip(events, decisions):
        event.technique = decision.technique
        event.technique_confidence = decision.confidence
        event.technique_source = decision.source
        event.technique_evidence = decision.evidence
        event.technique_candidates = [
            TechniqueCandidate(
                technique=label,
                confidence=confidence,
                evidence=["model-candidate"],
            )
            for label, confidence in decision.candidates
        ]
        event.related_note_index = None
        event.harmonic_type = (
            "natural" if decision.technique == "harmonic" else None
        )
        event.harmonic_touch_fret = None
    return events, model_info


def chord_score(pitch_classes: set[int], root: int, intervals: tuple[int, ...]) -> float:
    chord = {(root + interval) % 12 for interval in intervals}
    matched = len(chord & pitch_classes)
    foreign = len(pitch_classes - chord)
    return matched * 2.0 - foreign * 0.35


def infer_chords(notes: Iterable[NoteEvent], bpm: float) -> list[dict[str, object]]:
    note_list = list(notes)
    if not note_list:
        return []
    beat = 60 / bpm
    window = beat * 4
    duration = max(note.end for note in note_list)
    result: list[dict[str, object]] = []
    start = 0.0

    qualities = (("", (0, 4, 7)), ("m", (0, 3, 7)), ("7", (0, 4, 7, 10)))
    while start < duration:
        active = {
            note.pitch % 12
            for note in note_list
            if note.start < start + window and note.end > start
        }
        if len(active) < 2:
            result.append(
                {
                    "start": round(start, 4),
                    "duration": round(window, 4),
                    "chord": "N.C.",
                }
            )
            start += window
            continue
        best_name = "N.C."
        best_score = -100.0
        for root in range(12):
            for suffix, intervals in qualities:
                score = chord_score(active, root, intervals)
                if score > best_score:
                    best_score = score
                    best_name = f"{PITCH_NAMES[root]}{suffix}"
        result.append({"start": round(start, 4), "duration": round(window, 4), "chord": best_name})
        start += window
    return result


def json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path)
    source_group.add_argument("--url")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tab-model",
        type=Path,
        help="Optional trained StringTrace string/fret checkpoint.",
    )
    parser.add_argument(
        "--tab-model-mode",
        choices=("direct", "hybrid"),
        default=os.environ.get("STRINGTRACE_TAB_MODEL_MODE", "direct"),
        help="Use direct StringTrace decoding or Basic Pitch plus StringTrace.",
    )
    parser.add_argument(
        "--technique-model",
        type=Path,
        help="Optional calibrated note-technique checkpoint.",
    )
    parser.add_argument(
        "--technique-gate",
        type=Path,
        help="Required fixed validation/test gate for --technique-model.",
    )
    parser.add_argument(
        "--technique-bundle",
        type=Path,
        help="Optional selective multi-model technique bundle.",
    )
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if arguments.url:
        source, metadata = acquire_url(arguments.url, output)
    else:
        source = arguments.input.resolve()
        metadata = {"title": source.stem, "artist": "本地上传"}
    if not source.exists():
        raise FileNotFoundError(source)

    normalized = normalize_audio(source, output)
    stems = separate_audio(normalized, output)
    bpm, beat_times, onset_times = estimate_tempo(source)
    note_source, source_warning = transcription_source(normalized, stems)
    configured_model = arguments.tab_model
    if configured_model is None and os.environ.get("STRINGTRACE_TAB_MODEL"):
        configured_model = Path(os.environ["STRINGTRACE_TAB_MODEL"])
    if configured_model is not None:
        configured_model = configured_model.resolve()
        if not configured_model.is_file():
            raise FileNotFoundError(configured_model)
    notes, dropped_note_count, transcription_model = transcribe_guitar(
        note_source,
        configured_model,
        model_mode=arguments.tab_model_mode,
    )
    configured_technique_model = arguments.technique_model
    configured_technique_gate = arguments.technique_gate
    configured_technique_bundle = arguments.technique_bundle
    if (
        configured_technique_model is None
        and os.environ.get("STRINGTRACE_TECHNIQUE_MODEL")
    ):
        configured_technique_model = Path(
            os.environ["STRINGTRACE_TECHNIQUE_MODEL"]
        )
    if (
        configured_technique_gate is None
        and os.environ.get("STRINGTRACE_TECHNIQUE_GATE")
    ):
        configured_technique_gate = Path(
            os.environ["STRINGTRACE_TECHNIQUE_GATE"]
        )
    if (
        configured_technique_bundle is None
        and os.environ.get("STRINGTRACE_TECHNIQUE_BUNDLE")
    ):
        configured_technique_bundle = Path(
            os.environ["STRINGTRACE_TECHNIQUE_BUNDLE"]
        )
    if configured_technique_bundle is not None and (
        configured_technique_model is not None
        or configured_technique_gate is not None
    ):
        raise ValueError(
            "Technique bundle cannot be combined with a single model/gate"
        )
    if (configured_technique_model is None) != (
        configured_technique_gate is None
    ):
        raise ValueError(
            "Technique model and gate must be configured together"
        )
    if configured_technique_bundle is not None:
        configured_technique_bundle = configured_technique_bundle.resolve()
        if not configured_technique_bundle.is_file():
            raise FileNotFoundError(configured_technique_bundle)
        notes, technique_model = annotate_with_technique_bundle(
            note_source,
            notes,
            configured_technique_bundle,
        )
    elif configured_technique_model is not None:
        configured_technique_model = configured_technique_model.resolve()
        configured_technique_gate = configured_technique_gate.resolve()
        if not configured_technique_model.is_file():
            raise FileNotFoundError(configured_technique_model)
        if not configured_technique_gate.is_file():
            raise FileNotFoundError(configured_technique_gate)
        notes, technique_model = annotate_with_technique_model(
            note_source,
            notes,
            configured_technique_model,
            configured_technique_gate,
        )
    else:
        technique_model = {
            "name": "stringtrace-technique-baseline",
            "version": "heuristic-v4",
            "kind": "heuristic",
            "labels": list(TECHNIQUE_LABELS),
        }
    levels = {name: audio_rms(path) for name, path in stems.items()}
    waveforms = {name: waveform(path) for name, path in stems.items()}
    visualization = visualization_timeline(normalized)
    playable_stems = compress_stems(stems, output)
    normalized.unlink(missing_ok=True)
    warnings = []
    if (
        configured_technique_model is None
        and configured_technique_bundle is None
    ):
        warnings.append(
            "Technique labels are experimental acoustic heuristics, "
            "not reliable classifications."
        )
    else:
        warnings.append(
            "Only independently gated technique labels are applied; "
            "all other predictions remain unknown."
        )
    if configured_model is None:
        warnings.append(
            "String and fret positions use the Basic Pitch baseline and require review."
        )
    else:
        warnings.append(
            "String and fret positions use a trained model; publish only after benchmark gates pass."
        )
    if source_warning:
        warnings.append(source_warning)
    if dropped_note_count:
        warnings.append(
            f"{dropped_note_count} note events outside the 24-fret guitar range were discarded."
        )
    result = {
        "version": 9,
        "transcription_model": transcription_model,
        "technique_model": technique_model,
        "metadata": metadata,
        "bpm": bpm,
        "beats": beat_times,
        "onsets": onset_times,
        "stems": {name: str(path.resolve()) for name, path in playable_stems.items()},
        "levels": levels,
        "waveforms": waveforms,
        "visualization": visualization,
        "notes": [asdict(note) for note in notes],
        "chords": infer_chords(notes, bpm),
        "warnings": warnings,
    }
    result_path = output / "analysis.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps({"analysis": str(result_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
