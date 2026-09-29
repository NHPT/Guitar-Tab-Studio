from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile

from .prepare_guitarset_stems import (
    relocated_source_annotation,
    sha256_file,
    stratified_selection,
    validate_stem,
)
from .schema import TrackAnnotation, load_manifest, write_manifest


GENERATOR_VERSION = "procedural-accompaniment-v1"


def stable_track_seed(seed: int, track_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{track_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _pan(signal: np.ndarray, pan: float) -> np.ndarray:
    angle = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
    return np.column_stack((signal * math.cos(angle), signal * math.sin(angle)))


def procedural_accompaniment(
    frame_count: int,
    sample_rate: int,
    *,
    seed: int,
) -> np.ndarray:
    if frame_count <= 0 or sample_rate <= 0:
        raise ValueError("frame_count and sample_rate must be positive")
    rng = np.random.default_rng(seed)
    accompaniment = np.zeros((frame_count, 2), dtype=np.float32)
    tempo = float(rng.uniform(84, 156))
    beat_seconds = 60 / tempo
    beat_samples = max(1, round(beat_seconds * sample_rate))
    bass_notes = np.asarray((41.2, 49.0, 55.0, 65.4, 73.4, 82.4))
    chord_roots = np.asarray((110.0, 123.47, 146.83, 164.81, 196.0))

    for beat_index, start in enumerate(range(0, frame_count, beat_samples)):
        bass_length = min(
            frame_count - start,
            max(1, round(beat_samples * 0.82)),
        )
        bass_time = np.arange(bass_length, dtype=np.float32) / sample_rate
        bass_frequency = float(rng.choice(bass_notes))
        bass_envelope = np.exp(-3.2 * bass_time / max(beat_seconds, 1e-6))
        bass = (
            np.sin(2 * np.pi * bass_frequency * bass_time)
            + 0.28 * np.sin(4 * np.pi * bass_frequency * bass_time)
        ) * bass_envelope
        accompaniment[start : start + bass_length] += _pan(
            bass.astype(np.float32) * 0.5,
            float(rng.uniform(-0.35, 0.35)),
        )

        drum_length = min(
            frame_count - start,
            max(1, round(0.18 * sample_rate)),
        )
        drum_time = np.arange(drum_length, dtype=np.float32) / sample_rate
        if beat_index % 4 in (0, 2):
            phase = 2 * np.pi * (
                72 * drum_time - 18 * drum_time * drum_time
            )
            drum = np.sin(phase) * np.exp(-28 * drum_time)
            accompaniment[start : start + drum_length] += _pan(
                drum.astype(np.float32) * 0.9,
                0,
            )
        else:
            noise = rng.standard_normal(drum_length).astype(np.float32)
            high_passed = np.diff(noise, prepend=noise[0])
            snare = high_passed * np.exp(-22 * drum_time)
            accompaniment[start : start + drum_length] += _pan(
                snare.astype(np.float32) * 0.22,
                float(rng.choice((-0.35, 0.35))),
            )

        half_beat = max(1, beat_samples // 2)
        for hat_start in (start, start + half_beat):
            if hat_start >= frame_count:
                continue
            hat_length = min(
                frame_count - hat_start,
                max(1, round(0.055 * sample_rate)),
            )
            hat_time = np.arange(hat_length, dtype=np.float32) / sample_rate
            noise = rng.standard_normal(hat_length).astype(np.float32)
            hat = np.diff(noise, prepend=noise[0]) * np.exp(-75 * hat_time)
            accompaniment[hat_start : hat_start + hat_length] += _pan(
                hat.astype(np.float32) * 0.10,
                float(rng.uniform(-0.9, 0.9)),
            )

        if beat_index % 4 == 0:
            pad_length = min(
                frame_count - start,
                max(1, round(beat_samples * 3.8)),
            )
            pad_time = np.arange(pad_length, dtype=np.float32) / sample_rate
            root = float(rng.choice(chord_roots))
            attack = np.minimum(1.0, pad_time / 0.08)
            release = np.minimum(
                1.0,
                np.maximum(0.0, (pad_length / sample_rate - pad_time) / 0.2),
            )
            envelope = attack * release
            for interval, pan in ((1.0, -0.55), (1.25, 0.0), (1.5, 0.55)):
                tone = (
                    np.sin(2 * np.pi * root * interval * pad_time)
                    + 0.18
                    * np.sin(4 * np.pi * root * interval * pad_time)
                )
                accompaniment[start : start + pad_length] += _pan(
                    (tone * envelope * 0.10).astype(np.float32),
                    pan,
                )
    return accompaniment


def mix_source_audio(
    source_path: Path,
    mixture_path: Path,
    *,
    seed: int,
    snr_db: float,
) -> dict[str, float | int | str]:
    source, sample_rate = soundfile.read(
        source_path,
        dtype="float32",
        always_2d=True,
    )
    if source.shape[1] == 1:
        source = np.repeat(source, 2, axis=1)
    elif source.shape[1] > 2:
        source = np.repeat(source.mean(axis=1, keepdims=True), 2, axis=1)
    accompaniment = procedural_accompaniment(
        len(source),
        sample_rate,
        seed=seed,
    )
    source_rms = float(np.sqrt(np.mean(np.square(source))))
    accompaniment_rms = float(
        np.sqrt(np.mean(np.square(accompaniment)))
    )
    target_accompaniment_rms = source_rms / (10 ** (snr_db / 20))
    accompaniment *= target_accompaniment_rms / max(
        accompaniment_rms,
        1e-8,
    )
    scaled_accompaniment_rms = float(
        np.sqrt(np.mean(np.square(accompaniment)))
    )
    mixture = source + accompaniment
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.98:
        mixture *= 0.98 / peak
    mixture_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(
        mixture_path,
        mixture,
        sample_rate,
        subtype="PCM_16",
    )
    return {
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "target_snr_db": snr_db,
        "source_rms": source_rms,
        "accompaniment_rms": scaled_accompaniment_rms,
        "realized_snr_db": (
            20
            * math.log10(
                max(source_rms, 1e-8)
                / max(scaled_accompaniment_rms, 1e-8)
            )
        ),
        "mixture_sha256": sha256_file(mixture_path),
    }


def parse_snr_values(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("SNR values must be comma-separated numbers") from error
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError("At least one finite SNR value is required")
    return values


def build_mixture_stems(
    annotations: list[TrackAnnotation],
    source_manifest: Path,
    output_directory: Path,
    *,
    model: str,
    device: str,
    batch_size: int,
    seed: int,
    snr_values: Sequence[float],
) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    audio_directory = output_directory / "audio"
    metadata_directory = output_directory / "metadata"
    work_directory = output_directory / ".demucs-work"
    mixture_directory = work_directory / "mixtures"
    audio_directory.mkdir(parents=True, exist_ok=True)
    metadata_directory.mkdir(parents=True, exist_ok=True)
    work_directory.mkdir(parents=True, exist_ok=True)
    stems: dict[str, Path] = {}
    metadata: dict[str, dict[str, object]] = {}
    missing: list[TrackAnnotation] = []
    for annotation in annotations:
        track_seed = stable_track_seed(seed, annotation.track_id)
        snr_db = float(snr_values[track_seed % len(snr_values)])
        target = audio_directory / f"{annotation.track_id}.wav"
        metadata_path = metadata_directory / f"{annotation.track_id}.json"
        if target.is_file() and metadata_path.is_file():
            item_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if (
                item_metadata.get("generator") == GENERATOR_VERSION
                and int(item_metadata.get("seed", -1)) == track_seed
                and float(item_metadata.get("target_snr_db", math.nan))
                == snr_db
                and item_metadata.get("separation_model") == model
            ):
                validate_stem(target, annotation.duration)
                stems[annotation.track_id] = target
                metadata[annotation.track_id] = item_metadata
                continue
        missing.append(annotation)

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        mixtures: list[Path] = []
        batch_metadata: dict[str, dict[str, object]] = {}
        for annotation in batch:
            track_seed = stable_track_seed(seed, annotation.track_id)
            snr_db = float(snr_values[track_seed % len(snr_values)])
            mixture_path = mixture_directory / f"{annotation.track_id}.wav"
            mix_metadata = mix_source_audio(
                annotation.resolve_audio(source_manifest),
                mixture_path,
                seed=track_seed,
                snr_db=snr_db,
            )
            mixtures.append(mixture_path)
            batch_metadata[annotation.track_id] = {
                **mix_metadata,
                "separation_model": model,
                "source_audio_sha256": sha256_file(
                    annotation.resolve_audio(source_manifest)
                ),
            }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                model,
                "--two-stems",
                "guitar",
                "--other-method",
                "none",
                "--device",
                device,
                "--out",
                str(work_directory / "stems"),
                *(str(path) for path in mixtures),
            ],
            check=True,
        )
        for annotation, mixture_path in zip(batch, mixtures):
            generated = (
                work_directory
                / "stems"
                / model
                / mixture_path.stem
                / "guitar.wav"
            )
            validate_stem(generated, annotation.duration)
            target = audio_directory / f"{annotation.track_id}.wav"
            generated.replace(target)
            item_metadata = {
                **batch_metadata[annotation.track_id],
                "stem_sha256": sha256_file(target),
            }
            metadata_path = (
                metadata_directory / f"{annotation.track_id}.json"
            )
            metadata_path.write_text(
                json.dumps(item_metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            mixture_path.unlink()
            shutil.rmtree(generated.parent)
            stems[annotation.track_id] = target
            metadata[annotation.track_id] = item_metadata
        print(
            json.dumps(
                {
                    "stage": "mixture_separation",
                    "completed": min(start + len(batch), len(missing)),
                    "remaining": max(0, len(missing) - start - len(batch)),
                }
            ),
            flush=True,
        )
    shutil.rmtree(work_directory, ignore_errors=True)
    return stems, metadata


def derived_annotation(
    annotation: TrackAnnotation,
    stem_path: Path,
    mixture_metadata: dict[str, object],
    source_manifest: Path,
    output_manifest: Path,
    *,
    model: str,
) -> TrackAnnotation:
    relative_audio = Path(
        os.path.relpath(stem_path, output_manifest.parent)
    )
    return replace(
        annotation,
        track_id=f"{annotation.track_id}-mixture-demucs-guitar",
        audio=str(relative_audio),
        provenance={
            **annotation.provenance,
            "derived_from_track_id": annotation.track_id,
            "derived_from_audio_sha256": mixture_metadata[
                "source_audio_sha256"
            ],
            "mixture_sha256": mixture_metadata["mixture_sha256"],
            "audio_sha256": mixture_metadata["stem_sha256"],
            "audio_condition": "procedural-mixture-demucs-guitar-stem",
            "separation_model": model,
            "separation_mode": "two-stems-guitar",
            "annotation_alignment": "sample-aligned-no-offset",
            "mixture_generator": GENERATOR_VERSION,
            "mixture_seed": mixture_metadata["seed"],
            "target_snr_db": mixture_metadata["target_snr_db"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mix GuitarSet with deterministic non-guitar accompaniment and "
            "extract production-like guitar stems."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="htdemucs_6s")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=40)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--snr-db", default="-3,0,3")
    parser.add_argument("--include-source", action="store_true")
    arguments = parser.parse_args()

    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if arguments.train_limit < 0 or arguments.validation_limit < 0:
        parser.error("split limits must be non-negative")
    try:
        snr_values = parse_snr_values(arguments.snr_db)
    except ValueError as error:
        parser.error(str(error))

    source_manifest = arguments.manifest.resolve()
    output_directory = arguments.output_directory.resolve()
    output_manifest = arguments.output_manifest.resolve()
    annotations = load_manifest(source_manifest)
    selected = (
        stratified_selection(
            [
                annotation
                for annotation in annotations
                if annotation.split == "train"
            ],
            arguments.train_limit,
        )
        + stratified_selection(
            [
                annotation
                for annotation in annotations
                if annotation.split == "validation"
            ],
            arguments.validation_limit,
        )
    )
    stems, metadata = build_mixture_stems(
        selected,
        source_manifest,
        output_directory,
        model=arguments.model,
        device=arguments.device,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        snr_values=snr_values,
    )
    derived = [
        derived_annotation(
            annotation,
            stems[annotation.track_id],
            metadata[annotation.track_id],
            source_manifest,
            output_manifest,
            model=arguments.model,
        )
        for annotation in selected
    ]
    output_annotations = (
        [
            relocated_source_annotation(
                annotation,
                source_manifest,
                output_manifest,
            )
            for annotation in selected
        ]
        + derived
        if arguments.include_source
        else derived
    )
    output_annotations.sort(
        key=lambda annotation: (annotation.split, annotation.track_id)
    )
    write_manifest(output_manifest, output_annotations)
    split_counts = {
        split: sum(
            annotation.split == split
            for annotation in output_annotations
        )
        for split in ("train", "validation")
    }
    snr_counts = {
        str(value): sum(
            float(item["target_snr_db"]) == value
            for item in metadata.values()
        )
        for value in snr_values
    }
    report = {
        "schema_version": 1,
        "source_manifest": {
            "name": arguments.manifest.name,
            "sha256": sha256_file(source_manifest),
        },
        "derived_manifest": {
            "name": output_manifest.name,
            "sha256": sha256_file(output_manifest),
        },
        "separation_model": arguments.model,
        "separation_mode": "two-stems-guitar",
        "annotation_alignment": "sample-aligned-no-offset",
        "mixture_generator": GENERATOR_VERSION,
        "selection": {
            "method": "round-robin-group-family-role",
            "requested_train_limit": arguments.train_limit,
            "requested_validation_limit": arguments.validation_limit,
            "include_source": arguments.include_source,
            "seed": arguments.seed,
            "snr_db": list(snr_values),
        },
        "source_tracks": len(selected) if arguments.include_source else 0,
        "derived_tracks": len(derived),
        "tracks": len(output_annotations),
        "splits": split_counts,
        "snr_counts": snr_counts,
        "audio_bytes": sum(path.stat().st_size for path in stems.values()),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
