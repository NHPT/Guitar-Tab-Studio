from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import soundfile

from .schema import TrackAnnotation, load_manifest, write_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def track_family(track_id: str) -> str:
    parts = track_id.split("_")
    if len(parts) < 2:
        return "unknown"
    match = re.match(r"[A-Za-z]+", parts[1])
    return match.group(0) if match else parts[1]


def track_role(track_id: str) -> str:
    return track_id.rsplit("_", 1)[-1]


def stratified_selection(
    annotations: list[TrackAnnotation],
    limit: int | None,
) -> list[TrackAnnotation]:
    ordered = sorted(annotations, key=lambda annotation: annotation.track_id)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit <= 0:
        return []

    buckets: dict[tuple[str, str, str], list[TrackAnnotation]] = {}
    for annotation in ordered:
        key = (
            annotation.group_id,
            track_family(annotation.track_id),
            track_role(annotation.track_id),
        )
        buckets.setdefault(key, []).append(annotation)

    selected: list[TrackAnnotation] = []
    keys = sorted(buckets)
    depth = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            bucket = buckets[key]
            if depth >= len(bucket):
                continue
            selected.append(bucket[depth])
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
        depth += 1
    return sorted(selected, key=lambda annotation: annotation.track_id)


def validate_stem(path: Path, expected_duration: float) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing separated guitar stem: {path}")
    info = soundfile.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise RuntimeError(f"Invalid separated guitar stem: {path}")
    duration = info.frames / info.samplerate
    if abs(duration - expected_duration) > 0.1:
        raise RuntimeError(
            f"Stem duration mismatch for {path.name}: "
            f"{duration:.4f}s vs {expected_duration:.4f}s"
        )


def run_demucs(
    annotations: list[TrackAnnotation],
    source_manifest: Path,
    output_directory: Path,
    *,
    model: str,
    device: str,
    batch_size: int,
) -> dict[str, Path]:
    audio_directory = output_directory / "audio"
    work_directory = output_directory / ".demucs-work"
    audio_directory.mkdir(parents=True, exist_ok=True)
    work_directory.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    missing: list[TrackAnnotation] = []
    for annotation in annotations:
        target = audio_directory / f"{annotation.track_id}.wav"
        if target.is_file():
            validate_stem(target, annotation.duration)
            resolved[annotation.track_id] = target
        else:
            missing.append(annotation)

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        sources = [
            annotation.resolve_audio(source_manifest)
            for annotation in batch
        ]
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
                str(work_directory),
                *(str(source) for source in sources),
            ],
            check=True,
        )
        for annotation, source in zip(batch, sources):
            generated = (
                work_directory
                / model
                / source.stem
                / "guitar.wav"
            )
            validate_stem(generated, annotation.duration)
            target = audio_directory / f"{annotation.track_id}.wav"
            generated.replace(target)
            shutil.rmtree(generated.parent)
            resolved[annotation.track_id] = target
        print(
            json.dumps(
                {
                    "stage": "separation",
                    "completed": min(start + len(batch), len(missing)),
                    "remaining": max(0, len(missing) - start - len(batch)),
                }
            ),
            flush=True,
        )
    shutil.rmtree(work_directory, ignore_errors=True)
    return resolved


def derived_annotation(
    annotation: TrackAnnotation,
    stem_path: Path,
    source_manifest: Path,
    output_manifest: Path,
    *,
    model: str,
) -> TrackAnnotation:
    source_audio = annotation.resolve_audio(source_manifest)
    relative_audio = Path(
        os.path.relpath(stem_path, output_manifest.parent)
    )
    return replace(
        annotation,
        track_id=f"{annotation.track_id}-demucs-guitar",
        audio=str(relative_audio),
        provenance={
            **annotation.provenance,
            "derived_from_track_id": annotation.track_id,
            "derived_from_audio_sha256": sha256_file(source_audio),
            "audio_sha256": sha256_file(stem_path),
            "audio_condition": "demucs-guitar-stem",
            "separation_model": model,
            "separation_mode": "two-stems-guitar",
            "annotation_alignment": "sample-aligned-no-offset",
        },
    )


def relocated_source_annotation(
    annotation: TrackAnnotation,
    source_manifest: Path,
    output_manifest: Path,
) -> TrackAnnotation:
    relative_audio = Path(
        os.path.relpath(
            annotation.resolve_audio(source_manifest),
            output_manifest.parent,
        )
    )
    return replace(annotation, audio=str(relative_audio))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a labeled GuitarSet derivative with production-like "
            "Demucs guitar stems."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="htdemucs_6s")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=60)
    parser.add_argument("--validation-limit", type=int, default=60)
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include the selected original recordings beside each stem.",
    )
    arguments = parser.parse_args()

    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if arguments.train_limit < 0 or arguments.validation_limit < 0:
        parser.error("split limits must be non-negative")

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
    stems = run_demucs(
        selected,
        source_manifest,
        output_directory,
        model=arguments.model,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    derived = [
        derived_annotation(
            annotation,
            stems[annotation.track_id],
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
        "selection": {
            "method": "round-robin-group-family-role",
            "requested_train_limit": arguments.train_limit,
            "requested_validation_limit": arguments.validation_limit,
            "include_source": arguments.include_source,
        },
        "source_tracks": len(selected) if arguments.include_source else 0,
        "derived_tracks": len(derived),
        "tracks": len(output_annotations),
        "splits": split_counts,
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
