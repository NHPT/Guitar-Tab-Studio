#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "worker"))

from stringtrace_ml.beat_reference import create_review  # noqa: E402
from worker import estimate_tempo, tempo_map_from_beats  # noqa: E402

CORPUS_VERSION = "beat-pilot-v2-2026-10-21"


@dataclass(frozen=True)
class Recording:
    slug: str
    split: str
    clips: int
    time_signature: tuple[int, int]
    tags: tuple[str, ...]


RECORDINGS = (
    Recording(
        "dust-in-the-wind",
        "train",
        4,
        (4, 4),
        ("acoustic", "fingerstyle", "sixteenth-notes"),
    ),
    Recording(
        "hotel-california",
        "train",
        5,
        (4, 4),
        ("full-mix", "strong-drums", "electric-guitar"),
    ),
    Recording(
        "little-wing",
        "train",
        5,
        (4, 4),
        ("clean-electric", "expressive-timing", "rubato"),
    ),
    Recording(
        "sultans-of-swing",
        "train",
        5,
        (4, 4),
        ("clean-electric", "strong-drums", "syncopated"),
    ),
    Recording(
        "tears-in-heaven",
        "train",
        5,
        (4, 4),
        ("acoustic", "fingerstyle", "full-mix"),
    ),
    Recording(
        "black-bird",
        "validation",
        5,
        (4, 4),
        ("acoustic", "fingerstyle", "mixed-meter-candidate"),
    ),
    Recording(
        "wish-you-were-here",
        "validation",
        5,
        (4, 4),
        ("acoustic", "strumming", "full-mix"),
    ),
    Recording(
        "internationale",
        "test",
        4,
        (4, 4),
        ("distorted-electric", "strong-drums", "tempo-change"),
    ),
    Recording(
        "nothing-else-matters",
        "test",
        5,
        (6, 8),
        ("clean-electric", "compound-meter", "arpeggio"),
    ),
    Recording(
        "stairway-to-heaven",
        "test",
        5,
        (4, 4),
        ("mixed-texture", "tempo-change", "expressive-timing"),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_starts(
    recording_duration: float,
    count: int,
    clip_duration: float,
) -> list[float]:
    first = 8.0
    last = recording_duration - clip_duration - 8.0
    if last < first:
        raise ValueError("Recording is too short for the requested pilot clips")
    if count == 1:
        return [round((first + last) / 2, 3)]
    step = (last - first) / (count - 1)
    return [round(first + index * step, 3) for index in range(count)]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def retarget_beat_grid(
    beats: list[float],
    target_bpm: float,
) -> tuple[float, list[float]]:
    values = [
        float(value)
        for value in beats
        if math.isfinite(float(value)) and float(value) >= 0
    ]
    if len(values) < 2 or any(
        right <= left for left, right in zip(values, values[1:])
    ):
        raise ValueError("Beat grid must contain increasing finite times")
    if not math.isfinite(target_bpm) or not 20 <= target_bpm <= 320:
        raise ValueError("Target BPM must be between 20 and 320")
    intervals = [
        right - left
        for left, right in zip(values, values[1:])
        if 0.08 <= right - left <= 3.0
    ]
    if not intervals:
        raise ValueError("Beat grid does not contain usable intervals")
    source_bpm = 60.0 / statistics.median(intervals)
    source_step = source_bpm / target_bpm
    resampled: list[float] = []
    position = 0.0
    maximum_position = len(values) - 1
    while position <= maximum_position + 1e-9:
        left_index = min(int(math.floor(position)), len(values) - 1)
        right_index = min(left_index + 1, len(values) - 1)
        fraction = position - left_index
        at = values[left_index] + (
            values[right_index] - values[left_index]
        ) * fraction
        normalized = round(at, 4)
        if not resampled or normalized - resampled[-1] > 0.05:
            resampled.append(normalized)
        position += source_step
    return round(source_bpm, 4), resampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the 48-clip beat gold-standard pilot corpus.",
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".cache" / "beat-gold" / "pilot-v2",
    )
    parser.add_argument("--annotator-alias", default="reviewer-a")
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--pre-roll", type=float, default=4.0)
    parser.add_argument("--post-roll", type=float, default=4.0)
    parser.add_argument("--corpus-version", default=CORPUS_VERSION)
    parser.add_argument(
        "--recording",
        choices=[recording.slug for recording in RECORDINGS],
        help="Regenerate one recording while preserving the other manifest items.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Generate targeted reviewer files without updating the shared manifest.",
    )
    parser.add_argument(
        "--target-bpm",
        type=float,
        help=(
            "Retarget one recording's existing beat grid to this BPM and "
            "regenerate its review drafts."
        ),
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.skip_manifest and not arguments.recording:
        parser.error("--skip-manifest requires --recording")
    if arguments.target_bpm is not None and not arguments.recording:
        parser.error("--target-bpm requires --recording")
    if (
        arguments.target_bpm is not None
        and not 20 <= arguments.target_bpm <= 320
    ):
        parser.error("--target-bpm must be between 20 and 320")

    source_dir = arguments.source_dir.resolve()
    output_dir = arguments.output_dir.resolve()
    analysis_dir = output_dir / "analysis"
    review_dir = output_dir / "reviews" / arguments.annotator_alias
    manifest_clips: list[dict[str, object]] = []
    recordings = (
        tuple(
            recording
            for recording in RECORDINGS
            if recording.slug == arguments.recording
        )
        if arguments.recording
        else RECORDINGS
    )
    existing_manifest: dict[str, object] | None = None
    manifest_path = output_dir / "pilot-manifest.json"
    if arguments.recording and not arguments.skip_manifest:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Targeted regeneration requires an existing pilot manifest",
            )
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
        )
        if existing_manifest.get("annotator_alias") != arguments.annotator_alias:
            raise ValueError(
                "Targeted manifest regeneration must use its current annotator "
                "alias or pass --skip-manifest",
            )
        if not isinstance(existing_manifest.get("items"), list):
            raise ValueError("Existing pilot manifest items are invalid")
        existing_context = existing_manifest.get("context")
        if (
            existing_manifest.get("corpus_version")
            != arguments.corpus_version
            or float(existing_manifest.get("evaluation_duration", -1))
            != arguments.duration
            or not isinstance(existing_context, dict)
            or float(existing_context.get("pre_roll", -1))
            != arguments.pre_roll
            or float(existing_context.get("post_roll", -1))
            != arguments.post_roll
        ):
            raise ValueError(
                "Targeted regeneration parameters must match the existing "
                "pilot manifest"
            )

    for recording in recordings:
        audio_path = source_dir / f"{recording.slug}.media"
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing decoded source: {audio_path.name}")
        duration = float(soundfile.info(audio_path).duration)
        analysis_path = analysis_dir / f"{recording.slug}.json"
        previous_analysis = (
            json.loads(analysis_path.read_text(encoding="utf-8"))
            if analysis_path.is_file()
            else None
        )
        analysis: dict[str, object]
        if arguments.force or not analysis_path.exists():
            bpm, beats, onsets, tempo_map = estimate_tempo(audio_path)
            analysis = {
                "schema_version": 1,
                "recording_id": recording.slug,
                "source_audio_sha256": sha256_file(audio_path),
                "duration": round(duration, 6),
                "bpm": bpm,
                "beats": beats,
                "onsets": onsets,
                "tempo_map": tempo_map,
                "time_signature": list(recording.time_signature),
                "pickup_beats": 0,
                "timing_source": "pilot-mix-beat-candidate-v1",
                "verification_status": "quarantined",
                "benchmark_eligible": False,
            }
        else:
            assert previous_analysis is not None
            analysis = previous_analysis
        if arguments.target_bpm is not None:
            previous_bpm = float(analysis.get("bpm", 0))
            if abs(previous_bpm - arguments.target_bpm) > 0.01:
                source_bpm, retargeted_beats = retarget_beat_grid(
                    list(analysis.get("beats", [])),
                    arguments.target_bpm,
                )
                analysis = {
                    **analysis,
                    "bpm": round(arguments.target_bpm, 2),
                    "beats": retargeted_beats,
                    "tempo_map": tempo_map_from_beats(
                        retargeted_beats,
                        arguments.target_bpm,
                    ),
                    "timing_source": (
                        "pilot-mix-beat-candidate-v1-retargeted"
                    ),
                    "tempo_override": {
                        "source_bpm": source_bpm,
                        "target_bpm": round(arguments.target_bpm, 4),
                        "method": "time-axis-grid-resampling",
                    },
                    "verification_status": "quarantined",
                    "benchmark_eligible": False,
                }
        if analysis != previous_analysis:
            structure_reviews = sorted(
                (output_dir / "structure-reviews").glob(
                    f"*/{recording.slug}.json"
                )
            )
            if structure_reviews:
                raise ValueError(
                    "Analysis regeneration would invalidate existing "
                    f"structure reviews for {recording.slug}; reopen and "
                    "archive them before retrying"
                )
        write_json(analysis_path, analysis)

        for index, start in enumerate(
            clip_starts(duration, recording.clips, arguments.duration),
            start=1,
        ):
            clip_id = f"{recording.slug}-{index:02d}"
            review_path = review_dir / f"{clip_id}.json"
            clip_audio_path = review_path.with_suffix(".wav")
            if arguments.force or arguments.target_bpm is not None:
                review_path.unlink(missing_ok=True)
                clip_audio_path.unlink(missing_ok=True)
            if review_path.exists():
                review = json.loads(review_path.read_text(encoding="utf-8"))
            else:
                review = create_review(
                    audio_path=audio_path,
                    analysis_path=analysis_path,
                    review_path=review_path,
                    clip_id=clip_id,
                    group_id=f"{recording.slug}-recording",
                    annotator_alias=arguments.annotator_alias,
                    source_start=start,
                    duration=arguments.duration,
                    corpus_version=arguments.corpus_version,
                    pre_roll=arguments.pre_roll,
                    post_roll=arguments.post_roll,
                    time_signature=recording.time_signature,
                    tags=[recording.split, *recording.tags],
                )
            if review.get("corpus_version") != arguments.corpus_version:
                raise ValueError(
                    f"{clip_id} belongs to a different corpus version",
                )
            manifest_clips.append(
                {
                    "clip_id": clip_id,
                    "recording_id": recording.slug,
                    "split": recording.split,
                    "audio_source_start": review["audio_source_start"],
                    "audio_source_end": review["audio_source_end"],
                    "audio_duration": review["audio_duration"],
                    "evaluation_start": review["evaluation_start"],
                    "evaluation_end": review["evaluation_end"],
                    "evaluation_duration": review["evaluation_duration"],
                    "review": str(review_path.relative_to(output_dir)),
                    "audio": str(clip_audio_path.relative_to(output_dir)),
                    "audio_sha256": review["audio_sha256"],
                    "tags": review["tags"],
                    "status": review["status"],
                }
            )
        print(
            f"{recording.slug}: {recording.clips} clips, "
            f"{duration:.1f}s, {recording.split}",
        )

    if arguments.skip_manifest:
        print(
            json.dumps(
                {
                    "manifest": "skipped",
                    "recording": arguments.recording,
                    "annotator_alias": arguments.annotator_alias,
                    "clips": len(manifest_clips),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if existing_manifest is not None:
        replacements = {
            str(clip["clip_id"]): clip
            for clip in manifest_clips
        }
        existing_items = existing_manifest["items"]
        assert isinstance(existing_items, list)
        existing_ids = {
            str(item.get("clip_id"))
            for item in existing_items
            if isinstance(item, dict)
        }
        if replacements.keys() - existing_ids:
            raise ValueError(
                "Existing pilot manifest is missing targeted recording clips",
            )
        manifest_clips = [
            replacements.get(str(item.get("clip_id")), item)
            for item in existing_items
            if isinstance(item, dict)
        ]

    split_counts = {
        split: sum(clip["split"] == split for clip in manifest_clips)
        for split in ("train", "validation", "test")
    }
    immutable_corpus = {
        "corpus_version": arguments.corpus_version,
        "evaluation_duration": arguments.duration,
        "context": {
            "pre_roll": arguments.pre_roll,
            "post_roll": arguments.post_roll,
        },
        "items": [
            {
                key: clip[key]
                for key in (
                    "clip_id",
                    "recording_id",
                    "split",
                    "audio_source_start",
                    "audio_source_end",
                    "audio_duration",
                    "evaluation_start",
                    "evaluation_end",
                    "evaluation_duration",
                    "audio_sha256",
                )
            }
            for clip in manifest_clips
        ],
    }
    corpus_sha256 = hashlib.sha256(
        json.dumps(
            immutable_corpus,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ).hexdigest()
    manifest = (
        {
            **existing_manifest,
            "corpus_sha256": corpus_sha256,
            "split_counts": split_counts,
            "items": manifest_clips,
        }
        if existing_manifest is not None
        else {
            "schema_version": 2,
            "kind": "beat-gold-pilot-v2",
            "corpus_version": arguments.corpus_version,
            "corpus_sha256": corpus_sha256,
            "immutable": True,
            "verification_status": "draft",
            "benchmark_eligible": False,
            "annotator_alias": arguments.annotator_alias,
            "evaluation_duration": arguments.duration,
            "context": {
                "pre_roll": arguments.pre_roll,
                "post_roll": arguments.post_roll,
            },
            "recordings": len(RECORDINGS),
            "clips": len(manifest_clips),
            "split_counts": split_counts,
            "items": manifest_clips,
        }
    )
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "pilot-manifest.json"),
                "corpus_version": arguments.corpus_version,
                "corpus_sha256": corpus_sha256,
                "recordings": len(RECORDINGS),
                "clips": len(manifest_clips),
                "split_counts": split_counts,
                "verification_status": "draft",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
