#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "worker"))

from stringtrace_ml.beat_reference import create_review  # noqa: E402
from worker import estimate_tempo  # noqa: E402

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
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    source_dir = arguments.source_dir.resolve()
    output_dir = arguments.output_dir.resolve()
    analysis_dir = output_dir / "analysis"
    review_dir = output_dir / "reviews" / arguments.annotator_alias
    manifest_clips: list[dict[str, object]] = []

    for recording in RECORDINGS:
        audio_path = source_dir / f"{recording.slug}.media"
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing decoded source: {audio_path.name}")
        duration = float(soundfile.info(audio_path).duration)
        analysis_path = analysis_dir / f"{recording.slug}.json"
        if arguments.force or not analysis_path.exists():
            bpm, beats, onsets, tempo_map = estimate_tempo(audio_path)
            write_json(
                analysis_path,
                {
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
                },
            )

        for index, start in enumerate(
            clip_starts(duration, recording.clips, arguments.duration),
            start=1,
        ):
            clip_id = f"{recording.slug}-{index:02d}"
            review_path = review_dir / f"{clip_id}.json"
            clip_audio_path = review_path.with_suffix(".wav")
            if arguments.force:
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
    manifest = {
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
    write_json(output_dir / "pilot-manifest.json", manifest)
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
