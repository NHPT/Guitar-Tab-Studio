#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "worker"))

from stringtrace_ml.schema import (  # noqa: E402
    ChordRegion,
    LabeledEvent,
    TrackAnnotation,
    write_manifest,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the bundled score as a quarantined review candidate. "
            "The output is not benchmark-eligible until human approval."
        )
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=ROOT / "apps" / "api" / "assets" / "internationale-analysis.json",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=ROOT / "apps" / "api" / "assets" / "internationale.ogg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "services"
            / "worker"
            / "references"
            / "internationale"
            / "candidate.jsonl"
        ),
    )
    arguments = parser.parse_args()
    analysis = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    duration = float(analysis["metadata"]["duration"])
    relative_audio = os.path.relpath(
        arguments.audio.resolve(),
        arguments.output.parent.resolve(),
    )
    annotation = TrackAnnotation(
        track_id="internationale-full-song-candidate",
        group_id="internationale-tang-dynasty-recording",
        audio=relative_audio,
        split="test",
        duration=duration,
        events=[
            LabeledEvent(
                onset=float(note["start"]),
                offset=min(duration, float(note["end"])),
                string=int(note["string"]),
                fret=int(note["fret"]),
                technique=str(note.get("technique", "unknown")),
                confidence=float(note.get("confidence", 0.5)),
            )
            for note in analysis["notes"]
        ],
        chords=[
            ChordRegion(
                start=float(chord["start"]),
                end=min(
                    duration,
                    float(chord["start"]) + float(chord["duration"]),
                ),
                chord=str(chord["chord"]),
            )
            for chord in analysis["chords"]
        ],
        bpm=float(analysis["bpm"]),
        beats=[float(beat) for beat in analysis["beats"]],
        time_signature=tuple(analysis.get("time_signature", [4, 4])),
        provenance={
            "dataset": "GTS-Builtin-Review-Candidate",
            "source_type": "model-generated-candidate",
            "verification_status": "quarantined",
            "review_status": "draft",
            "required_review_layers": [
                "beat-and-measure",
                "onset-and-pitch",
                "string-and-fret",
                "chord",
                "technique",
            ],
            "analysis_sha256": sha256(arguments.analysis),
            "source_audio_sha256": sha256(arguments.audio),
            "candidate_generator": (
                "percussion-downbeat-grid-v2+acoustic-onset-gate-v1"
                "+sequence-optimizer-v3"
            ),
            "pickup_beats": int(analysis.get("pickup_beats", 0)),
            "benchmark_eligible": False,
        },
    )
    annotation.validate()
    write_manifest(arguments.output, [annotation])
    print(
        json.dumps(
            {
                "output": arguments.output.name,
                "events": len(annotation.events),
                "beats": len(annotation.beats),
                "chords": len(annotation.chords),
                "verification_status": "quarantined",
                "benchmark_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
