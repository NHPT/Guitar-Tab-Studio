from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import model_identity
from .metrics import evaluate_events
from .schema import (
    LabeledEvent,
    load_manifest,
    reject_quarantined_references,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Worker analysis.json against a held-out reference clip.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--track-id")
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    annotations = load_manifest(arguments.manifest, "test")
    if arguments.track_id:
        annotations = [
            annotation
            for annotation in annotations
            if annotation.track_id == arguments.track_id
        ]
    if len(annotations) != 1:
        raise ValueError(
            "Exactly one test annotation must be selected for one Worker analysis"
        )
    reject_quarantined_references(annotations)
    annotation = annotations[0]
    source_start = float(annotation.provenance.get("source_start", 0))
    source_end = source_start + annotation.duration
    analysis = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    predictions = [
        LabeledEvent(
            onset=round(float(note["start"]) - source_start, 6),
            offset=round(
                min(source_end, float(note["end"])) - source_start,
                6,
            ),
            string=int(note["string"]),
            fret=int(note["fret"]),
            technique=str(note.get("technique", "unknown")),
            confidence=float(note.get("confidence", 1.0)),
        )
        for note in analysis.get("notes", [])
        if source_start <= float(note["start"]) < source_end
        and float(note["end"]) > source_start
    ]
    report = {
        "track_id": annotation.track_id,
        "model": model_identity(
            analysis.get(
                "transcription_model",
                {
                    "name": "legacy-worker",
                    "version": analysis.get("version"),
                    "kind": "baseline",
                },
            )
        ),
        "reference_events": len(annotation.events),
        "predicted_events": len(predictions),
        "metrics": evaluate_events(
            annotation.events,
            predictions,
            onset_tolerance=arguments.onset_tolerance,
            beats=annotation.beats,
            excluded_ranges=annotation.excluded_ranges,
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
