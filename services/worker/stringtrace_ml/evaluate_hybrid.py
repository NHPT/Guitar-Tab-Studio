from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

from worker import transcribe_basic_pitch

from .evaluate import aggregate, aggregate_decoder_errors, model_identity
from .features import extract_file_cqt
from .hybrid import PitchEvent, PositionAssignmentConfig, assign_pitch_events
from .inference import load_checkpoint, predict_probabilities_with_position
from .metrics import evaluate_events
from .schema import (
    load_manifest,
    reject_quarantined_references,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Basic Pitch note detection with StringTrace fretboard "
            "position assignment."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--group-tolerance", type=float, default=0.045)
    parser.add_argument("--activity-delay", type=float, default=0.023)
    parser.add_argument("--activity-context", type=float, default=0.046)
    parser.add_argument("--onset-context", type=float, default=0.023)
    parser.add_argument("--onset-weight", type=float, default=1.0)
    parser.add_argument("--position-head-weight", type=float, default=1.0)
    parser.add_argument("--continuity-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    annotations = load_manifest(arguments.manifest, arguments.split)
    if not annotations:
        raise RuntimeError(f"The manifest contains no {arguments.split} tracks")
    reject_quarantined_references(annotations)
    assignment_config = PositionAssignmentConfig(
        group_tolerance=arguments.group_tolerance,
        activity_delay=arguments.activity_delay,
        activity_context=arguments.activity_context,
        onset_context=arguments.onset_context,
        onset_weight=arguments.onset_weight,
        position_head_weight=arguments.position_head_weight,
        continuity_weight=arguments.continuity_weight,
    )
    device = torch.device(arguments.device)
    loaded = load_checkpoint(arguments.checkpoint, device)
    track_results: list[dict[str, object]] = []
    for index, annotation in enumerate(annotations, start=1):
        audio_path = annotation.resolve_audio(arguments.manifest.resolve())
        with contextlib.redirect_stdout(sys.stderr):
            notes, dropped_notes = transcribe_basic_pitch(audio_path)
        pitch_events = [
            PitchEvent(
                onset=note.start,
                offset=note.end,
                pitch=note.pitch,
                technique="unknown",
                confidence=note.confidence,
            )
            for note in notes
        ]
        features = extract_file_cqt(audio_path, loaded.feature_config)
        (
            fret_probabilities,
            onset_probabilities,
            position_probabilities,
        ) = predict_probabilities_with_position(
            loaded, features, device=device
        )
        predictions = assign_pitch_events(
            pitch_events,
            fret_probabilities,
            onset_probabilities,
            loaded.feature_config,
            position_probabilities=position_probabilities,
            config=assignment_config,
        )
        track_results.append(
            {
                "track_id": annotation.track_id,
                "reference_events": len(annotation.events),
                "predicted_events": len(predictions),
                "dropped_out_of_range_pitch_events": dropped_notes,
                "metrics": evaluate_events(
                    annotation.events,
                    predictions,
                    onset_tolerance=arguments.onset_tolerance,
                    beats=annotation.beats,
                    excluded_ranges=annotation.excluded_ranges,
                ),
            }
        )
        print(
            json.dumps(
                {
                    "stage": "hybrid_evaluation",
                    "tracks": index,
                    "total_tracks": len(annotations),
                    "track_id": annotation.track_id,
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    metric_names = [
        "onset",
        "pitch",
        "tablature",
        "repeated_tablature",
    ]
    if all(
        "beat_relative_onset" in result["metrics"]
        for result in track_results
    ):
        metric_names.append("beat_relative_onset")
    report = {
        "split": arguments.split,
        "track_count": len(track_results),
        "pitch_frontend": {
            "name": "spotify-basic-pitch",
            "version": "0.4",
        },
        "position_model": model_identity(loaded.metadata),
        "assignment": assignment_config.to_dict(),
        "aggregate": {
            metric_name: aggregate(track_results, metric_name)
            for metric_name in metric_names
        }
        | {
            "decoder_errors": aggregate_decoder_errors(track_results),
        },
        "tracks": track_results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
