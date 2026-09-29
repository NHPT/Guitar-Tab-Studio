from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .features import extract_file_cqt
from .inference import decode_events, load_checkpoint, predict_probabilities
from .metrics import evaluate_events
from .schema import load_manifest, reject_quarantined_references


def model_identity(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: metadata[key]
        for key in (
            "name",
            "version",
            "kind",
            "epoch",
            "validation_loss",
            "attack_fret_weight",
            "attack_fret_frames",
            "pitch_shift_probability",
            "pitch_shift_min",
            "pitch_shift_max",
            "fret_distance_weight",
            "pitch_consistency_weight",
            "position_loss_weight",
            "pitch_conditioned_position",
            "selection_metric",
            "selection_metric_value",
        )
        if key in metadata
    }


def aggregate(
    track_results: list[dict[str, object]],
    metric_name: str,
) -> dict[str, int | float | None]:
    true_positive = sum(
        int(result["metrics"][metric_name]["true_positive"])
        for result in track_results
    )
    false_positive = sum(
        int(result["metrics"][metric_name]["false_positive"])
        for result in track_results
    )
    false_negative = sum(
        int(result["metrics"][metric_name]["false_negative"])
        for result in track_results
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    weighted_errors = [
        (
            float(result["metrics"][metric_name]["mean_onset_error_ms"]),
            int(result["metrics"][metric_name]["true_positive"]),
        )
        for result in track_results
        if result["metrics"][metric_name]["mean_onset_error_ms"] is not None
    ]
    error_weight = sum(weight for _error, weight in weighted_errors)
    result: dict[str, int | float | None] = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_onset_error_ms": (
            sum(error * weight for error, weight in weighted_errors) / error_weight
            if error_weight
            else None
        ),
    }
    weighted_beat_errors = [
        (
            float(result["metrics"][metric_name]["mean_beat_error"]),
            int(result["metrics"][metric_name]["true_positive"]),
        )
        for result in track_results
        if result["metrics"][metric_name].get("mean_beat_error") is not None
    ]
    beat_error_weight = sum(weight for _error, weight in weighted_beat_errors)
    if weighted_beat_errors:
        result["mean_beat_error"] = (
            sum(
                error * weight
                for error, weight in weighted_beat_errors
            )
            / beat_error_weight
        )
    return result


def aggregate_decoder_errors(
    track_results: list[dict[str, object]],
) -> dict[str, object]:
    names = (
        "exact_tablature",
        "same_pitch_wrong_string",
        "same_string_wrong_fret",
        "wrong_string_and_pitch",
        "spurious_onset",
        "missed_onset",
    )
    string_confusion = [[0 for _predicted in range(6)] for _reference in range(6)]
    for track_result in track_results:
        errors = track_result["metrics"]["decoder_errors"]
        for reference_string in range(6):
            for predicted_string in range(6):
                string_confusion[reference_string][predicted_string] += int(
                    errors["string_confusion"][reference_string][predicted_string]
                )
    return {
        **{
            name: sum(
                int(result["metrics"]["decoder_errors"][name])
                for result in track_results
            )
            for name in names
        },
        "string_confusion": string_confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained StringTrace checkpoint on held-out tracks.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--onset-threshold", type=float, default=0.60)
    parser.add_argument("--activity-threshold", type=float, default=0.25)
    parser.add_argument("--minimum-onset-gap", type=float, default=0.035)
    parser.add_argument("--activity-context", type=float, default=0.046)
    parser.add_argument("--activity-delay", type=float, default=0.046)
    parser.add_argument("--minimum-event-confidence", type=float, default=0.42)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    annotations = load_manifest(arguments.manifest, arguments.split)
    if not annotations:
        raise RuntimeError(f"The manifest contains no {arguments.split} tracks")
    reject_quarantined_references(annotations)
    device = torch.device(arguments.device)
    loaded = load_checkpoint(arguments.checkpoint, device)
    report_model = model_identity(loaded.metadata)
    track_results: list[dict[str, object]] = []
    for index, annotation in enumerate(annotations, start=1):
        audio_path = annotation.resolve_audio(arguments.manifest.resolve())
        features = extract_file_cqt(audio_path, loaded.feature_config)
        fret_probabilities, onset_probabilities = predict_probabilities(
            loaded,
            features,
            device=device,
        )
        predictions = decode_events(
            fret_probabilities,
            onset_probabilities,
            loaded.feature_config,
            onset_threshold=arguments.onset_threshold,
            activity_threshold=arguments.activity_threshold,
            minimum_onset_gap=arguments.minimum_onset_gap,
            activity_context=arguments.activity_context,
            activity_delay=arguments.activity_delay,
            minimum_event_confidence=arguments.minimum_event_confidence,
        )
        track_results.append(
            {
                "track_id": annotation.track_id,
                "reference_events": len(annotation.events),
                "predicted_events": len(predictions),
                "metrics": evaluate_events(
                    annotation.events,
                    predictions,
                    onset_tolerance=arguments.onset_tolerance,
                    beats=annotation.beats,
                    excluded_ranges=annotation.excluded_ranges,
                ),
                "model": report_model,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "evaluation",
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
        "decoder": {
            "onset_threshold": arguments.onset_threshold,
            "activity_threshold": arguments.activity_threshold,
            "minimum_onset_gap": arguments.minimum_onset_gap,
            "activity_context": arguments.activity_context,
            "activity_delay": arguments.activity_delay,
            "minimum_event_confidence": arguments.minimum_event_confidence,
        },
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
