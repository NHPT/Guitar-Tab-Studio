from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .evaluate import aggregate, model_identity
from .features import extract_file_cqt
from .inference import decode_events, load_checkpoint, predict_probabilities
from .metrics import evaluate_events
from .schema import load_manifest, reject_quarantined_references


METRIC_NAMES = (
    "onset",
    "pitch",
    "tablature",
    "repeated_tablature",
)


@dataclass(frozen=True)
class DecoderConfig:
    onset_threshold: float
    activity_threshold: float
    minimum_onset_gap: float
    activity_context: float
    activity_delay: float
    minimum_event_confidence: float


def parse_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one number")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate StringTrace decoder parameters with one inference pass "
            "per held-out track."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--onset-thresholds",
        type=parse_grid,
        default=[0.60],
    )
    parser.add_argument(
        "--activity-thresholds",
        type=parse_grid,
        default=[0.25],
    )
    parser.add_argument(
        "--minimum-onset-gaps",
        type=parse_grid,
        default=[0.035],
    )
    parser.add_argument(
        "--activity-contexts",
        type=parse_grid,
        default=[0.046],
    )
    parser.add_argument(
        "--activity-delays",
        type=parse_grid,
        default=[0.046],
    )
    parser.add_argument(
        "--minimum-event-confidences",
        type=parse_grid,
        default=[0.42],
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    configurations = [
        DecoderConfig(*values)
        for values in itertools.product(
            arguments.onset_thresholds,
            arguments.activity_thresholds,
            arguments.minimum_onset_gaps,
            arguments.activity_contexts,
            arguments.activity_delays,
            arguments.minimum_event_confidences,
        )
    ]
    if len(configurations) > 500:
        raise ValueError(
            f"Decoder grid contains {len(configurations)} combinations; "
            "the maximum is 500"
        )

    annotations = load_manifest(arguments.manifest, arguments.split)
    if not annotations:
        raise RuntimeError(f"The manifest contains no {arguments.split} tracks")
    reject_quarantined_references(annotations)
    device = torch.device(arguments.device)
    loaded = load_checkpoint(arguments.checkpoint, device)
    results_by_configuration: list[list[dict[str, object]]] = [
        [] for _configuration in configurations
    ]

    for track_index, annotation in enumerate(annotations, start=1):
        features = extract_file_cqt(
            annotation.resolve_audio(arguments.manifest.resolve()),
            loaded.feature_config,
        )
        fret_probabilities, onset_probabilities = predict_probabilities(
            loaded,
            features,
            device=device,
        )
        for configuration_index, configuration in enumerate(configurations):
            predictions = decode_events(
                fret_probabilities,
                onset_probabilities,
                loaded.feature_config,
                **asdict(configuration),
            )
            results_by_configuration[configuration_index].append(
                {
                    "predicted_events": len(predictions),
                    "metrics": evaluate_events(
                        annotation.events,
                        predictions,
                        onset_tolerance=arguments.onset_tolerance,
                        excluded_ranges=annotation.excluded_ranges,
                        include_diagnostics=False,
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "stage": "calibration",
                    "tracks": track_index,
                    "total_tracks": len(annotations),
                    "track_id": annotation.track_id,
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    sweep = []
    for configuration, track_results in zip(
        configurations,
        results_by_configuration,
    ):
        sweep.append(
            {
                "decoder": asdict(configuration),
                "predicted_events": sum(
                    int(result["predicted_events"])
                    for result in track_results
                ),
                "aggregate": {
                    metric_name: aggregate(track_results, metric_name)
                    for metric_name in METRIC_NAMES
                },
            }
        )
    sweep.sort(
        key=lambda result: (
            result["aggregate"]["tablature"]["f1"],
            result["aggregate"]["repeated_tablature"]["recall"],
            result["aggregate"]["onset"]["f1"],
        ),
        reverse=True,
    )
    report = {
        "split": arguments.split,
        "track_count": len(annotations),
        "configuration_count": len(configurations),
        "selection_metric": (
            "tablature.f1, repeated_tablature.recall, onset.f1"
        ),
        "model": model_identity(loaded.metadata),
        "best": sweep[0],
        "sweep": sweep,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
