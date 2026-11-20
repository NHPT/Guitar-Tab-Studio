from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .candidate_sequence import candidate_sequence_activity_audit
from .data_sources import manifest_identity, sha256_file
from .features import FeatureConfig
from .onset_position import OnsetPositionConfig, OnsetPositionDataset
from .schema import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit candidate activity and repeated-positive supervision "
            "before sequence training."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--positive-only-dataset",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--candidate-onset-threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--candidate-frame-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--candidate-minimum-note-length",
        type=float,
        default=70.0,
    )
    parser.add_argument("--repeat-interval", type=float, default=0.75)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-stride", type=int, default=16)
    arguments = parser.parse_args()

    for name in (
        "candidate_onset_threshold",
        "candidate_frame_threshold",
    ):
        value = getattr(arguments, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(
                f"--{name.replace('_', '-')} must be in 0..1"
            )
    for name in ("candidate_minimum_note_length", "repeat_interval"):
        value = getattr(arguments, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if not 0 < arguments.sequence_stride <= arguments.sequence_length:
        parser.error("--sequence-stride must be in 1..sequence-length")

    checkpoint = torch.load(
        arguments.position_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported onset-position checkpoint format")
    feature_config = FeatureConfig.from_dict(
        dict(checkpoint.get("feature_config", {}))
    )
    position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("model_config", {}))
    )
    common = {
        "split": arguments.split,
        "feature_config": feature_config,
        "config": position_config,
        "feature_cache_directory": arguments.feature_cache,
        "candidate_cache_directory": arguments.candidate_cache,
        "candidate_onset_threshold": arguments.candidate_onset_threshold,
        "candidate_frame_threshold": arguments.candidate_frame_threshold,
        "candidate_minimum_note_length": (
            arguments.candidate_minimum_note_length
        ),
    }
    raw_dataset = OnsetPositionDataset(
        arguments.manifest,
        **common,
    )
    positive_only = sorted(set(arguments.positive_only_dataset))
    governed_dataset = OnsetPositionDataset(
        arguments.manifest,
        positive_only_datasets=set(positive_only),
        **common,
    )
    annotations = load_manifest(arguments.manifest.resolve())
    report = {
        "schema_version": 1,
        "kind": "candidate-sequence-activity-data-audit",
        "manifest": manifest_identity(
            arguments.manifest.resolve(),
            annotations,
            purpose="training",
        ),
        "split": arguments.split,
        "position_checkpoint": {
            "name": arguments.position_checkpoint.name,
            "sha256": sha256_file(arguments.position_checkpoint),
        },
        "candidate_frontend": {
            "kind": "spotify-basic-pitch-0.4",
            "onset_threshold": arguments.candidate_onset_threshold,
            "frame_threshold": arguments.candidate_frame_threshold,
            "minimum_note_length": (
                arguments.candidate_minimum_note_length
            ),
        },
        "positive_only_datasets": positive_only,
        "raw": candidate_sequence_activity_audit(
            raw_dataset,
            repeat_interval=arguments.repeat_interval,
            sequence_length=arguments.sequence_length,
            sequence_stride=arguments.sequence_stride,
        ),
        "governed": candidate_sequence_activity_audit(
            governed_dataset,
            repeat_interval=arguments.repeat_interval,
            sequence_length=arguments.sequence_length,
            sequence_stride=arguments.sequence_stride,
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "raw": report["raw"]["totals"],
                "governed": report["governed"]["totals"],
                "filtered_negative_groups": report["governed"][
                    "filtered_negative_groups"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
