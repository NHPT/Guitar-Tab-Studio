from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from .candidate_activity import CandidateActivityConfig, CandidateActivityHead
from .features import FeatureConfig, extract_file_cqt
from .onset_position import OnsetPositionConfig, OnsetPositionNet
from .prepare_guitarset_stems import sha256_file
from .schema import LabeledEvent


BUNDLE_KIND = "stringtrace-onset-position-research-bundle"
BUNDLE_FORMAT_VERSION = 1


def default_research_config() -> dict[str, object]:
    return {
        "primary_frontend": {
            "onset_threshold": 0.35,
            "frame_threshold": 0.25,
            "minimum_note_length": 70.0,
        },
        "supplemental_frontend": {
            "onset_threshold": 0.25,
            "frame_threshold": 0.2,
            "minimum_note_length": 70.0,
        },
        "assignment": {
            "mode": "direct",
            "minimum_direct_probability": 0.5,
            "minimum_activity_probability": 0.2,
            "minimum_repeat_probability": 0.44,
            "repeat_minimum_direct_probability": 0.45,
        },
        "supplemental_assignment": {
            "mode": "direct",
            "minimum_direct_probability": 0.7,
            "minimum_activity_probability": 0.85,
        },
        "merge_onset_tolerance": 0.05,
    }


@dataclass(frozen=True)
class LoadedOnsetPositionBundle:
    position_model: OnsetPositionNet
    primary_activity_head: CandidateActivityHead
    supplemental_activity_head: CandidateActivityHead
    feature_config: FeatureConfig
    position_config: OnsetPositionConfig
    inference_config: dict[str, object]
    metadata: dict[str, object]


def _load_checkpoint(path: Path, expected_name: str) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{expected_name} checkpoint must be a dictionary")
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError(f"Unsupported {expected_name} checkpoint format")
    return checkpoint


def _compact_position_checkpoint(
    checkpoint: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "model_state": checkpoint["model_state"],
        "model_config": checkpoint.get("model_config", {}),
        "feature_config": checkpoint.get("feature_config", {}),
        "metadata": checkpoint.get("metadata", {}),
    }


def _compact_activity_checkpoint(
    checkpoint: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "model_state": checkpoint["model_state"],
        "activity_config": checkpoint.get("activity_config", {}),
        "position_config": checkpoint.get("position_config", {}),
        "feature_config": checkpoint.get("feature_config", {}),
        "metadata": checkpoint.get("metadata", {}),
    }


def _validate_activity_checkpoint(
    checkpoint: dict[str, object],
    *,
    name: str,
    position_config: OnsetPositionConfig,
    feature_config: FeatureConfig,
    position_sha256: str,
) -> None:
    activity_position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("position_config", {}))
    )
    if activity_position_config != position_config:
        raise ValueError(f"{name} position configuration differs")
    activity_feature_config = FeatureConfig.from_dict(
        dict(checkpoint.get("feature_config", {}))
    )
    if activity_feature_config != feature_config:
        raise ValueError(f"{name} feature configuration differs")
    metadata = dict(checkpoint.get("metadata", {}))
    base_checkpoint = dict(metadata.get("base_checkpoint", {}))
    if base_checkpoint.get("sha256") != position_sha256:
        raise ValueError(f"{name} targets a different position checkpoint")


def build_research_bundle(
    position_checkpoint_path: Path,
    primary_activity_checkpoint_path: Path,
    supplemental_activity_checkpoint_path: Path,
    output_path: Path,
    *,
    version: str,
) -> dict[str, object]:
    position_path = position_checkpoint_path.resolve()
    primary_path = primary_activity_checkpoint_path.resolve()
    supplemental_path = supplemental_activity_checkpoint_path.resolve()
    position_checkpoint = _load_checkpoint(position_path, "position")
    primary_checkpoint = _load_checkpoint(primary_path, "primary activity")
    supplemental_checkpoint = _load_checkpoint(
        supplemental_path,
        "supplemental activity",
    )
    position_config = OnsetPositionConfig.from_dict(
        dict(position_checkpoint.get("model_config", {}))
    )
    feature_config = FeatureConfig.from_dict(
        dict(position_checkpoint.get("feature_config", {}))
    )
    position_sha256 = sha256_file(position_path)
    _validate_activity_checkpoint(
        primary_checkpoint,
        name="Primary activity checkpoint",
        position_config=position_config,
        feature_config=feature_config,
        position_sha256=position_sha256,
    )
    _validate_activity_checkpoint(
        supplemental_checkpoint,
        name="Supplemental activity checkpoint",
        position_config=position_config,
        feature_config=feature_config,
        position_sha256=position_sha256,
    )
    source_checkpoints = {
        "position": {
            "name": position_path.name,
            "sha256": position_sha256,
        },
        "primary_activity": {
            "name": primary_path.name,
            "sha256": sha256_file(primary_path),
        },
        "supplemental_activity": {
            "name": supplemental_path.name,
            "sha256": sha256_file(supplemental_path),
        },
    }
    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "kind": BUNDLE_KIND,
        "position_checkpoint": _compact_position_checkpoint(
            position_checkpoint
        ),
        "primary_activity_checkpoint": _compact_activity_checkpoint(
            primary_checkpoint
        ),
        "supplemental_activity_checkpoint": _compact_activity_checkpoint(
            supplemental_checkpoint
        ),
        "inference_config": default_research_config(),
        "metadata": {
            "name": "stringtrace-onset-position-research",
            "version": version,
            "release_channel": "research",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoints": source_checkpoints,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    return {
        "bundle": output_path.name,
        "kind": BUNDLE_KIND,
        "version": version,
        "sha256": sha256_file(output_path),
        "source_checkpoints": source_checkpoints,
    }


def _load_activity_head(
    checkpoint: dict[str, object],
    *,
    name: str,
    position_config: OnsetPositionConfig,
    feature_config: FeatureConfig,
    position_sha256: str,
    device: torch.device,
) -> CandidateActivityHead:
    _validate_activity_checkpoint(
        checkpoint,
        name=name,
        position_config=position_config,
        feature_config=feature_config,
        position_sha256=position_sha256,
    )
    activity_config = CandidateActivityConfig.from_dict(
        dict(checkpoint.get("activity_config", {}))
    )
    head = CandidateActivityHead(position_config, activity_config)
    head.load_state_dict(checkpoint["model_state"])
    head.to(device)
    head.eval()
    return head


def _probability(value: object, name: str) -> float:
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in 0..1")
    return result


def _positive(value: object, name: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_inference_config(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Bundle inference configuration is invalid")
    result = dict(value)
    for section_name in ("primary_frontend", "supplemental_frontend"):
        section = dict(result.get(section_name, {}))
        _probability(
            section.get("onset_threshold"),
            f"{section_name}.onset_threshold",
        )
        _probability(
            section.get("frame_threshold"),
            f"{section_name}.frame_threshold",
        )
        _positive(
            section.get("minimum_note_length"),
            f"{section_name}.minimum_note_length",
        )
    assignment = dict(result.get("assignment", {}))
    supplemental_assignment = dict(
        result.get("supplemental_assignment", {})
    )
    if assignment.get("mode") != "direct":
        raise ValueError("Bundle primary assignment mode must be direct")
    if supplemental_assignment.get("mode") != "direct":
        raise ValueError("Bundle supplemental assignment mode must be direct")
    for name in (
        "minimum_direct_probability",
        "minimum_activity_probability",
        "minimum_repeat_probability",
        "repeat_minimum_direct_probability",
    ):
        _probability(assignment.get(name), f"assignment.{name}")
    for name in (
        "minimum_direct_probability",
        "minimum_activity_probability",
    ):
        _probability(
            supplemental_assignment.get(name),
            f"supplemental_assignment.{name}",
        )
    _positive(
        result.get("merge_onset_tolerance"),
        "merge_onset_tolerance",
    )
    return result


def load_research_bundle(
    path: Path,
    device: torch.device,
) -> LoadedOnsetPositionBundle:
    bundle = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("Onset-position bundle must be a dictionary")
    if int(bundle.get("format_version", 0)) != BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsupported onset-position bundle format")
    if bundle.get("kind") != BUNDLE_KIND:
        raise ValueError("Unsupported onset-position bundle kind")
    metadata = dict(bundle.get("metadata", {}))
    if metadata.get("release_channel") != "research":
        raise ValueError("Onset-position bundle must use the research channel")
    source_checkpoints = dict(metadata.get("source_checkpoints", {}))
    position_identity = dict(source_checkpoints.get("position", {}))
    position_sha256 = str(position_identity.get("sha256", ""))
    if len(position_sha256) != 64:
        raise ValueError("Bundle position checkpoint identity is invalid")

    position_checkpoint = dict(bundle.get("position_checkpoint", {}))
    if int(position_checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported embedded position checkpoint format")
    feature_config = FeatureConfig.from_dict(
        dict(position_checkpoint.get("feature_config", {}))
    )
    position_config = OnsetPositionConfig.from_dict(
        dict(position_checkpoint.get("model_config", {}))
    )
    position_model = OnsetPositionNet(feature_config, position_config)
    position_model.load_state_dict(position_checkpoint["model_state"])
    position_model.to(device)
    position_model.eval()

    primary_checkpoint = dict(
        bundle.get("primary_activity_checkpoint", {})
    )
    supplemental_checkpoint = dict(
        bundle.get("supplemental_activity_checkpoint", {})
    )
    primary_head = _load_activity_head(
        primary_checkpoint,
        name="Primary activity checkpoint",
        position_config=position_config,
        feature_config=feature_config,
        position_sha256=position_sha256,
        device=device,
    )
    supplemental_head = _load_activity_head(
        supplemental_checkpoint,
        name="Supplemental activity checkpoint",
        position_config=position_config,
        feature_config=feature_config,
        position_sha256=position_sha256,
        device=device,
    )
    return LoadedOnsetPositionBundle(
        position_model=position_model,
        primary_activity_head=primary_head,
        supplemental_activity_head=supplemental_head,
        feature_config=feature_config,
        position_config=position_config,
        inference_config=_validate_inference_config(
            bundle.get("inference_config")
        ),
        metadata=metadata,
    )


def transcribe_with_research_bundle(
    audio_path: Path,
    primary_events: list[dict[str, float | int]],
    supplemental_events: list[dict[str, float | int]],
    loaded: LoadedOnsetPositionBundle,
    device: torch.device,
) -> list[LabeledEvent]:
    from .evaluate_onset_position import assign_detected_events, merge_predictions

    features = extract_file_cqt(audio_path, loaded.feature_config)
    config = loaded.inference_config
    assignment = dict(config["assignment"])
    primary = assign_detected_events(
        primary_events,
        features,
        loaded.position_model,
        loaded.feature_config,
        loaded.position_config,
        device,
        mode="direct",
        minimum_direct_probability=float(
            assignment["minimum_direct_probability"]
        ),
        activity_head=loaded.primary_activity_head,
        minimum_activity_probability=float(
            assignment["minimum_activity_probability"]
        ),
        minimum_repeat_probability=float(
            assignment["minimum_repeat_probability"]
        ),
        repeat_minimum_direct_probability=float(
            assignment["repeat_minimum_direct_probability"]
        ),
    )
    supplemental_assignment = dict(config["supplemental_assignment"])
    supplemental = assign_detected_events(
        supplemental_events,
        features,
        loaded.position_model,
        loaded.feature_config,
        loaded.position_config,
        device,
        mode="direct",
        minimum_direct_probability=float(
            supplemental_assignment["minimum_direct_probability"]
        ),
        activity_head=loaded.supplemental_activity_head,
        minimum_activity_probability=float(
            supplemental_assignment["minimum_activity_probability"]
        ),
        minimum_repeat_probability=1.0,
        repeat_minimum_direct_probability=float(
            supplemental_assignment["minimum_direct_probability"]
        ),
    )
    return merge_predictions(
        primary,
        supplemental,
        onset_tolerance=float(config["merge_onset_tolerance"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single-file research bundle for onset-position "
            "transcription."
        )
    )
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--primary-activity-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--supplemental-activity-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args()
    print(
        build_research_bundle(
            arguments.position_checkpoint,
            arguments.primary_activity_checkpoint,
            arguments.supplemental_activity_checkpoint,
            arguments.output,
            version=arguments.version,
        )
    )


if __name__ == "__main__":
    main()
