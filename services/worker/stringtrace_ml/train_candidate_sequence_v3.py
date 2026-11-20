from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .analyze_candidate_sequence_assignment import assignment_audit
from .candidate_margin import (
    calibration_partition_digest,
    stratified_calibration_partition,
)
from .candidate_sequence import (
    CandidateSequenceGroup,
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    candidate_sequence_loss,
    collate_candidate_sequence_windows,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
    sequence_window_ranges,
)
from .data_sources import manifest_identity, sha256_file
from .features import FeatureConfig
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    basic_pitch_cache_path,
)
from .prepare_candidate_cache import validate_candidate_cache
from .schema import TrackAnnotation, load_manifest
from .train import choose_device
from .train_candidate_sequence import (
    append_log,
    load_position_model,
    move_batch,
)
from .train_candidate_sequence_activity import (
    refine_groups_with_string_sequence,
    window_sampling_plan,
)


PREREGISTRATION_SHA256 = (
    "e6594381f5bc45a99afa92f29a775f1362e60d6bb4ef5c61c5863f6497f0173d"
)
SOURCE_LEDGER_VERSION = "2026-11-06.2"
SOURCE_LEDGER_SHA256 = (
    "de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a"
)

FROZEN_ARTIFACTS: dict[str, tuple[str, str]] = {
    "formal_manifest": (
        "governed-v1/formal-training-manifest-v3.jsonl",
        "cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d",
    ),
    "formal_composition": (
        "reports/governed-v1-formal-training-v3-composition.json",
        "6d3b2731e1699df887eabf947b309e665cd196ba93955dfb395a4a76320e4cb0",
    ),
    "formal_source_audit": (
        "reports/governed-v1-formal-training-v3-source-audit.json",
        "ed171d832719ddfac766aabe3e45161e0961f0d0eb819812a052dcdebebad770",
    ),
    "cross_source_isolation": (
        "reports/governed-v1-formal-training-v3-cross-source-isolation.json",
        "59dbc6a2d5f60ff88dc52d4980389f2be78e13128fe54067e5cc84ab103e2d17",
    ),
    "p3_manifest": (
        (
            "governed-v1/"
            "guitar-techs-p3-music-position-train-candidate-v1.jsonl"
        ),
        "f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8",
    ),
    "p3_candidate_audit": (
        "reports/guitar-techs-p3-music-position-candidate-v1-audit.json",
        "266bef7f7879198595b934ce38ec52320eb16c61f70dae61929ebb50adcaff39",
    ),
    "p3_candidate_cache_report": (
        "reports/guitar-techs-p3-music-candidate-cache-v1.json",
        "f167413ca2a25c6d233fddf90979bc459d3b3c14f5e9d31acedd71c0d84860eb",
    ),
    "p3_sequence_coverage": (
        "reports/guitar-techs-p3-music-sequence-coverage-v1.json",
        "6c12bd0fdeb5776bed997a4d3efd8ba90bdf825d78b9127f25d393ec2fe5a728",
    ),
    "p3_string_v1_audit": (
        "reports/guitar-techs-p3-music-string-v1-training-audit.json",
        "48d6afde3ac03e8c7b83809dbfac19e07fb67776143039b563281b38ec01f732",
    ),
    "position_checkpoint": (
        "checkpoints/guitarset-candidate-activity-residual-v1.pt",
        "7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e",
    ),
    "string_v1_checkpoint": (
        "checkpoints/guitarset-candidate-sequence-string-v1.pt",
        "392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982",
    ),
    "governed_string_v1_baseline": (
        "reports/candidate-sequence-margin-v1-governed-baseline.json",
        "7f21f70f70da1f63baa9a448bdad34f113d920277e434ac61452efa8f6e633cc",
    ),
    "agpt_string_v1_baseline": (
        "reports/candidate-sequence-activity-v1-agpt-baseline.json",
        "ec5b83495fef9aab6e019f28f2cebfa2bdcd0e6611997e6eb5b2640fb1c37096",
    ),
    "guitar_techs_p2_string_v1_baseline": (
        "reports/candidate-sequence-activity-v1-guitar-techs-baseline.json",
        "d8bef975b37501ec394304737f8e233064e3ec62027b01fd5d2e2c3a6492ca2d",
    ),
}

P3_GROUP_ID = "guitar-techs-P3"
CALIBRATION_FRACTION = 0.2
SEED = 20261122
SEQUENCE_LENGTH = 32
SEQUENCE_STRIDE = 16
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
MINIMUM_LEARNING_RATE = 2.5e-6
GRADIENT_NORM_CAP = 5.0
P3_SAMPLING_WEIGHT = 4.08955223880597
FRONTEND_ONSET_THRESHOLD = 0.35
FRONTEND_FRAME_THRESHOLD = 0.25
FRONTEND_MINIMUM_NOTE_LENGTH = 70.0
REPEAT_INTERVAL = 0.75
EXPECTED_PARTITION_DIGEST = (
    "c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f"
)
EXPECTED_TRAINABLE_PARAMETERS = 93190

EXPECTED_BASELINE: dict[str, float | int] = {
    "aggregate_accuracy": 0.8150606043986179,
    "aggregate_wrong_strings": 3372,
    "repeat_true_positive": 7552,
    "guitarset_accuracy": 0.963257429563875,
    "agpt_accuracy": 0.4627375613922699,
    "guitar_techs_p1_accuracy": 0.4409090909090909,
    "cross_source_macro_accuracy": 0.45182332615068044,
}


@dataclass(frozen=True)
class V3TrackSelection:
    calibration_track_ids: frozenset[str]
    gradient_track_ids: frozenset[str]
    p3_track_ids: frozenset[str]
    partition_keys: frozenset[tuple[str, ...]]
    partition_counts: dict[str, object]


def _dataset_name(annotation: TrackAnnotation) -> str:
    return str(annotation.provenance.get("dataset", "unknown"))


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def verify_artifact_hashes(
    root: Path,
    artifacts: dict[str, tuple[str, str]],
) -> dict[str, dict[str, str]]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Training root does not exist: {root}")
    result: dict[str, dict[str, str]] = {}
    for name, (relative_name, expected_sha256) in sorted(artifacts.items()):
        relative_path = Path(relative_name)
        if relative_path.is_absolute():
            raise ValueError(f"Frozen artifact path must be relative: {name}")
        path = (resolved_root / relative_path).resolve()
        if not _inside_root(path, resolved_root):
            raise ValueError(f"Frozen artifact escapes the training root: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen artifact {name}: {path.name}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Frozen artifact hash mismatch for {name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        result[name] = {
            "path": relative_path.as_posix(),
            "sha256": actual_sha256,
        }
    return result


def verify_file_hash(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> dict[str, str]:
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Missing {name}: {resolved_path.name}")
    actual_sha256 = sha256_file(resolved_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{name} hash mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return {"name": resolved_path.name, "sha256": actual_sha256}


def candidate_cache_set_identity(
    manifest_path: Path,
    cache_directory: Path,
    *,
    onset_threshold: float = FRONTEND_ONSET_THRESHOLD,
    frame_threshold: float = FRONTEND_FRAME_THRESHOLD,
    minimum_note_length: float = FRONTEND_MINIMUM_NOTE_LENGTH,
) -> dict[str, int | str]:
    annotations = load_manifest(manifest_path)
    identities: list[tuple[str, str]] = []
    events = 0
    dropped = 0
    for annotation in annotations:
        audio_path = annotation.resolve_audio(manifest_path)
        cache_path = basic_pitch_cache_path(
            audio_path,
            cache_directory,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen candidate cache: {cache_path.name}"
            )
        counts = validate_candidate_cache(cache_path)
        events += counts["events"]
        dropped += counts["dropped"]
        identities.append((cache_path.name, sha256_file(cache_path)))
    digest = hashlib.sha256()
    for name, cache_sha256 in sorted(identities):
        digest.update(f"{name}:{cache_sha256}\n".encode())
    return {
        "audio_files": len(annotations),
        "events": events,
        "dropped": dropped,
        "cache_set_sha256": digest.hexdigest(),
    }


def validate_candidate_cache_set(
    actual: dict[str, int | str],
    expected: dict[str, int | str],
) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Frozen candidate cache {key} mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


def select_v3_tracks(
    annotations: list[TrackAnnotation],
    *,
    p3_group_id: str = P3_GROUP_ID,
    fraction: float = CALIBRATION_FRACTION,
    seed: int = SEED,
) -> V3TrackSelection:
    track_ids = [annotation.track_id for annotation in annotations]
    if len(set(track_ids)) != len(track_ids):
        duplicates = sorted(
            track_id
            for track_id, count in Counter(track_ids).items()
            if count > 1
        )
        raise ValueError(
            "v3 requires globally unique training track IDs: "
            + ", ".join(duplicates)
        )
    p3_annotations = [
        annotation
        for annotation in annotations
        if annotation.group_id == p3_group_id
    ]
    if not p3_annotations:
        raise ValueError("v3 training manifest contains no P3 annotations")
    if any(_dataset_name(annotation) != "Guitar-TECHS" for annotation in p3_annotations):
        raise ValueError("P3 annotations must belong to Guitar-TECHS")

    partition_annotations = [
        annotation
        for annotation in annotations
        if annotation.group_id != p3_group_id
    ]
    selected_indexes, selected_keys, partition_counts = (
        stratified_calibration_partition(
            partition_annotations,
            fraction=fraction,
            seed=seed,
            stratify_by_group_datasets={"GuitarSet"},
        )
    )
    calibration_track_ids = frozenset(
        partition_annotations[index].track_id
        for index in selected_indexes
    )
    p3_track_ids = frozenset(
        annotation.track_id for annotation in p3_annotations
    )
    gradient_track_ids = frozenset(
        annotation.track_id
        for annotation in partition_annotations
        if (
            _dataset_name(annotation) == "GuitarSet"
            and annotation.track_id not in calibration_track_ids
        )
    ) | p3_track_ids
    if calibration_track_ids & gradient_track_ids:
        raise ValueError("v3 calibration and gradient tracks overlap")
    return V3TrackSelection(
        calibration_track_ids=calibration_track_ids,
        gradient_track_ids=gradient_track_ids,
        p3_track_ids=p3_track_ids,
        partition_keys=frozenset(selected_keys),
        partition_counts=partition_counts,
    )


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(
            f"Frozen {name} mismatch: expected {expected!r}, got {actual!r}"
        )


def _selection_counts(
    annotations: list[TrackAnnotation],
    track_ids: frozenset[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for annotation in annotations:
        if annotation.track_id not in track_ids:
            continue
        label = (
            "Guitar-TECHS P3"
            if annotation.group_id == P3_GROUP_ID
            else _dataset_name(annotation)
        )
        counts[label] += 1
    return dict(sorted(counts.items()))


def candidate_group_window_counts(
    dataset: OnsetPositionDataset,
) -> dict[str, dict[str, int]]:
    groups_by_annotation: Counter[int] = Counter(
        group.annotation_index for group in dataset.groups
    )
    result: dict[str, Counter[str]] = {}
    for annotation_index, annotation in enumerate(dataset.annotations):
        label = (
            "Guitar-TECHS P3"
            if annotation.group_id == P3_GROUP_ID
            else _dataset_name(annotation)
        )
        counts = result.setdefault(label, Counter())
        group_count = groups_by_annotation[annotation_index]
        counts["tracks"] += 1
        counts["groups"] += group_count
        counts["windows"] += len(
            sequence_window_ranges(
                group_count,
                sequence_length=SEQUENCE_LENGTH,
                sequence_stride=SEQUENCE_STRIDE,
            )
        )
    return {
        label: dict(sorted(counts.items()))
        for label, counts in sorted(result.items())
    }


def _position_config_from_checkpoint(path: Path) -> OnsetPositionConfig:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported onset-position checkpoint format")
    return OnsetPositionConfig.from_dict(
        dict(checkpoint.get("model_config", {}))
    )


def build_input_audit(
    training_root: Path,
    preregistration: Path,
    *,
    artifacts: dict[str, tuple[str, str]] = FROZEN_ARTIFACTS,
    source_ledger_path: Path | None = None,
) -> dict[str, object]:
    root = training_root.resolve()
    artifact_report = verify_artifact_hashes(root, artifacts)
    preregistration_report = verify_file_hash(
        preregistration,
        PREREGISTRATION_SHA256,
        name="v3 preregistration",
    )
    ledger_path = (
        source_ledger_path.resolve()
        if source_ledger_path is not None
        else Path(__file__).with_name("source_ledger.json").resolve()
    )
    ledger_report = verify_file_hash(
        ledger_path,
        SOURCE_LEDGER_SHA256,
        name="source ledger",
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    _require_equal(
        "source ledger version",
        ledger.get("ledger_version"),
        SOURCE_LEDGER_VERSION,
    )

    formal_manifest = root / artifacts["formal_manifest"][0]
    annotations = load_manifest(formal_manifest)
    train_annotations = [
        annotation for annotation in annotations if annotation.split == "train"
    ]
    validation_annotations = [
        annotation
        for annotation in annotations
        if annotation.split == "validation"
    ]
    _require_equal("formal-v3 tracks", len(annotations), 1117)
    _require_equal("formal-v3 train tracks", len(train_annotations), 997)
    _require_equal(
        "formal-v3 validation tracks",
        len(validation_annotations),
        120,
    )
    _require_equal(
        "formal-v3 events",
        sum(len(annotation.events) for annotation in annotations),
        131502,
    )
    _require_equal(
        "formal-v3 groups",
        len({annotation.group_id for annotation in annotations}),
        144,
    )

    selection = select_v3_tracks(train_annotations)
    partition_digest = calibration_partition_digest(
        set(selection.partition_keys)
    )
    _require_equal(
        "calibration partition digest",
        partition_digest,
        EXPECTED_PARTITION_DIGEST,
    )
    _require_equal(
        "calibration tracks",
        len(selection.calibration_track_ids),
        268,
    )
    _require_equal("gradient tracks", len(selection.gradient_track_ids), 408)
    gradient_counts = _selection_counts(
        train_annotations,
        selection.gradient_track_ids,
    )
    _require_equal(
        "gradient source counts",
        gradient_counts,
        {"Guitar-TECHS P3": 24, "GuitarSet": 384},
    )

    p3_manifest = root / artifacts["p3_manifest"][0]
    p3_annotations = load_manifest(p3_manifest)
    _require_equal("P3 tracks", len(p3_annotations), 24)
    _require_equal(
        "P3 events",
        sum(len(annotation.events) for annotation in p3_annotations),
        3882,
    )
    _require_equal(
        "P3 manifest track identity",
        {annotation.track_id for annotation in p3_annotations},
        set(selection.p3_track_ids),
    )
    candidate_cache_directory = root / "governed-v1/.basic-pitch-cache"
    p3_cache = candidate_cache_set_identity(
        p3_manifest,
        candidate_cache_directory,
    )
    validate_candidate_cache_set(
        p3_cache,
        {
            "audio_files": 24,
            "events": 6685,
            "dropped": 157,
            "cache_set_sha256": (
                "c592ed2656c29fc4e0c4546a683b7c500c5b8f229fb431ea20eae70f2931b91a"
            ),
        },
    )

    position_checkpoint = root / artifacts["position_checkpoint"][0]
    position_config = _position_config_from_checkpoint(position_checkpoint)
    gradient_dataset = OnsetPositionDataset(
        formal_manifest,
        split="train",
        config=position_config,
        candidate_cache_directory=candidate_cache_directory,
        candidate_onset_threshold=FRONTEND_ONSET_THRESHOLD,
        candidate_frame_threshold=FRONTEND_FRAME_THRESHOLD,
        candidate_minimum_note_length=FRONTEND_MINIMUM_NOTE_LENGTH,
        included_track_ids=set(selection.gradient_track_ids),
        candidate_ambiguity_weight=1.25,
        candidate_rare_pair_max_weight=1.5,
        candidate_cross_string_weight=2.0,
        candidate_cross_string_interval=2.0,
        candidate_max_supervision_weight=4.0,
    )
    sequence_counts = candidate_group_window_counts(gradient_dataset)
    _require_equal(
        "gradient sequence counts",
        sequence_counts,
        {
            "Guitar-TECHS P3": {
                "groups": 3416,
                "tracks": 24,
                "windows": 201,
            },
            "GuitarSet": {
                "groups": 55946,
                "tracks": 384,
                "windows": 3288,
            },
        },
    )
    weighted_p3_share = (
        201 * P3_SAMPLING_WEIGHT
        / (3288 + 201 * P3_SAMPLING_WEIGHT)
    )
    if not math.isclose(weighted_p3_share, 0.2, abs_tol=1e-15):
        raise ValueError(
            "Frozen P3 sampler weight does not produce 20% probability mass"
        )

    return {
        "schema_version": 1,
        "kind": "candidate-sequence-string-v3-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_read": False,
        "cross_source_validation_read": False,
        "preregistration": preregistration_report,
        "source_ledger": {
            **ledger_report,
            "version": SOURCE_LEDGER_VERSION,
        },
        "artifacts": artifact_report,
        "formal_manifest": {
            "tracks": len(annotations),
            "events": sum(
                len(annotation.events) for annotation in annotations
            ),
            "train_tracks": len(train_annotations),
            "validation_tracks": len(validation_annotations),
            "groups": len(
                {annotation.group_id for annotation in annotations}
            ),
        },
        "partition": {
            "fraction": CALIBRATION_FRACTION,
            "seed": SEED,
            "selected_key_sha256": partition_digest,
            "selected_tracks": len(selection.calibration_track_ids),
            "datasets": selection.partition_counts,
            "p3_excluded_from_construction": True,
        },
        "gradient": {
            "tracks": len(selection.gradient_track_ids),
            "sources": gradient_counts,
            "sequence": sequence_counts,
        },
        "p3_candidate_cache": p3_cache,
        "sampling": {
            "replacement": True,
            "draws_per_epoch": 3489,
            "guitarset_weight": 1.0,
            "p3_weight": P3_SAMPLING_WEIGHT,
            "expected_p3_share": weighted_p3_share,
        },
    }


def write_input_audit(path: Path, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def calibration_metrics(
    model: CandidateSequenceStringModel,
    groups: list[CandidateSequenceGroup],
    annotations: list[TrackAnnotation],
    device: torch.device,
    *,
    inference_batch_size: int,
) -> dict[str, object]:
    refined_groups = refine_groups_with_string_sequence(
        groups,
        model,
        device,
        batch_size=inference_batch_size,
    )
    audit = assignment_audit(refined_groups, annotations)
    aggregate = dict(audit["aggregate"])
    by_dataset = {
        str(name): dict(values)
        for name, values in dict(audit["by_dataset"]).items()
    }
    required_datasets = {"GuitarSet", "AG-PT-set", "Guitar-TECHS"}
    missing_datasets = required_datasets - set(by_dataset)
    if missing_datasets:
        raise ValueError(
            "Calibration audit is missing frozen datasets: "
            + ", ".join(sorted(missing_datasets))
        )
    guitarset_accuracy = float(
        by_dataset["GuitarSet"]["accepted_string_accuracy"]
    )
    agpt_accuracy = float(
        by_dataset["AG-PT-set"]["accepted_string_accuracy"]
    )
    guitar_techs_p1_accuracy = float(
        by_dataset["Guitar-TECHS"]["accepted_string_accuracy"]
    )
    return {
        "aggregate_accuracy": float(
            aggregate["accepted_string_accuracy"]
        ),
        "aggregate_wrong_strings": int(aggregate["wrong_string"]),
        "repeat_true_positive": int(
            aggregate["repeat_correct_string"]
        ),
        "guitarset_accuracy": guitarset_accuracy,
        "agpt_accuracy": agpt_accuracy,
        "guitar_techs_p1_accuracy": guitar_techs_p1_accuracy,
        "cross_source_macro_accuracy": (
            agpt_accuracy + guitar_techs_p1_accuracy
        )
        / 2,
        "assignment_audit": audit,
    }


def validate_epoch_zero(
    metrics: dict[str, object],
    expected: dict[str, float | int] = EXPECTED_BASELINE,
) -> None:
    for name, expected_value in expected.items():
        if name not in metrics:
            raise ValueError(f"Epoch-zero metric is missing: {name}")
        actual_value = metrics[name]
        if isinstance(expected_value, int):
            if int(actual_value) != expected_value:
                raise ValueError(
                    f"Epoch-zero {name} mismatch: expected {expected_value}, "
                    f"got {actual_value}"
                )
            continue
        if not math.isclose(
            float(actual_value),
            expected_value,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Epoch-zero {name} mismatch: expected {expected_value}, "
                f"got {actual_value}"
            )


def epoch_is_eligible(
    metrics: dict[str, object],
    baseline: dict[str, float | int] = EXPECTED_BASELINE,
) -> bool:
    return (
        float(metrics["aggregate_accuracy"])
        > float(baseline["aggregate_accuracy"])
        and int(metrics["aggregate_wrong_strings"])
        < int(baseline["aggregate_wrong_strings"])
        and int(metrics["repeat_true_positive"])
        >= int(baseline["repeat_true_positive"])
        and float(metrics["guitarset_accuracy"])
        >= float(baseline["guitarset_accuracy"])
        and float(metrics["cross_source_macro_accuracy"])
        >= float(baseline["cross_source_macro_accuracy"])
        and float(metrics["agpt_accuracy"])
        >= float(baseline["agpt_accuracy"]) - 0.005
        and float(metrics["guitar_techs_p1_accuracy"])
        >= float(baseline["guitar_techs_p1_accuracy"]) - 0.005
    )


def epoch_selection_key(
    epoch: int,
    metrics: dict[str, object],
) -> tuple[float, int, float, int]:
    return (
        float(metrics["aggregate_accuracy"]),
        int(metrics["repeat_true_positive"]),
        float(metrics["cross_source_macro_accuracy"]),
        -epoch,
    )


def run_training_epoch(
    model: CandidateSequenceStringModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float | int]:
    model.train()
    loss_total = 0.0
    correct_total = 0
    candidate_total = 0
    batches = 0
    for source_batch in loader:
        batch = move_batch(source_batch, device)
        optimizer.zero_grad(set_to_none=True)
        residuals = model(
            batch["pitches"],
            batch["confidences"],
            batch["durations"],
            batch["base_string_logits"],
            batch["valid_strings"],
            batch["candidate_mask"],
            batch["group_mask"],
            batch["group_onsets"],
        )
        loss, correct, total = candidate_sequence_loss(
            batch["base_string_logits"],
            residuals,
            batch["string_targets"],
            batch["target_weights"],
            batch["candidate_mask"],
            batch["valid_strings"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_NORM_CAP,
        )
        optimizer.step()
        loss_total += float(loss.detach().cpu())
        correct_total += correct
        candidate_total += total
        batches += 1
    return {
        "loss": loss_total / max(1, batches),
        "window_candidate_oracle_string_accuracy": (
            correct_total / max(1, candidate_total)
        ),
        "window_active_candidates": candidate_total,
    }


def _fixed_output_paths(root: Path) -> dict[str, Path]:
    return {
        "audit": root / "reports/candidate-sequence-string-v3-input-audit.json",
        "output": (
            root
            / "checkpoints/guitarset-candidate-sequence-string-v3.pt"
        ),
        "log": root / "reports/candidate-sequence-string-v3-training.jsonl",
    }


def _validate_sequence_model(model: CandidateSequenceStringModel) -> int:
    _require_equal(
        "sequence length",
        model.config.sequence_length,
        SEQUENCE_LENGTH,
    )
    _require_equal(
        "sequence stride",
        model.config.sequence_stride,
        SEQUENCE_STRIDE,
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    _require_equal(
        "trainable sequence parameters",
        trainable_parameters,
        EXPECTED_TRAINABLE_PARAMETERS,
    )
    return trainable_parameters


def _build_training_datasets(
    root: Path,
    formal_manifest: Path,
    position_config: OnsetPositionConfig,
    selection: V3TrackSelection,
) -> tuple[OnsetPositionDataset, OnsetPositionDataset]:
    common = {
        "split": "train",
        "config": position_config,
        "feature_cache_directory": root / "governed-v1/.feature-cache",
        "candidate_cache_directory": root / "governed-v1/.basic-pitch-cache",
        "candidate_onset_threshold": FRONTEND_ONSET_THRESHOLD,
        "candidate_frame_threshold": FRONTEND_FRAME_THRESHOLD,
        "candidate_minimum_note_length": FRONTEND_MINIMUM_NOTE_LENGTH,
    }
    training_dataset = OnsetPositionDataset(
        formal_manifest,
        **common,
        included_track_ids=set(selection.gradient_track_ids),
        candidate_ambiguity_weight=1.25,
        candidate_rare_pair_max_weight=1.5,
        candidate_cross_string_weight=2.0,
        candidate_cross_string_interval=2.0,
        candidate_max_supervision_weight=4.0,
    )
    calibration_dataset = OnsetPositionDataset(
        formal_manifest,
        **common,
        included_track_ids=set(selection.calibration_track_ids),
        positive_only_datasets={"AG-PT-set", "Guitar-TECHS"},
    )
    return training_dataset, calibration_dataset


def _checkpoint_payload(
    *,
    model: CandidateSequenceStringModel,
    position_config: OnsetPositionConfig,
    feature_config: FeatureConfig,
    position_metadata: dict[str, object],
    initial_string_metadata: dict[str, object],
    root: Path,
    formal_manifest: Path,
    audit_path: Path,
    epoch: int,
    baseline: dict[str, object],
    selected: dict[str, object],
    selection: V3TrackSelection,
    sampling_plan: dict[str, object],
    sequence_counts: dict[str, dict[str, int]],
    cache_report: dict[str, object],
    trainable_parameters: int,
) -> dict[str, object]:
    position_path = root / FROZEN_ARTIFACTS["position_checkpoint"][0]
    string_path = root / FROZEN_ARTIFACTS["string_v1_checkpoint"][0]
    annotations = load_manifest(formal_manifest)
    return {
        "format_version": 1,
        "model_state": model.state_dict(),
        "sequence_config": model.config.to_dict(),
        "position_config": position_config.to_dict(),
        "feature_config": feature_config.to_dict(),
        "metadata": {
            "name": "stringtrace-candidate-sequence-string",
            "version": "candidate-sequence-string-v3",
            "kind": "trained",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_checkpoint": {
                "name": position_path.name,
                "sha256": sha256_file(position_path),
            },
            "base_model_metadata": position_metadata,
            "initialized_string_checkpoint": {
                "name": string_path.name,
                "sha256": sha256_file(string_path),
            },
            "initialized_string_metadata": initial_string_metadata,
            "input_audit": {
                "name": audit_path.name,
                "sha256": sha256_file(audit_path),
            },
            "training_data": manifest_identity(
                formal_manifest,
                annotations,
                purpose="training",
            ),
            "seed": SEED,
            "training_device": "cpu",
            "epoch": epoch,
            "selection_metric": (
                "training_only_assignment_accuracy_repeat_macro"
            ),
            "selection_metric_value": float(
                selected["aggregate_accuracy"]
            ),
            "calibration_gate_passed": True,
            "calibration_baseline": baseline,
            "calibration_selected": selected,
            "trainable_scope": "candidate-sequence-string-full-model",
            "trainable_parameters": trainable_parameters,
            "objective": {
                "kind": "per_candidate_set_valued_cross_entropy",
                "activity_threshold": 0.5,
                "ambiguity_weight": 1.25,
                "rare_pair_max_weight": 1.5,
                "cross_string_weight": 2.0,
                "cross_string_interval": 2.0,
                "maximum_supervision_weight": 4.0,
                "gradient_norm_cap": GRADIENT_NORM_CAP,
            },
            "optimization": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "scheduler": "cosine",
                "minimum_learning_rate": MINIMUM_LEARNING_RATE,
            },
            "partition": {
                "fraction": CALIBRATION_FRACTION,
                "seed": SEED,
                "selected_key_sha256": calibration_partition_digest(
                    set(selection.partition_keys)
                ),
                "selected_tracks": len(selection.calibration_track_ids),
                "p3_excluded_from_construction": True,
                "datasets": selection.partition_counts,
            },
            "sampling": sampling_plan,
            "candidate_frontend": {
                "kind": "spotify-basic-pitch-0.4",
                "onset_threshold": FRONTEND_ONSET_THRESHOLD,
                "frame_threshold": FRONTEND_FRAME_THRESHOLD,
                "minimum_note_length": FRONTEND_MINIMUM_NOTE_LENGTH,
            },
            "sequence_data": sequence_counts,
            "positive_only_calibration_datasets": [
                "AG-PT-set",
                "Guitar-TECHS",
            ],
            "feature_cache": cache_report,
        },
    }


def run_training(
    training_root: Path,
    audit_path: Path,
    *,
    precompute_device_name: str,
    precompute_batch_size: int,
    inference_batch_size: int,
) -> dict[str, object]:
    root = training_root.resolve()
    formal_manifest = root / FROZEN_ARTIFACTS["formal_manifest"][0]
    train_annotations = load_manifest(formal_manifest, split="train")
    selection = select_v3_tracks(train_annotations)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    precompute_device = choose_device(precompute_device_name)
    training_device = torch.device("cpu")
    position_path = root / FROZEN_ARTIFACTS["position_checkpoint"][0]
    string_path = root / FROZEN_ARTIFACTS["string_v1_checkpoint"][0]
    (
        position_model,
        feature_config,
        position_config,
        position_metadata,
    ) = load_position_model(position_path, precompute_device)
    model, initial_string_metadata = load_candidate_sequence_model(
        string_path,
        position_path,
        position_config,
        feature_config,
        training_device,
    )
    trainable_parameters = _validate_sequence_model(model)
    training_dataset, calibration_dataset = _build_training_datasets(
        root,
        formal_manifest,
        position_config,
        selection,
    )
    cache_report = {
        "training": training_dataset.prepare_feature_cache(),
        "calibration": calibration_dataset.prepare_feature_cache(),
    }

    print(
        json.dumps(
            {
                "stage": "precompute_v3_groups",
                "training_groups": len(training_dataset),
                "calibration_groups": len(calibration_dataset),
                "precompute_device": str(precompute_device),
                "training_device": "cpu",
            }
        ),
        flush=True,
    )
    training_groups = precompute_candidate_sequence_groups(
        training_dataset,
        position_model,
        precompute_device,
        batch_size=precompute_batch_size,
        repeat_interval=REPEAT_INTERVAL,
    )
    calibration_groups = precompute_candidate_sequence_groups(
        calibration_dataset,
        position_model,
        precompute_device,
        batch_size=precompute_batch_size,
        repeat_interval=REPEAT_INTERVAL,
    )
    training_windows = CandidateSequenceWindowDataset(
        training_groups,
        model.config,
    )
    sequence_counts = candidate_group_window_counts(training_dataset)
    _require_equal(
        "precomputed training groups",
        len(training_groups),
        55946 + 3416,
    )
    _require_equal("training windows", len(training_windows), 3489)
    sampling_weights, sampling_plan = window_sampling_plan(
        training_windows,
        training_dataset.annotations,
        {
            "GuitarSet": 1.0,
            "Guitar-TECHS": P3_SAMPLING_WEIGHT,
        },
    )
    _require_equal("sampler draws", sampling_plan["num_samples"], 3489)
    expected_shares = dict(sampling_plan["expected_dataset_share"])
    if not math.isclose(
        float(expected_shares["Guitar-TECHS"]),
        0.2,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("P3 replacement sampler share is not exactly 20%")
    loader = DataLoader(
        training_windows,
        batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(
            sampling_weights,
            num_samples=len(training_windows),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        ),
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        collate_fn=collate_candidate_sequence_windows,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MINIMUM_LEARNING_RATE,
    )

    output_paths = _fixed_output_paths(root)
    log_path = output_paths["log"]
    output_path = output_paths["output"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    baseline = calibration_metrics(
        model,
        calibration_groups,
        calibration_dataset.annotations,
        training_device,
        inference_batch_size=inference_batch_size,
    )
    validate_epoch_zero(baseline)
    append_log(
        log_path,
        {
            "epoch": 0,
            "stage": "frozen_string_v1_baseline",
            "calibration": baseline,
        },
    )
    print(
        json.dumps({"epoch": 0, "calibration": baseline}),
        flush=True,
    )

    best_key: tuple[float, int, float, int] | None = None
    best_epoch = 0
    best_metrics: dict[str, object] | None = None
    for epoch in range(1, EPOCHS + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_training_epoch(
            model,
            loader,
            training_device,
            optimizer,
        )
        metrics = calibration_metrics(
            model,
            calibration_groups,
            calibration_dataset.annotations,
            training_device,
            inference_batch_size=inference_batch_size,
        )
        eligible = epoch_is_eligible(metrics)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "calibration": metrics,
            "eligible": eligible,
        }
        append_log(log_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        selection_key = epoch_selection_key(epoch, metrics)
        if eligible and (best_key is None or selection_key > best_key):
            best_key = selection_key
            best_epoch = epoch
            best_metrics = metrics
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _checkpoint_payload(
                    model=model,
                    position_config=position_config,
                    feature_config=feature_config,
                    position_metadata=position_metadata,
                    initial_string_metadata=initial_string_metadata,
                    root=root,
                    formal_manifest=formal_manifest,
                    audit_path=audit_path,
                    epoch=epoch,
                    baseline=baseline,
                    selected=metrics,
                    selection=selection,
                    sampling_plan=sampling_plan,
                    sequence_counts=sequence_counts,
                    cache_report=cache_report,
                    trainable_parameters=trainable_parameters,
                ),
                output_path,
            )
        scheduler.step()

    result: dict[str, object] = {
        "calibration_gate_passed": best_metrics is not None,
        "best_epoch": best_epoch,
        "baseline": baseline,
        "selected": best_metrics,
        "trainable_parameters": trainable_parameters,
    }
    if best_metrics is not None:
        result["output"] = output_path.name
        result["sha256"] = sha256_file(output_path)
    else:
        result["output"] = None
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and run the frozen candidate sequence string v3 "
            "training contract."
        )
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--precompute-device", default="auto")
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    for name in ("precompute_batch_size", "inference_batch_size"):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    root = arguments.training_root.resolve()
    output_paths = _fixed_output_paths(root)
    if not arguments.audit_only:
        stale = [
            path
            for path in (output_paths["output"], output_paths["log"])
            if path.exists()
        ]
        if stale:
            raise FileExistsError(
                "Refusing to overwrite stale v3 training output: "
                + ", ".join(path.name for path in stale)
            )

    audit = build_input_audit(root, arguments.preregistration)
    write_input_audit(output_paths["audit"], audit)
    print(
        json.dumps(
            {
                "stage": "input_audit",
                "passed": True,
                "output": output_paths["audit"].name,
            }
        ),
        flush=True,
    )
    if arguments.audit_only:
        return
    run_training(
        root,
        output_paths["audit"],
        precompute_device_name=arguments.precompute_device,
        precompute_batch_size=arguments.precompute_batch_size,
        inference_batch_size=arguments.inference_batch_size,
    )


if __name__ == "__main__":
    main()
