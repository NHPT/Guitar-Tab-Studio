from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .analyze_candidate_gradient_conflict import (
    identity_sha256,
    windows_by_track,
)
from .analyze_candidate_guitarset_partition_gradient_conflict import (
    recording_family_root,
)
from .candidate_sequence import (
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    collate_candidate_sequence_windows,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import sha256_file
from .features import FeatureConfig
from .onset_position import OnsetPositionConfig
from .schema import TrackAnnotation, load_manifest
from .train import choose_device
from .train_candidate_sequence import append_log, load_position_model
from .train_candidate_sequence_v3 import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MINIMUM_LEARNING_RATE,
    REPEAT_INTERVAL,
    SEED,
    V3TrackSelection,
    WEIGHT_DECAY,
    _build_training_datasets,
    _require_equal,
    _validate_sequence_model,
    calibration_metrics,
    candidate_group_window_counts,
    epoch_is_eligible,
    epoch_selection_key,
    select_v3_tracks,
    validate_epoch_zero,
    verify_file_hash,
)
from .train_candidate_sequence_v4 import (
    source_gradient_diagnostics,
    window_source_names,
)
from .train_candidate_sequence_v6 import (
    GUITARSET_WINDOWS,
    P3_WINDOWS,
    window_identities,
    window_track_ids,
)
from .train_candidate_sequence_v7 import (
    drift_was_prevented,
    run_anchored_training_epoch,
    snapshot_anchor,
    validate_calibration_report,
)
from .train_candidate_sequence_v8 import (
    EXPECTED_P3_SCHEDULE,
    P3_GENERATOR_SEED,
    checkpoint_payload as v8_checkpoint_payload,
    p3_injection_count,
    window_active_candidate_exposures,
)


PREREGISTRATION_SHA256 = (
    "302a77e8225a3f9d0950ebadf60d7a7c0da379db129eb1fb1900b13f35e2e66a"
)
V8_PREREGISTRATION_SHA256 = (
    "97014e726ec595ef9f689adba76fe69340afb2639a9e4b2edce338b67d7cb0ba"
)
V8_INPUT_AUDIT_SHA256 = (
    "417760f62cb5576bde0393c3015715867ff444fdb97bafd9d83ae528ac96a901"
)
V8_TRAINING_LOG_SHA256 = (
    "30850a58f3a8847e390303da69ecc5e1f4356cd01cfc84e1e3ff73f327e49a63"
)
V8_RESULT_SHA256 = (
    "bc54b167d6d2eba7b2ab0ebdfeedaeabb8023928fef1a892818179fb41a74202"
)
V8_CHECKPOINT_SHA256 = (
    "0bf00091daaaf6c8428c982d1bf0e30165460eeae821a96a7552bbad83aad726"
)
V7_CALIBRATION_SHA256 = (
    "460bf75ad6b581ed6d162c658e4e000074269871f3415cb4e5d30b49e11c527f"
)
FORMAL_MANIFEST_SHA256 = (
    "cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d"
)
P3_MANIFEST_SHA256 = (
    "f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8"
)
POSITION_CHECKPOINT_SHA256 = (
    "7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e"
)
STRING_V1_CHECKPOINT_SHA256 = (
    "392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982"
)
SOURCE_LEDGER_SHA256 = (
    "de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a"
)
EXPECTED_BASE_EXPOSURES = 109_007
ATTRIBUTION_METRICS = (
    "aggregate_accuracy",
    "aggregate_wrong_strings",
    "repeat_true_positive",
    "guitarset_accuracy",
    "agpt_accuracy",
    "guitar_techs_p1_accuracy",
    "cross_source_macro_accuracy",
)


def fixed_output_paths(root: Path) -> dict[str, Path]:
    return {
        "audit": root / "reports/candidate-sequence-string-v9-input-audit.json",
        "output": (
            root
            / "checkpoints/guitarset-candidate-sequence-string-v9.pt"
        ),
        "log": root / "reports/candidate-sequence-string-v9-training.jsonl",
    }


def _verify_evidence(
    root: Path,
    preregistration: Path,
) -> dict[str, dict[str, str]]:
    docs = preregistration.parent
    files = {
        "preregistration": (
            preregistration,
            PREREGISTRATION_SHA256,
        ),
        "v8_preregistration": (
            docs / "candidate-sequence-string-v8-preregistration.md",
            V8_PREREGISTRATION_SHA256,
        ),
        "v8_input_audit": (
            root / "reports/candidate-sequence-string-v8-input-audit.json",
            V8_INPUT_AUDIT_SHA256,
        ),
        "v8_training_log": (
            root / "reports/candidate-sequence-string-v8-training.jsonl",
            V8_TRAINING_LOG_SHA256,
        ),
        "v8_result": (
            docs / "candidate-sequence-string-v8-result.md",
            V8_RESULT_SHA256,
        ),
        "v8_checkpoint": (
            root / "checkpoints/guitarset-candidate-sequence-string-v8.pt",
            V8_CHECKPOINT_SHA256,
        ),
        "v7_anchor_calibration": (
            root
            / "reports/candidate-sequence-string-v7-anchor-calibration.json",
            V7_CALIBRATION_SHA256,
        ),
        "formal_manifest": (
            root / "governed-v1/formal-training-manifest-v3.jsonl",
            FORMAL_MANIFEST_SHA256,
        ),
        "p3_manifest": (
            root
            / (
                "governed-v1/"
                "guitar-techs-p3-music-position-train-candidate-v1.jsonl"
            ),
            P3_MANIFEST_SHA256,
        ),
        "position_checkpoint": (
            root
            / "checkpoints/guitarset-candidate-activity-residual-v1.pt",
            POSITION_CHECKPOINT_SHA256,
        ),
        "string_v1_checkpoint": (
            root
            / "checkpoints/guitarset-candidate-sequence-string-v1.pt",
            STRING_V1_CHECKPOINT_SHA256,
        ),
        "source_ledger": (
            Path(__file__).with_name("source_ledger.json"),
            SOURCE_LEDGER_SHA256,
        ),
    }
    return {
        name: verify_file_hash(path, expected, name=name.replace("_", " "))
        for name, (path, expected) in files.items()
    }


class GuitarSetReplayInjectionSampler(Sampler[int]):
    def __init__(
        self,
        guitarset_indices: Iterable[int],
        *,
        base_generator: torch.Generator,
        injection_generator: torch.Generator,
    ) -> None:
        self.guitarset_indices = tuple(guitarset_indices)
        if not self.guitarset_indices:
            raise ValueError("GuitarSet replay injection requires indexes")
        if len(set(self.guitarset_indices)) != len(
            self.guitarset_indices
        ):
            raise ValueError("GuitarSet replay indexes must be unique")
        if min(self.guitarset_indices) < 0:
            raise ValueError("GuitarSet replay indexes must be non-negative")
        self.base_generator = base_generator
        self.injection_generator = injection_generator
        self.epoch = 0
        self.last_indices: list[int] = []
        self.last_base_indices: list[int] = []
        self.last_injected_indices: list[int] = []
        self.last_insertion_positions: list[int] = []

    def __iter__(self) -> Iterator[int]:
        self.epoch += 1
        injected_count = p3_injection_count(
            self.epoch,
            guitarset_windows=len(self.guitarset_indices),
        )
        base_order = torch.randperm(
            len(self.guitarset_indices),
            generator=self.base_generator,
        ).tolist()
        self.last_base_indices = [
            self.guitarset_indices[index] for index in base_order
        ]
        injected_order = torch.randperm(
            len(self.guitarset_indices),
            generator=self.injection_generator,
        ).tolist()
        self.last_injected_indices = [
            self.guitarset_indices[index]
            for index in injected_order[:injected_count]
        ]
        self.last_insertion_positions = [
            math.floor(
                (index + 1)
                * len(self.last_base_indices)
                / (injected_count + 1)
            )
            for index in range(injected_count)
        ]
        if (
            len(set(self.last_insertion_positions))
            != injected_count
        ):
            raise RuntimeError(
                "GuitarSet replay insertion positions are not unique"
            )
        injected_by_position = dict(
            zip(
                self.last_insertion_positions,
                self.last_injected_indices,
            )
        )
        combined: list[int] = []
        for position, base_index in enumerate(self.last_base_indices):
            if position in injected_by_position:
                combined.append(injected_by_position[position])
            combined.append(base_index)
        if len(combined) != len(self.last_base_indices) + injected_count:
            raise RuntimeError(
                "GuitarSet replay injection changed draw counts"
            )
        self.last_indices = combined
        return iter(self.last_indices)

    def __len__(self) -> int:
        next_epoch = self.epoch + 1
        return len(self.guitarset_indices) + p3_injection_count(
            next_epoch,
            guitarset_windows=len(self.guitarset_indices),
        )


def _partition_report(
    indexes: list[int] | tuple[int, ...],
    identities: list[str],
    track_ids: list[str],
    active_exposures: list[int],
    annotations_by_id: dict[str, TrackAnnotation],
) -> dict[str, object]:
    local_track_ids = [track_ids[index] for index in indexes]
    annotations = [
        annotations_by_id[track_id] for track_id in local_track_ids
    ]
    families = {
        recording_family_root(annotation) for annotation in annotations
    }
    return {
        "draws": len(indexes),
        "unique_windows": len(set(indexes)),
        "duplicate_draws": len(indexes) - len(set(indexes)),
        "window_identity_sha256": identity_sha256(
            identities[index] for index in indexes
        ),
        "active_candidate_exposures": sum(
            active_exposures[index] for index in indexes
        ),
        "tracks": {
            "count": len(set(local_track_ids)),
            "identity_sha256": identity_sha256(local_track_ids),
        },
        "players": dict(
            sorted(
                Counter(
                    annotation.group_id for annotation in annotations
                ).items()
            )
        ),
        "variants": dict(
            sorted(
                Counter(
                    (
                        "derived"
                        if annotation.provenance.get(
                            "derived_from_track_id"
                        )
                        else "original"
                    )
                    for annotation in annotations
                ).items()
            )
        ),
        "families": {
            "count": len(families),
            "identity_sha256": identity_sha256(families),
        },
    }


def replay_coverage_report(
    sampler: GuitarSetReplayInjectionSampler,
    identities: list[str],
    track_ids: list[str],
    active_exposures: list[int],
    annotations_by_id: dict[str, TrackAnnotation],
) -> dict[str, object]:
    if not (
        len(identities) == len(track_ids) == len(active_exposures)
    ):
        raise ValueError("replay coverage metadata does not align")
    expected_draws = (
        len(sampler.last_base_indices)
        + len(sampler.last_injected_indices)
    )
    if len(sampler.last_indices) != expected_draws:
        raise RuntimeError(
            "GuitarSet replay sampler has not completed an epoch"
        )
    expected = set(sampler.guitarset_indices)
    observed_base = set(sampler.last_base_indices)
    combined = _partition_report(
        sampler.last_indices,
        identities,
        track_ids,
        active_exposures,
        annotations_by_id,
    )
    base = _partition_report(
        sampler.last_base_indices,
        identities,
        track_ids,
        active_exposures,
        annotations_by_id,
    )
    injected = _partition_report(
        sampler.last_injected_indices,
        identities,
        track_ids,
        active_exposures,
        annotations_by_id,
    )
    base.update(
        {
            "eligible_window_identity_sha256": identity_sha256(
                identities[index] for index in sampler.guitarset_indices
            ),
            "omitted_eligible_windows": len(expected - observed_base),
            "unexpected_windows": len(observed_base - expected),
        }
    )
    return {
        "draws": len(sampler.last_indices),
        "batches": math.ceil(len(sampler.last_indices) / BATCH_SIZE),
        "counts": {"GuitarSet": len(sampler.last_indices)},
        "shares": {"GuitarSet": 1.0},
        "p3_optimizer_windows": 0,
        "base": base,
        "injected": injected,
        "combined": combined,
        "insertion_positions": list(
            sampler.last_insertion_positions
        ),
    }


def validate_replay_coverage(
    report: dict[str, object],
    *,
    epoch: int,
) -> None:
    injected_expected = p3_injection_count(epoch)
    base = dict(report["base"])
    injected = dict(report["injected"])
    combined = dict(report["combined"])
    for name, actual, expected in (
        ("batches", report["batches"], 208),
        ("base draws", base["draws"], GUITARSET_WINDOWS),
        ("base unique windows", base["unique_windows"], GUITARSET_WINDOWS),
        ("base duplicate draws", base["duplicate_draws"], 0),
        ("base omitted windows", base["omitted_eligible_windows"], 0),
        ("base unexpected windows", base["unexpected_windows"], 0),
        ("base tracks", dict(base["tracks"])["count"], 384),
        ("base families", dict(base["families"])["count"], 234),
        (
            "base active exposures",
            base["active_candidate_exposures"],
            EXPECTED_BASE_EXPOSURES,
        ),
        ("injected draws", injected["draws"], injected_expected),
        (
            "injected unique windows",
            injected["unique_windows"],
            injected_expected,
        ),
        ("injected duplicate draws", injected["duplicate_draws"], 0),
        (
            "combined draws",
            combined["draws"],
            GUITARSET_WINDOWS + injected_expected,
        ),
        (
            "combined unique windows",
            combined["unique_windows"],
            GUITARSET_WINDOWS,
        ),
        (
            "combined replay duplicates",
            combined["duplicate_draws"],
            injected_expected,
        ),
        ("P3 optimizer windows", report["p3_optimizer_windows"], 0),
    ):
        _require_equal(f"epoch {epoch} {name}", actual, expected)
    _require_equal(
        f"epoch {epoch} base identity",
        base["window_identity_sha256"],
        base["eligible_window_identity_sha256"],
    )
    if int(injected["active_candidate_exposures"]) <= 0:
        raise ValueError(
            f"epoch {epoch} injected replay has no active exposures"
        )
    _require_equal(
        f"epoch {epoch} combined active exposures",
        combined["active_candidate_exposures"],
        (
            int(base["active_candidate_exposures"])
            + int(injected["active_candidate_exposures"])
        ),
    )


def source_content_attribution(
    v8_metrics: dict[str, object],
    v9_metrics: dict[str, object],
) -> str:
    def no_worse(
        left: dict[str, object],
        right: dict[str, object],
    ) -> tuple[bool, bool]:
        comparisons = []
        for name in ATTRIBUTION_METRICS:
            left_value = float(left[name])
            right_value = float(right[name])
            if name == "aggregate_wrong_strings":
                comparisons.append(
                    (left_value <= right_value, left_value < right_value)
                )
            else:
                comparisons.append(
                    (left_value >= right_value, left_value > right_value)
                )
        return (
            all(item[0] for item in comparisons),
            any(item[1] for item in comparisons),
        )

    v8_no_worse, v8_strict = no_worse(v8_metrics, v9_metrics)
    v9_no_worse, v9_strict = no_worse(v9_metrics, v8_metrics)
    if v8_no_worse and v8_strict:
        return "p3_content_advantage"
    if v9_no_worse and (v9_strict or not v8_strict):
        return "extra_steps_sufficient"
    return "mixed_inconclusive"


def build_input_audit(
    training_root: Path,
    preregistration: Path,
) -> dict[str, object]:
    root = training_root.resolve()
    preregistration = preregistration.resolve()
    evidence = _verify_evidence(root, preregistration)

    calibration_path = (
        root
        / "reports/candidate-sequence-string-v7-anchor-calibration.json"
    )
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    radii = validate_calibration_report(calibration)
    v8_audit = json.loads(
        (
            root / "reports/candidate-sequence-string-v8-input-audit.json"
        ).read_text(encoding="utf-8")
    )
    _require_equal(
        "v8 audit kind",
        v8_audit.get("kind"),
        "candidate-sequence-string-v8-input-audit",
    )
    _require_equal("v8 audit passed", v8_audit.get("passed"), True)
    _require_equal(
        "v8 audit calibration hash",
        dict(v8_audit["frozen_anchor"])["sha256"],
        V7_CALIBRATION_SHA256,
    )

    v8_log_path = (
        root / "reports/candidate-sequence-string-v8-training.jsonl"
    )
    v8_rows = [
        json.loads(line)
        for line in v8_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require_equal(
        "v8 training epochs",
        [row.get("epoch") for row in v8_rows],
        [0, 1, 2, 3],
    )
    _require_equal(
        "v8 drift decisions",
        [row.get("drift_prevented") for row in v8_rows[1:]],
        [True, False, True],
    )
    _require_equal(
        "v8 production decisions",
        [row.get("production_eligible") for row in v8_rows[1:]],
        [True, False, True],
    )
    _require_equal(
        "v8 P3 schedule",
        [
            int(dict(row["sampling"])["p3"]["draws"])
            for row in v8_rows[1:]
        ],
        list(EXPECTED_P3_SCHEDULE),
    )
    v8_epoch_one = {
        name: v8_rows[1]["calibration"][name]
        for name in ATTRIBUTION_METRICS
    }
    v8_checkpoint_path = (
        root / "checkpoints/guitarset-candidate-sequence-string-v8.pt"
    )
    checkpoint = torch.load(
        v8_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    metadata = dict(checkpoint["metadata"])
    _require_equal(
        "v8 checkpoint version",
        metadata.get("version"),
        "candidate-sequence-string-v8",
    )
    _require_equal("v8 checkpoint epoch", metadata.get("epoch"), 1)
    experiment = dict(metadata["experiment"])
    _require_equal(
        "v8 checkpoint calibration hash",
        experiment.get("anchor_calibration_sha256"),
        V7_CALIBRATION_SHA256,
    )
    _require_equal(
        "v8 checkpoint radii",
        dict(experiment.get("radii", {})),
        radii,
    )

    return {
        "schema_version": 1,
        "kind": "candidate-sequence-string-v9-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_result_read": False,
        "cross_source_validation_result_read": False,
        "evidence": evidence,
        "frozen_anchor": {
            "name": calibration_path.name,
            "sha256": V7_CALIBRATION_SHA256,
            "radii": radii,
            "recalibrated": False,
        },
        "v8_matched_epoch_one": {
            "drift_prevented": True,
            "production_eligible": True,
            "metrics": v8_epoch_one,
        },
        "single_policy_change_from_v8": {
            "kind": "injected-window-source",
            "v8_source": "Guitar-TECHS P3",
            "v9_source": "GuitarSet",
            "injection_counts_changed": False,
            "insertion_positions_changed": False,
            "optimizer_steps_changed": False,
            "anchor_changed": False,
            "task_loss_changed": False,
        },
        "sampling": {
            "base_guitarset_windows_per_epoch": GUITARSET_WINDOWS,
            "injected_guitarset_windows_by_epoch": list(
                EXPECTED_P3_SCHEDULE
            ),
            "p3_optimizer_windows": [0, 0, 0],
            "base_seed": SEED,
            "injection_seed": P3_GENERATOR_SEED,
            "batches_per_epoch": [208, 208, 208],
            "total_optimizer_steps": 624,
        },
        "source_attribution_rule": {
            "primary_epoch": 1,
            "metrics": list(ATTRIBUTION_METRICS),
            "outcomes": [
                "p3_content_advantage",
                "extra_steps_sufficient",
                "mixed_inconclusive",
            ],
        },
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_locked_inputs(
    audit_path: Path,
    calibration_path: Path,
) -> tuple[dict[str, float], str, dict[str, object]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for name, expected in {
        "kind": "candidate-sequence-string-v9-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
    }.items():
        _require_equal(f"v9 input audit {name}", audit.get(name), expected)
    frozen = dict(audit.get("frozen_anchor", {}))
    actual_hash = sha256_file(calibration_path)
    _require_equal(
        "locked v7 calibration filename",
        frozen.get("name"),
        calibration_path.name,
    )
    _require_equal(
        "locked v7 calibration hash",
        frozen.get("sha256"),
        actual_hash,
    )
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    radii = validate_calibration_report(calibration)
    _require_equal(
        "locked v9 anchor radii",
        dict(frozen.get("radii", {})),
        radii,
    )
    v8_epoch_one = dict(
        dict(audit["v8_matched_epoch_one"])["metrics"]
    )
    _require_equal(
        "locked v8 comparison metrics",
        set(v8_epoch_one),
        set(ATTRIBUTION_METRICS),
    )
    return radii, actual_hash, v8_epoch_one


def checkpoint_payload(
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
    gradient_diagnostics: dict[str, object],
    radii: dict[str, float],
    calibration_sha256: str,
) -> dict[str, object]:
    payload = v8_checkpoint_payload(
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
        selected=selected,
        selection=selection,
        sampling_plan=sampling_plan,
        sequence_counts=sequence_counts,
        cache_report=cache_report,
        trainable_parameters=trainable_parameters,
        gradient_diagnostics=gradient_diagnostics,
        radii=radii,
        calibration_sha256=calibration_sha256,
    )
    metadata = dict(payload["metadata"])
    metadata["version"] = "candidate-sequence-string-v9"
    metadata["experiment"] = {
        "kind": "anchored-equal-step-guitarset-injection-control",
        "single_policy_change_from_v8": True,
        "injected_source": "GuitarSet",
        "initialization": "string-v1",
        "anchor_center": "string-v1",
        "radii": radii,
        "anchor_calibration_sha256": calibration_sha256,
        "projection_after_adamw": True,
        "optimizer_moments_projected": False,
        "base_guitarset_windows_per_epoch": GUITARSET_WINDOWS,
        "injected_guitarset_windows_by_epoch": list(
            EXPECTED_P3_SCHEDULE
        ),
        "p3_optimizer_windows": 0,
        "total_optimizer_steps": 624,
    }
    payload["metadata"] = metadata
    return payload


def run_training(
    training_root: Path,
    audit_path: Path,
    *,
    precompute_device_name: str,
    precompute_batch_size: int,
    inference_batch_size: int,
) -> dict[str, object]:
    root = training_root.resolve()
    calibration_path = (
        root
        / "reports/candidate-sequence-string-v7-anchor-calibration.json"
    )
    radii, calibration_sha256, v8_epoch_one = load_locked_inputs(
        audit_path,
        calibration_path,
    )
    formal_manifest = (
        root / "governed-v1/formal-training-manifest-v3.jsonl"
    )
    train_annotations = load_manifest(formal_manifest, split="train")
    selection = select_v3_tracks(train_annotations)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    precompute_device = choose_device(precompute_device_name)
    training_device = torch.device("cpu")
    position_path = (
        root / "checkpoints/guitarset-candidate-activity-residual-v1.pt"
    )
    string_path = (
        root / "checkpoints/guitarset-candidate-sequence-string-v1.pt"
    )
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
    anchor = snapshot_anchor(model)
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
                "stage": "precompute_v9_groups",
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
        55_946 + 3_416,
    )
    _require_equal(
        "materialized training windows",
        len(training_windows),
        GUITARSET_WINDOWS + P3_WINDOWS,
    )
    source_names = window_source_names(
        training_windows,
        training_dataset.annotations,
    )
    guitarset_indices = tuple(
        index
        for index, name in enumerate(source_names)
        if name == "GuitarSet"
    )
    p3_indices = tuple(
        index
        for index, name in enumerate(source_names)
        if name == "Guitar-TECHS"
    )
    _require_equal(
        "eligible GuitarSet windows",
        len(guitarset_indices),
        GUITARSET_WINDOWS,
    )
    _require_equal("diagnostic-only P3 windows", len(p3_indices), P3_WINDOWS)
    identities = window_identities(
        training_windows,
        training_dataset.annotations,
    )
    track_ids = window_track_ids(
        training_windows,
        training_dataset.annotations,
    )
    active_exposures = window_active_candidate_exposures(training_windows)
    sampler = GuitarSetReplayInjectionSampler(
        guitarset_indices,
        base_generator=torch.Generator().manual_seed(SEED),
        injection_generator=torch.Generator().manual_seed(
            P3_GENERATOR_SEED
        ),
    )
    loader = DataLoader(
        training_windows,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        collate_fn=collate_candidate_sequence_windows,
    )
    annotations_by_id = {
        annotation.track_id: annotation
        for annotation in training_dataset.annotations
    }
    track_windows = windows_by_track(
        training_groups,
        training_dataset.annotations,
        model.config,
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
    sampling_plan: dict[str, object] = {
        "kind": "guitarset-full-coverage-with-equal-step-replay",
        "base_guitarset_windows_per_epoch": GUITARSET_WINDOWS,
        "base_guitarset_replacement": False,
        "base_window_identity_sha256": identity_sha256(
            identities[index] for index in guitarset_indices
        ),
        "injected_guitarset_windows_by_epoch": list(
            EXPECTED_P3_SCHEDULE
        ),
        "injected_replacement_within_epoch": False,
        "p3_optimizer_windows": 0,
        "base_seed": SEED,
        "injection_seed": P3_GENERATOR_SEED,
        "total_optimizer_steps": 624,
    }

    output_paths = fixed_output_paths(root)
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
    baseline_gradients = source_gradient_diagnostics(
        model,
        track_windows,
        train_annotations,
        batch_size=BATCH_SIZE,
    )
    append_log(
        log_path,
        {
            "epoch": 0,
            "stage": "frozen_string_v1_baseline",
            "calibration": baseline,
            "anchor_calibration": {
                "sha256": calibration_sha256,
                "radii": radii,
            },
            "sampling_plan": sampling_plan,
            "source_gradients": baseline_gradients,
        },
    )
    print(
        json.dumps(
            {
                "epoch": 0,
                "calibration": baseline,
                "anchor_calibration": {
                    "sha256": calibration_sha256,
                    "radii": radii,
                },
                "sampling_plan": sampling_plan,
                "source_gradients": baseline_gradients,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    best_key: tuple[float, int, float, int] | None = None
    best_epoch = 0
    best_metrics: dict[str, object] | None = None
    selected_gradients: dict[str, object] | None = None
    drift_prevented_epochs: list[int] = []
    cumulative_base_exposures = 0
    cumulative_injected_exposures = 0
    epoch_one_attribution: str | None = None
    for epoch in range(1, EPOCHS + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics, anchor_metrics = run_anchored_training_epoch(
            model,
            loader,
            training_device,
            optimizer,
            anchor,
            radii,
        )
        actual_sampling = replay_coverage_report(
            sampler,
            identities,
            track_ids,
            active_exposures,
            annotations_by_id,
        )
        validate_replay_coverage(actual_sampling, epoch=epoch)
        cumulative_base_exposures += int(
            dict(actual_sampling["base"])[
                "active_candidate_exposures"
            ]
        )
        cumulative_injected_exposures += int(
            dict(actual_sampling["injected"])[
                "active_candidate_exposures"
            ]
        )
        cumulative_sampling = {
            "base_draws": epoch * GUITARSET_WINDOWS,
            "injected_draws": sum(
                EXPECTED_P3_SCHEDULE[:epoch]
            ),
            "base_active_candidate_exposures": (
                cumulative_base_exposures
            ),
            "injected_active_candidate_exposures": (
                cumulative_injected_exposures
            ),
            "p3_optimizer_windows": 0,
        }
        metrics = calibration_metrics(
            model,
            calibration_groups,
            calibration_dataset.annotations,
            training_device,
            inference_batch_size=inference_batch_size,
        )
        gradient_diagnostics = source_gradient_diagnostics(
            model,
            track_windows,
            train_annotations,
            batch_size=BATCH_SIZE,
        )
        drift_prevented = drift_was_prevented(metrics, baseline)
        production_eligible = epoch_is_eligible(metrics, baseline)
        if drift_prevented:
            drift_prevented_epochs.append(epoch)
        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "anchor": anchor_metrics,
            "sampling": actual_sampling,
            "cumulative_sampling": cumulative_sampling,
            "calibration": metrics,
            "source_gradients": gradient_diagnostics,
            "drift_prevented": drift_prevented,
            "production_eligible": production_eligible,
            "eligible": production_eligible,
        }
        if epoch == 1:
            epoch_one_attribution = source_content_attribution(
                v8_epoch_one,
                metrics,
            )
            record["matched_v8_epoch_one"] = {
                "v8_metrics": v8_epoch_one,
                "classification": epoch_one_attribution,
            }
        append_log(log_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        selection_key = epoch_selection_key(epoch, metrics)
        if (
            production_eligible
            and (best_key is None or selection_key > best_key)
        ):
            best_key = selection_key
            best_epoch = epoch
            best_metrics = metrics
            selected_gradients = gradient_diagnostics
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                checkpoint_payload(
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
                    gradient_diagnostics=gradient_diagnostics,
                    radii=radii,
                    calibration_sha256=calibration_sha256,
                ),
                output_path,
            )
        scheduler.step()

    if epoch_one_attribution is None:
        raise RuntimeError("v9 source attribution was not computed")
    result: dict[str, object] = {
        "source_content_attribution": epoch_one_attribution,
        "drift_prevented": bool(drift_prevented_epochs),
        "drift_prevented_epochs": drift_prevented_epochs,
        "production_gate_passed": best_metrics is not None,
        "best_epoch": best_epoch,
        "baseline": baseline,
        "baseline_source_gradients": baseline_gradients,
        "selected": best_metrics,
        "selected_source_gradients": selected_gradients,
        "trainable_parameters": trainable_parameters,
        "anchor_calibration_sha256": calibration_sha256,
        "radii": radii,
        "sampling": sampling_plan,
        "cumulative_sampling": {
            "base_active_candidate_exposures": (
                cumulative_base_exposures
            ),
            "injected_active_candidate_exposures": (
                cumulative_injected_exposures
            ),
        },
    }
    if best_metrics is not None:
        result["output"] = output_path.name
        result["sha256"] = sha256_file(output_path)
    else:
        result["output"] = None
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _refuse_existing(paths: Iterable[Path], stage: str) -> None:
    stale = [path for path in paths if path.exists()]
    if stale:
        raise FileExistsError(
            f"Refusing to overwrite stale v9 {stage} output: "
            + ", ".join(path.name for path in stale)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or train candidate sequence string v9 as the "
            "equal-step GuitarSet injection control for v8."
        )
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--precompute-device", default="auto")
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    arguments = parser.parse_args()
    for name in ("precompute_batch_size", "inference_batch_size"):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    root = arguments.training_root.resolve()
    outputs = fixed_output_paths(root)
    if arguments.audit_only:
        _refuse_existing([outputs["audit"]], "input audit")
        audit = build_input_audit(root, arguments.preregistration)
        write_json(outputs["audit"], audit)
        print(
            json.dumps(
                {
                    "stage": "input_audit",
                    "passed": True,
                    "output": outputs["audit"].name,
                    "sha256": sha256_file(outputs["audit"]),
                }
            ),
            flush=True,
        )
        return

    if not outputs["audit"].is_file():
        raise FileNotFoundError("Frozen v9 input audit does not exist")
    _refuse_existing(
        [outputs["output"], outputs["log"]],
        "training",
    )
    run_training(
        root,
        outputs["audit"],
        precompute_device_name=arguments.precompute_device,
        precompute_batch_size=arguments.precompute_batch_size,
        inference_batch_size=arguments.inference_batch_size,
    )


if __name__ == "__main__":
    main()
