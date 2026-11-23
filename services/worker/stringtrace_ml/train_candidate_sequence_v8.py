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
    checkpoint_payload as v7_checkpoint_payload,
    drift_was_prevented,
    run_anchored_training_epoch,
    snapshot_anchor,
    validate_calibration_report,
)


PREREGISTRATION_SHA256 = (
    "97014e726ec595ef9f689adba76fe69340afb2639a9e4b2edce338b67d7cb0ba"
)
V7_PREREGISTRATION_SHA256 = (
    "49aa18e6ab3a57ccaae13c7bda555eac770fe704f4b3be1c2d55cc6e08012f0b"
)
V7_CALIBRATION_SHA256 = (
    "460bf75ad6b581ed6d162c658e4e000074269871f3415cb4e5d30b49e11c527f"
)
V7_INPUT_AUDIT_SHA256 = (
    "2974b36060b22e00ba585828d5981eb953a1a9b344054bd34baeace2acc3aeb7"
)
V7_TRAINING_LOG_SHA256 = (
    "c53101d8f17d80fbcb7e8ec7f889900549ec16df52f74ba797061412d7b302e1"
)
V7_RESULT_SHA256 = (
    "6c2e41e058e05ad7a194f64e799bcac41d8ebd06273984b27ee403ffdf0bae2e"
)
V7_CHECKPOINT_SHA256 = (
    "3ff0012a25fcf7e5ffa7f72e62be8a5d922e7cfff41a55299586589211afd8a3"
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
P3_TARGET_SHARE = 0.01
P3_GENERATOR_SEED = 20261123
EXPECTED_GUITARSET_EXPOSURES = 109_007
EXPECTED_P3_SCHEDULE = (33, 33, 34)


def fixed_output_paths(root: Path) -> dict[str, Path]:
    return {
        "audit": root / "reports/candidate-sequence-string-v8-input-audit.json",
        "output": (
            root
            / "checkpoints/guitarset-candidate-sequence-string-v8.pt"
        ),
        "log": root / "reports/candidate-sequence-string-v8-training.jsonl",
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
        "v7_preregistration": (
            docs / "candidate-sequence-string-v7-preregistration.md",
            V7_PREREGISTRATION_SHA256,
        ),
        "v7_anchor_calibration": (
            root
            / "reports/candidate-sequence-string-v7-anchor-calibration.json",
            V7_CALIBRATION_SHA256,
        ),
        "v7_input_audit": (
            root / "reports/candidate-sequence-string-v7-input-audit.json",
            V7_INPUT_AUDIT_SHA256,
        ),
        "v7_training_log": (
            root / "reports/candidate-sequence-string-v7-training.jsonl",
            V7_TRAINING_LOG_SHA256,
        ),
        "v7_result": (
            docs / "candidate-sequence-string-v7-result.md",
            V7_RESULT_SHA256,
        ),
        "v7_checkpoint": (
            root / "checkpoints/guitarset-candidate-sequence-string-v7.pt",
            V7_CHECKPOINT_SHA256,
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


def p3_injection_count(
    epoch: int,
    *,
    guitarset_windows: int = GUITARSET_WINDOWS,
    target_share: float = P3_TARGET_SHARE,
) -> int:
    if epoch <= 0:
        raise ValueError("injection epoch must be positive")
    if guitarset_windows <= 0:
        raise ValueError("GuitarSet window count must be positive")
    if not math.isfinite(target_share) or not 0 < target_share < 1:
        raise ValueError("P3 target share must be in (0, 1)")
    expected = guitarset_windows * target_share / (1 - target_share)

    def cumulative(local_epoch: int) -> int:
        return math.floor(local_epoch * expected + 0.5)

    return cumulative(epoch) - cumulative(epoch - 1)


class P3InjectionSampler(Sampler[int]):
    def __init__(
        self,
        guitarset_indices: Iterable[int],
        p3_indices: Iterable[int],
        *,
        guitarset_generator: torch.Generator,
        p3_generator: torch.Generator,
    ) -> None:
        self.guitarset_indices = tuple(guitarset_indices)
        self.p3_indices = tuple(p3_indices)
        if not self.guitarset_indices or not self.p3_indices:
            raise ValueError("P3 injection requires both source partitions")
        if len(set(self.guitarset_indices)) != len(self.guitarset_indices):
            raise ValueError("GuitarSet injection indexes must be unique")
        if len(set(self.p3_indices)) != len(self.p3_indices):
            raise ValueError("P3 injection indexes must be unique")
        if set(self.guitarset_indices) & set(self.p3_indices):
            raise ValueError("source injection partitions must be disjoint")
        if min((*self.guitarset_indices, *self.p3_indices)) < 0:
            raise ValueError("injection indexes must be non-negative")
        self.guitarset_generator = guitarset_generator
        self.p3_generator = p3_generator
        self.epoch = 0
        self.last_indices: list[int] = []
        self.last_guitarset_indices: list[int] = []
        self.last_p3_indices: list[int] = []

    def __iter__(self) -> Iterator[int]:
        self.epoch += 1
        p3_count = p3_injection_count(
            self.epoch,
            guitarset_windows=len(self.guitarset_indices),
        )
        if p3_count > len(self.p3_indices):
            raise ValueError("P3 injection exceeds the eligible partition")
        guitarset_order = torch.randperm(
            len(self.guitarset_indices),
            generator=self.guitarset_generator,
        ).tolist()
        self.last_guitarset_indices = [
            self.guitarset_indices[index] for index in guitarset_order
        ]
        p3_order = torch.randperm(
            len(self.p3_indices),
            generator=self.p3_generator,
        ).tolist()
        self.last_p3_indices = [
            self.p3_indices[index] for index in p3_order[:p3_count]
        ]

        insertion_positions = [
            math.floor(
                (index + 1)
                * len(self.last_guitarset_indices)
                / (p3_count + 1)
            )
            for index in range(p3_count)
        ]
        if len(set(insertion_positions)) != p3_count:
            raise RuntimeError("P3 insertion positions are not unique")
        p3_by_position = dict(
            zip(insertion_positions, self.last_p3_indices)
        )
        combined: list[int] = []
        for position, guitarset_index in enumerate(
            self.last_guitarset_indices
        ):
            if position in p3_by_position:
                combined.append(p3_by_position[position])
            combined.append(guitarset_index)
        if len(combined) != len(self.last_guitarset_indices) + p3_count:
            raise RuntimeError("P3 injection did not preserve draw counts")
        self.last_indices = combined
        return iter(self.last_indices)

    def __len__(self) -> int:
        next_epoch = self.epoch + 1
        return len(self.guitarset_indices) + p3_injection_count(
            next_epoch,
            guitarset_windows=len(self.guitarset_indices),
        )


def window_active_candidate_exposures(
    windows: CandidateSequenceWindowDataset,
) -> list[int]:
    result: list[int] = []
    for window in windows.windows:
        total = 0
        for group_index in window:
            group = windows.groups[group_index]
            total += int(
                (group.string_targets.sum(dim=1) > 0).sum().item()
            )
        result.append(total)
    return result


def injection_coverage_report(
    sampler: P3InjectionSampler,
    identities: list[str],
    source_names: list[str],
    track_ids: list[str],
    active_exposures: list[int],
    annotations_by_id: dict[str, TrackAnnotation],
) -> dict[str, object]:
    if not (
        len(identities)
        == len(source_names)
        == len(track_ids)
        == len(active_exposures)
    ):
        raise ValueError("injection coverage metadata does not align")
    expected_draws = (
        len(sampler.last_guitarset_indices)
        + len(sampler.last_p3_indices)
    )
    if len(sampler.last_indices) != expected_draws:
        raise RuntimeError("P3 injection sampler has not completed an epoch")

    guitarset_expected = set(sampler.guitarset_indices)
    guitarset_observed = set(sampler.last_guitarset_indices)
    p3_expected = set(sampler.p3_indices)
    p3_observed = set(sampler.last_p3_indices)
    source_counts = Counter(
        source_names[index] for index in sampler.last_indices
    )
    exposure_counts = Counter()
    for index in sampler.last_indices:
        exposure_counts[source_names[index]] += active_exposures[index]

    guitarset_track_ids = [
        track_ids[index] for index in sampler.last_guitarset_indices
    ]
    guitarset_annotations = [
        annotations_by_id[track_id] for track_id in guitarset_track_ids
    ]
    guitarset_families = {
        recording_family_root(annotation)
        for annotation in guitarset_annotations
    }
    p3_track_ids = [
        track_ids[index] for index in sampler.last_p3_indices
    ]
    return {
        "target_p3_share": P3_TARGET_SHARE,
        "draws": len(sampler.last_indices),
        "batches": math.ceil(len(sampler.last_indices) / BATCH_SIZE),
        "counts": dict(sorted(source_counts.items())),
        "shares": {
            name: count / len(sampler.last_indices)
            for name, count in sorted(source_counts.items())
        },
        "active_candidate_exposures": dict(sorted(exposure_counts.items())),
        "guitarset": {
            "draws": len(sampler.last_guitarset_indices),
            "unique_windows": len(guitarset_observed),
            "duplicate_draws": (
                len(sampler.last_guitarset_indices)
                - len(guitarset_observed)
            ),
            "omitted_eligible_windows": len(
                guitarset_expected - guitarset_observed
            ),
            "unexpected_windows": len(
                guitarset_observed - guitarset_expected
            ),
            "eligible_window_identity_sha256": identity_sha256(
                identities[index] for index in sampler.guitarset_indices
            ),
            "observed_window_identity_sha256": identity_sha256(
                identities[index]
                for index in sampler.last_guitarset_indices
            ),
            "tracks": {
                "count": len(set(guitarset_track_ids)),
                "identity_sha256": identity_sha256(guitarset_track_ids),
            },
            "players": dict(
                sorted(
                    Counter(
                        annotation.group_id
                        for annotation in guitarset_annotations
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
                        for annotation in guitarset_annotations
                    ).items()
                )
            ),
            "families": {
                "count": len(guitarset_families),
                "identity_sha256": identity_sha256(guitarset_families),
            },
        },
        "p3": {
            "eligible_windows": len(p3_expected),
            "draws": len(sampler.last_p3_indices),
            "unique_windows": len(p3_observed),
            "duplicate_draws": (
                len(sampler.last_p3_indices) - len(p3_observed)
            ),
            "unexpected_windows": len(p3_observed - p3_expected),
            "selected_window_identity_sha256": identity_sha256(
                identities[index] for index in sampler.last_p3_indices
            ),
            "tracks": {
                "count": len(set(p3_track_ids)),
                "identity_sha256": identity_sha256(p3_track_ids),
            },
        },
    }


def validate_injection_coverage(
    report: dict[str, object],
    *,
    epoch: int,
) -> None:
    p3_expected = p3_injection_count(epoch)
    guitarset = dict(report["guitarset"])
    p3 = dict(report["p3"])
    for name, actual, expected in (
        ("total batches", report["batches"], 208),
        ("GuitarSet draws", guitarset["draws"], GUITARSET_WINDOWS),
        (
            "GuitarSet unique windows",
            guitarset["unique_windows"],
            GUITARSET_WINDOWS,
        ),
        ("GuitarSet duplicates", guitarset["duplicate_draws"], 0),
        (
            "GuitarSet omissions",
            guitarset["omitted_eligible_windows"],
            0,
        ),
        ("GuitarSet unexpected windows", guitarset["unexpected_windows"], 0),
        ("GuitarSet tracks", dict(guitarset["tracks"])["count"], 384),
        ("GuitarSet families", dict(guitarset["families"])["count"], 234),
        ("P3 eligible windows", p3["eligible_windows"], P3_WINDOWS),
        ("P3 draws", p3["draws"], p3_expected),
        ("P3 unique windows", p3["unique_windows"], p3_expected),
        ("P3 duplicate draws", p3["duplicate_draws"], 0),
        ("P3 unexpected windows", p3["unexpected_windows"], 0),
    ):
        _require_equal(f"epoch {epoch} {name}", actual, expected)
    _require_equal(
        f"epoch {epoch} GuitarSet identity",
        guitarset["observed_window_identity_sha256"],
        guitarset["eligible_window_identity_sha256"],
    )
    _require_equal(
        f"epoch {epoch} source counts",
        report["counts"],
        {
            "Guitar-TECHS": p3_expected,
            "GuitarSet": GUITARSET_WINDOWS,
        },
    )
    exposures = dict(report["active_candidate_exposures"])
    _require_equal(
        f"epoch {epoch} GuitarSet active exposures",
        exposures["GuitarSet"],
        EXPECTED_GUITARSET_EXPOSURES,
    )
    if int(exposures.get("Guitar-TECHS", 0)) <= 0:
        raise ValueError(f"epoch {epoch} has no active P3 exposures")


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
    v7_audit = json.loads(
        (
            root / "reports/candidate-sequence-string-v7-input-audit.json"
        ).read_text(encoding="utf-8")
    )
    _require_equal(
        "v7 audit kind",
        v7_audit.get("kind"),
        "candidate-sequence-string-v7-input-audit",
    )
    _require_equal("v7 audit passed", v7_audit.get("passed"), True)
    _require_equal(
        "v7 audit calibration hash",
        dict(v7_audit["anchor_calibration"])["sha256"],
        V7_CALIBRATION_SHA256,
    )

    v7_log_path = (
        root / "reports/candidate-sequence-string-v7-training.jsonl"
    )
    v7_rows = [
        json.loads(line)
        for line in v7_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require_equal(
        "v7 training epochs",
        [row.get("epoch") for row in v7_rows],
        [0, 1, 2, 3],
    )
    _require_equal(
        "v7 drift decisions",
        [row.get("drift_prevented") for row in v7_rows[1:]],
        [False, False, True],
    )
    _require_equal(
        "v7 production decisions",
        [row.get("production_eligible") for row in v7_rows[1:]],
        [False, True, True],
    )
    v7_checkpoint_path = (
        root / "checkpoints/guitarset-candidate-sequence-string-v7.pt"
    )
    checkpoint = torch.load(
        v7_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    metadata = dict(checkpoint["metadata"])
    _require_equal(
        "v7 checkpoint version",
        metadata.get("version"),
        "candidate-sequence-string-v7",
    )
    _require_equal("v7 checkpoint epoch", metadata.get("epoch"), 3)
    experiment = dict(metadata["experiment"])
    _require_equal(
        "v7 checkpoint calibration hash",
        experiment.get("anchor_calibration_sha256"),
        V7_CALIBRATION_SHA256,
    )
    _require_equal(
        "v7 checkpoint radii",
        dict(experiment.get("radii", {})),
        radii,
    )

    schedule = [
        p3_injection_count(epoch) for epoch in range(1, EPOCHS + 1)
    ]
    _require_equal(
        "v8 frozen P3 schedule",
        tuple(schedule),
        EXPECTED_P3_SCHEDULE,
    )
    return {
        "schema_version": 1,
        "kind": "candidate-sequence-string-v8-input-audit",
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
        "v7_outcome": {
            "drift_prevented_epochs": [3],
            "production_eligible_epochs": [2, 3],
            "selected_epoch": 3,
            "checkpoint_sha256": V7_CHECKPOINT_SHA256,
        },
        "single_policy_change_from_v7": {
            "kind": "p3-window-injection",
            "v7_p3_windows": [0, 0, 0],
            "v8_p3_windows": schedule,
            "target_p3_share": P3_TARGET_SHARE,
            "guitarset_replay_changed": False,
            "anchor_changed": False,
            "task_loss_changed": False,
        },
        "sampling": {
            "guitarset_windows_per_epoch": GUITARSET_WINDOWS,
            "guitarset_replacement": False,
            "p3_eligible_windows": P3_WINDOWS,
            "p3_replacement_within_epoch": False,
            "p3_windows_by_epoch": schedule,
            "guitarset_seed": SEED,
            "p3_seed": P3_GENERATOR_SEED,
            "batches_per_epoch": [208, 208, 208],
            "pooled_p3_share": sum(schedule)
            / (EPOCHS * GUITARSET_WINDOWS + sum(schedule)),
        },
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_locked_anchor(
    audit_path: Path,
    calibration_path: Path,
) -> tuple[dict[str, float], str]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for name, expected in {
        "kind": "candidate-sequence-string-v8-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
    }.items():
        _require_equal(f"v8 input audit {name}", audit.get(name), expected)
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
    report = json.loads(calibration_path.read_text(encoding="utf-8"))
    radii = validate_calibration_report(report)
    _require_equal(
        "locked v8 anchor radii",
        dict(frozen.get("radii", {})),
        radii,
    )
    return radii, actual_hash


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
    payload = v7_checkpoint_payload(
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
    metadata["version"] = "candidate-sequence-string-v8"
    metadata["experiment"] = {
        "kind": "anchored-one-percent-p3-injection-control",
        "single_policy_change_from_v7": True,
        "initialization": "string-v1",
        "anchor_center": "string-v1",
        "radii": radii,
        "anchor_calibration_sha256": calibration_sha256,
        "projection_after_adamw": True,
        "optimizer_moments_projected": False,
        "guitarset_windows_per_epoch": GUITARSET_WINDOWS,
        "guitarset_replacement": False,
        "p3_windows_by_epoch": list(EXPECTED_P3_SCHEDULE),
        "p3_replacement_within_epoch": False,
        "target_p3_share": P3_TARGET_SHARE,
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
    radii, calibration_sha256 = load_locked_anchor(
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
                "stage": "precompute_v8_groups",
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
    _require_equal("eligible P3 windows", len(p3_indices), P3_WINDOWS)
    identities = window_identities(
        training_windows,
        training_dataset.annotations,
    )
    track_ids = window_track_ids(
        training_windows,
        training_dataset.annotations,
    )
    active_exposures = window_active_candidate_exposures(training_windows)
    sampler = P3InjectionSampler(
        guitarset_indices,
        p3_indices,
        guitarset_generator=torch.Generator().manual_seed(SEED),
        p3_generator=torch.Generator().manual_seed(P3_GENERATOR_SEED),
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
    schedule = [
        p3_injection_count(epoch) for epoch in range(1, EPOCHS + 1)
    ]
    sampling_plan: dict[str, object] = {
        "kind": "guitarset-full-coverage-with-p3-injection",
        "guitarset_windows_per_epoch": GUITARSET_WINDOWS,
        "guitarset_replacement": False,
        "guitarset_window_identity_sha256": identity_sha256(
            identities[index] for index in guitarset_indices
        ),
        "p3_eligible_windows": P3_WINDOWS,
        "p3_windows_by_epoch": schedule,
        "p3_replacement_within_epoch": False,
        "target_p3_share": P3_TARGET_SHARE,
        "pooled_actual_p3_share": sum(schedule)
        / (EPOCHS * GUITARSET_WINDOWS + sum(schedule)),
        "guitarset_seed": SEED,
        "p3_seed": P3_GENERATOR_SEED,
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
    cumulative_counts: Counter[str] = Counter()
    cumulative_exposures: Counter[str] = Counter()
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
        actual_sampling = injection_coverage_report(
            sampler,
            identities,
            source_names,
            track_ids,
            active_exposures,
            annotations_by_id,
        )
        validate_injection_coverage(actual_sampling, epoch=epoch)
        cumulative_counts.update(actual_sampling["counts"])
        cumulative_exposures.update(
            actual_sampling["active_candidate_exposures"]
        )
        cumulative_sampling = {
            "counts": dict(sorted(cumulative_counts.items())),
            "shares": {
                name: count / sum(cumulative_counts.values())
                for name, count in sorted(cumulative_counts.items())
            },
            "active_candidate_exposures": dict(
                sorted(cumulative_exposures.items())
            ),
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
        record = {
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

    _require_equal(
        "cumulative P3 draws",
        cumulative_counts["Guitar-TECHS"],
        sum(EXPECTED_P3_SCHEDULE),
    )
    result: dict[str, object] = {
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
            "counts": dict(sorted(cumulative_counts.items())),
            "active_candidate_exposures": dict(
                sorted(cumulative_exposures.items())
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
            f"Refusing to overwrite stale v8 {stage} output: "
            + ", ".join(path.name for path in stale)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or train candidate sequence string v8 with frozen "
            "v7 trust regions and nominal 1% P3 injection."
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
        raise FileNotFoundError("Frozen v8 input audit does not exist")
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
