from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterator

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
from .train_candidate_sequence_activity import window_sampling_plan
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
    run_training_epoch,
    select_v3_tracks,
    validate_epoch_zero,
    verify_file_hash,
)
from .train_candidate_sequence_v4 import (
    source_gradient_diagnostics,
    window_source_names,
)
from .train_candidate_sequence_v5 import (
    build_input_audit as build_v5_input_audit,
    checkpoint_payload as v5_checkpoint_payload,
)


PREREGISTRATION_SHA256 = (
    "5be7175bb428ee48d6abd8c2f5f52c4d2b75498014ef984bf00a0f06cfcac11e"
)
V5_PREREGISTRATION_SHA256 = (
    "e6d19577e722b4eaae0a52dbd260a64e29231439628fba0894872eb22abbd6a8"
)
V5_INPUT_AUDIT_SHA256 = (
    "e086978cef0cabd4783f34ac3d21d94b364f52b0a237ae203cc456800d15ed77"
)
V5_TRAINING_LOG_SHA256 = (
    "f01406d0fc1c97eb88ca416d8832a90991640ddb402f0f22571b15049ea22ffb"
)
V5_RESULT_SHA256 = (
    "8737657e7e6de870934ded990ac81a345033f34f7fe652400e8ffe99bb2126cc"
)
GUITARSET_AUDIT_PLAN_SHA256 = (
    "345932175548254502be20049ec222149901a136508ef0685b0cf833ab157866"
)
GUITARSET_AUDIT_REPORT_SHA256 = (
    "ed6c9a95e58f1e3da2f0777070a884d2ebb1640c1b90660cf0740f6e36342e7f"
)
GUITARSET_AUDIT_RESULT_SHA256 = (
    "4d47faeb1f2957380f5e62cd500a3c0539c9f8d906a223b1746d0043b48ccb2f"
)
P3_EXPECTED_SHARE = 0.0
P3_SAMPLING_WEIGHT = 0.0
GUITARSET_WINDOWS = 3288
P3_WINDOWS = 201


class RecordingPermutationSampler(Sampler[int]):
    def __init__(
        self,
        indices: list[int] | tuple[int, ...],
        *,
        generator: torch.Generator,
    ) -> None:
        if not indices:
            raise ValueError("permutation sampler requires eligible indexes")
        if len(set(indices)) != len(indices):
            raise ValueError("permutation sampler indexes must be unique")
        if min(indices) < 0:
            raise ValueError("permutation sampler indexes must be non-negative")
        self.indices = tuple(indices)
        self.generator = generator
        self.last_indices: list[int] = []

    def __iter__(self) -> Iterator[int]:
        order = torch.randperm(
            len(self.indices),
            generator=self.generator,
        ).tolist()
        self.last_indices = [self.indices[index] for index in order]
        return iter(self.last_indices)

    def __len__(self) -> int:
        return len(self.indices)


def fixed_output_paths(root: Path) -> dict[str, Path]:
    return {
        "audit": root / "reports/candidate-sequence-string-v6-input-audit.json",
        "output": (
            root
            / "checkpoints/guitarset-candidate-sequence-string-v6.pt"
        ),
        "log": root / "reports/candidate-sequence-string-v6-training.jsonl",
    }


def _verify_evidence_file(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> dict[str, str]:
    return verify_file_hash(path, expected_sha256, name=name)


def build_input_audit(
    training_root: Path,
    preregistration: Path,
) -> dict[str, object]:
    root = training_root.resolve()
    preregistration = preregistration.resolve()
    preregistration_report = _verify_evidence_file(
        preregistration,
        PREREGISTRATION_SHA256,
        name="v6 preregistration",
    )
    docs_directory = preregistration.parent
    v5_preregistration = (
        docs_directory / "candidate-sequence-string-v5-preregistration.md"
    )
    v5_preregistration_report = _verify_evidence_file(
        v5_preregistration,
        V5_PREREGISTRATION_SHA256,
        name="v5 preregistration",
    )
    v5_contract = build_v5_input_audit(root, v5_preregistration)

    v5_input_path = (
        root / "reports/candidate-sequence-string-v5-input-audit.json"
    )
    v5_training_path = (
        root / "reports/candidate-sequence-string-v5-training.jsonl"
    )
    v5_result_path = docs_directory / "candidate-sequence-string-v5-result.md"
    v5_input_report = _verify_evidence_file(
        v5_input_path,
        V5_INPUT_AUDIT_SHA256,
        name="v5 input audit",
    )
    v5_training_report = _verify_evidence_file(
        v5_training_path,
        V5_TRAINING_LOG_SHA256,
        name="v5 training log",
    )
    v5_result_report = _verify_evidence_file(
        v5_result_path,
        V5_RESULT_SHA256,
        name="v5 result",
    )
    v5_rows = [
        json.loads(line)
        for line in v5_training_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    _require_equal(
        "v5 training epochs",
        [row.get("epoch") for row in v5_rows],
        [0, 1, 2, 3],
    )
    _require_equal(
        "v5 ineligible epochs",
        [row.get("eligible") for row in v5_rows[1:]],
        [False, False, False],
    )
    v5_guitarset_draws = sum(
        int(row["sampling"]["counts"].get("GuitarSet", 0))
        for row in v5_rows[1:]
    )
    v5_p3_draws = sum(
        int(row["sampling"]["counts"].get("Guitar-TECHS", 0))
        for row in v5_rows[1:]
    )
    _require_equal("v5 total GuitarSet draws", v5_guitarset_draws, 10467)
    _require_equal("v5 total P3 draws", v5_p3_draws, 0)
    v5_checkpoint = (
        root / "checkpoints/guitarset-candidate-sequence-string-v5.pt"
    )
    if v5_checkpoint.exists():
        raise ValueError("v5 checkpoint exists despite failed training gate")

    guitarset_plan_path = (
        docs_directory
        / "candidate-sequence-guitarset-gradient-conflict-v1-audit-plan.md"
    )
    guitarset_result_path = (
        docs_directory
        / "candidate-sequence-guitarset-gradient-conflict-v1-result.md"
    )
    guitarset_report_path = (
        root
        / "reports/candidate-sequence-guitarset-gradient-conflict-v1.json"
    )
    guitarset_plan_report = _verify_evidence_file(
        guitarset_plan_path,
        GUITARSET_AUDIT_PLAN_SHA256,
        name="GuitarSet gradient audit plan",
    )
    guitarset_result_report = _verify_evidence_file(
        guitarset_result_path,
        GUITARSET_AUDIT_RESULT_SHA256,
        name="GuitarSet gradient audit result",
    )
    guitarset_report_file = _verify_evidence_file(
        guitarset_report_path,
        GUITARSET_AUDIT_REPORT_SHA256,
        name="GuitarSet gradient audit report",
    )
    guitarset_report = json.loads(
        guitarset_report_path.read_text(encoding="utf-8")
    )
    for name, expected in {
        "kind": "candidate-sequence-guitarset-gradient-conflict-v1",
        "passed": True,
        "training_only": True,
        "optimizer_created": False,
        "parameter_mutation_detected": False,
        "governed_validation_result_read": False,
        "cross_source_validation_result_read": False,
    }.items():
        _require_equal(
            f"GuitarSet gradient audit {name}",
            guitarset_report.get(name),
            expected,
        )
    comparisons = dict(guitarset_report.get("comparisons", {}))
    _require_equal(
        "GuitarSet gradient objectives",
        set(comparisons),
        {"operational", "common_unweighted"},
    )
    expected_modules = {
        "token_encoder",
        "sequence_encoder",
        "residual_hidden",
        "residual_output",
        "all",
    }
    evidence: dict[str, dict[str, float | str]] = {}
    for objective, modules in comparisons.items():
        _require_equal(
            f"{objective} GuitarSet gradient modules",
            set(modules),
            expected_modules,
        )
        evidence[objective] = {}
        for module_name, values in modules.items():
            primary = dict(values["equal_track_mean"])
            _require_equal(
                f"{objective} {module_name} classification",
                primary.get("classification"),
                "inconclusive",
            )
            if module_name == "all":
                evidence[objective] = {
                    "classification": str(primary["classification"]),
                    "cosine": float(primary["cosine"]),
                    "calibration_to_gradient_norm_ratio": float(
                        primary["calibration_to_gradient_norm_ratio"]
                    ),
                }

    v5_sampling = dict(v5_contract["sampling"])
    _require_equal(
        "v5 replacement sampling",
        v5_sampling["replacement"],
        True,
    )
    _require_equal(
        "v5 draws per epoch",
        v5_sampling["draws_per_epoch"],
        3489,
    )
    _require_equal(
        "v5 expected P3 share",
        v5_sampling["expected_p3_share"],
        0.0,
    )
    return {
        "schema_version": 1,
        "kind": "candidate-sequence-string-v6-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_read": False,
        "cross_source_validation_read": False,
        "preregistration": preregistration_report,
        "v5_evidence": {
            "preregistration": v5_preregistration_report,
            "input_audit": v5_input_report,
            "training_log": v5_training_report,
            "result": v5_result_report,
            "epochs": 3,
            "eligible_epochs": 0,
            "guitarset_draws": v5_guitarset_draws,
            "p3_draws": v5_p3_draws,
            "checkpoint_created": False,
        },
        "guitarset_gradient_evidence": {
            "plan": guitarset_plan_report,
            "report": guitarset_report_file,
            "result": guitarset_result_report,
            "all_parameters": evidence,
        },
        "base_input_contract": v5_contract,
        "single_policy_change_from_v5": {
            "kind": "guitarset-replay-policy",
            "v5_replacement": True,
            "v6_replacement": False,
            "v5_draws_per_epoch": 3489,
            "v6_draws_per_epoch": GUITARSET_WINDOWS,
            "v5_batches_per_epoch": math.ceil(3489 / BATCH_SIZE),
            "v6_batches_per_epoch": math.ceil(
                GUITARSET_WINDOWS / BATCH_SIZE
            ),
        },
        "sampling": {
            "replacement": False,
            "draws_per_epoch": GUITARSET_WINDOWS,
            "eligible_guitarset_windows": GUITARSET_WINDOWS,
            "ineligible_p3_windows": P3_WINDOWS,
            "guitarset_weight": 1.0,
            "p3_weight": P3_SAMPLING_WEIGHT,
            "expected_p3_share": P3_EXPECTED_SHARE,
            "expected_p3_draws": 0,
            "expected_unique_guitarset_windows": GUITARSET_WINDOWS,
            "expected_duplicate_draws": 0,
            "expected_omitted_windows": 0,
            "p3_diagnostic_only": True,
        },
    }


def write_input_audit(path: Path, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def window_track_ids(
    windows: CandidateSequenceWindowDataset,
    annotations: list[TrackAnnotation],
) -> list[str]:
    result: list[str] = []
    for window in windows.windows:
        annotation_indexes = {
            windows.groups[group_index].annotation_index
            for group_index in window
        }
        if len(annotation_indexes) != 1:
            raise RuntimeError("sequence window crosses annotation boundary")
        result.append(
            annotations[next(iter(annotation_indexes))].track_id
        )
    return result


def window_identities(
    windows: CandidateSequenceWindowDataset,
    annotations: list[TrackAnnotation],
) -> list[str]:
    track_ids = window_track_ids(windows, annotations)
    return [
        (
            f"{window_index}:{track_id}:"
            f"{window[0]}:{window[-1]}:{len(window)}"
        )
        for window_index, (track_id, window) in enumerate(
            zip(track_ids, windows.windows)
        )
    ]


def full_coverage_report(
    sampler: RecordingPermutationSampler,
    eligible_indices: tuple[int, ...],
    identities: list[str],
    source_names: list[str],
    track_ids: list[str],
    annotations_by_id: dict[str, TrackAnnotation],
) -> dict[str, object]:
    if not (
        len(identities) == len(source_names) == len(track_ids)
    ):
        raise ValueError("window coverage metadata does not align")
    if len(sampler.last_indices) != len(sampler):
        raise RuntimeError("permutation sampler has not completed an epoch")
    eligible = set(eligible_indices)
    observed = set(sampler.last_indices)
    duplicate_draws = len(sampler.last_indices) - len(observed)
    omitted = eligible - observed
    unexpected = observed - eligible
    sampled_track_ids = [
        track_ids[index] for index in sampler.last_indices
    ]
    sampled_annotations = [
        annotations_by_id[track_id] for track_id in sampled_track_ids
    ]
    source_counts = Counter(
        source_names[index] for index in sampler.last_indices
    )
    player_counts = Counter(
        annotation.group_id for annotation in sampled_annotations
    )
    variant_counts = Counter(
        (
            "derived"
            if annotation.provenance.get("derived_from_track_id")
            else "original"
        )
        for annotation in sampled_annotations
    )
    sampled_families = [
        recording_family_root(annotation)
        for annotation in sampled_annotations
    ]
    unique_tracks = tuple(sorted(set(sampled_track_ids)))
    unique_families = tuple(sorted(set(sampled_families)))
    return {
        "replacement": False,
        "draws": len(sampler.last_indices),
        "unique_windows": len(observed),
        "duplicate_draws": duplicate_draws,
        "omitted_eligible_windows": len(omitted),
        "unexpected_windows": len(unexpected),
        "eligible_window_identity_sha256": identity_sha256(
            identities[index] for index in eligible_indices
        ),
        "observed_window_identity_sha256": identity_sha256(
            identities[index] for index in sampler.last_indices
        ),
        "counts": dict(sorted(source_counts.items())),
        "shares": {
            name: count / len(sampler.last_indices)
            for name, count in sorted(source_counts.items())
        },
        "tracks": {
            "count": len(unique_tracks),
            "identity_sha256": identity_sha256(unique_tracks),
        },
        "players": dict(sorted(player_counts.items())),
        "variants": dict(sorted(variant_counts.items())),
        "families": {
            "count": len(unique_families),
            "identity_sha256": identity_sha256(unique_families),
        },
    }


def validate_full_coverage(
    report: dict[str, object],
    *,
    epoch: int,
) -> None:
    for name, actual, expected in (
        ("draws", report["draws"], GUITARSET_WINDOWS),
        ("unique windows", report["unique_windows"], GUITARSET_WINDOWS),
        ("duplicate draws", report["duplicate_draws"], 0),
        ("omitted windows", report["omitted_eligible_windows"], 0),
        ("unexpected windows", report["unexpected_windows"], 0),
        ("track coverage", dict(report["tracks"])["count"], 384),
        ("family coverage", dict(report["families"])["count"], 234),
    ):
        _require_equal(f"epoch {epoch} {name}", actual, expected)
    _require_equal(
        f"epoch {epoch} source counts",
        report["counts"],
        {"GuitarSet": GUITARSET_WINDOWS},
    )
    _require_equal(
        f"epoch {epoch} source shares",
        report["shares"],
        {"GuitarSet": 1.0},
    )
    _require_equal(
        f"epoch {epoch} window identity",
        report["observed_window_identity_sha256"],
        report["eligible_window_identity_sha256"],
    )


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
) -> dict[str, object]:
    payload = v5_checkpoint_payload(
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
    )
    metadata = dict(payload["metadata"])
    metadata["version"] = "candidate-sequence-string-v6"
    metadata["experiment"] = {
        "kind": "guitarset-full-coverage-replay-control",
        "single_policy_change_from_v5": True,
        "replacement": False,
        "draws_per_epoch": GUITARSET_WINDOWS,
        "all_eligible_windows_once": True,
        "p3_expected_sampler_share": P3_EXPECTED_SHARE,
        "p3_sampling_weight": P3_SAMPLING_WEIGHT,
        "p3_training_gradient": False,
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
                "stage": "precompute_v6_groups",
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
    _require_equal(
        "materialized training windows",
        len(training_windows),
        GUITARSET_WINDOWS + P3_WINDOWS,
    )
    sampling_weights, base_sampling_plan = window_sampling_plan(
        training_windows,
        training_dataset.annotations,
        {
            "GuitarSet": 1.0,
            "Guitar-TECHS": P3_SAMPLING_WEIGHT,
        },
    )
    source_names = window_source_names(
        training_windows,
        training_dataset.annotations,
    )
    _require_equal(
        "window source counts",
        dict(base_sampling_plan["window_counts"]),
        {"Guitar-TECHS": P3_WINDOWS, "GuitarSet": GUITARSET_WINDOWS},
    )
    expected_shares = dict(
        base_sampling_plan["expected_dataset_share"]
    )
    _require_equal(
        "full-coverage expected P3 share",
        float(expected_shares["Guitar-TECHS"]),
        P3_EXPECTED_SHARE,
    )
    eligible_indices = tuple(
        index
        for index, weight in enumerate(sampling_weights)
        if weight > 0
    )
    _require_equal(
        "eligible GuitarSet windows",
        len(eligible_indices),
        GUITARSET_WINDOWS,
    )
    if any(
        not math.isclose(weight, 1.0 if index in eligible_indices else 0.0)
        for index, weight in enumerate(sampling_weights)
    ):
        raise ValueError("v6 window eligibility weights changed")
    if {
        source_names[index] for index in eligible_indices
    } != {"GuitarSet"}:
        raise ValueError("v6 eligible windows contain a non-GuitarSet source")

    identities = window_identities(
        training_windows,
        training_dataset.annotations,
    )
    track_ids = window_track_ids(
        training_windows,
        training_dataset.annotations,
    )
    eligible_identity = identity_sha256(
        identities[index] for index in eligible_indices
    )
    sampling_plan = {
        **base_sampling_plan,
        "replacement": False,
        "num_samples": len(eligible_indices),
        "eligible_window_identity_sha256": eligible_identity,
        "all_eligible_windows_once": True,
        "expected_duplicate_draws": 0,
        "expected_omitted_windows": 0,
    }
    sampler = RecordingPermutationSampler(
        eligible_indices,
        generator=torch.Generator().manual_seed(SEED),
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
            "source_gradients": baseline_gradients,
        },
    )
    print(
        json.dumps(
            {
                "epoch": 0,
                "calibration": baseline,
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
    for epoch in range(1, EPOCHS + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_training_epoch(
            model,
            loader,
            training_device,
            optimizer,
        )
        actual_sampling = full_coverage_report(
            sampler,
            eligible_indices,
            identities,
            source_names,
            track_ids,
            annotations_by_id,
        )
        validate_full_coverage(actual_sampling, epoch=epoch)
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
        eligible = epoch_is_eligible(metrics)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "sampling": actual_sampling,
            "calibration": metrics,
            "source_gradients": gradient_diagnostics,
            "eligible": eligible,
        }
        append_log(log_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        selection_key = epoch_selection_key(epoch, metrics)
        if eligible and (best_key is None or selection_key > best_key):
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
                ),
                output_path,
            )
        scheduler.step()

    result: dict[str, object] = {
        "calibration_gate_passed": best_metrics is not None,
        "best_epoch": best_epoch,
        "baseline": baseline,
        "baseline_source_gradients": baseline_gradients,
        "selected": best_metrics,
        "selected_source_gradients": selected_gradients,
        "trainable_parameters": trainable_parameters,
        "expected_p3_sampler_share": P3_EXPECTED_SHARE,
        "sampling": sampling_plan,
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
            "Run candidate sequence string v6 as deterministic "
            "nonreplacement GuitarSet full-coverage replay."
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
    output_paths = fixed_output_paths(root)
    if not arguments.audit_only:
        stale = [
            path
            for path in (output_paths["output"], output_paths["log"])
            if path.exists()
        ]
        if stale:
            raise FileExistsError(
                "Refusing to overwrite stale v6 training output: "
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
