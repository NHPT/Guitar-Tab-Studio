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
from torch.utils.data import DataLoader, WeightedRandomSampler

from .analyze_candidate_gradient_conflict import (
    AUDIT_PLAN_SHA256,
    EXPECTED_PANEL_SHA256,
    EXPECTED_SELECTION_SHA256,
    assert_parameters_unchanged,
    cosine_from_grams,
    extract_unit_gradient,
    identity_sha256,
    module_gram_matrices,
    parameter_modules,
    parameters_snapshot,
    select_audit_units,
    sum_gram_matrices,
    verify_audit_plan,
    windows_by_track,
)
from .candidate_margin import calibration_partition_digest
from .candidate_sequence import (
    CandidateSequenceGroup,
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    collate_candidate_sequence_windows,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import sha256_file
from .features import FeatureConfig
from .onset_position import OnsetPositionConfig, OnsetPositionDataset
from .schema import TrackAnnotation, load_manifest
from .train import choose_device
from .train_candidate_sequence import append_log, load_position_model
from .train_candidate_sequence_activity import window_sampling_plan
from .train_candidate_sequence_v3 import (
    BATCH_SIZE,
    CALIBRATION_FRACTION,
    EPOCHS,
    EXPECTED_PARTITION_DIGEST,
    FRONTEND_FRAME_THRESHOLD,
    FRONTEND_MINIMUM_NOTE_LENGTH,
    FRONTEND_ONSET_THRESHOLD,
    LEARNING_RATE,
    MINIMUM_LEARNING_RATE,
    REPEAT_INTERVAL,
    SEED,
    SOURCE_LEDGER_SHA256,
    SOURCE_LEDGER_VERSION,
    V3TrackSelection,
    WEIGHT_DECAY,
    _build_training_datasets,
    _checkpoint_payload as v3_checkpoint_payload,
    _position_config_from_checkpoint,
    _require_equal,
    _selection_counts,
    _validate_sequence_model,
    calibration_metrics,
    candidate_cache_set_identity,
    candidate_group_window_counts,
    epoch_is_eligible,
    epoch_selection_key,
    run_training_epoch,
    select_v3_tracks,
    validate_candidate_cache_set,
    validate_epoch_zero,
    verify_artifact_hashes,
    verify_file_hash,
)


PREREGISTRATION_SHA256 = (
    "f92176cbe06069839f0099e485d8c79a6460643dc432a08d23554ec490f9f3dd"
)
GRADIENT_CONFLICT_REPORT_SHA256 = (
    "f7ca64af7037c5e4f319f0a52ce609cb88720fd1fdaadc24e83ba3e53ec30c33"
)
P3_EXPECTED_SHARE = 0.01
P3_SAMPLING_WEIGHT = 0.1652344338911503

FROZEN_ARTIFACTS: dict[str, tuple[str, str]] = {
    "formal_manifest": (
        "governed-v1/formal-training-manifest-v3.jsonl",
        "cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d",
    ),
    "p3_manifest": (
        "governed-v1/guitar-techs-p3-music-position-train-candidate-v1.jsonl",
        "f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8",
    ),
    "v3_input_audit": (
        "reports/candidate-sequence-string-v3-input-audit.json",
        "7dbb822c25e408acd2cfc72cbd3ee712b1b5f5bfbd0ff9e91b66bbab668a7bd4",
    ),
    "gradient_conflict_report": (
        "reports/candidate-sequence-gradient-conflict-v1.json",
        GRADIENT_CONFLICT_REPORT_SHA256,
    ),
    "position_checkpoint": (
        "checkpoints/guitarset-candidate-activity-residual-v1.pt",
        "7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e",
    ),
    "string_v1_checkpoint": (
        "checkpoints/guitarset-candidate-sequence-string-v1.pt",
        "392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982",
    ),
}


class RecordingWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self,
        weights: list[float],
        num_samples: int,
        *,
        generator: torch.Generator,
    ) -> None:
        super().__init__(
            weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        self.last_indices: list[int] = []

    def __iter__(self) -> Iterator[int]:
        self.last_indices = list(super().__iter__())
        return iter(self.last_indices)


def expected_p3_share(
    guitarset_windows: int,
    p3_windows: int,
    p3_weight: float,
) -> float:
    if min(guitarset_windows, p3_windows) <= 0:
        raise ValueError("source window counts must be positive")
    if not math.isfinite(p3_weight) or p3_weight <= 0:
        raise ValueError("P3 sampling weight must be positive and finite")
    return (
        p3_windows * p3_weight
        / (guitarset_windows + p3_windows * p3_weight)
    )


def window_source_names(
    windows: CandidateSequenceWindowDataset,
    annotations: list[TrackAnnotation],
) -> list[str]:
    result: list[str] = []
    for window in windows.windows:
        annotation_index = windows.groups[window[0]].annotation_index
        result.append(
            str(
                annotations[annotation_index].provenance.get(
                    "dataset",
                    "unknown",
                )
            )
        )
    return result


def sampled_source_report(
    sampler: RecordingWeightedRandomSampler,
    source_names: list[str],
) -> dict[str, object]:
    if len(source_names) != len(sampler.weights):
        raise ValueError("sampler weights and source names do not align")
    if len(sampler.last_indices) != sampler.num_samples:
        raise RuntimeError("sampler has not completed an epoch")
    counts = Counter(source_names[index] for index in sampler.last_indices)
    return {
        "draws": len(sampler.last_indices),
        "counts": dict(sorted(counts.items())),
        "shares": {
            name: count / len(sampler.last_indices)
            for name, count in sorted(counts.items())
        },
    }


def build_input_audit(
    training_root: Path,
    preregistration: Path,
    gradient_audit_plan: Path,
    *,
    artifacts: dict[str, tuple[str, str]] = FROZEN_ARTIFACTS,
    source_ledger_path: Path | None = None,
) -> dict[str, object]:
    root = training_root.resolve()
    artifact_report = verify_artifact_hashes(root, artifacts)
    preregistration_report = verify_file_hash(
        preregistration,
        PREREGISTRATION_SHA256,
        name="v4 preregistration",
    )
    gradient_plan_report = verify_audit_plan(
        gradient_audit_plan,
        AUDIT_PLAN_SHA256,
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

    v3_audit_path = root / artifacts["v3_input_audit"][0]
    v3_audit = json.loads(v3_audit_path.read_text(encoding="utf-8"))
    for name, expected in {
        "kind": "candidate-sequence-string-v3-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_read": False,
        "cross_source_validation_read": False,
    }.items():
        _require_equal(f"v3 input audit {name}", v3_audit.get(name), expected)

    conflict_path = root / artifacts["gradient_conflict_report"][0]
    conflict = json.loads(conflict_path.read_text(encoding="utf-8"))
    for name, expected in {
        "kind": "candidate-sequence-gradient-conflict-v1",
        "passed": True,
        "training_only": True,
        "optimizer_created": False,
        "parameter_mutation_detected": False,
        "governed_validation_result_read": False,
        "cross_source_validation_result_read": False,
    }.items():
        _require_equal(
            f"gradient conflict report {name}",
            conflict.get(name),
            expected,
        )
    comparisons = dict(conflict.get("comparisons", {}))
    expected_modules = {
        "token_encoder",
        "sequence_encoder",
        "residual_hidden",
        "residual_output",
        "all",
    }
    _require_equal(
        "gradient conflict modules",
        set(comparisons),
        expected_modules,
    )
    for module_name, values in comparisons.items():
        source = dict(values["guitarset_vs_p3"])
        modalities = dict(values["p3_direct_vs_mic"])
        _require_equal(
            f"{module_name} source classification",
            source.get("classification"),
            "inconclusive",
        )
        _require_equal(
            f"{module_name} modality classification",
            modalities.get("classification"),
            "aligned",
        )

    formal_manifest = root / artifacts["formal_manifest"][0]
    annotations = load_manifest(formal_manifest)
    train_annotations = [
        annotation
        for annotation in annotations
        if annotation.split == "train"
    ]
    validation_annotations = [
        annotation
        for annotation in annotations
        if annotation.split == "validation"
    ]
    _require_equal("formal-v3 tracks", len(annotations), 1117)
    _require_equal("formal-v3 train tracks", len(train_annotations), 997)
    _require_equal("formal-v3 validation tracks", len(validation_annotations), 120)
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
    _require_equal("calibration tracks", len(selection.calibration_track_ids), 268)
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

    diagnostic_selection = select_audit_units(train_annotations)
    _require_equal(
        "diagnostic GuitarSet selection",
        identity_sha256(diagnostic_selection.guitarset_track_ids),
        EXPECTED_SELECTION_SHA256,
    )
    for name, track_ids in diagnostic_selection.panels.items():
        _require_equal(
            f"diagnostic {name} selection",
            identity_sha256(track_ids),
            EXPECTED_PANEL_SHA256[name],
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
    expected_sequence_counts = {
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
    }
    _require_equal(
        "gradient sequence counts",
        sequence_counts,
        expected_sequence_counts,
    )
    p3_share = expected_p3_share(3288, 201, P3_SAMPLING_WEIGHT)
    if not math.isclose(
        p3_share,
        P3_EXPECTED_SHARE,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("frozen P3 weight does not produce 1% sampler mass")

    return {
        "schema_version": 1,
        "kind": "candidate-sequence-string-v4-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_read": False,
        "cross_source_validation_read": False,
        "preregistration": preregistration_report,
        "gradient_audit_plan": gradient_plan_report,
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
        "gradient_diagnostic": {
            "guitarset_tracks": len(
                diagnostic_selection.guitarset_track_ids
            ),
            "guitarset_track_identity_sha256": identity_sha256(
                diagnostic_selection.guitarset_track_ids
            ),
            "p3_pairs": len(diagnostic_selection.p3_pairs),
            "epochs": [0, 1, 2, 3],
            "zero_update": True,
        },
        "gradient_conflict_evidence": {
            module_name: {
                "cosine": float(
                    values["guitarset_vs_p3"]["cosine"]
                ),
                "p3_to_guitarset_norm_ratio": float(
                    values["guitarset_vs_p3"][
                        "p3_to_guitarset_norm_ratio"
                    ]
                ),
                "classification": values["guitarset_vs_p3"][
                    "classification"
                ],
            }
            for module_name, values in sorted(comparisons.items())
        },
        "p3_candidate_cache": p3_cache,
        "sampling": {
            "replacement": True,
            "draws_per_epoch": 3489,
            "guitarset_weight": 1.0,
            "p3_weight": P3_SAMPLING_WEIGHT,
            "expected_p3_share": p3_share,
            "only_change_from_v3": True,
        },
    }


def write_input_audit(path: Path, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_gradient_diagnostics(
    model: CandidateSequenceStringModel,
    track_windows: dict[str, list[list[CandidateSequenceGroup]]],
    train_annotations: list[TrackAnnotation],
    *,
    batch_size: int,
) -> dict[str, object]:
    selection = select_audit_units(train_annotations)
    modules = parameter_modules(model)
    snapshot = parameters_snapshot(model)
    unit_by_track = {}
    extraction_order = [
        *selection.guitarset_track_ids,
        *[
            track_id
            for pair in selection.p3_pairs
            for track_id in (pair.direct_track_id, pair.mic_track_id)
        ],
    ]
    for track_id in extraction_order:
        unit_by_track[track_id] = extract_unit_gradient(
            model,
            track_windows[track_id],
            modules,
            track_id=track_id,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )
    assert_parameters_unchanged(model, snapshot)

    guitarset_units = [
        unit_by_track[track_id]
        for track_id in selection.guitarset_track_ids
    ]
    direct_units = [
        unit_by_track[pair.direct_track_id]
        for pair in selection.p3_pairs
    ]
    mic_units = [
        unit_by_track[pair.mic_track_id]
        for pair in selection.p3_pairs
    ]
    grams = {
        module_name: module_gram_matrices(
            guitarset_units,
            direct_units,
            mic_units,
            module_name,
        )
        for module_name in modules
    }
    grams["all"] = sum_gram_matrices(list(grams.values()))
    guitarset_counts = np.ones(len(guitarset_units), dtype=np.float64)
    p3_counts = np.ones(len(selection.p3_pairs), dtype=np.float64)

    report: dict[str, object] = {}
    for module_name, module_grams in grams.items():
        cosine, guitarset_norm, p3_norm = cosine_from_grams(
            module_grams["guitarset"],
            module_grams["paired"],
            module_grams["guitarset_paired"],
            guitarset_counts,
            p3_counts,
        )
        modality_cosine, direct_norm, mic_norm = cosine_from_grams(
            module_grams["direct"],
            module_grams["mic"],
            module_grams["direct_mic"],
            p3_counts,
            p3_counts,
        )
        report[module_name] = {
            "guitarset_vs_p3_cosine": cosine,
            "guitarset_mean_gradient_norm": guitarset_norm,
            "p3_mean_gradient_norm": p3_norm,
            "p3_to_guitarset_norm_ratio": p3_norm / guitarset_norm,
            "p3_direct_vs_mic_cosine": modality_cosine,
            "p3_direct_mean_gradient_norm": direct_norm,
            "p3_mic_mean_gradient_norm": mic_norm,
        }
    return {
        "zero_update": True,
        "guitarset_units": len(guitarset_units),
        "p3_paired_units": len(selection.p3_pairs),
        "modules": report,
    }


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
    payload = v3_checkpoint_payload(
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
    )
    metadata = dict(payload["metadata"])
    metadata["version"] = "candidate-sequence-string-v4"
    metadata["experiment"] = {
        "kind": "p3-sampler-mass-scale-control",
        "single_change_from_v3": True,
        "p3_expected_sampler_share": P3_EXPECTED_SHARE,
        "p3_sampling_weight": P3_SAMPLING_WEIGHT,
    }
    metadata["source_gradient_diagnostics"] = gradient_diagnostics
    payload["metadata"] = metadata
    return payload


def fixed_output_paths(root: Path) -> dict[str, Path]:
    return {
        "audit": root / "reports/candidate-sequence-string-v4-input-audit.json",
        "output": (
            root
            / "checkpoints/guitarset-candidate-sequence-string-v4.pt"
        ),
        "log": root / "reports/candidate-sequence-string-v4-training.jsonl",
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
                "stage": "precompute_v4_groups",
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
        P3_EXPECTED_SHARE,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("P3 replacement sampler share is not exactly 1%")

    sampler = RecordingWeightedRandomSampler(
        sampling_weights,
        len(training_windows),
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
    source_names = window_source_names(
        training_windows,
        training_dataset.annotations,
    )
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
        actual_sampling = sampled_source_report(sampler, source_names)
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
            "Audit and run candidate sequence string v4 with one-percent "
            "Guitar-TECHS P3 sampler mass."
        )
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--gradient-audit-plan", type=Path, required=True)
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
                "Refusing to overwrite stale v4 training output: "
                + ", ".join(path.name for path in stale)
            )

    audit = build_input_audit(
        root,
        arguments.preregistration,
        arguments.gradient_audit_plan,
    )
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
