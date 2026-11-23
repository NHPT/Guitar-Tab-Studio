from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .analyze_candidate_gradient_conflict import windows_by_track
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
from .schema import load_manifest
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
    RecordingWeightedRandomSampler,
    build_input_audit as build_v4_input_audit,
    checkpoint_payload as v4_checkpoint_payload,
    sampled_source_report,
    source_gradient_diagnostics,
    window_source_names,
)


PREREGISTRATION_SHA256 = (
    "e6d19577e722b4eaae0a52dbd260a64e29231439628fba0894872eb22abbd6a8"
)
V4_INPUT_AUDIT_SHA256 = (
    "87b9e4d3d7bf01c971479e66712672ab6a55d6ce5ec07ca807d9ae1dbd3761da"
)
V4_TRAINING_LOG_SHA256 = (
    "ad09d70c580a5490f0e6e45ef58cfae2913912d1236c50a71e0951571b432f0e"
)
V4_RESULT_SHA256 = (
    "d9c80902616bbca684ee44355192da5f3f9208bd7a60edfc5b9b9b39f64deb26"
)
P3_EXPECTED_SHARE = 0.0
P3_SAMPLING_WEIGHT = 0.0


def fixed_output_paths(root: Path) -> dict[str, Path]:
    return {
        "audit": root / "reports/candidate-sequence-string-v5-input-audit.json",
        "output": (
            root
            / "checkpoints/guitarset-candidate-sequence-string-v5.pt"
        ),
        "log": root / "reports/candidate-sequence-string-v5-training.jsonl",
    }


def build_input_audit(
    training_root: Path,
    preregistration: Path,
) -> dict[str, object]:
    root = training_root.resolve()
    preregistration = preregistration.resolve()
    preregistration_report = verify_file_hash(
        preregistration,
        PREREGISTRATION_SHA256,
        name="v5 preregistration",
    )
    docs_directory = preregistration.parent
    v4_preregistration = (
        docs_directory / "candidate-sequence-string-v4-preregistration.md"
    )
    gradient_audit_plan = (
        docs_directory
        / "candidate-sequence-gradient-conflict-v1-audit-plan.md"
    )
    v4_result = docs_directory / "candidate-sequence-string-v4-result.md"
    v4_contract = build_v4_input_audit(
        root,
        v4_preregistration,
        gradient_audit_plan,
    )
    v4_result_report = verify_file_hash(
        v4_result,
        V4_RESULT_SHA256,
        name="v4 result",
    )
    v4_input_path = (
        root / "reports/candidate-sequence-string-v4-input-audit.json"
    )
    v4_input_report = verify_file_hash(
        v4_input_path,
        V4_INPUT_AUDIT_SHA256,
        name="v4 input audit",
    )
    v4_training_path = (
        root / "reports/candidate-sequence-string-v4-training.jsonl"
    )
    v4_training_report = verify_file_hash(
        v4_training_path,
        V4_TRAINING_LOG_SHA256,
        name="v4 training log",
    )
    v4_rows = [
        json.loads(line)
        for line in v4_training_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    _require_equal(
        "v4 training epochs",
        [row.get("epoch") for row in v4_rows],
        [0, 1, 2, 3],
    )
    _require_equal(
        "v4 ineligible epochs",
        [row.get("eligible") for row in v4_rows[1:]],
        [False, False, False],
    )
    p3_draws = sum(
        int(row["sampling"]["counts"].get("Guitar-TECHS", 0))
        for row in v4_rows[1:]
    )
    _require_equal("v4 total P3 draws", p3_draws, 102)
    v4_checkpoint = (
        root / "checkpoints/guitarset-candidate-sequence-string-v4.pt"
    )
    if v4_checkpoint.exists():
        raise ValueError("v4 checkpoint exists despite failed training gate")

    sampling = dict(v4_contract["sampling"])
    _require_equal("v4 expected P3 share", sampling["expected_p3_share"], 0.01)
    return {
        "schema_version": 1,
        "kind": "candidate-sequence-string-v5-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_read": False,
        "cross_source_validation_read": False,
        "preregistration": preregistration_report,
        "v4_evidence": {
            "input_audit": v4_input_report,
            "training_log": v4_training_report,
            "result": v4_result_report,
            "epochs": 3,
            "eligible_epochs": 0,
            "p3_draws": p3_draws,
            "checkpoint_created": False,
        },
        "base_input_contract": v4_contract,
        "single_change_from_v4": {
            "kind": "p3-sampler-mass",
            "v4_expected_p3_share": 0.01,
            "v5_expected_p3_share": P3_EXPECTED_SHARE,
            "v4_p3_weight": sampling["p3_weight"],
            "v5_p3_weight": P3_SAMPLING_WEIGHT,
        },
        "sampling": {
            "replacement": True,
            "draws_per_epoch": 3489,
            "guitarset_weight": 1.0,
            "p3_weight": P3_SAMPLING_WEIGHT,
            "expected_p3_share": P3_EXPECTED_SHARE,
            "expected_p3_draws": 0,
            "p3_diagnostic_only": True,
        },
    }


def write_input_audit(path: Path, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    payload = v4_checkpoint_payload(
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
    metadata["version"] = "candidate-sequence-string-v5"
    metadata["experiment"] = {
        "kind": "guitarset-only-replay-control",
        "single_change_from_v4": True,
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
                "stage": "precompute_v5_groups",
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
    _require_equal(
        "GuitarSet-only expected P3 share",
        float(expected_shares["Guitar-TECHS"]),
        P3_EXPECTED_SHARE,
    )

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
        p3_draws = int(
            actual_sampling["counts"].get("Guitar-TECHS", 0)
        )
        if p3_draws:
            raise RuntimeError(
                f"GuitarSet-only epoch sampled {p3_draws} P3 windows"
            )
        _require_equal(
            f"epoch {epoch} GuitarSet draws",
            int(actual_sampling["counts"].get("GuitarSet", 0)),
            3489,
        )
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
            "Run candidate sequence string v5 as a GuitarSet-only replay "
            "control with zero P3 training mass."
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
                "Refusing to overwrite stale v5 training output: "
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
