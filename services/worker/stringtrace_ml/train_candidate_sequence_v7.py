from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .analyze_candidate_gradient_conflict import (
    MODULE_PREFIXES,
    assert_parameters_unchanged,
    identity_sha256,
    module_parameter_report,
    parameter_modules,
    parameters_snapshot,
    windows_by_track,
)
from .candidate_sequence import (
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    candidate_sequence_loss,
    collate_candidate_sequence_windows,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import sha256_file
from .features import FeatureConfig
from .onset_position import OnsetPositionConfig
from .schema import load_manifest
from .train import choose_device
from .train_candidate_sequence import (
    append_log,
    load_position_model,
    move_batch,
)
from .train_candidate_sequence_activity import window_sampling_plan
from .train_candidate_sequence_v3 import (
    BATCH_SIZE,
    EPOCHS,
    GRADIENT_NORM_CAP,
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
    P3_EXPECTED_SHARE,
    P3_SAMPLING_WEIGHT,
    P3_WINDOWS,
    RecordingPermutationSampler,
    build_input_audit as build_v6_input_audit,
    checkpoint_payload as v6_checkpoint_payload,
    full_coverage_report,
    validate_full_coverage,
    window_identities,
    window_track_ids,
)


PREREGISTRATION_SHA256 = (
    "49aa18e6ab3a57ccaae13c7bda555eac770fe704f4b3be1c2d55cc6e08012f0b"
)
V6_PREREGISTRATION_SHA256 = (
    "5be7175bb428ee48d6abd8c2f5f52c4d2b75498014ef984bf00a0f06cfcac11e"
)
V6_INPUT_AUDIT_SHA256 = (
    "81f8d0712f86076c22dc446761e706cc63d226eff321550d28da72baf9e889b9"
)
V6_TRAINING_LOG_SHA256 = (
    "cccfddda70b3ff960eea0cd09e6a57d3892ae9a69e2bf453a94077aff90ac520"
)
V6_RESULT_SHA256 = (
    "24d8101d976f202c34dca03292f70b11ad84eaff556c39918e35345ab8bd6a6d"
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
EXPECTED_GUITARSET_GROUPS = 55_946
EXPECTED_ACTIVE_EXPOSURES = 109_007
ADAM_EPSILON = 1e-8

SOURCE_LOCAL_GRADIENT_REFERENCE = {
    "token_encoder": 0.021608929409185153,
    "sequence_encoder": 0.011811496851312856,
    "residual_hidden": 0.004798116867593484,
    "residual_output": 0.026636370019020944,
    "all": 0.03659201308134063,
}


def fixed_output_paths(root: Path) -> dict[str, Path]:
    return {
        "audit": root / "reports/candidate-sequence-string-v7-input-audit.json",
        "calibration": (
            root
            / "reports/candidate-sequence-string-v7-anchor-calibration.json"
        ),
        "output": (
            root
            / "checkpoints/guitarset-candidate-sequence-string-v7.pt"
        ),
        "log": root / "reports/candidate-sequence-string-v7-training.jsonl",
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
        "v6_preregistration": (
            docs / "candidate-sequence-string-v6-preregistration.md",
            V6_PREREGISTRATION_SHA256,
        ),
        "v6_input_audit": (
            root / "reports/candidate-sequence-string-v6-input-audit.json",
            V6_INPUT_AUDIT_SHA256,
        ),
        "v6_training_log": (
            root / "reports/candidate-sequence-string-v6-training.jsonl",
            V6_TRAINING_LOG_SHA256,
        ),
        "v6_result": (
            docs / "candidate-sequence-string-v6-result.md",
            V6_RESULT_SHA256,
        ),
        "guitarset_gradient_audit_plan": (
            docs
            / "candidate-sequence-guitarset-gradient-conflict-v1-audit-plan.md",
            GUITARSET_AUDIT_PLAN_SHA256,
        ),
        "guitarset_gradient_audit_report": (
            root
            / "reports/candidate-sequence-guitarset-gradient-conflict-v1.json",
            GUITARSET_AUDIT_REPORT_SHA256,
        ),
        "guitarset_gradient_audit_result": (
            docs
            / "candidate-sequence-guitarset-gradient-conflict-v1-result.md",
            GUITARSET_AUDIT_RESULT_SHA256,
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


def _squared_norm(tensors: Iterable[torch.Tensor]) -> float:
    return sum(
        float(torch.sum(tensor.detach().to(torch.float64).square()).cpu())
        for tensor in tensors
    )


def virtual_adamw_update(
    parameters: list[tuple[str, torch.Tensor]],
    gradients: dict[str, torch.Tensor],
    *,
    clip_scale: float,
    learning_rate: float = LEARNING_RATE,
    epsilon: float = ADAM_EPSILON,
    weight_decay: float = WEIGHT_DECAY,
) -> dict[str, object]:
    if not math.isfinite(clip_scale) or not 0 < clip_scale <= 1:
        raise ValueError("gradient clip scale must be in (0, 1]")
    adaptive_squared = 0.0
    decay_squared = 0.0
    update_squared = 0.0
    raw_squared = 0.0
    clipped_squared = 0.0
    updates: dict[str, torch.Tensor] = {}
    for name, parameter in parameters:
        if name not in gradients:
            raise ValueError(f"missing calibration gradient for {name}")
        raw = gradients[name].detach().to(torch.float64)
        if raw.shape != parameter.shape:
            raise ValueError(f"calibration gradient shape changed for {name}")
        clipped = raw * clip_scale
        theta = parameter.detach().to(torch.float64)
        adaptive = (
            -learning_rate * clipped / (clipped.abs() + epsilon)
        )
        decay = -learning_rate * weight_decay * theta
        update = adaptive + decay
        if not torch.isfinite(update).all():
            raise ValueError(f"non-finite virtual update for {name}")
        updates[name] = update
        raw_squared += _squared_norm([raw])
        clipped_squared += _squared_norm([clipped])
        adaptive_squared += _squared_norm([adaptive])
        decay_squared += _squared_norm([decay])
        update_squared += _squared_norm([update])
    return {
        "raw_task_gradient_norm": math.sqrt(raw_squared),
        "clipped_task_gradient_norm": math.sqrt(clipped_squared),
        "adaptive_update_norm": math.sqrt(adaptive_squared),
        "weight_decay_update_norm": math.sqrt(decay_squared),
        "virtual_update_norm": math.sqrt(update_squared),
        "updates": updates,
    }


def _calibration_gradient(
    model: CandidateSequenceStringModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    model.eval()
    model.zero_grad(set_to_none=True)
    active_exposures = 0
    weighted_loss_sum = 0.0
    batches = 0
    for source_batch in loader:
        batch = move_batch(source_batch, device)
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
        loss, _correct, total = candidate_sequence_loss(
            batch["base_string_logits"],
            residuals,
            batch["string_targets"],
            batch["target_weights"],
            batch["candidate_mask"],
            batch["valid_strings"],
        )
        if total <= 0:
            continue
        weighted_loss = loss * total
        weighted_loss.backward()
        active_exposures += total
        weighted_loss_sum += float(weighted_loss.detach().cpu())
        batches += 1
    if active_exposures <= 0:
        raise ValueError("anchor calibration found no active exposures")
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing calibration gradient for {name}")
        gradients[name] = (
            parameter.grad.detach().cpu().to(torch.float64)
            / active_exposures
        )
    model.zero_grad(set_to_none=True)
    return gradients, {
        "batches": batches,
        "active_candidate_exposures": active_exposures,
        "weighted_loss_sum": weighted_loss_sum,
        "mean_weighted_loss": weighted_loss_sum / active_exposures,
    }


def calibrate_anchor(
    training_root: Path,
    preregistration: Path,
    *,
    precompute_device_name: str,
    precompute_batch_size: int,
) -> dict[str, object]:
    root = training_root.resolve()
    preregistration = preregistration.resolve()
    evidence = _verify_evidence(root, preregistration)
    formal_manifest = (
        root / "governed-v1/formal-training-manifest-v3.jsonl"
    )
    annotations = load_manifest(formal_manifest, split="train")
    selection = select_v3_tracks(annotations)

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
        _feature_config,
        position_config,
        _position_metadata,
    ) = load_position_model(position_path, precompute_device)
    model, _initial_string_metadata = load_candidate_sequence_model(
        string_path,
        position_path,
        position_config,
        _feature_config,
        training_device,
    )
    _validate_sequence_model(model)
    snapshot = parameters_snapshot(model)
    training_dataset, _calibration_dataset = _build_training_datasets(
        root,
        formal_manifest,
        position_config,
        selection,
    )
    cache_report = training_dataset.prepare_feature_cache()
    print(
        json.dumps(
            {
                "stage": "precompute_v7_calibration_groups",
                "training_groups": len(training_dataset),
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
    windows = CandidateSequenceWindowDataset(
        training_groups,
        model.config,
    )
    _require_equal(
        "precomputed training groups",
        len(training_groups),
        EXPECTED_GUITARSET_GROUPS + 3_416,
    )
    _require_equal(
        "materialized training windows",
        len(windows),
        GUITARSET_WINDOWS + P3_WINDOWS,
    )
    sampling_weights, sampling_plan = window_sampling_plan(
        windows,
        training_dataset.annotations,
        {
            "GuitarSet": 1.0,
            "Guitar-TECHS": P3_SAMPLING_WEIGHT,
        },
    )
    source_names = window_source_names(
        windows,
        training_dataset.annotations,
    )
    eligible_indices = tuple(
        index
        for index, weight in enumerate(sampling_weights)
        if weight > 0
    )
    _require_equal(
        "calibration eligible windows",
        len(eligible_indices),
        GUITARSET_WINDOWS,
    )
    _require_equal(
        "calibration eligible sources",
        {source_names[index] for index in eligible_indices},
        {"GuitarSet"},
    )
    eligible_group_count = sum(
        training_dataset.annotations[
            group.annotation_index
        ].provenance.get("dataset")
        == "GuitarSet"
        for group in training_groups
    )
    _require_equal(
        "eligible GuitarSet groups",
        eligible_group_count,
        EXPECTED_GUITARSET_GROUPS,
    )
    loader = DataLoader(
        windows,
        batch_size=BATCH_SIZE,
        sampler=eligible_indices,
        num_workers=0,
        collate_fn=collate_candidate_sequence_windows,
    )
    gradients, objective = _calibration_gradient(
        model,
        loader,
        training_device,
    )
    _require_equal(
        "calibration batches",
        objective["batches"],
        math.ceil(GUITARSET_WINDOWS / BATCH_SIZE),
    )
    _require_equal(
        "calibration active candidate exposures",
        objective["active_candidate_exposures"],
        EXPECTED_ACTIVE_EXPOSURES,
    )

    modules = parameter_modules(model)
    raw_total = math.sqrt(
        _squared_norm(gradients.values())
    )
    if not math.isfinite(raw_total) or raw_total <= 0:
        raise ValueError("anchor calibration has an invalid full gradient")
    clip_scale = min(1.0, GRADIENT_NORM_CAP / raw_total)
    module_reports: dict[str, dict[str, object]] = {}
    all_adaptive_squared = 0.0
    all_decay_squared = 0.0
    all_update_squared = 0.0
    for module_name, parameters in modules.items():
        local = virtual_adamw_update(
            [
                (name, parameter.detach().cpu())
                for name, parameter in parameters
            ],
            gradients,
            clip_scale=clip_scale,
        )
        update_norm = float(local["virtual_update_norm"])
        if not math.isfinite(update_norm) or update_norm <= 0:
            raise ValueError(
                f"invalid trust radius for module {module_name}"
            )
        reference = SOURCE_LOCAL_GRADIENT_REFERENCE[module_name]
        module_reports[module_name] = {
            key: value
            for key, value in local.items()
            if key != "updates"
        }
        module_reports[module_name].update(
            {
                "radius": update_norm,
                "source_local_reference_norm": reference,
                "raw_norm_minus_reference": (
                    float(local["raw_task_gradient_norm"]) - reference
                ),
                "raw_to_reference_ratio": (
                    float(local["raw_task_gradient_norm"]) / reference
                ),
            }
        )
        all_adaptive_squared += (
            float(local["adaptive_update_norm"]) ** 2
        )
        all_decay_squared += (
            float(local["weight_decay_update_norm"]) ** 2
        )
        all_update_squared += update_norm**2

    all_report = {
        "raw_task_gradient_norm": raw_total,
        "clipped_task_gradient_norm": raw_total * clip_scale,
        "adaptive_update_norm": math.sqrt(all_adaptive_squared),
        "weight_decay_update_norm": math.sqrt(all_decay_squared),
        "virtual_update_norm": math.sqrt(all_update_squared),
        "source_local_reference_norm": (
            SOURCE_LOCAL_GRADIENT_REFERENCE["all"]
        ),
        "raw_norm_minus_reference": (
            raw_total - SOURCE_LOCAL_GRADIENT_REFERENCE["all"]
        ),
        "raw_to_reference_ratio": (
            raw_total / SOURCE_LOCAL_GRADIENT_REFERENCE["all"]
        ),
    }
    assert_parameters_unchanged(model, snapshot)
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "candidate-sequence-string-v7-anchor-calibration",
        "passed": True,
        "training_only": True,
        "optimizer_created": False,
        "checkpoint_created": False,
        "parameter_mutation_detected": False,
        "governed_validation_result_read": False,
        "cross_source_validation_result_read": False,
        "evidence": evidence,
        "parameter_partition": module_parameter_report(modules),
        "dataset": {
            "precomputed_groups": len(training_groups),
            "eligible_guitarset_groups": eligible_group_count,
            "materialized_windows": len(windows),
            "eligible_guitarset_windows": len(eligible_indices),
            "ineligible_p3_windows": P3_WINDOWS,
            "cache": cache_report,
            "sampling": sampling_plan,
        },
        "objective": objective,
        "optimizer_scale": {
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": ADAM_EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "global_clip_scale": clip_scale,
        },
        "modules": module_reports,
        "all": all_report,
        "radii": {
            name: float(values["radius"])
            for name, values in module_reports.items()
        },
        "zero_parameter_mutation_confirmed": True,
    }
    return report


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_calibration_report(
    report: dict[str, object],
) -> dict[str, float]:
    for name, expected in {
        "kind": "candidate-sequence-string-v7-anchor-calibration",
        "passed": True,
        "training_only": True,
        "optimizer_created": False,
        "checkpoint_created": False,
        "parameter_mutation_detected": False,
        "zero_parameter_mutation_confirmed": True,
    }.items():
        _require_equal(f"anchor calibration {name}", report.get(name), expected)
    objective = dict(report.get("objective", {}))
    _require_equal(
        "anchor calibration exposure denominator",
        objective.get("active_candidate_exposures"),
        EXPECTED_ACTIVE_EXPOSURES,
    )
    radii = {
        str(name): float(value)
        for name, value in dict(report.get("radii", {})).items()
    }
    _require_equal(
        "anchor calibration radius modules",
        set(radii),
        set(MODULE_PREFIXES),
    )
    for name, radius in radii.items():
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"invalid frozen radius for {name}")
    return radii


def build_input_audit(
    training_root: Path,
    preregistration: Path,
    calibration_path: Path,
) -> dict[str, object]:
    root = training_root.resolve()
    preregistration = preregistration.resolve()
    calibration_path = calibration_path.resolve()
    evidence = _verify_evidence(root, preregistration)
    v6_preregistration = (
        preregistration.parent
        / "candidate-sequence-string-v6-preregistration.md"
    )
    v6_contract = build_v6_input_audit(root, v6_preregistration)
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    radii = validate_calibration_report(calibration)
    calibration_identity = {
        "name": calibration_path.name,
        "sha256": sha256_file(calibration_path),
    }
    return {
        "schema_version": 1,
        "kind": "candidate-sequence-string-v7-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_result_read": False,
        "cross_source_validation_result_read": False,
        "evidence": evidence,
        "anchor_calibration": {
            **calibration_identity,
            "radii": radii,
            "active_candidate_exposures": EXPECTED_ACTIVE_EXPOSURES,
            "parameter_mutation_detected": False,
        },
        "base_input_contract": v6_contract,
        "single_policy_change_from_v6": {
            "kind": "post-adamw-module-trust-region",
            "center": "string-v1",
            "module_count": len(radii),
            "optimizer_moments_projected": False,
            "task_gradient_changed": False,
            "replay_changed": False,
        },
        "sampling": {
            "replacement": False,
            "draws_per_epoch": GUITARSET_WINDOWS,
            "eligible_guitarset_windows": GUITARSET_WINDOWS,
            "ineligible_p3_windows": P3_WINDOWS,
            "expected_p3_share": P3_EXPECTED_SHARE,
            "all_eligible_windows_once": True,
        },
    }


def load_locked_calibration(
    audit_path: Path,
    calibration_path: Path,
) -> tuple[dict[str, float], str]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for name, expected in {
        "kind": "candidate-sequence-string-v7-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
    }.items():
        _require_equal(f"v7 input audit {name}", audit.get(name), expected)
    identity = dict(audit.get("anchor_calibration", {}))
    actual_hash = sha256_file(calibration_path)
    _require_equal(
        "locked anchor calibration filename",
        identity.get("name"),
        calibration_path.name,
    )
    _require_equal(
        "locked anchor calibration hash",
        identity.get("sha256"),
        actual_hash,
    )
    report = json.loads(calibration_path.read_text(encoding="utf-8"))
    radii = validate_calibration_report(report)
    _require_equal(
        "locked anchor radii",
        dict(identity.get("radii", {})),
        radii,
    )
    return radii, actual_hash


def snapshot_anchor(
    model: CandidateSequenceStringModel,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def module_displacement_norm(
    parameters: list[tuple[str, torch.nn.Parameter]],
    anchor: dict[str, torch.Tensor],
) -> float:
    squared = 0.0
    for name, parameter in parameters:
        delta = parameter.detach().to(torch.float64) - anchor[name].to(
            device=parameter.device,
            dtype=torch.float64,
        )
        squared += float(torch.sum(delta.square()).cpu())
    return math.sqrt(squared)


def project_modules_to_anchor(
    modules: dict[str, list[tuple[str, torch.nn.Parameter]]],
    anchor: dict[str, torch.Tensor],
    radii: dict[str, float],
) -> dict[str, dict[str, float | bool]]:
    if set(modules) != set(radii):
        raise ValueError("anchor modules and radii do not align")
    result: dict[str, dict[str, float | bool]] = {}
    with torch.no_grad():
        for module_name, parameters in modules.items():
            radius = radii[module_name]
            before = module_displacement_norm(parameters, anchor)
            projected = before > radius
            refinements = 0
            if projected:
                scale = radius / before
                for name, parameter in parameters:
                    center = anchor[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    parameter.copy_(
                        center + (parameter - center) * scale
                    )
            after = module_displacement_norm(parameters, anchor)
            tolerance = max(1e-12, radius * 1e-6)
            while (
                projected
                and after > radius + tolerance
                and refinements < 16
            ):
                inset = min(
                    radius * 0.5,
                    tolerance * (2 ** (refinements + 1)),
                )
                correction = (radius - inset) / after
                for name, parameter in parameters:
                    center = anchor[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    parameter.copy_(
                        center + (parameter - center) * correction
                    )
                refinements += 1
                after = module_displacement_norm(parameters, anchor)
            if after > radius + tolerance:
                raise RuntimeError(
                    f"{module_name} trust-region cap failed: "
                    f"{after} > {radius} + {tolerance}"
                )
            result[module_name] = {
                "before": before,
                "after": after,
                "projected": projected,
                "radius": radius,
                "numerical_refinements": refinements,
            }
    return result


def _module_gradient_norms(
    modules: dict[str, list[tuple[str, torch.nn.Parameter]]],
) -> dict[str, float]:
    squared_by_module: dict[str, float] = {}
    for module_name, parameters in modules.items():
        missing = [
            name for name, parameter in parameters if parameter.grad is None
        ]
        if missing:
            raise RuntimeError(
                "missing task gradients for " + ", ".join(missing)
            )
        squared_by_module[module_name] = _squared_norm(
            parameter.grad
            for _, parameter in parameters
        )
    return {
        **{
            name: math.sqrt(value)
            for name, value in squared_by_module.items()
        },
        "all": math.sqrt(sum(squared_by_module.values())),
    }


def run_anchored_training_epoch(
    model: CandidateSequenceStringModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    anchor: dict[str, torch.Tensor],
    radii: dict[str, float],
) -> tuple[dict[str, float | int], dict[str, object]]:
    model.train()
    modules = parameter_modules(model)
    loss_total = 0.0
    correct_total = 0
    candidate_total = 0
    batches = 0
    projection_counts = {name: 0 for name in modules}
    max_before = {name: 0.0 for name in modules}
    max_after = {name: 0.0 for name in modules}
    gradient_sums = {name: 0.0 for name in (*modules, "all")}
    gradient_maxima = {name: 0.0 for name in (*modules, "all")}
    last_step: dict[str, dict[str, float | bool]] | None = None

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
        gradient_norms = _module_gradient_norms(modules)
        for name, value in gradient_norms.items():
            gradient_sums[name] += value
            gradient_maxima[name] = max(gradient_maxima[name], value)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_NORM_CAP,
        )
        optimizer.step()
        last_step = project_modules_to_anchor(
            modules,
            anchor,
            radii,
        )
        for name, values in last_step.items():
            projection_counts[name] += int(bool(values["projected"]))
            max_before[name] = max(
                max_before[name],
                float(values["before"]),
            )
            max_after[name] = max(
                max_after[name],
                float(values["after"]),
            )
        loss_total += float(loss.detach().cpu())
        correct_total += correct
        candidate_total += total
        batches += 1

    if batches <= 0 or last_step is None:
        raise RuntimeError("anchored training epoch contained no batches")
    anchor_report: dict[str, object] = {}
    for name, values in last_step.items():
        final_displacement = float(values["after"])
        radius = radii[name]
        tolerance = max(1e-12, radius * 1e-6)
        anchor_report[name] = {
            "radius": radius,
            "last_pre_projection_displacement": float(values["before"]),
            "last_post_projection_displacement": final_displacement,
            "maximum_pre_projection_displacement": max_before[name],
            "maximum_post_projection_displacement": max_after[name],
            "projection_count": projection_counts[name],
            "projection_fraction": projection_counts[name] / batches,
            "last_projection_numerical_refinements": int(
                values["numerical_refinements"]
            ),
            "epoch_end_displacement": final_displacement,
            "boundary_occupancy": final_displacement / radius,
            "at_boundary": abs(final_displacement - radius) <= tolerance,
            "task_gradient_norm_before_clipping_mean": (
                gradient_sums[name] / batches
            ),
            "task_gradient_norm_before_clipping_maximum": (
                gradient_maxima[name]
            ),
        }
    anchor_report["all"] = {
        "task_gradient_norm_before_clipping_mean": (
            gradient_sums["all"] / batches
        ),
        "task_gradient_norm_before_clipping_maximum": (
            gradient_maxima["all"]
        ),
        "optimizer_steps": batches,
    }
    train = {
        "loss": loss_total / batches,
        "window_candidate_oracle_string_accuracy": (
            correct_total / max(1, candidate_total)
        ),
        "window_active_candidates": candidate_total,
    }
    return train, anchor_report


def drift_was_prevented(
    metrics: dict[str, object],
    baseline: dict[str, object],
) -> bool:
    return (
        float(metrics["aggregate_accuracy"])
        >= float(baseline["aggregate_accuracy"])
        and int(metrics["aggregate_wrong_strings"])
        <= int(baseline["aggregate_wrong_strings"])
        and int(metrics["repeat_true_positive"])
        >= int(baseline["repeat_true_positive"])
        and float(metrics["guitarset_accuracy"])
        >= float(baseline["guitarset_accuracy"])
        and float(metrics["agpt_accuracy"])
        >= float(baseline["agpt_accuracy"])
        and float(metrics["guitar_techs_p1_accuracy"])
        >= float(baseline["guitar_techs_p1_accuracy"])
        and float(metrics["cross_source_macro_accuracy"])
        >= float(baseline["cross_source_macro_accuracy"])
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
    radii: dict[str, float],
    calibration_sha256: str,
) -> dict[str, object]:
    payload = v6_checkpoint_payload(
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
    metadata["version"] = "candidate-sequence-string-v7"
    metadata["experiment"] = {
        "kind": "string-v1-module-trust-region-control",
        "single_policy_change_from_v6": True,
        "anchor_center": "string-v1",
        "radii": radii,
        "anchor_calibration_sha256": calibration_sha256,
        "projection_after_adamw": True,
        "optimizer_moments_projected": False,
        "replacement": False,
        "draws_per_epoch": GUITARSET_WINDOWS,
        "p3_training_gradient": False,
    }
    payload["metadata"] = metadata
    return payload


def run_training(
    training_root: Path,
    audit_path: Path,
    calibration_path: Path,
    *,
    precompute_device_name: str,
    precompute_batch_size: int,
    inference_batch_size: int,
) -> dict[str, object]:
    root = training_root.resolve()
    radii, calibration_sha256 = load_locked_calibration(
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
                "stage": "precompute_v7_groups",
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
        EXPECTED_GUITARSET_GROUPS + 3_416,
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
            "anchor_calibration": {
                "sha256": calibration_sha256,
                "radii": radii,
            },
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


def _refuse_existing(paths: Iterable[Path], stage: str) -> None:
    stale = [path for path in paths if path.exists()]
    if stale:
        raise FileExistsError(
            f"Refusing to overwrite stale v7 {stage} output: "
            + ", ".join(path.name for path in stale)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate, audit, or train candidate sequence string v7 "
            "with frozen string-v1 module trust regions."
        )
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--precompute-device", default="auto")
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibrate-only", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    arguments = parser.parse_args()
    for name in ("precompute_batch_size", "inference_batch_size"):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    root = arguments.training_root.resolve()
    outputs = fixed_output_paths(root)
    if arguments.calibrate_only:
        _refuse_existing([outputs["calibration"]], "calibration")
        report = calibrate_anchor(
            root,
            arguments.preregistration,
            precompute_device_name=arguments.precompute_device,
            precompute_batch_size=arguments.precompute_batch_size,
        )
        write_json(outputs["calibration"], report)
        print(
            json.dumps(
                {
                    "stage": "anchor_calibration",
                    "passed": True,
                    "output": outputs["calibration"].name,
                    "sha256": sha256_file(outputs["calibration"]),
                    "radii": report["radii"],
                }
            ),
            flush=True,
        )
        return

    if not outputs["calibration"].is_file():
        raise FileNotFoundError(
            "Frozen v7 anchor calibration report does not exist"
        )
    if arguments.audit_only:
        _refuse_existing([outputs["audit"]], "input audit")
        audit = build_input_audit(
            root,
            arguments.preregistration,
            outputs["calibration"],
        )
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
        raise FileNotFoundError("Frozen v7 input audit does not exist")
    _refuse_existing(
        [outputs["output"], outputs["log"]],
        "training",
    )
    run_training(
        root,
        outputs["audit"],
        outputs["calibration"],
        precompute_device_name=arguments.precompute_device,
        precompute_batch_size=arguments.precompute_batch_size,
        inference_batch_size=arguments.inference_batch_size,
    )


if __name__ == "__main__":
    main()
