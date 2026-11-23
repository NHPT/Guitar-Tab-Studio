from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .candidate_sequence import (
    CandidateSequenceConfig,
    CandidateSequenceGroup,
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    candidate_sequence_loss,
    collate_candidate_sequence_windows,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import sha256_file
from .onset_position import OnsetPositionDataset, basic_pitch_cache_path
from .prepare_candidate_cache import validate_candidate_cache
from .schema import TrackAnnotation, load_manifest
from .train import choose_device
from .train_candidate_sequence import load_position_model, move_batch
from .train_candidate_sequence_v3 import (
    EXPECTED_TRAINABLE_PARAMETERS,
    FRONTEND_FRAME_THRESHOLD,
    FRONTEND_MINIMUM_NOTE_LENGTH,
    FRONTEND_ONSET_THRESHOLD,
    P3_GROUP_ID,
    REPEAT_INTERVAL,
    select_v3_tracks,
    verify_artifact_hashes,
    verify_file_hash,
)


AUDIT_PLAN_SHA256 = (
    "3987e2f213745b9c47a4dab7b05d05bccff8e141353812b458c8a5c23dcf909e"
)
SOURCE_LEDGER_VERSION = "2026-11-06.2"
SOURCE_LEDGER_SHA256 = (
    "de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a"
)
AUDIT_SEED = 20261123
BOOTSTRAP_REPLICATES = 10_000
GUITARSET_TRACKS_PER_PLAYER = 12
PANEL_COUNT = 4

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
    "position_checkpoint": (
        "checkpoints/guitarset-candidate-activity-residual-v1.pt",
        "7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e",
    ),
    "string_v1_checkpoint": (
        "checkpoints/guitarset-candidate-sequence-string-v1.pt",
        "392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982",
    ),
}

EXPECTED_SELECTION_SHA256 = (
    "74f07d7a2cade3fc9b448122d690568e0865a4eef0e44ec153a10cd25e4d6fed"
)
EXPECTED_PANEL_SHA256 = {
    "panel-1": "5c89de1d264ead52438cff322a3ab92e1f46291565b91045e728d1465c490ce7",
    "panel-2": "8665ec9cf55cad5864b2044638866538899a7b7393980c0dd837ab7f9d769fae",
    "panel-3": "5fbd835584832190c2c6e35153d713df715eed95c4f0a884965688606e51b1c4",
    "panel-4": "a6b3473a6a0fbd293dd1eac4b7e865763201caef2fe99d930c87e1082e7bf793",
}
EXPECTED_GUITARSET_CACHE = {
    "audio_files": 48,
    "events": 14691,
    "dropped": 139,
    "cache_set_sha256": (
        "38c09ffe13d65f38c97fc7ad0ab0523664ba8ddb6907f51e2829022d0f8d07fc"
    ),
}
EXPECTED_P3_CACHE = {
    "audio_files": 24,
    "events": 6685,
    "dropped": 157,
    "cache_set_sha256": (
        "c592ed2656c29fc4e0c4546a683b7c500c5b8f229fb431ea20eae70f2931b91a"
    ),
}

MODULE_PREFIXES = {
    "token_encoder": "token_encoder.",
    "sequence_encoder": "sequence_encoder.",
    "residual_hidden": "residual_head.0.",
    "residual_output": "residual_head.3.",
}


@dataclass(frozen=True)
class P3Pair:
    source_label: str
    direct_track_id: str
    mic_track_id: str


@dataclass(frozen=True)
class GradientAuditSelection:
    guitarset_track_ids: tuple[str, ...]
    panels: dict[str, tuple[str, ...]]
    p3_pairs: tuple[P3Pair, ...]


@dataclass
class UnitGradient:
    track_id: str
    vectors: dict[str, torch.Tensor]
    groups: int
    windows: int
    active_candidate_exposures: int
    weighted_loss_sum: float

    def metadata(self) -> dict[str, int | float | str]:
        return {
            "track_id": self.track_id,
            "groups": self.groups,
            "windows": self.windows,
            "active_candidate_exposures": self.active_candidate_exposures,
            "weighted_loss_sum": self.weighted_loss_sum,
            "mean_weighted_loss": (
                self.weighted_loss_sum
                / max(1, self.active_candidate_exposures)
            ),
        }


def identity_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(f"{value}\n".encode())
    return digest.hexdigest()


def select_guitarset_panels(
    annotations: list[TrackAnnotation],
    eligible_track_ids: set[str] | frozenset[str],
    *,
    seed: int = AUDIT_SEED,
    tracks_per_player: int = GUITARSET_TRACKS_PER_PLAYER,
    panel_count: int = PANEL_COUNT,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    if tracks_per_player <= 0 or panel_count <= 0:
        raise ValueError("track and panel counts must be positive")
    if tracks_per_player % panel_count:
        raise ValueError("tracks per player must divide evenly into panels")

    by_player: dict[str, list[str]] = {}
    for annotation in annotations:
        if (
            annotation.track_id in eligible_track_ids
            and annotation.provenance.get("dataset") == "GuitarSet"
        ):
            by_player.setdefault(annotation.group_id, []).append(
                annotation.track_id
            )
    if not by_player:
        raise ValueError("gradient audit found no eligible GuitarSet tracks")

    panel_size = tracks_per_player // panel_count
    selected: list[str] = []
    panels: dict[str, list[str]] = {
        f"panel-{index + 1}": [] for index in range(panel_count)
    }
    for player, track_ids in sorted(by_player.items()):
        if len(set(track_ids)) != len(track_ids):
            raise ValueError(f"duplicate GuitarSet track in {player}")
        if len(track_ids) < tracks_per_player:
            raise ValueError(
                f"{player} has only {len(track_ids)} eligible tracks"
            )
        ranked = sorted(
            track_ids,
            key=lambda track_id: (
                hashlib.sha256(
                    f"{seed}:{track_id}".encode()
                ).hexdigest(),
                track_id,
            ),
        )
        chosen = ranked[:tracks_per_player]
        selected.extend(chosen)
        for panel_index in range(panel_count):
            start = panel_index * panel_size
            panels[f"panel-{panel_index + 1}"].extend(
                chosen[start : start + panel_size]
            )
    return (
        tuple(sorted(selected)),
        {
            name: tuple(track_ids)
            for name, track_ids in sorted(panels.items())
        },
    )


def pair_p3_tracks(
    annotations: list[TrackAnnotation],
) -> tuple[P3Pair, ...]:
    by_label: dict[str, dict[str, str]] = {}
    for annotation in annotations:
        if (
            annotation.group_id != P3_GROUP_ID
            or annotation.provenance.get("dataset") != "Guitar-TECHS"
        ):
            raise ValueError("P3 pair input contains a non-P3 annotation")
        source_label = str(annotation.provenance.get("source_label", ""))
        modality = str(annotation.provenance.get("audio_modality", ""))
        if not source_label or modality not in {"directinput", "micamp"}:
            raise ValueError(
                f"invalid P3 pairing metadata for {annotation.track_id}"
            )
        modalities = by_label.setdefault(source_label, {})
        if modality in modalities:
            raise ValueError(
                f"duplicate P3 {modality} track for {source_label}"
            )
        modalities[modality] = annotation.track_id

    pairs: list[P3Pair] = []
    for source_label, modalities in sorted(by_label.items()):
        if set(modalities) != {"directinput", "micamp"}:
            raise ValueError(
                f"P3 source {source_label} does not have both modalities"
            )
        pairs.append(
            P3Pair(
                source_label=source_label,
                direct_track_id=modalities["directinput"],
                mic_track_id=modalities["micamp"],
            )
        )
    return tuple(pairs)


def select_audit_units(
    annotations: list[TrackAnnotation],
) -> GradientAuditSelection:
    v3_selection = select_v3_tracks(annotations)
    guitarset_track_ids, panels = select_guitarset_panels(
        annotations,
        v3_selection.gradient_track_ids,
    )
    p3_annotations = [
        annotation
        for annotation in annotations
        if annotation.track_id in v3_selection.p3_track_ids
    ]
    pairs = pair_p3_tracks(p3_annotations)

    if len(guitarset_track_ids) != 48:
        raise ValueError("gradient audit requires exactly 48 GuitarSet tracks")
    if len(pairs) != 12:
        raise ValueError("gradient audit requires exactly 12 P3 pairs")
    if identity_sha256(guitarset_track_ids) != EXPECTED_SELECTION_SHA256:
        raise ValueError("frozen GuitarSet gradient selection changed")
    for name, track_ids in panels.items():
        if len(track_ids) != 12:
            raise ValueError(f"{name} must contain exactly 12 tracks")
        if identity_sha256(track_ids) != EXPECTED_PANEL_SHA256[name]:
            raise ValueError(f"frozen {name} selection changed")
    return GradientAuditSelection(
        guitarset_track_ids=guitarset_track_ids,
        panels=panels,
        p3_pairs=pairs,
    )


def parameter_modules(
    model: CandidateSequenceStringModel,
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    modules = {name: [] for name in MODULE_PREFIXES}
    for parameter_name, parameter in model.named_parameters():
        matches = [
            module_name
            for module_name, prefix in MODULE_PREFIXES.items()
            if parameter_name.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"parameter {parameter_name!r} maps to {len(matches)} modules"
            )
        if not parameter.requires_grad:
            raise ValueError(
                f"sequence parameter is unexpectedly frozen: {parameter_name}"
            )
        modules[matches[0]].append((parameter_name, parameter))
    if any(not parameters for parameters in modules.values()):
        raise ValueError("gradient audit module partition is incomplete")
    return modules


def module_parameter_report(
    modules: dict[str, list[tuple[str, torch.nn.Parameter]]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    total = 0
    tensor_count = 0
    for name, parameters in modules.items():
        count = sum(parameter.numel() for _, parameter in parameters)
        total += count
        tensor_count += len(parameters)
        result[name] = {
            "parameters": count,
            "tensors": [parameter_name for parameter_name, _ in parameters],
        }
    result["all"] = {
        "parameters": total,
        "tensor_count": tensor_count,
    }
    if total != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError(
            "unexpected candidate sequence parameter count: "
            f"{total} != {EXPECTED_TRAINABLE_PARAMETERS}"
        )
    return result


def verify_audit_plan(
    path: Path,
    expected_sha256: str = AUDIT_PLAN_SHA256,
) -> dict[str, str]:
    return verify_file_hash(
        path,
        expected_sha256,
        name="gradient conflict audit plan",
    )


def candidate_cache_identity(
    annotations: list[TrackAnnotation],
    manifest_path: Path,
    cache_directory: Path,
) -> dict[str, int | str]:
    identities: list[tuple[str, str]] = []
    events = 0
    dropped = 0
    for annotation in annotations:
        cache_path = basic_pitch_cache_path(
            annotation.resolve_audio(manifest_path),
            cache_directory,
            onset_threshold=FRONTEND_ONSET_THRESHOLD,
            frame_threshold=FRONTEND_FRAME_THRESHOLD,
            minimum_note_length=FRONTEND_MINIMUM_NOTE_LENGTH,
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"missing frozen candidate cache: {cache_path.name}"
            )
        counts = validate_candidate_cache(cache_path)
        events += counts["events"]
        dropped += counts["dropped"]
        identities.append((cache_path.name, sha256_file(cache_path)))
    digest = hashlib.sha256()
    for name, value in sorted(identities):
        digest.update(f"{name}:{value}\n".encode())
    return {
        "audio_files": len(annotations),
        "events": events,
        "dropped": dropped,
        "cache_set_sha256": digest.hexdigest(),
    }


def require_mapping_equal(
    name: str,
    actual: dict[str, int | str],
    expected: dict[str, int | str],
) -> None:
    if actual != expected:
        raise ValueError(
            f"frozen {name} mismatch: expected {expected!r}, got {actual!r}"
        )


def build_input_contract(
    root: Path,
    plan_path: Path,
) -> tuple[
    dict[str, object],
    list[TrackAnnotation],
    GradientAuditSelection,
]:
    artifact_report = verify_artifact_hashes(root, FROZEN_ARTIFACTS)
    plan_report = verify_audit_plan(plan_path)
    ledger_path = Path(__file__).with_name("source_ledger.json")
    ledger_report = verify_file_hash(
        ledger_path,
        SOURCE_LEDGER_SHA256,
        name="source ledger",
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("ledger_version") != SOURCE_LEDGER_VERSION:
        raise ValueError("frozen source ledger version changed")

    v3_audit_path = root / FROZEN_ARTIFACTS["v3_input_audit"][0]
    v3_audit = json.loads(v3_audit_path.read_text(encoding="utf-8"))
    expected_flags = {
        "kind": "candidate-sequence-string-v3-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_read": False,
        "cross_source_validation_read": False,
    }
    for name, expected in expected_flags.items():
        if v3_audit.get(name) != expected:
            raise ValueError(f"v3 input audit flag changed: {name}")

    manifest_path = root / FROZEN_ARTIFACTS["formal_manifest"][0]
    annotations = load_manifest(manifest_path, split="train")
    selection = select_audit_units(annotations)
    selected_ids = set(selection.guitarset_track_ids)
    selected_guitarset = [
        annotation
        for annotation in annotations
        if annotation.track_id in selected_ids
    ]
    p3_ids = {
        track_id
        for pair in selection.p3_pairs
        for track_id in (pair.direct_track_id, pair.mic_track_id)
    }
    p3_manifest_path = root / FROZEN_ARTIFACTS["p3_manifest"][0]
    p3_manifest_annotations = load_manifest(p3_manifest_path, split="train")
    if {item.track_id for item in p3_manifest_annotations} != p3_ids:
        raise ValueError("P3 manifest identity differs from formal-v3")

    cache_directory = root / "governed-v1/.basic-pitch-cache"
    guitarset_cache = candidate_cache_identity(
        selected_guitarset,
        manifest_path,
        cache_directory,
    )
    p3_cache = candidate_cache_identity(
        p3_manifest_annotations,
        p3_manifest_path,
        cache_directory,
    )
    require_mapping_equal(
        "GuitarSet candidate cache",
        guitarset_cache,
        EXPECTED_GUITARSET_CACHE,
    )
    require_mapping_equal(
        "P3 candidate cache",
        p3_cache,
        EXPECTED_P3_CACHE,
    )

    return (
        {
            "audit_plan": plan_report,
            "artifacts": artifact_report,
            "source_ledger": {
                **ledger_report,
                "version": SOURCE_LEDGER_VERSION,
            },
            "v3_input_audit_flags": expected_flags,
            "candidate_caches": {
                "guitarset": guitarset_cache,
                "p3": p3_cache,
            },
        },
        annotations,
        selection,
    )


def windows_by_track(
    groups: list[CandidateSequenceGroup],
    annotations: list[TrackAnnotation],
    config: CandidateSequenceConfig,
) -> dict[str, list[list[CandidateSequenceGroup]]]:
    dataset = CandidateSequenceWindowDataset(groups, config)
    result: dict[str, list[list[CandidateSequenceGroup]]] = {}
    for index in range(len(dataset)):
        window = dataset[index]
        annotation_indexes = {
            group.annotation_index for group in window
        }
        if len(annotation_indexes) != 1:
            raise RuntimeError("sequence window crosses annotation boundary")
        annotation_index = next(iter(annotation_indexes))
        track_id = annotations[annotation_index].track_id
        result.setdefault(track_id, []).append(window)
    return result


def extract_unit_gradient(
    model: CandidateSequenceStringModel,
    windows: list[list[CandidateSequenceGroup]],
    modules: dict[str, list[tuple[str, torch.nn.Parameter]]],
    *,
    track_id: str,
    batch_size: int,
    device: torch.device,
) -> UnitGradient:
    if batch_size <= 0:
        raise ValueError("gradient batch size must be positive")
    if not windows:
        raise ValueError(f"track {track_id} has no sequence windows")
    model.eval()
    model.zero_grad(set_to_none=True)
    active_candidate_exposures = 0
    weighted_loss_sum = 0.0
    group_indexes: set[tuple[int, float]] = set()

    for start in range(0, len(windows), batch_size):
        local_windows = windows[start : start + batch_size]
        for window in local_windows:
            group_indexes.update(
                (group.annotation_index, group.onset) for group in window
            )
        batch = move_batch(
            collate_candidate_sequence_windows(local_windows),
            device,
        )
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
        weighted = loss * total
        weighted.backward()
        active_candidate_exposures += total
        weighted_loss_sum += float(weighted.detach().cpu())

    if active_candidate_exposures <= 0:
        raise ValueError(f"track {track_id} has no active candidates")
    vectors: dict[str, torch.Tensor] = {}
    for module_name, parameters in modules.items():
        parts: list[torch.Tensor] = []
        for parameter_name, parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError(
                    f"missing gradient for parameter {parameter_name}"
                )
            parts.append(
                parameter.grad.detach().reshape(-1).to(
                    device="cpu",
                    dtype=torch.float64,
                )
                / active_candidate_exposures
            )
        vectors[module_name] = torch.cat(parts)
    model.zero_grad(set_to_none=True)
    return UnitGradient(
        track_id=track_id,
        vectors=vectors,
        groups=len(group_indexes),
        windows=len(windows),
        active_candidate_exposures=active_candidate_exposures,
        weighted_loss_sum=weighted_loss_sum,
    )


def parameters_snapshot(
    model: CandidateSequenceStringModel,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def assert_parameters_unchanged(
    model: CandidateSequenceStringModel,
    snapshot: dict[str, torch.Tensor],
) -> None:
    current_names = {name for name, _ in model.named_parameters()}
    if current_names != set(snapshot):
        raise RuntimeError("sequence model parameter set changed")
    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach().cpu(), snapshot[name])
    ]
    if changed:
        raise RuntimeError(
            "gradient audit mutated parameters: " + ", ".join(changed)
        )


def unit_matrix(
    units: list[UnitGradient],
    module_name: str,
) -> torch.Tensor:
    if not units:
        raise ValueError("gradient unit matrix cannot be empty")
    return torch.stack([unit.vectors[module_name] for unit in units])


def module_gram_matrices(
    guitarset: list[UnitGradient],
    direct: list[UnitGradient],
    mic: list[UnitGradient],
    module_name: str,
) -> dict[str, np.ndarray]:
    guitarset_matrix = unit_matrix(guitarset, module_name)
    direct_matrix = unit_matrix(direct, module_name)
    mic_matrix = unit_matrix(mic, module_name)
    paired_matrix = (direct_matrix + mic_matrix) / 2
    return {
        "guitarset": (guitarset_matrix @ guitarset_matrix.T).numpy(),
        "paired": (paired_matrix @ paired_matrix.T).numpy(),
        "guitarset_paired": (
            guitarset_matrix @ paired_matrix.T
        ).numpy(),
        "direct": (direct_matrix @ direct_matrix.T).numpy(),
        "mic": (mic_matrix @ mic_matrix.T).numpy(),
        "direct_mic": (direct_matrix @ mic_matrix.T).numpy(),
    }


def sum_gram_matrices(
    matrices: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not matrices:
        raise ValueError("at least one module Gram matrix is required")
    keys = set(matrices[0])
    if any(set(matrix) != keys for matrix in matrices):
        raise ValueError("module Gram matrix keys differ")
    return {
        key: sum(
            (matrix[key] for matrix in matrices[1:]),
            matrices[0][key].copy(),
        )
        for key in sorted(keys)
    }


def cosine_from_grams(
    aa: np.ndarray,
    bb: np.ndarray,
    ab: np.ndarray,
    a_counts: np.ndarray,
    b_counts: np.ndarray,
) -> tuple[float, float, float]:
    a_counts = np.asarray(a_counts, dtype=np.float64)
    b_counts = np.asarray(b_counts, dtype=np.float64)
    if (
        aa.shape != (a_counts.size, a_counts.size)
        or bb.shape != (b_counts.size, b_counts.size)
        or ab.shape != (a_counts.size, b_counts.size)
    ):
        raise ValueError("Gram matrix and count shapes do not align")
    if a_counts.sum() <= 0 or b_counts.sum() <= 0:
        raise ValueError("gradient mean counts must be positive")
    a_squared = float(a_counts @ aa @ a_counts)
    b_squared = float(b_counts @ bb @ b_counts)
    if a_squared <= 0 or b_squared <= 0:
        raise ValueError("gradient mean has zero norm")
    dot = float(a_counts @ ab @ b_counts)
    a_norm = math.sqrt(a_squared) / float(a_counts.sum())
    b_norm = math.sqrt(b_squared) / float(b_counts.sum())
    cosine = dot / math.sqrt(a_squared * b_squared)
    return cosine, a_norm, b_norm


def bootstrap_cosines_from_grams(
    aa: np.ndarray,
    bb: np.ndarray,
    ab: np.ndarray,
    a_counts: np.ndarray,
    b_counts: np.ndarray,
) -> np.ndarray:
    a_counts = np.asarray(a_counts, dtype=np.float64)
    b_counts = np.asarray(b_counts, dtype=np.float64)
    if a_counts.ndim != 2 or b_counts.ndim != 2:
        raise ValueError("bootstrap counts must be two-dimensional")
    if a_counts.shape[0] != b_counts.shape[0]:
        raise ValueError("bootstrap replicate counts differ")
    numerator = np.einsum(
        "bi,ij,bj->b",
        a_counts,
        ab,
        b_counts,
        optimize=True,
    )
    a_squared = np.einsum(
        "bi,ij,bj->b",
        a_counts,
        aa,
        a_counts,
        optimize=True,
    )
    b_squared = np.einsum(
        "bi,ij,bj->b",
        b_counts,
        bb,
        b_counts,
        optimize=True,
    )
    denominator = np.sqrt(a_squared * b_squared)
    if bool(np.any(denominator <= 0)):
        raise ValueError("bootstrap gradient mean has zero norm")
    return numerator / denominator


def classify_interval(lower: float, upper: float) -> str:
    if upper < 0:
        return "conflict"
    if lower > 0:
        return "aligned"
    return "inconclusive"


def distribution_summary(values: np.ndarray) -> dict[str, int | float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("distribution summary requires a nonempty vector")
    if not bool(np.isfinite(values).all()):
        raise ValueError("distribution contains a non-finite value")
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "maximum": float(values.max()),
    }


def comparison_report(
    grams: dict[str, np.ndarray],
    guitarset_track_ids: tuple[str, ...],
    panels: dict[str, tuple[str, ...]],
    pair_labels: tuple[str, ...],
    guitarset_bootstrap_counts: np.ndarray,
    p3_bootstrap_counts: np.ndarray,
) -> dict[str, object]:
    guitarset_ones = np.ones(len(guitarset_track_ids), dtype=np.float64)
    p3_ones = np.ones(len(pair_labels), dtype=np.float64)
    cosine, guitarset_norm, p3_norm = cosine_from_grams(
        grams["guitarset"],
        grams["paired"],
        grams["guitarset_paired"],
        guitarset_ones,
        p3_ones,
    )
    bootstrap = bootstrap_cosines_from_grams(
        grams["guitarset"],
        grams["paired"],
        grams["guitarset_paired"],
        guitarset_bootstrap_counts,
        p3_bootstrap_counts,
    )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])

    track_index = {
        track_id: index
        for index, track_id in enumerate(guitarset_track_ids)
    }
    panel_cosines: dict[str, float] = {}
    for panel_name, panel_track_ids in panels.items():
        counts = np.zeros(len(guitarset_track_ids), dtype=np.float64)
        for track_id in panel_track_ids:
            counts[track_index[track_id]] = 1
        panel_cosines[panel_name] = cosine_from_grams(
            grams["guitarset"],
            grams["paired"],
            grams["guitarset_paired"],
            counts,
            p3_ones,
        )[0]

    modality_cosine, direct_norm, mic_norm = cosine_from_grams(
        grams["direct"],
        grams["mic"],
        grams["direct_mic"],
        p3_ones,
        p3_ones,
    )
    modality_bootstrap = bootstrap_cosines_from_grams(
        grams["direct"],
        grams["mic"],
        grams["direct_mic"],
        p3_bootstrap_counts,
        p3_bootstrap_counts,
    )
    modality_lower, modality_upper = np.quantile(
        modality_bootstrap,
        [0.025, 0.975],
    )

    direct_squared = np.diag(grams["direct"])
    mic_squared = np.diag(grams["mic"])
    paired_squared = np.diag(grams["paired"])
    guitarset_squared = np.diag(grams["guitarset"])
    pair_denominator = np.sqrt(direct_squared * mic_squared)
    if bool(np.any(pair_denominator <= 0)):
        raise ValueError("P3 modality pair has a zero-norm gradient")
    pair_cosines = np.diag(grams["direct_mic"]) / pair_denominator

    return {
        "guitarset_vs_p3": {
            "cosine": cosine,
            "bootstrap_95_ci": [float(lower), float(upper)],
            "classification": classify_interval(
                float(lower),
                float(upper),
            ),
            "guitarset_mean_gradient_norm": guitarset_norm,
            "p3_mean_gradient_norm": p3_norm,
            "p3_to_guitarset_norm_ratio": p3_norm / guitarset_norm,
            "guitarset_unit_norms": distribution_summary(
                np.sqrt(guitarset_squared)
            ),
            "p3_pair_unit_norms": distribution_summary(
                np.sqrt(paired_squared)
            ),
            "panel_cosines": panel_cosines,
        },
        "p3_direct_vs_mic": {
            "cosine": modality_cosine,
            "bootstrap_95_ci": [
                float(modality_lower),
                float(modality_upper),
            ],
            "classification": classify_interval(
                float(modality_lower),
                float(modality_upper),
            ),
            "direct_mean_gradient_norm": direct_norm,
            "mic_mean_gradient_norm": mic_norm,
            "mic_to_direct_norm_ratio": mic_norm / direct_norm,
            "within_pair_cosines": {
                source_label: float(pair_cosine)
                for source_label, pair_cosine in zip(
                    pair_labels,
                    pair_cosines,
                )
            },
            "within_pair_summary": distribution_summary(pair_cosines),
        },
    }


def selection_report(
    selection: GradientAuditSelection,
) -> dict[str, object]:
    return {
        "seed": AUDIT_SEED,
        "guitarset": {
            "tracks": list(selection.guitarset_track_ids),
            "track_count": len(selection.guitarset_track_ids),
            "track_identity_sha256": identity_sha256(
                selection.guitarset_track_ids
            ),
        },
        "panels": {
            name: {
                "tracks": list(track_ids),
                "track_count": len(track_ids),
                "track_identity_sha256": identity_sha256(track_ids),
            }
            for name, track_ids in selection.panels.items()
        },
        "p3_pairs": [
            {
                "source_label": pair.source_label,
                "direct_track_id": pair.direct_track_id,
                "mic_track_id": pair.mic_track_id,
            }
            for pair in selection.p3_pairs
        ],
    }


def run_audit(
    training_root: Path,
    plan_path: Path,
    *,
    precompute_device_name: str,
    precompute_batch_size: int,
    gradient_batch_size: int,
) -> dict[str, object]:
    root = training_root.resolve()
    input_contract, annotations, selection = build_input_contract(
        root,
        plan_path.resolve(),
    )
    random.seed(AUDIT_SEED)
    np.random.seed(AUDIT_SEED)
    torch.manual_seed(AUDIT_SEED)

    precompute_device = choose_device(precompute_device_name)
    gradient_device = torch.device("cpu")
    position_path = root / FROZEN_ARTIFACTS["position_checkpoint"][0]
    string_path = root / FROZEN_ARTIFACTS["string_v1_checkpoint"][0]
    (
        position_model,
        feature_config,
        position_config,
        _position_metadata,
    ) = load_position_model(position_path, precompute_device)
    model, _string_metadata = load_candidate_sequence_model(
        string_path,
        position_path,
        position_config,
        feature_config,
        gradient_device,
    )
    modules = parameter_modules(model)
    module_report = module_parameter_report(modules)
    parameter_snapshot = parameters_snapshot(model)

    selected_track_ids = set(selection.guitarset_track_ids)
    selected_track_ids.update(
        track_id
        for pair in selection.p3_pairs
        for track_id in (pair.direct_track_id, pair.mic_track_id)
    )
    manifest_path = root / FROZEN_ARTIFACTS["formal_manifest"][0]
    dataset = OnsetPositionDataset(
        manifest_path,
        split="train",
        config=position_config,
        feature_cache_directory=root / "governed-v1/.feature-cache",
        candidate_cache_directory=root / "governed-v1/.basic-pitch-cache",
        candidate_onset_threshold=FRONTEND_ONSET_THRESHOLD,
        candidate_frame_threshold=FRONTEND_FRAME_THRESHOLD,
        candidate_minimum_note_length=FRONTEND_MINIMUM_NOTE_LENGTH,
        included_track_ids=selected_track_ids,
        candidate_ambiguity_weight=1.25,
        candidate_rare_pair_max_weight=1.5,
        candidate_cross_string_weight=2.0,
        candidate_cross_string_interval=2.0,
        candidate_max_supervision_weight=4.0,
    )
    if {item.track_id for item in dataset.annotations} != selected_track_ids:
        raise ValueError("gradient dataset track identity changed")
    feature_cache_report = dataset.prepare_feature_cache()
    print(
        json.dumps(
            {
                "stage": "precompute_gradient_audit_groups",
                "tracks": len(dataset.annotations),
                "groups": len(dataset),
                "precompute_device": str(precompute_device),
            }
        ),
        flush=True,
    )
    groups = precompute_candidate_sequence_groups(
        dataset,
        position_model,
        precompute_device,
        batch_size=precompute_batch_size,
        repeat_interval=REPEAT_INTERVAL,
    )
    track_windows = windows_by_track(
        groups,
        dataset.annotations,
        model.config,
    )
    if set(track_windows) != selected_track_ids:
        missing = sorted(selected_track_ids - set(track_windows))
        raise ValueError(
            "selected tracks have no sequence windows: " + ", ".join(missing)
        )

    unit_by_track: dict[str, UnitGradient] = {}
    extraction_order = [
        *selection.guitarset_track_ids,
        *[
            track_id
            for pair in selection.p3_pairs
            for track_id in (pair.direct_track_id, pair.mic_track_id)
        ],
    ]
    for index, track_id in enumerate(extraction_order, start=1):
        unit = extract_unit_gradient(
            model,
            track_windows[track_id],
            modules,
            track_id=track_id,
            batch_size=gradient_batch_size,
            device=gradient_device,
        )
        unit_by_track[track_id] = unit
        print(
            json.dumps(
                {
                    "stage": "extract_unit_gradient",
                    "unit": index,
                    "units": len(extraction_order),
                    **unit.metadata(),
                }
            ),
            flush=True,
        )
    assert_parameters_unchanged(model, parameter_snapshot)

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
    module_grams = {
        module_name: module_gram_matrices(
            guitarset_units,
            direct_units,
            mic_units,
            module_name,
        )
        for module_name in MODULE_PREFIXES
    }
    module_grams["all"] = sum_gram_matrices(
        list(module_grams.values())
    )

    rng = np.random.default_rng(AUDIT_SEED)
    guitarset_bootstrap_counts = rng.multinomial(
        len(guitarset_units),
        np.full(len(guitarset_units), 1 / len(guitarset_units)),
        size=BOOTSTRAP_REPLICATES,
    )
    p3_bootstrap_counts = rng.multinomial(
        len(selection.p3_pairs),
        np.full(
            len(selection.p3_pairs),
            1 / len(selection.p3_pairs),
        ),
        size=BOOTSTRAP_REPLICATES,
    )
    pair_labels = tuple(
        pair.source_label for pair in selection.p3_pairs
    )
    comparisons = {
        module_name: comparison_report(
            grams,
            selection.guitarset_track_ids,
            selection.panels,
            pair_labels,
            guitarset_bootstrap_counts,
            p3_bootstrap_counts,
        )
        for module_name, grams in module_grams.items()
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "candidate-sequence-gradient-conflict-v1",
        "passed": True,
        "training_only": True,
        "optimizer_created": False,
        "parameter_mutation_detected": False,
        "governed_validation_result_read": False,
        "cross_source_validation_result_read": False,
        "inputs": input_contract,
        "selection": selection_report(selection),
        "model": {
            "initialization": "candidate-sequence-string-v1",
            "mode": "eval",
            "gradient_device": str(gradient_device),
            "precompute_device": str(precompute_device),
            "module_partition": module_report,
        },
        "objective": {
            "kind": "per_candidate_set_valued_cross_entropy",
            "sequence_length": model.config.sequence_length,
            "sequence_stride": model.config.sequence_stride,
            "candidate_count_normalized_per_track": True,
            "all_track_windows_once": True,
        },
        "bootstrap": {
            "seed": AUDIT_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "percentile-95",
            "guitarset_units": len(guitarset_units),
            "p3_paired_units": len(selection.p3_pairs),
            "modality_resampling": "shared-paired-performance-counts",
        },
        "precompute": {
            "feature_cache": feature_cache_report,
            "groups": len(groups),
            "windows": sum(len(value) for value in track_windows.values()),
        },
        "units": {
            "guitarset": [
                unit.metadata() for unit in guitarset_units
            ],
            "p3": [
                {
                    "source_label": pair.source_label,
                    "direct": unit_by_track[
                        pair.direct_track_id
                    ].metadata(),
                    "mic": unit_by_track[pair.mic_track_id].metadata(),
                }
                for pair in selection.p3_pairs
            ],
        },
        "comparisons": comparisons,
    }
    output_path = (
        root / "reports/candidate-sequence-gradient-conflict-v1.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "gradient_conflict_audit_complete",
                "output": output_path.name,
                "sha256": sha256_file(output_path),
            }
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen string-v1 GuitarSet and Guitar-TECHS P3 "
            "gradients without updating model parameters."
        )
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--audit-plan", type=Path, required=True)
    parser.add_argument("--precompute-device", default="auto")
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--gradient-batch-size", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.precompute_batch_size <= 0:
        parser.error("--precompute-batch-size must be positive")
    if arguments.gradient_batch_size <= 0:
        parser.error("--gradient-batch-size must be positive")
    run_audit(
        arguments.training_root,
        arguments.audit_plan,
        precompute_device_name=arguments.precompute_device,
        precompute_batch_size=arguments.precompute_batch_size,
        gradient_batch_size=arguments.gradient_batch_size,
    )


if __name__ == "__main__":
    main()
