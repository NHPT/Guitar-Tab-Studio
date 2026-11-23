from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from .analyze_candidate_gradient_conflict import (
    MODULE_PREFIXES,
    UnitGradient,
    assert_parameters_unchanged,
    bootstrap_cosines_from_grams,
    candidate_cache_identity,
    classify_interval,
    cosine_from_grams,
    distribution_summary,
    extract_unit_gradient,
    identity_sha256,
    module_parameter_report,
    parameter_modules,
    parameters_snapshot,
    require_mapping_equal,
    sum_gram_matrices,
    windows_by_track,
)
from .candidate_sequence import (
    CandidateSequenceGroup,
    CandidateSequenceStringModel,
    load_candidate_sequence_model,
    precompute_candidate_sequence_groups,
)
from .data_sources import sha256_file
from .onset_position import OnsetPositionDataset
from .schema import TrackAnnotation, load_manifest
from .train import choose_device
from .train_candidate_sequence import load_position_model
from .train_candidate_sequence_v3 import (
    EXPECTED_TRAINABLE_PARAMETERS,
    FRONTEND_FRAME_THRESHOLD,
    FRONTEND_MINIMUM_NOTE_LENGTH,
    FRONTEND_ONSET_THRESHOLD,
    REPEAT_INTERVAL,
    select_v3_tracks,
    verify_artifact_hashes,
    verify_file_hash,
)


AUDIT_PLAN_SHA256 = (
    "345932175548254502be20049ec222149901a136508ef0685b0cf833ab157866"
)
SOURCE_LEDGER_VERSION = "2026-11-06.2"
SOURCE_LEDGER_SHA256 = (
    "de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a"
)
AUDIT_SEED = 20261124
BOOTSTRAP_REPLICATES = 10_000

FROZEN_ARTIFACTS: dict[str, tuple[str, str]] = {
    "formal_manifest": (
        "governed-v1/formal-training-manifest-v3.jsonl",
        "cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d",
    ),
    "v5_input_audit": (
        "reports/candidate-sequence-string-v5-input-audit.json",
        "e086978cef0cabd4783f34ac3d21d94b364f52b0a237ae203cc456800d15ed77",
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

EXPECTED_TRACK_IDENTITIES = {
    "all": "b36882afc7c5a8e407aea1f467aac29cd5ea7d46682f31f8a8e0302e4e653639",
    "gradient": (
        "e1839751cee5f6e57149d9c93871211b833a9d714bac649df0d4113e00258709"
    ),
    "calibration": (
        "f5ad4b495801fca64848029e209ce7bad5a056aaccef44e5e088b067ea231335"
    ),
}
EXPECTED_FAMILY_IDENTITY = (
    "c9c1a1535103a523e3fdd9127108e04e160082bf626c549d6af95a43057999bd"
)
EXPECTED_FAMILY_ALLOCATIONS = {
    "both_gradient": 150,
    "original_calibration_derived_gradient": 49,
    "original_gradient_derived_calibration": 35,
    "both_calibration": 6,
}
EXPECTED_GRADIENT_CACHE = {
    "audio_files": 384,
    "events": 113710,
    "dropped": 823,
    "cache_set_sha256": (
        "4aedd3c1e181b7d2470e42d8a5f1c748e3ad629536d1df3f648c0e6111b1dfb5"
    ),
}
EXPECTED_CALIBRATION_CACHE = {
    "audio_files": 96,
    "events": 27445,
    "dropped": 230,
    "cache_set_sha256": (
        "57c7b74f47cb7f21fd9bffe192bbd25ccb1fb4145c1ad868eb567a6a10efa70c"
    ),
}

MODULE_NAMES = (*MODULE_PREFIXES, "all")


@dataclass(frozen=True)
class GuitarSetPartition:
    gradient_track_ids: tuple[str, ...]
    calibration_track_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    family_by_track: dict[str, str]
    player_by_track: dict[str, str]
    variant_by_track: dict[str, str]
    family_allocations: dict[str, int]


def recording_family_root(annotation: TrackAnnotation) -> str:
    derived_from = annotation.provenance.get("derived_from_track_id")
    if derived_from is None:
        return annotation.track_id
    if not isinstance(derived_from, str) or not derived_from:
        raise ValueError(
            f"invalid derived family root for {annotation.track_id}"
        )
    return derived_from


def _variant(annotation: TrackAnnotation) -> str:
    return (
        "derived"
        if annotation.provenance.get("derived_from_track_id") is not None
        else "original"
    )


def _allocation_name(
    members: list[TrackAnnotation],
    calibration_ids: set[str],
) -> str:
    by_variant = {_variant(annotation): annotation for annotation in members}
    original_calibration = (
        by_variant["original"].track_id in calibration_ids
    )
    derived_calibration = by_variant["derived"].track_id in calibration_ids
    if not original_calibration and not derived_calibration:
        return "both_gradient"
    if original_calibration and not derived_calibration:
        return "original_calibration_derived_gradient"
    if not original_calibration and derived_calibration:
        return "original_gradient_derived_calibration"
    return "both_calibration"


def select_guitarset_partition(
    annotations: list[TrackAnnotation],
) -> GuitarSetPartition:
    selection = select_v3_tracks(annotations)
    guitarset = [
        annotation
        for annotation in annotations
        if annotation.provenance.get("dataset") == "GuitarSet"
    ]
    gradient_ids = tuple(
        sorted(
            annotation.track_id
            for annotation in guitarset
            if annotation.track_id in selection.gradient_track_ids
        )
    )
    calibration_ids = tuple(
        sorted(
            annotation.track_id
            for annotation in guitarset
            if annotation.track_id in selection.calibration_track_ids
        )
    )
    all_ids = tuple(sorted((*gradient_ids, *calibration_ids)))
    if len(gradient_ids) != 384 or len(calibration_ids) != 96:
        raise ValueError("GuitarSet partition must contain 384/96 tracks")
    if set(gradient_ids) & set(calibration_ids):
        raise ValueError("GuitarSet partition overlaps")
    for name, track_ids in (
        ("all", all_ids),
        ("gradient", gradient_ids),
        ("calibration", calibration_ids),
    ):
        if identity_sha256(track_ids) != EXPECTED_TRACK_IDENTITIES[name]:
            raise ValueError(f"frozen {name} track identity changed")

    by_family: dict[str, list[TrackAnnotation]] = {}
    for annotation in guitarset:
        by_family.setdefault(
            recording_family_root(annotation),
            [],
        ).append(annotation)
    family_ids = tuple(sorted(by_family))
    if len(family_ids) != 240:
        raise ValueError("GuitarSet partition must contain 240 families")
    if identity_sha256(family_ids) != EXPECTED_FAMILY_IDENTITY:
        raise ValueError("frozen GuitarSet family identity changed")

    family_by_track: dict[str, str] = {}
    player_by_track: dict[str, str] = {}
    variant_by_track: dict[str, str] = {}
    calibration_set = set(calibration_ids)
    allocations: Counter[str] = Counter()
    for family_id, members in sorted(by_family.items()):
        if len(members) != 2:
            raise ValueError(
                f"GuitarSet family {family_id} does not contain two tracks"
            )
        variants = {_variant(annotation) for annotation in members}
        if variants != {"original", "derived"}:
            raise ValueError(
                f"GuitarSet family {family_id} variants are incomplete"
            )
        if {
            str(annotation.provenance.get("annotation_sha256", ""))
            for annotation in members
        }.__len__() != 1:
            raise ValueError(
                f"GuitarSet family {family_id} annotations differ"
            )
        if len({annotation.group_id for annotation in members}) != 1:
            raise ValueError(
                f"GuitarSet family {family_id} crosses player groups"
            )
        allocations[_allocation_name(members, calibration_set)] += 1
        for annotation in members:
            family_by_track[annotation.track_id] = family_id
            player_by_track[annotation.track_id] = annotation.group_id
            variant_by_track[annotation.track_id] = _variant(annotation)
    allocation_report = dict(sorted(allocations.items()))
    if allocation_report != EXPECTED_FAMILY_ALLOCATIONS:
        raise ValueError("frozen GuitarSet family allocation changed")
    return GuitarSetPartition(
        gradient_track_ids=gradient_ids,
        calibration_track_ids=calibration_ids,
        family_ids=family_ids,
        family_by_track=family_by_track,
        player_by_track=player_by_track,
        variant_by_track=variant_by_track,
        family_allocations=allocation_report,
    )


def verify_audit_plan(path: Path) -> dict[str, str]:
    return verify_file_hash(
        path,
        AUDIT_PLAN_SHA256,
        name="GuitarSet partition gradient audit plan",
    )


def build_input_contract(
    root: Path,
    plan_path: Path,
) -> tuple[dict[str, object], list[TrackAnnotation], GuitarSetPartition]:
    artifacts = verify_artifact_hashes(root, FROZEN_ARTIFACTS)
    plan = verify_audit_plan(plan_path)
    ledger_path = Path(__file__).with_name("source_ledger.json")
    ledger_report = verify_file_hash(
        ledger_path,
        SOURCE_LEDGER_SHA256,
        name="source ledger",
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("ledger_version") != SOURCE_LEDGER_VERSION:
        raise ValueError("frozen source ledger version changed")

    v5_audit_path = root / FROZEN_ARTIFACTS["v5_input_audit"][0]
    v5_audit = json.loads(v5_audit_path.read_text(encoding="utf-8"))
    expected_flags = {
        "kind": "candidate-sequence-string-v5-input-audit",
        "passed": True,
        "training_only": True,
        "training_started": False,
        "governed_validation_read": False,
        "cross_source_validation_read": False,
    }
    for name, expected in expected_flags.items():
        if v5_audit.get(name) != expected:
            raise ValueError(f"v5 input audit flag changed: {name}")
    expected_sampling = {
        "expected_p3_draws": 0,
        "expected_p3_share": 0,
        "p3_weight": 0,
        "replacement": True,
    }
    sampling = v5_audit.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("v5 input audit sampling is missing")
    for name, expected in expected_sampling.items():
        if sampling.get(name) != expected:
            raise ValueError(f"v5 sampling field changed: {name}")

    manifest_path = root / FROZEN_ARTIFACTS["formal_manifest"][0]
    annotations = load_manifest(manifest_path, split="train")
    partition = select_guitarset_partition(annotations)
    by_id = {annotation.track_id: annotation for annotation in annotations}
    cache_directory = root / "governed-v1/.basic-pitch-cache"
    gradient_cache = candidate_cache_identity(
        [by_id[track_id] for track_id in partition.gradient_track_ids],
        manifest_path,
        cache_directory,
    )
    calibration_cache = candidate_cache_identity(
        [by_id[track_id] for track_id in partition.calibration_track_ids],
        manifest_path,
        cache_directory,
    )
    require_mapping_equal(
        "GuitarSet gradient candidate cache",
        gradient_cache,
        EXPECTED_GRADIENT_CACHE,
    )
    require_mapping_equal(
        "GuitarSet calibration candidate cache",
        calibration_cache,
        EXPECTED_CALIBRATION_CACHE,
    )
    return (
        {
            "audit_plan": plan,
            "artifacts": artifacts,
            "source_ledger": {
                **ledger_report,
                "version": SOURCE_LEDGER_VERSION,
            },
            "v5_input_audit_flags": expected_flags,
            "v5_sampling": {
                name: sampling[name] for name in expected_sampling
            },
            "candidate_caches": {
                "gradient": gradient_cache,
                "calibration": calibration_cache,
            },
        },
        annotations,
        partition,
    )


def _compact_unit(unit: UnitGradient) -> UnitGradient:
    unit.vectors = {
        name: vector.to(dtype=torch.float32)
        for name, vector in unit.vectors.items()
    }
    return unit


def unweighted_windows(
    windows: list[list[CandidateSequenceGroup]],
) -> list[list[CandidateSequenceGroup]]:
    return [
        [
            replace(
                group,
                target_weights=torch.ones_like(group.target_weights),
            )
            for group in window
        ]
        for window in windows
    ]


def extract_partition_units(
    model: CandidateSequenceStringModel,
    track_windows: dict[str, list[list[CandidateSequenceGroup]]],
    track_ids: tuple[str, ...],
    modules: dict[str, list[tuple[str, torch.nn.Parameter]]],
    *,
    source: str,
    include_unweighted: bool,
    gradient_batch_size: int,
    device: torch.device,
) -> tuple[list[UnitGradient], list[UnitGradient] | None]:
    operational: list[UnitGradient] = []
    common_unweighted: list[UnitGradient] | None = (
        [] if include_unweighted else None
    )
    for index, track_id in enumerate(track_ids, start=1):
        windows = track_windows.get(track_id)
        if not windows:
            raise ValueError(f"selected track has no windows: {track_id}")
        unit = _compact_unit(
            extract_unit_gradient(
                model,
                windows,
                modules,
                track_id=track_id,
                batch_size=gradient_batch_size,
                device=device,
            )
        )
        operational.append(unit)
        if common_unweighted is not None:
            common_unweighted.append(
                _compact_unit(
                    extract_unit_gradient(
                        model,
                        unweighted_windows(windows),
                        modules,
                        track_id=track_id,
                        batch_size=gradient_batch_size,
                        device=device,
                    )
                )
            )
        print(
            json.dumps(
                {
                    "stage": "extract_partition_gradient",
                    "source": source,
                    "unit": index,
                    "units": len(track_ids),
                    "objectives": (
                        ["operational", "common_unweighted"]
                        if include_unweighted
                        else ["operational"]
                    ),
                    **unit.metadata(),
                }
            ),
            flush=True,
        )
    return operational, common_unweighted


def _unit_matrix(
    units: list[UnitGradient],
    module_name: str,
) -> torch.Tensor:
    if not units:
        raise ValueError("gradient unit matrix cannot be empty")
    return torch.stack(
        [unit.vectors[module_name] for unit in units]
    ).to(dtype=torch.float64)


def partition_gram_matrices(
    gradient: list[UnitGradient],
    calibration: list[UnitGradient],
    module_name: str,
) -> dict[str, np.ndarray]:
    gradient_matrix = _unit_matrix(gradient, module_name)
    calibration_matrix = _unit_matrix(calibration, module_name)
    result = {
        "gradient": (gradient_matrix @ gradient_matrix.T).numpy(),
        "calibration": (
            calibration_matrix @ calibration_matrix.T
        ).numpy(),
        "gradient_calibration": (
            gradient_matrix @ calibration_matrix.T
        ).numpy(),
    }
    del gradient_matrix
    del calibration_matrix
    return result


def objective_gram_matrices(
    gradient: list[UnitGradient],
    calibration: list[UnitGradient],
) -> dict[str, dict[str, np.ndarray]]:
    result = {
        module_name: partition_gram_matrices(
            gradient,
            calibration,
            module_name,
        )
        for module_name in MODULE_PREFIXES
    }
    result["all"] = sum_gram_matrices(list(result.values()))
    return result


def expand_family_bootstrap_counts(
    track_ids: tuple[str, ...],
    family_by_track: dict[str, str],
    family_ids: tuple[str, ...],
    family_counts: np.ndarray,
) -> np.ndarray:
    family_counts = np.asarray(family_counts)
    if family_counts.ndim != 2:
        raise ValueError("family bootstrap counts must be two-dimensional")
    if family_counts.shape[1] != len(family_ids):
        raise ValueError("family bootstrap counts do not align")
    family_index = {
        family_id: index for index, family_id in enumerate(family_ids)
    }
    try:
        indexes = [
            family_index[family_by_track[track_id]]
            for track_id in track_ids
        ]
    except KeyError as error:
        raise ValueError("track is missing a frozen family") from error
    return family_counts[:, indexes]


def _mask_counts(
    track_ids: tuple[str, ...],
    predicate: Callable[[str], bool],
) -> np.ndarray:
    counts = np.asarray(
        [float(predicate(track_id)) for track_id in track_ids],
        dtype=np.float64,
    )
    if counts.sum() <= 0:
        raise ValueError("comparison subset is empty")
    return counts


def _comparison_slice(
    grams: dict[str, np.ndarray],
    gradient_counts: np.ndarray,
    calibration_counts: np.ndarray,
    gradient_bootstrap_counts: np.ndarray | None = None,
    calibration_bootstrap_counts: np.ndarray | None = None,
) -> dict[str, object]:
    cosine, gradient_norm, calibration_norm = cosine_from_grams(
        grams["gradient"],
        grams["calibration"],
        grams["gradient_calibration"],
        gradient_counts,
        calibration_counts,
    )
    result: dict[str, object] = {
        "cosine": cosine,
        "gradient_mean_norm": gradient_norm,
        "calibration_mean_norm": calibration_norm,
        "calibration_to_gradient_norm_ratio": (
            calibration_norm / gradient_norm
        ),
    }
    if (
        gradient_bootstrap_counts is not None
        and calibration_bootstrap_counts is not None
    ):
        bootstrap = bootstrap_cosines_from_grams(
            grams["gradient"],
            grams["calibration"],
            grams["gradient_calibration"],
            gradient_bootstrap_counts,
            calibration_bootstrap_counts,
        )
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        result.update(
            {
                "bootstrap_95_ci": [float(lower), float(upper)],
                "classification": classify_interval(
                    float(lower),
                    float(upper),
                ),
            }
        )
    return result


def split_family_report(
    grams: dict[str, np.ndarray],
    partition: GuitarSetPartition,
) -> dict[str, object]:
    gradient_index = {
        track_id: index
        for index, track_id in enumerate(partition.gradient_track_ids)
    }
    calibration_index = {
        track_id: index
        for index, track_id in enumerate(partition.calibration_track_ids)
    }
    gradient_by_family: dict[str, str] = {
        partition.family_by_track[track_id]: track_id
        for track_id in partition.gradient_track_ids
        if sum(
            partition.family_by_track[candidate]
            == partition.family_by_track[track_id]
            for candidate in partition.gradient_track_ids
        )
        == 1
    }
    calibration_by_family = {
        partition.family_by_track[track_id]: track_id
        for track_id in partition.calibration_track_ids
        if sum(
            partition.family_by_track[candidate]
            == partition.family_by_track[track_id]
            for candidate in partition.calibration_track_ids
        )
        == 1
    }
    split_families = tuple(
        sorted(set(gradient_by_family) & set(calibration_by_family))
    )
    if len(split_families) != 84:
        raise ValueError("expected 84 split GuitarSet families")
    gradient_diagonal = np.diag(grams["gradient"])
    calibration_diagonal = np.diag(grams["calibration"])
    values: list[float] = []
    by_orientation: dict[str, list[float]] = {
        "original_gradient_derived_calibration": [],
        "original_calibration_derived_gradient": [],
    }
    for family_id in split_families:
        gradient_track = gradient_by_family[family_id]
        calibration_track = calibration_by_family[family_id]
        gradient_row = gradient_index[gradient_track]
        calibration_column = calibration_index[calibration_track]
        denominator = np.sqrt(
            gradient_diagonal[gradient_row]
            * calibration_diagonal[calibration_column]
        )
        if denominator <= 0:
            raise ValueError("split family has a zero-norm gradient")
        cosine = float(
            grams["gradient_calibration"][
                gradient_row,
                calibration_column,
            ]
            / denominator
        )
        values.append(cosine)
        orientation = (
            "original_gradient_derived_calibration"
            if partition.variant_by_track[gradient_track] == "original"
            else "original_calibration_derived_gradient"
        )
        by_orientation[orientation].append(cosine)
    return {
        "families": len(split_families),
        "all": distribution_summary(np.asarray(values)),
        "by_orientation": {
            name: distribution_summary(np.asarray(local_values))
            for name, local_values in by_orientation.items()
        },
    }


def partition_comparison_report(
    grams: dict[str, np.ndarray],
    partition: GuitarSetPartition,
    gradient_units: list[UnitGradient],
    calibration_units: list[UnitGradient],
    gradient_bootstrap_counts: np.ndarray,
    calibration_bootstrap_counts: np.ndarray,
) -> dict[str, object]:
    gradient_ones = np.ones(
        len(partition.gradient_track_ids),
        dtype=np.float64,
    )
    calibration_ones = np.ones(
        len(partition.calibration_track_ids),
        dtype=np.float64,
    )
    primary = _comparison_slice(
        grams,
        gradient_ones,
        calibration_ones,
        gradient_bootstrap_counts,
        calibration_bootstrap_counts,
    )
    primary["gradient_unit_norms"] = distribution_summary(
        np.sqrt(np.diag(grams["gradient"]))
    )
    primary["calibration_unit_norms"] = distribution_summary(
        np.sqrt(np.diag(grams["calibration"]))
    )

    gradient_exposures = np.asarray(
        [unit.active_candidate_exposures for unit in gradient_units],
        dtype=np.float64,
    )
    calibration_exposures = np.asarray(
        [unit.active_candidate_exposures for unit in calibration_units],
        dtype=np.float64,
    )
    exposure_weighted = _comparison_slice(
        grams,
        gradient_exposures,
        calibration_exposures,
    )
    exposure_weighted["gradient_exposures"] = int(
        gradient_exposures.sum()
    )
    exposure_weighted["calibration_exposures"] = int(
        calibration_exposures.sum()
    )

    players = sorted(set(partition.player_by_track.values()))
    player_report = {
        player: _comparison_slice(
            grams,
            _mask_counts(
                partition.gradient_track_ids,
                lambda track_id, player=player: (
                    partition.player_by_track[track_id] == player
                ),
            ),
            _mask_counts(
                partition.calibration_track_ids,
                lambda track_id, player=player: (
                    partition.player_by_track[track_id] == player
                ),
            ),
        )
        for player in players
    }

    variants: dict[str, object] = {}
    for variant in ("original", "derived"):
        gradient_mask = _mask_counts(
            partition.gradient_track_ids,
            lambda track_id, variant=variant: (
                partition.variant_by_track[track_id] == variant
            ),
        )
        calibration_mask = _mask_counts(
            partition.calibration_track_ids,
            lambda track_id, variant=variant: (
                partition.variant_by_track[track_id] == variant
            ),
        )
        variants[variant] = _comparison_slice(
            grams,
            gradient_mask,
            calibration_mask,
            gradient_bootstrap_counts * gradient_mask,
            calibration_bootstrap_counts * calibration_mask,
        )
    return {
        "equal_track_mean": primary,
        "active_candidate_exposure_weighted": exposure_weighted,
        "players": player_report,
        "variants": variants,
        "split_families": split_family_report(grams, partition),
    }


def objective_agreement_report(
    operational: list[UnitGradient],
    common_unweighted: list[UnitGradient],
    module_name: str,
) -> dict[str, float]:
    if len(operational) != len(common_unweighted):
        raise ValueError("objective agreement units do not align")
    if [unit.track_id for unit in operational] != [
        unit.track_id for unit in common_unweighted
    ]:
        raise ValueError("objective agreement track order changed")
    if module_name == "all":
        operational_squared = 0.0
        unweighted_squared = 0.0
        dot = 0.0
        for local_module in MODULE_PREFIXES:
            local = objective_agreement_report(
                operational,
                common_unweighted,
                local_module,
            )
            operational_squared += local["operational_mean_norm"] ** 2
            unweighted_squared += (
                local["common_unweighted_mean_norm"] ** 2
            )
            dot += local["dot"]
        operational_norm = float(np.sqrt(operational_squared))
        unweighted_norm = float(np.sqrt(unweighted_squared))
    else:
        operational_mean = _unit_matrix(
            operational,
            module_name,
        ).mean(dim=0)
        unweighted_mean = _unit_matrix(
            common_unweighted,
            module_name,
        ).mean(dim=0)
        dot = float(torch.dot(operational_mean, unweighted_mean))
        operational_norm = float(torch.linalg.vector_norm(operational_mean))
        unweighted_norm = float(torch.linalg.vector_norm(unweighted_mean))
    if operational_norm <= 0 or unweighted_norm <= 0:
        raise ValueError("objective agreement mean has zero norm")
    return {
        "cosine": dot / (operational_norm * unweighted_norm),
        "operational_mean_norm": operational_norm,
        "common_unweighted_mean_norm": unweighted_norm,
        "common_to_operational_norm_ratio": (
            unweighted_norm / operational_norm
        ),
        "dot": dot,
    }


def partition_report(
    partition: GuitarSetPartition,
) -> dict[str, object]:
    def source_report(track_ids: tuple[str, ...]) -> dict[str, object]:
        return {
            "track_count": len(track_ids),
            "track_identity_sha256": identity_sha256(track_ids),
            "original_tracks": sum(
                partition.variant_by_track[track_id] == "original"
                for track_id in track_ids
            ),
            "derived_tracks": sum(
                partition.variant_by_track[track_id] == "derived"
                for track_id in track_ids
            ),
            "players": dict(
                sorted(
                    Counter(
                        partition.player_by_track[track_id]
                        for track_id in track_ids
                    ).items()
                )
            ),
            "tracks": list(track_ids),
        }

    return {
        "fraction": 0.2,
        "seed": 20261122,
        "gradient": source_report(partition.gradient_track_ids),
        "calibration": source_report(partition.calibration_track_ids),
        "families": {
            "count": len(partition.family_ids),
            "identity_sha256": identity_sha256(partition.family_ids),
            "allocations": partition.family_allocations,
        },
    }


def _build_dataset(
    root: Path,
    manifest_path: Path,
    position_config: object,
    track_ids: tuple[str, ...],
    *,
    operational_weights: bool,
) -> OnsetPositionDataset:
    weight_arguments = (
        {
            "candidate_ambiguity_weight": 1.25,
            "candidate_rare_pair_max_weight": 1.5,
            "candidate_cross_string_weight": 2.0,
            "candidate_cross_string_interval": 2.0,
            "candidate_max_supervision_weight": 4.0,
        }
        if operational_weights
        else {}
    )
    return OnsetPositionDataset(
        manifest_path,
        split="train",
        config=position_config,
        feature_cache_directory=root / "governed-v1/.feature-cache",
        candidate_cache_directory=root / "governed-v1/.basic-pitch-cache",
        candidate_onset_threshold=FRONTEND_ONSET_THRESHOLD,
        candidate_frame_threshold=FRONTEND_FRAME_THRESHOLD,
        candidate_minimum_note_length=FRONTEND_MINIMUM_NOTE_LENGTH,
        included_track_ids=set(track_ids),
        **weight_arguments,
    )


def _prepare_partition_windows(
    dataset: OnsetPositionDataset,
    position_model: torch.nn.Module,
    precompute_device: torch.device,
    sequence_model: CandidateSequenceStringModel,
    *,
    precompute_batch_size: int,
    source: str,
) -> tuple[
    dict[str, list[list[CandidateSequenceGroup]]],
    dict[str, object],
]:
    feature_cache = dataset.prepare_feature_cache()
    print(
        json.dumps(
            {
                "stage": "precompute_partition_groups",
                "source": source,
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
        sequence_model.config,
    )
    report = {
        "tracks": len(dataset.annotations),
        "groups": len(groups),
        "windows": sum(len(windows) for windows in track_windows.values()),
        "feature_cache": feature_cache,
        "candidate_supervision": dataset.candidate_supervision,
    }
    return track_windows, report


def _metadata(units: Iterable[UnitGradient]) -> list[dict[str, object]]:
    return [unit.metadata() for unit in units]


def run_audit(
    training_root: Path,
    plan_path: Path,
    *,
    precompute_device_name: str,
    precompute_batch_size: int,
    gradient_batch_size: int,
) -> dict[str, object]:
    root = training_root.resolve()
    input_contract, _annotations, partition = build_input_contract(
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
    if module_report["all"]["parameters"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("sequence parameter count changed")
    snapshot = parameters_snapshot(model)
    manifest_path = root / FROZEN_ARTIFACTS["formal_manifest"][0]

    gradient_dataset = _build_dataset(
        root,
        manifest_path,
        position_config,
        partition.gradient_track_ids,
        operational_weights=True,
    )
    if tuple(
        sorted(annotation.track_id for annotation in gradient_dataset.annotations)
    ) != partition.gradient_track_ids:
        raise ValueError("gradient dataset track identity changed")
    gradient_windows, gradient_precompute = _prepare_partition_windows(
        gradient_dataset,
        position_model,
        precompute_device,
        model,
        precompute_batch_size=precompute_batch_size,
        source="gradient",
    )
    operational_gradient, unweighted_gradient = extract_partition_units(
        model,
        gradient_windows,
        partition.gradient_track_ids,
        modules,
        source="gradient",
        include_unweighted=True,
        gradient_batch_size=gradient_batch_size,
        device=gradient_device,
    )
    if unweighted_gradient is None:
        raise RuntimeError("unweighted gradient extraction was skipped")
    del gradient_windows
    del gradient_dataset
    gc.collect()

    calibration_dataset = _build_dataset(
        root,
        manifest_path,
        position_config,
        partition.calibration_track_ids,
        operational_weights=False,
    )
    if tuple(
        sorted(
            annotation.track_id
            for annotation in calibration_dataset.annotations
        )
    ) != partition.calibration_track_ids:
        raise ValueError("calibration dataset track identity changed")
    calibration_windows, calibration_precompute = (
        _prepare_partition_windows(
            calibration_dataset,
            position_model,
            precompute_device,
            model,
            precompute_batch_size=precompute_batch_size,
            source="calibration",
        )
    )
    calibration_units, skipped = extract_partition_units(
        model,
        calibration_windows,
        partition.calibration_track_ids,
        modules,
        source="calibration",
        include_unweighted=False,
        gradient_batch_size=gradient_batch_size,
        device=gradient_device,
    )
    if skipped is not None:
        raise RuntimeError("calibration extraction returned extra units")
    del calibration_windows
    del calibration_dataset
    del position_model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    assert_parameters_unchanged(model, snapshot)

    rng = np.random.default_rng(AUDIT_SEED)
    family_counts = rng.multinomial(
        len(partition.family_ids),
        np.full(
            len(partition.family_ids),
            1 / len(partition.family_ids),
        ),
        size=BOOTSTRAP_REPLICATES,
    )
    gradient_bootstrap_counts = expand_family_bootstrap_counts(
        partition.gradient_track_ids,
        partition.family_by_track,
        partition.family_ids,
        family_counts,
    )
    calibration_bootstrap_counts = expand_family_bootstrap_counts(
        partition.calibration_track_ids,
        partition.family_by_track,
        partition.family_ids,
        family_counts,
    )

    operational_grams = objective_gram_matrices(
        operational_gradient,
        calibration_units,
    )
    common_unweighted_grams = objective_gram_matrices(
        unweighted_gradient,
        calibration_units,
    )
    comparisons = {
        "operational": {
            module_name: partition_comparison_report(
                operational_grams[module_name],
                partition,
                operational_gradient,
                calibration_units,
                gradient_bootstrap_counts,
                calibration_bootstrap_counts,
            )
            for module_name in MODULE_NAMES
        },
        "common_unweighted": {
            module_name: partition_comparison_report(
                common_unweighted_grams[module_name],
                partition,
                unweighted_gradient,
                calibration_units,
                gradient_bootstrap_counts,
                calibration_bootstrap_counts,
            )
            for module_name in MODULE_NAMES
        },
    }
    objective_agreement = {
        module_name: objective_agreement_report(
            operational_gradient,
            unweighted_gradient,
            module_name,
        )
        for module_name in MODULE_NAMES
    }
    for value in objective_agreement.values():
        value.pop("dot")

    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "candidate-sequence-guitarset-gradient-conflict-v1",
        "passed": True,
        "training_only": True,
        "optimizer_created": False,
        "parameter_mutation_detected": False,
        "governed_validation_result_read": False,
        "cross_source_validation_result_read": False,
        "inputs": input_contract,
        "partition": partition_report(partition),
        "model": {
            "initialization": "candidate-sequence-string-v1",
            "mode": "eval",
            "gradient_device": str(gradient_device),
            "precompute_device": str(precompute_device),
            "module_partition": module_report,
        },
        "objectives": {
            "operational": {
                "gradient": {
                    "ambiguity_weight": 1.25,
                    "rare_pair_max_weight": 1.5,
                    "cross_string_weight": 2.0,
                    "cross_string_interval": 2.0,
                    "maximum_supervision_weight": 4.0,
                    "weight_context": "384-source-local-GuitarSet",
                },
                "calibration": {
                    "target_weights": "unit",
                    "weight_context": "96-source-local-GuitarSet",
                },
            },
            "common_unweighted": {
                "gradient_target_weights": "unit",
                "calibration_target_weights": "unit",
            },
            "sequence_length": model.config.sequence_length,
            "sequence_stride": model.config.sequence_stride,
            "candidate_count_normalized_per_track": True,
            "all_track_windows_once": True,
        },
        "bootstrap": {
            "seed": AUDIT_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "percentile-95",
            "family_units": len(partition.family_ids),
            "resampling": "shared-recording-family-counts",
        },
        "precompute": {
            "gradient": gradient_precompute,
            "calibration": calibration_precompute,
        },
        "units": {
            "gradient_operational": _metadata(operational_gradient),
            "gradient_common_unweighted": _metadata(unweighted_gradient),
            "calibration": _metadata(calibration_units),
        },
        "comparisons": comparisons,
        "gradient_objective_agreement": objective_agreement,
    }
    output_path = (
        root
        / "reports/candidate-sequence-guitarset-gradient-conflict-v1.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "guitarset_partition_gradient_audit_complete",
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
            "Compare frozen string-v1 gradients for the 384/96 GuitarSet "
            "training partition without updating model parameters."
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
