from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .candidate_margin import (
    calibration_partition_digest,
    stratified_calibration_partition,
)
from .data_sources import manifest_identity, sha256_file
from .model import OPEN_STRING_PITCHES, PITCH_COUNT, PITCH_MIN
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    candidate_harmonic_cqt_indexes,
    candidate_position_queries,
    normalize_candidate_attack_evidence,
)
from .schema import load_manifest
from .train_candidate_sequence import load_position_model


REAL_DATASETS = ("GuitarSet", "AG-PT-set", "Guitar-TECHS")


@dataclass(frozen=True)
class ProbeExample:
    rank: int
    identity: str
    partition: str
    dataset: str
    pitch: int
    string: int
    raw: np.ndarray
    invariant: np.ndarray


def _rank(seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _push_example(
    buckets: dict[
        tuple[str, str, int, int],
        list[tuple[int, str, ProbeExample]],
    ],
    example: ProbeExample,
    *,
    maximum_per_bucket: int,
) -> None:
    key = (
        example.partition,
        example.dataset,
        example.pitch,
        example.string,
    )
    bucket = buckets.setdefault(key, [])
    item = (-example.rank, example.identity, example)
    if len(bucket) < maximum_per_bucket:
        heapq.heappush(bucket, item)
    elif example.rank < -bucket[0][0]:
        heapq.heapreplace(bucket, item)


def _gather_harmonic_evidence(
    features: torch.Tensor,
    query_batch_indexes: torch.Tensor,
    query_pitch_indexes: torch.Tensor,
    harmonic_indexes: torch.Tensor,
    harmonic_valid: torch.Tensor,
) -> torch.Tensor:
    indexes = harmonic_indexes[query_pitch_indexes].flatten(1)
    valid = harmonic_valid[query_pitch_indexes].flatten(1)
    selected_features = features[query_batch_indexes]
    gathered = torch.gather(
        selected_features,
        3,
        indexes[:, None, None, :].expand(
            -1,
            selected_features.shape[1],
            selected_features.shape[2],
            -1,
        ),
    )
    return gathered * valid[:, None, None, :]


def collect_examples(
    dataset: OnsetPositionDataset,
    position_config: OnsetPositionConfig,
    calibration_annotation_indexes: set[int],
    *,
    batch_size: int,
    seed: int,
    maximum_per_bucket: int,
) -> tuple[list[ProbeExample], dict[str, int]]:
    harmonic_indexes, harmonic_valid = candidate_harmonic_cqt_indexes(
        dataset.feature_config,
        harmonic_count=position_config.candidate_spectral_harmonics,
        radius=position_config.candidate_spectral_radius,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    buckets: dict[
        tuple[str, str, int, int],
        list[tuple[int, str, ProbeExample]],
    ] = {}
    counts: Counter[str] = Counter()
    group_offset = 0
    for batch_index, batch in enumerate(loader):
        features = batch["features"]
        pitch_masks = batch["pitch_mask"]
        (
            query_batch_indexes,
            query_pitch_indexes,
            targets,
        ) = candidate_position_queries(
            batch["event_pitches"],
            batch["event_strings"],
            pitch_masks,
            categorical=True,
        )
        target_strings = targets[:, 1:]
        active = targets[:, 0] < 0.5
        single_target = (target_strings > 0).sum(dim=1) == 1
        selected = active & single_target
        counts["queries"] += int(targets.shape[0])
        counts["active_queries"] += int(active.sum())
        counts["multi_target_excluded"] += int(
            (active & ~single_target).sum()
        )
        if bool(selected.any()):
            query_batch_indexes = query_batch_indexes[selected]
            query_pitch_indexes = query_pitch_indexes[selected]
            target_strings = target_strings[selected]
            gathered = _gather_harmonic_evidence(
                features,
                query_batch_indexes,
                query_pitch_indexes,
                harmonic_indexes,
                harmonic_valid,
            )
            raw = gathered.flatten(1)
            invariant = normalize_candidate_attack_evidence(
                raw,
                frame_count=position_config.frame_count,
                frames_before=position_config.frames_before,
                harmonic_count=(
                    position_config.candidate_spectral_harmonics
                ),
                radius=position_config.candidate_spectral_radius,
            )
            for row in range(raw.shape[0]):
                local_group_index = int(query_batch_indexes[row])
                group = dataset.groups[group_offset + local_group_index]
                annotation_index = group.annotation_index
                annotation = dataset.annotations[annotation_index]
                source = str(
                    annotation.provenance.get("dataset", "unknown")
                )
                if source not in REAL_DATASETS:
                    counts["non_real_source_excluded"] += 1
                    continue
                pitch = PITCH_MIN + int(query_pitch_indexes[row])
                string = int(target_strings[row].argmax())
                onset_millis = round(float(group.onset or 0.0) * 1000)
                identity = (
                    f"{source}:{annotation.group_id}:"
                    f"{annotation.track_id}:{onset_millis}:"
                    f"{pitch}:{string}"
                )
                partition = (
                    "calibration"
                    if annotation_index
                    in calibration_annotation_indexes
                    else "train"
                )
                example = ProbeExample(
                    rank=_rank(seed, identity),
                    identity=identity,
                    partition=partition,
                    dataset=source,
                    pitch=pitch,
                    string=string,
                    raw=raw[row].numpy().astype(
                        np.float32,
                        copy=True,
                    ),
                    invariant=invariant[row].numpy().astype(
                        np.float32,
                        copy=True,
                    ),
                )
                _push_example(
                    buckets,
                    example,
                    maximum_per_bucket=maximum_per_bucket,
                )
                counts["eligible_examples"] += 1
        group_offset += features.shape[0]
        if (batch_index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage": "collect",
                        "batches": batch_index + 1,
                        "groups": group_offset,
                        "retained": sum(
                            len(bucket) for bucket in buckets.values()
                        ),
                    }
                ),
                flush=True,
            )
    examples = sorted(
        (
            item[2]
            for bucket in buckets.values()
            for item in bucket
        ),
        key=lambda item: (
            item.partition,
            item.dataset,
            item.rank,
            item.identity,
        ),
    )
    counts["retained_examples"] = len(examples)
    return examples, dict(sorted(counts.items()))


def _arrays(
    examples: list[ProbeExample],
    *,
    partition: str,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        example
        for example in examples
        if example.partition == partition
    ]
    if representation not in ("raw", "invariant"):
        raise ValueError("unknown candidate representation")
    evidence = np.stack(
        [getattr(example, representation) for example in selected]
    )
    datasets = np.asarray(
        [example.dataset for example in selected],
        dtype=object,
    )
    pitches = np.asarray(
        [example.pitch for example in selected],
        dtype=np.int64,
    )
    strings = np.asarray(
        [example.string for example in selected],
        dtype=np.int64,
    )
    return evidence, datasets, pitches, strings


def _source_probe_indexes(
    datasets: np.ndarray,
    pitches: np.ndarray,
    *,
    maximum_per_source_pitch: int,
) -> tuple[np.ndarray, list[int], dict[str, int]]:
    indexes: list[int] = []
    shared_pitches: list[int] = []
    counts: Counter[str] = Counter()
    for pitch in sorted(set(pitches.tolist())):
        by_source = {
            source: np.flatnonzero(
                (datasets == source) & (pitches == pitch)
            )
            for source in REAL_DATASETS
        }
        selected_count = min(
            maximum_per_source_pitch,
            *(len(source_indexes) for source_indexes in by_source.values()),
        )
        if selected_count <= 0:
            continue
        shared_pitches.append(pitch)
        for source, source_indexes in by_source.items():
            indexes.extend(source_indexes[:selected_count].tolist())
            counts[source] += selected_count
    if not indexes:
        raise ValueError("source probe has no shared-pitch examples")
    return (
        np.asarray(indexes, dtype=np.int64),
        shared_pitches,
        dict(sorted(counts.items())),
    )


def _classifier(seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-5,
            random_state=seed,
            average=True,
        ),
    )


def source_probe(
    train_evidence: np.ndarray,
    train_datasets: np.ndarray,
    train_pitches: np.ndarray,
    calibration_evidence: np.ndarray,
    calibration_datasets: np.ndarray,
    calibration_pitches: np.ndarray,
    *,
    seed: int,
    maximum_per_source_pitch: int,
) -> dict[str, object]:
    train_indexes, train_shared, train_counts = _source_probe_indexes(
        train_datasets,
        train_pitches,
        maximum_per_source_pitch=maximum_per_source_pitch,
    )
    calibration_indexes, calibration_shared, calibration_counts = (
        _source_probe_indexes(
            calibration_datasets,
            calibration_pitches,
            maximum_per_source_pitch=maximum_per_source_pitch,
        )
    )
    shared_pitches = sorted(set(train_shared) & set(calibration_shared))
    train_indexes, _train_shared, train_counts = _source_probe_indexes(
        train_datasets[
            np.isin(train_pitches, shared_pitches)
        ],
        train_pitches[
            np.isin(train_pitches, shared_pitches)
        ],
        maximum_per_source_pitch=maximum_per_source_pitch,
    )
    train_base_indexes = np.flatnonzero(
        np.isin(train_pitches, shared_pitches)
    )
    train_indexes = train_base_indexes[train_indexes]
    calibration_indexes, _calibration_shared, calibration_counts = (
        _source_probe_indexes(
            calibration_datasets[
                np.isin(calibration_pitches, shared_pitches)
            ],
            calibration_pitches[
                np.isin(calibration_pitches, shared_pitches)
            ],
            maximum_per_source_pitch=maximum_per_source_pitch,
        )
    )
    calibration_base_indexes = np.flatnonzero(
        np.isin(calibration_pitches, shared_pitches)
    )
    calibration_indexes = calibration_base_indexes[calibration_indexes]

    model = _classifier(seed)
    model.fit(
        train_evidence[train_indexes],
        train_datasets[train_indexes],
    )
    predictions = model.predict(
        calibration_evidence[calibration_indexes]
    )
    targets = calibration_datasets[calibration_indexes]
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro")
        ),
        "shared_pitches": shared_pitches,
        "train_counts": train_counts,
        "calibration_counts": calibration_counts,
    }


def _string_inputs(
    evidence: np.ndarray,
    pitches: np.ndarray,
) -> np.ndarray:
    pitch_one_hot = np.zeros(
        (pitches.shape[0], PITCH_COUNT),
        dtype=np.float32,
    )
    pitch_one_hot[
        np.arange(pitches.shape[0]),
        pitches - PITCH_MIN,
    ] = 1
    return np.concatenate((evidence, pitch_one_hot), axis=1)


def _masked_string_predictions(
    model: object,
    evidence: np.ndarray,
    pitches: np.ndarray,
) -> np.ndarray:
    scores = model.decision_function(
        _string_inputs(evidence, pitches)
    )
    classes = model[-1].classes_
    complete_scores = np.full(
        (evidence.shape[0], len(OPEN_STRING_PITCHES)),
        -np.inf,
        dtype=np.float64,
    )
    complete_scores[:, classes.astype(np.int64)] = scores
    for string_index, open_pitch in enumerate(OPEN_STRING_PITCHES):
        frets = pitches - open_pitch
        complete_scores[
            (frets < 0) | (frets > 24),
            string_index,
        ] = -np.inf
    return complete_scores.argmax(axis=1)


def string_probe(
    train_evidence: np.ndarray,
    train_datasets: np.ndarray,
    train_pitches: np.ndarray,
    train_strings: np.ndarray,
    calibration_evidence: np.ndarray,
    calibration_datasets: np.ndarray,
    calibration_pitches: np.ndarray,
    calibration_strings: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    dataset_counts = Counter(train_datasets.tolist())
    source_count = len(dataset_counts)
    sample_weights = np.asarray(
        [
            len(train_datasets)
            / (source_count * dataset_counts[source])
            for source in train_datasets
        ],
        dtype=np.float64,
    )
    model = _classifier(seed)
    model.fit(
        _string_inputs(train_evidence, train_pitches),
        train_strings,
        sgdclassifier__sample_weight=sample_weights,
    )
    predictions = _masked_string_predictions(
        model,
        calibration_evidence,
        calibration_pitches,
    )
    by_dataset: dict[str, dict[str, float | int]] = {}
    for source in REAL_DATASETS:
        selected = calibration_datasets == source
        correct = int(
            (predictions[selected] == calibration_strings[selected]).sum()
        )
        total = int(selected.sum())
        by_dataset[source] = {
            "accuracy": correct / max(1, total),
            "correct": correct,
            "total": total,
        }
    return {
        "macro_source_accuracy": float(
            np.mean(
                [
                    float(metrics["accuracy"])
                    for metrics in by_dataset.values()
                ]
            )
        ),
        "pooled_accuracy": float(
            (predictions == calibration_strings).mean()
        ),
        "correct": int(
            (predictions == calibration_strings).sum()
        ),
        "total": int(calibration_strings.shape[0]),
        "by_dataset": by_dataset,
    }


def _representation_metrics(
    examples: list[ProbeExample],
    representation: str,
    *,
    seed: int,
    maximum_per_source_pitch: int,
) -> dict[str, object]:
    train = _arrays(
        examples,
        partition="train",
        representation=representation,
    )
    calibration = _arrays(
        examples,
        partition="calibration",
        representation=representation,
    )
    return {
        "source_probe": source_probe(
            train[0],
            train[1],
            train[2],
            calibration[0],
            calibration[1],
            calibration[2],
            seed=seed,
            maximum_per_source_pitch=maximum_per_source_pitch,
        ),
        "string_probe": string_probe(
            train[0],
            train[1],
            train[2],
            train[3],
            calibration[0],
            calibration[1],
            calibration[2],
            calibration[3],
            seed=seed,
        ),
    }


def _gate(
    raw: dict[str, object],
    invariant: dict[str, object],
) -> dict[str, object]:
    raw_source = dict(raw["source_probe"])
    invariant_source = dict(invariant["source_probe"])
    raw_string = dict(raw["string_probe"])
    invariant_string = dict(invariant["string_probe"])
    raw_by_dataset = dict(raw_string["by_dataset"])
    invariant_by_dataset = dict(invariant_string["by_dataset"])

    raw_source_accuracy = float(raw_source["balanced_accuracy"])
    invariant_source_accuracy = float(
        invariant_source["balanced_accuracy"]
    )
    chance = 1 / len(REAL_DATASETS)
    source_absolute_reduction = (
        raw_source_accuracy - invariant_source_accuracy
    )
    raw_excess = max(0.0, raw_source_accuracy - chance)
    invariant_excess = max(
        0.0,
        invariant_source_accuracy - chance,
    )
    source_excess_reduction = (
        1 - invariant_excess / raw_excess
        if raw_excess > 0
        else 0.0
    )
    per_source_deltas = {
        source: (
            float(invariant_by_dataset[source]["accuracy"])
            - float(raw_by_dataset[source]["accuracy"])
        )
        for source in REAL_DATASETS
    }
    cross_source_delta = float(
        np.mean(
            [
                per_source_deltas["AG-PT-set"],
                per_source_deltas["Guitar-TECHS"],
            ]
        )
    )
    checks = {
        "source_balanced_accuracy_reduction_at_least_0_05": (
            source_absolute_reduction >= 0.05
        ),
        "source_excess_over_chance_reduction_at_least_20_percent": (
            source_excess_reduction >= 0.20
        ),
        "macro_string_accuracy_drop_at_most_0_005": (
            float(invariant_string["macro_source_accuracy"])
            >= float(raw_string["macro_source_accuracy"]) - 0.005
        ),
        "every_source_string_accuracy_drop_at_most_0_01": all(
            delta >= -0.01 for delta in per_source_deltas.values()
        ),
        "cross_source_string_accuracy_strictly_improves": (
            cross_source_delta > 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source_balanced_accuracy_absolute_reduction": (
            source_absolute_reduction
        ),
        "source_excess_over_chance_reduction": (
            source_excess_reduction
        ),
        "string_accuracy_delta_by_dataset": per_source_deltas,
        "cross_source_string_accuracy_delta": cross_source_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit candidate harmonic representations for source leakage "
            "and string-position separability."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--maximum-per-bucket", type=int, default=64)
    parser.add_argument(
        "--maximum-per-source-pitch",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--candidate-onset-threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--candidate-frame-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--candidate-minimum-note-length",
        type=float,
        default=70.0,
    )
    arguments = parser.parse_args()
    if min(
        arguments.batch_size,
        arguments.maximum_per_bucket,
        arguments.maximum_per_source_pitch,
    ) <= 0:
        raise ValueError("audit batch and sample limits must be positive")
    if not math.isfinite(arguments.fraction):
        raise ValueError("audit fraction must be finite")

    device = torch.device("cpu")
    (
        _position_model,
        feature_config,
        position_config,
        position_metadata,
    ) = load_position_model(arguments.position_checkpoint, device)
    dataset = OnsetPositionDataset(
        arguments.manifest,
        split="train",
        feature_config=feature_config,
        config=position_config,
        feature_cache_directory=arguments.feature_cache,
        candidate_cache_directory=arguments.candidate_cache,
        candidate_onset_threshold=arguments.candidate_onset_threshold,
        candidate_frame_threshold=arguments.candidate_frame_threshold,
        candidate_minimum_note_length=(
            arguments.candidate_minimum_note_length
        ),
        positive_only_datasets={"AG-PT-set", "Guitar-TECHS"},
    )
    cache = dataset.prepare_feature_cache()
    (
        calibration_annotation_indexes,
        calibration_keys,
        partition_counts,
    ) = stratified_calibration_partition(
        dataset.annotations,
        fraction=arguments.fraction,
        seed=arguments.seed,
        stratify_by_group_datasets={"GuitarSet"},
    )
    examples, collection = collect_examples(
        dataset,
        position_config,
        calibration_annotation_indexes,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        maximum_per_bucket=arguments.maximum_per_bucket,
    )
    retained_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for example in examples:
        retained_counts[example.partition][example.dataset] += 1

    raw_metrics = _representation_metrics(
        examples,
        "raw",
        seed=arguments.seed,
        maximum_per_source_pitch=(
            arguments.maximum_per_source_pitch
        ),
    )
    invariant_metrics = _representation_metrics(
        examples,
        "invariant",
        seed=arguments.seed,
        maximum_per_source_pitch=(
            arguments.maximum_per_source_pitch
        ),
    )
    gate = _gate(raw_metrics, invariant_metrics)
    annotations = load_manifest(arguments.manifest.resolve())
    report = {
        "version": "candidate-source-invariant-representation-v1-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_identity(
            arguments.manifest.resolve(),
            annotations,
            purpose="training",
        ),
        "position_checkpoint": {
            "name": arguments.position_checkpoint.name,
            "sha256": sha256_file(arguments.position_checkpoint),
            "metadata": position_metadata,
        },
        "partition": {
            "fraction": arguments.fraction,
            "seed": arguments.seed,
            "selected_key_sha256": calibration_partition_digest(
                calibration_keys
            ),
            "selected_annotations": len(
                calibration_annotation_indexes
            ),
            "datasets": partition_counts,
        },
        "candidate_frontend": {
            "onset_threshold": arguments.candidate_onset_threshold,
            "frame_threshold": arguments.candidate_frame_threshold,
            "minimum_note_length": (
                arguments.candidate_minimum_note_length
            ),
        },
        "representation": {
            "raw": (
                "three-channel absolute harmonic CQT trajectories"
            ),
            "invariant": (
                "pre-onset magnitude subtraction and per-candidate "
                "per-channel max-absolute normalization"
            ),
            "frames_before": position_config.frames_before,
            "frames_after": position_config.frames_after,
            "harmonics": (
                position_config.candidate_spectral_harmonics
            ),
            "radius": position_config.candidate_spectral_radius,
        },
        "sampling": {
            "maximum_per_partition_dataset_pitch_string": (
                arguments.maximum_per_bucket
            ),
            "maximum_per_source_pitch_for_source_probe": (
                arguments.maximum_per_source_pitch
            ),
            "collection": collection,
            "retained": {
                partition: dict(sorted(counts.items()))
                for partition, counts in sorted(
                    retained_counts.items()
                )
            },
        },
        "cache": cache,
        "raw": raw_metrics,
        "invariant": invariant_metrics,
        "gate": gate,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "gate": gate,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
