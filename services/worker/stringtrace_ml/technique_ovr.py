from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate_technique import sha256_file
from .technique_descriptors import (
    TechniqueDescriptorConfig,
    TechniqueDescriptorDataset,
)
from .technique_schema import TECHNIQUE_LABELS
from .train_technique_svm import dataset_matrix


def binary_sample_weights(
    references: np.ndarray,
    domains: np.ndarray,
    class_index: int,
) -> np.ndarray:
    positive = references == class_index
    counts: dict[tuple[str, bool], int] = {}
    for domain, is_positive in zip(domains, positive):
        key = (str(domain), bool(is_positive))
        counts[key] = counts.get(key, 0) + 1
    weights = np.asarray(
        [
            1 / counts[(str(domain), bool(is_positive))]
            for domain, is_positive in zip(domains, positive)
        ],
        dtype=np.float64,
    )
    return weights / weights.mean()


def binary_metrics(
    accepted: np.ndarray,
    references: np.ndarray,
    class_index: int,
) -> dict[str, int | float]:
    positive = references == class_index
    true_positive = int(np.sum(accepted & positive))
    false_positive = int(np.sum(accepted & ~positive))
    false_negative = int(np.sum(~accepted & positive))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "support": int(positive.sum()),
        "accepted": int(accepted.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def select_binary_threshold(
    scores: np.ndarray,
    references: np.ndarray,
    domains: np.ndarray,
    class_index: int,
    *,
    minimum_precision: float,
    minimum_predictions: int,
    minimum_domain_predictions: int,
) -> dict[str, object]:
    best_threshold: float | None = None
    best_metrics: dict[str, int | float] | None = None
    best_domains: dict[str, dict[str, int | float]] = {}
    for threshold in np.unique(scores)[::-1]:
        accepted = scores >= threshold
        overall = binary_metrics(accepted, references, class_index)
        if (
            int(overall["accepted"]) < minimum_predictions
            or float(overall["precision"]) < minimum_precision
        ):
            continue
        domain_results = {}
        domains_pass = True
        for domain in sorted(set(domains)):
            selected = domains == domain
            result = binary_metrics(
                accepted[selected],
                references[selected],
                class_index,
            )
            domain_results[str(domain)] = result
            if int(result["support"]) >= minimum_domain_predictions:
                domains_pass = domains_pass and (
                    int(result["accepted"]) >= minimum_domain_predictions
                    and float(result["precision"]) >= minimum_precision
                )
            else:
                domains_pass = domains_pass and (
                    int(result["false_positive"]) == 0
                )
        if not domains_pass:
            continue
        best_threshold = float(threshold)
        best_metrics = overall
        best_domains = domain_results
    return {
        "supported": best_threshold is not None,
        "threshold": best_threshold,
        "validation_predictions": (
            int(best_metrics["accepted"]) if best_metrics is not None else 0
        ),
        "validation_precision": (
            float(best_metrics["precision"]) if best_metrics is not None else 0.0
        ),
        "validation": best_metrics,
        "validation_domains": best_domains,
    }


def evaluate_ovr(
    models: dict[str, Pipeline],
    thresholds: dict[str, dict[str, object]],
    features: np.ndarray,
    references: np.ndarray,
    domains: np.ndarray,
) -> dict[str, object]:
    score_columns = [
        np.asarray(models[label].decision_function(features))
        for label in TECHNIQUE_LABELS
    ]
    scores = np.stack(score_columns, axis=1)
    accepted = np.zeros_like(scores, dtype=bool)
    for class_index, label in enumerate(TECHNIQUE_LABELS):
        gate = thresholds[label]
        threshold = gate.get("threshold")
        if gate.get("supported") and threshold is not None:
            accepted[:, class_index] = scores[:, class_index] >= float(
                threshold
            )
    accepted_count = accepted.sum(axis=1)
    decisions = np.where(
        accepted_count == 1,
        accepted.argmax(axis=1),
        -1,
    )

    def report_for(mask: np.ndarray) -> dict[str, object]:
        selected_decisions = decisions[mask]
        selected_references = references[mask]
        per_class = {
            label: {
                "supported": bool(thresholds[label]["supported"]),
                **binary_metrics(
                    selected_decisions == class_index,
                    selected_references,
                    class_index,
                ),
            }
            for class_index, label in enumerate(TECHNIQUE_LABELS)
        }
        decided = selected_decisions >= 0
        return {
            "events": int(mask.sum()),
            "accepted": int(decided.sum()),
            "unknown": int(np.sum(selected_decisions < 0)),
            "conflicts": int(np.sum(accepted_count[mask] > 1)),
            "coverage": float(decided.mean()) if len(decided) else 0.0,
            "accuracy": (
                float(
                    np.mean(
                        selected_decisions[decided]
                        == selected_references[decided]
                    )
                )
                if np.any(decided)
                else 0.0
            ),
            "per_class": per_class,
        }

    return {
        "selective": report_for(np.ones(len(references), dtype=bool)),
        "by_dataset": {
            str(domain): {
                "selective": report_for(domains == domain),
            }
            for domain in sorted(set(domains))
        },
    }


def load_ovr_checkpoint(
    path: Path,
) -> tuple[
    dict[str, Pipeline],
    dict[str, dict[str, object]],
    TechniqueDescriptorConfig,
    dict[str, object],
]:
    with path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported technique OVR checkpoint format")
    return (
        dict(checkpoint["models"]),
        dict(checkpoint["gates"]),
        TechniqueDescriptorConfig.from_dict(checkpoint["feature_config"]),
        dict(checkpoint["metadata"]),
    )


def train_main(arguments: argparse.Namespace) -> None:
    config = TechniqueDescriptorConfig()
    train_dataset = TechniqueDescriptorDataset(
        arguments.manifest,
        split="train",
        config=config,
        cache_directory=arguments.feature_cache,
    )
    validation_dataset = TechniqueDescriptorDataset(
        arguments.manifest,
        split="validation",
        config=config,
        cache_directory=arguments.feature_cache,
    )
    print(
        json.dumps(
            {
                "stage": "feature_cache",
                "train": train_dataset.prepare_feature_cache(),
                "validation": validation_dataset.prepare_feature_cache(),
            }
        ),
        flush=True,
    )
    train_features, train_references = dataset_matrix(train_dataset)
    validation_features, validation_references = dataset_matrix(
        validation_dataset
    )
    train_domains = train_dataset.dataset_names()
    validation_domains = validation_dataset.dataset_names()
    models = {}
    gates = {}
    trials = {}
    for class_index, label in enumerate(TECHNIQUE_LABELS):
        weights = binary_sample_weights(
            train_references,
            train_domains,
            class_index,
        )
        best_score = float("-inf")
        best_model = None
        label_trials = []
        for c_value in arguments.c_values:
            model = Pipeline(
                (
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(C=c_value, kernel="rbf", class_weight=None),
                    ),
                )
            )
            binary_train = train_references == class_index
            model.fit(
                train_features,
                binary_train,
                classifier__sample_weight=weights,
            )
            scores = np.asarray(
                model.decision_function(validation_features)
            )
            domain_ap = {}
            for domain in sorted(set(validation_domains)):
                selected = validation_domains == domain
                binary_reference = (
                    validation_references[selected] == class_index
                )
                if np.any(binary_reference):
                    domain_ap[str(domain)] = float(
                        average_precision_score(
                            binary_reference,
                            scores[selected],
                        )
                    )
            selection_score = min(domain_ap.values(), default=0.0)
            label_trials.append(
                {
                    "c": c_value,
                    "selection_score": selection_score,
                    "average_precision_by_dataset": domain_ap,
                }
            )
            if selection_score > best_score:
                best_score = selection_score
                best_model = model
        if best_model is None:
            raise RuntimeError(f"No OVR model trained for {label}")
        models[label] = best_model
        validation_scores = np.asarray(
            best_model.decision_function(validation_features)
        )
        gates[label] = select_binary_threshold(
            validation_scores,
            validation_references,
            validation_domains,
            class_index,
            minimum_precision=arguments.minimum_precision,
            minimum_predictions=arguments.minimum_predictions,
            minimum_domain_predictions=arguments.minimum_domain_predictions,
        )
        trials[label] = label_trials
        print(
            json.dumps(
                {
                    "label": label,
                    "selection_score": best_score,
                    "gate": gates[label],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    validation = evaluate_ovr(
        models,
        gates,
        validation_features,
        validation_references,
        validation_domains,
    )
    checkpoint = {
        "format_version": 1,
        "models": models,
        "gates": gates,
        "feature_config": config.to_dict(),
        "metadata": {
            "name": "stringtrace-technique-ovr",
            "version": "onset-descriptor-rbf-ovr-v1",
            "kind": "trained",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "labels": list(TECHNIQUE_LABELS),
            "sklearn_version": sklearn.__version__,
            "selection_metric": "minimum_average_precision_by_dataset",
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("wb") as stream:
        pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "split": "validation",
        "checkpoint": {
            "name": arguments.output.name,
            "sha256": sha256_file(arguments.output),
        },
        "model": checkpoint["metadata"],
        "gates": gates,
        "selection": {
            "minimum_precision": arguments.minimum_precision,
            "minimum_predictions": arguments.minimum_predictions,
            "minimum_domain_predictions": arguments.minimum_domain_predictions,
        },
        **validation,
        "trials": trials,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_main(arguments: argparse.Namespace) -> None:
    models, gates, config, metadata = load_ovr_checkpoint(
        arguments.checkpoint
    )
    dataset = TechniqueDescriptorDataset(
        arguments.manifest,
        split=arguments.split,
        config=config,
        cache_directory=arguments.feature_cache,
    )
    dataset.prepare_feature_cache()
    features, references = dataset_matrix(dataset)
    result = evaluate_ovr(
        models,
        gates,
        features,
        references,
        dataset.dataset_names(),
    )
    report = {
        "split": arguments.split,
        "checkpoint": {
            "name": arguments.checkpoint.name,
            "sha256": sha256_file(arguments.checkpoint),
        },
        "model": metadata,
        "gates": gates,
        **result,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or evaluate one-vs-rest technique classifiers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--report", type=Path, required=True)
    train.add_argument("--feature-cache", type=Path, required=True)
    train.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=(0.03, 0.1, 0.3, 1.0, 3.0),
    )
    train.add_argument("--minimum-precision", type=float, default=0.95)
    train.add_argument("--minimum-predictions", type=int, default=20)
    train.add_argument("--minimum-domain-predictions", type=int, default=10)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--feature-cache", type=Path, required=True)
    evaluate.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "train":
        train_main(arguments)
    else:
        evaluate_main(arguments)


if __name__ == "__main__":
    main()
