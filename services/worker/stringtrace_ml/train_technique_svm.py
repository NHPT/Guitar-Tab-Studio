from __future__ import annotations

import argparse
import json
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate_technique import _confusion
from .technique_descriptors import (
    TechniqueDescriptorConfig,
    TechniqueDescriptorDataset,
)
from .technique_model import (
    TechniqueEventDataset,
    TechniqueFeatureConfig,
)
from .technique_schema import TECHNIQUE_LABELS
from .train_technique import classification_metrics


def dataset_matrix(
    dataset: TechniqueEventDataset | TechniqueDescriptorDataset,
) -> tuple[np.ndarray, np.ndarray]:
    summaries = []
    labels = []
    for index in range(len(dataset)):
        item = dataset[index]
        summaries.append(item["summary"].numpy())
        labels.append(int(item["label"]))
    return (
        np.asarray(summaries, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def evaluate_classifier(
    model: Pipeline,
    features: np.ndarray,
    references: np.ndarray,
) -> dict[str, object]:
    predictions = model.predict(features)
    return classification_metrics(_confusion(references, predictions))


def full_decision_scores(
    model: Pipeline,
    features: np.ndarray,
) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    classes = np.asarray(classifier.classes_, dtype=np.int64)
    raw_scores = np.asarray(
        model.decision_function(features),
        dtype=np.float32,
    )
    scores = np.full(
        (len(features), len(TECHNIQUE_LABELS)),
        -1e6,
        dtype=np.float32,
    )
    if raw_scores.ndim == 1:
        scores[:, classes[0]] = -raw_scores
        scores[:, classes[1]] = raw_scores
    else:
        scores[:, classes] = raw_scores
    return scores


def dataset_metrics(
    model: Pipeline,
    features: np.ndarray,
    references: np.ndarray,
    datasets: np.ndarray,
) -> dict[str, dict[str, object]]:
    return {
        dataset: evaluate_classifier(
            model,
            features[datasets == dataset],
            references[datasets == dataset],
        )
        for dataset in sorted(set(datasets))
    }


def present_label_macro_f1(metrics: dict[str, object]) -> float:
    values = [
        float(class_metrics["f1"])
        for class_metrics in metrics["per_class"].values()
        if int(class_metrics["support"]) > 0
    ]
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a compact note-level technique SVM.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument(
        "--features",
        choices=("cqt-summary", "descriptors"),
        default="descriptors",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=(0.1, 0.3, 1.0, 3.0, 10.0),
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--balance-exponent", type=float, default=1.0)
    parser.add_argument("--balance-dataset", action="store_true")
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    if arguments.features == "descriptors":
        feature_config = TechniqueDescriptorConfig()
        train_dataset = TechniqueDescriptorDataset(
            arguments.manifest,
            split="train",
            config=feature_config,
            cache_directory=arguments.feature_cache,
        )
        validation_dataset = TechniqueDescriptorDataset(
            arguments.manifest,
            split="validation",
            config=feature_config,
            cache_directory=arguments.feature_cache,
        )
    else:
        feature_config = TechniqueFeatureConfig()
        train_dataset = TechniqueEventDataset(
            arguments.manifest,
            split="train",
            feature_config=feature_config,
            cache_directory=arguments.feature_cache,
            augment=False,
        )
        validation_dataset = TechniqueEventDataset(
            arguments.manifest,
            split="validation",
            feature_config=feature_config,
            cache_directory=arguments.feature_cache,
            augment=False,
        )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("SVM training requires train and validation events")
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
    train_weights = (
        train_dataset.balanced_sample_weights(
            exponent=arguments.balance_exponent,
            balance_dataset=arguments.balance_dataset,
        )
        if isinstance(train_dataset, TechniqueDescriptorDataset)
        else np.asarray(
            train_dataset.balanced_sample_weights(
                exponent=arguments.balance_exponent,
                balance_dataset=arguments.balance_dataset,
            )
        )
    )
    validation_datasets = np.asarray(
        [
            str(
                validation_dataset.tracks[track_index].provenance.get(
                    "dataset",
                    "unknown",
                )
            )
            for track_index, _event_index in validation_dataset.events
        ]
    )

    trials = []
    best_model: Pipeline | None = None
    best_metrics: dict[str, object] | None = None
    best_by_dataset: dict[str, dict[str, object]] | None = None
    best_c = 0.0
    best_selection_score = float("-inf")
    for c_value in arguments.c_values:
        model = Pipeline(
            (
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=c_value,
                        kernel="rbf",
                        class_weight=None,
                        decision_function_shape="ovr",
                        random_state=arguments.seed,
                    ),
                ),
            )
        )
        model.fit(
            train_features,
            train_references,
            classifier__sample_weight=train_weights,
        )
        metrics = evaluate_classifier(
            model,
            validation_features,
            validation_references,
        )
        by_dataset = dataset_metrics(
            model,
            validation_features,
            validation_references,
            validation_datasets,
        )
        selection_score = min(
            present_label_macro_f1(dataset_result)
            for dataset_result in by_dataset.values()
        )
        trial = {
            "c": c_value,
            "validation": metrics,
            "validation_by_dataset": by_dataset,
            "selection_score": selection_score,
        }
        trials.append(trial)
        print(json.dumps(trial, ensure_ascii=False), flush=True)
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_c = c_value
            best_model = model
            best_metrics = metrics
            best_by_dataset = by_dataset

    if (
        best_model is None
        or best_metrics is None
        or best_by_dataset is None
    ):
        raise RuntimeError("No SVM candidate was trained")
    checkpoint = {
        "format_version": 1,
        "model": best_model,
        "feature_kind": arguments.features,
        "feature_config": feature_config.to_dict(),
        "metadata": {
            "name": "stringtrace-technique-svm",
            "version": (
                "onset-descriptors-rbf-v2"
                if arguments.features == "descriptors"
                else "cqt-temporal-summary-rbf-v1"
            ),
            "kind": "trained",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": arguments.manifest.resolve().parent.name,
            "manifest_name": arguments.manifest.name,
            "labels": list(TECHNIQUE_LABELS),
            "seed": arguments.seed,
            "selection_metric_value": best_selection_score,
            "selection_metric": "minimum_present_label_macro_f1_by_dataset",
            "selected_c": best_c,
            "balance_exponent": arguments.balance_exponent,
            "balance_dataset": arguments.balance_dataset,
            "sklearn_version": sklearn.__version__,
            "training_groups": sorted(
                {track.group_id for track in train_dataset.tracks}
            ),
            "validation_groups": sorted(
                {track.group_id for track in validation_dataset.tracks}
            ),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("wb") as stream:
        pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "checkpoint": arguments.output.name,
        "train_events": len(train_dataset),
        "validation_events": len(validation_dataset),
        "feature_size": int(train_features.shape[1]),
        "feature_kind": arguments.features,
        "selected_c": best_c,
        "selection_score": best_selection_score,
        "balance_exponent": arguments.balance_exponent,
        "balance_dataset": arguments.balance_dataset,
        "best_validation": best_metrics,
        "best_validation_by_dataset": best_by_dataset,
        "trials": trials,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
