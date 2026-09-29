from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline

from .evaluate_technique import (
    _confusion,
    evaluate_confidence_gates,
    expected_calibration_error,
    fit_temperature,
    metrics_by_domain,
    select_confidence_gates,
    sha256_file,
)
from .technique_descriptors import (
    TechniqueDescriptorConfig,
    TechniqueDescriptorDataset,
)
from .technique_model import (
    TechniqueEventDataset,
    TechniqueFeatureConfig,
)
from .train_technique import classification_metrics
from .train_technique_svm import dataset_matrix, full_decision_scores


def load_svm_checkpoint(
    path: Path,
) -> tuple[
    Pipeline,
    str,
    TechniqueFeatureConfig | TechniqueDescriptorConfig,
    dict[str, object],
]:
    with path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported technique SVM checkpoint format")
    model = checkpoint.get("model")
    if not isinstance(model, Pipeline):
        raise ValueError("Technique SVM checkpoint contains an invalid model")
    feature_kind = str(checkpoint.get("feature_kind", "cqt-summary"))
    if feature_kind == "descriptors":
        feature_config = TechniqueDescriptorConfig.from_dict(
            checkpoint["feature_config"]
        )
    elif feature_kind == "cqt-summary":
        feature_config = TechniqueFeatureConfig.from_dict(
            checkpoint["feature_config"]
        )
    else:
        raise ValueError(f"Unsupported SVM feature kind: {feature_kind}")
    return model, feature_kind, feature_config, dict(
        checkpoint.get("metadata", {})
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate a note-level technique SVM.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-predictions", type=int, default=20)
    parser.add_argument("--minimum-class-f1", type=float, default=0.65)
    parser.add_argument("--minimum-domain-predictions", type=int, default=10)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    model, feature_kind, feature_config, metadata = load_svm_checkpoint(
        arguments.checkpoint
    )
    if feature_kind == "descriptors":
        if not isinstance(feature_config, TechniqueDescriptorConfig):
            raise ValueError("Descriptor checkpoint has an invalid config")
        dataset = TechniqueDescriptorDataset(
            arguments.manifest,
            split=arguments.split,
            config=feature_config,
            cache_directory=arguments.feature_cache,
        )
    else:
        if not isinstance(feature_config, TechniqueFeatureConfig):
            raise ValueError("CQT checkpoint has an invalid config")
        dataset = TechniqueEventDataset(
            arguments.manifest,
            split=arguments.split,
            feature_config=feature_config,
            cache_directory=arguments.feature_cache,
        )
    if not dataset:
        raise RuntimeError(f"The manifest contains no {arguments.split} events")
    features, references = dataset_matrix(dataset)
    domains = np.asarray(
        [
            str(
                dataset.tracks[track_index].provenance.get(
                    "dataset",
                    "unknown",
                )
            )
            for track_index, _event_index in dataset.events
        ]
    )
    logits = torch.from_numpy(full_decision_scores(model, features))
    references_tensor = torch.from_numpy(references)
    checkpoint_identity = {
        "name": arguments.checkpoint.name,
        "sha256": sha256_file(arguments.checkpoint),
    }

    gate_source = None
    if arguments.gate_report is not None:
        gate_source = json.loads(
            arguments.gate_report.read_text(encoding="utf-8")
        )
        if gate_source.get("checkpoint") != checkpoint_identity:
            raise ValueError(
                "Gate report was produced for a different checkpoint"
            )
        temperature = float(gate_source["calibration"]["temperature"])
    else:
        if arguments.split != "validation":
            parser.error(
                "A non-validation split requires --gate-report so the "
                "test data cannot select its own thresholds"
            )
        temperature = fit_temperature(logits, references_tensor)
    probabilities = torch.softmax(logits / temperature, dim=-1).numpy()
    raw_metrics = classification_metrics(
        _confusion(references, probabilities.argmax(axis=1))
    )
    gate_result = (
        {
            "selection": {
                **dict(gate_source["selection"]),
                "applied_from": arguments.gate_report.name,
            },
            "gates": dict(gate_source["gates"]),
            "selective": evaluate_confidence_gates(
                probabilities,
                references,
                dict(gate_source["gates"]),
            ),
            "raw": raw_metrics,
        }
        if gate_source is not None
        else select_confidence_gates(
            probabilities,
            references,
            minimum_precision=arguments.minimum_precision,
            minimum_predictions=arguments.minimum_predictions,
            minimum_class_f1=arguments.minimum_class_f1,
            source=arguments.split,
            domains=domains,
            minimum_domain_predictions=arguments.minimum_domain_predictions,
        )
    )
    gate_result["by_dataset"] = metrics_by_domain(
        probabilities,
        references,
        domains,
        dict(gate_result["gates"]),
    )
    report = {
        "split": arguments.split,
        "events": len(dataset),
        "checkpoint": checkpoint_identity,
        "model": metadata,
        "calibration": {
            "temperature": temperature,
            "ece": expected_calibration_error(
                probabilities,
                references,
            ),
        },
        **gate_result,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
