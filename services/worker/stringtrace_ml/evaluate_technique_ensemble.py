from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_technique import (
    _confusion,
    evaluate_confidence_gates,
    metrics_by_domain,
    select_confidence_gates,
    sha256_file,
)
from .evaluate_technique_svm import load_svm_checkpoint
from .technique_descriptors import TechniqueDescriptorDataset
from .technique_model import TechniqueEventDataset
from .train import choose_device
from .train_technique import classification_metrics
from .train_technique_svm import dataset_matrix


def checkpoint_identity(path: Path) -> dict[str, str]:
    return {"name": path.name, "sha256": sha256_file(path)}


def agreement_probabilities(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if left.shape != right.shape:
        raise ValueError("Ensemble probability matrices must have equal shape")
    agreement = left.argmax(axis=1) == right.argmax(axis=1)
    combined = np.zeros_like(left)
    scores = np.sqrt(left * right)
    scores /= np.maximum(1e-8, scores.sum(axis=1, keepdims=True))
    combined[agreement] = scores[agreement]
    return combined, agreement


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agreement between neural and SVM technique models.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--neural-checkpoint", type=Path, required=True)
    parser.add_argument("--svm-checkpoint", type=Path, required=True)
    parser.add_argument("--neural-calibration-report", type=Path, required=True)
    parser.add_argument("--svm-calibration-report", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cqt-cache", type=Path)
    parser.add_argument("--descriptor-cache", type=Path)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--minimum-predictions", type=int, default=20)
    parser.add_argument("--minimum-class-f1", type=float, default=0.0)
    parser.add_argument("--minimum-domain-predictions", type=int, default=10)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    neural_calibration = json.loads(
        arguments.neural_calibration_report.read_text(encoding="utf-8")
    )
    svm_calibration = json.loads(
        arguments.svm_calibration_report.read_text(encoding="utf-8")
    )
    neural_identity = checkpoint_identity(arguments.neural_checkpoint)
    svm_identity = checkpoint_identity(arguments.svm_checkpoint)
    if neural_calibration.get("checkpoint") != neural_identity:
        raise ValueError("Neural calibration report checkpoint mismatch")
    if svm_calibration.get("checkpoint") != svm_identity:
        raise ValueError("SVM calibration report checkpoint mismatch")

    device = choose_device(arguments.device)
    from .evaluate_technique import load_model

    neural_model, neural_config, neural_metadata = load_model(
        arguments.neural_checkpoint,
        device,
    )
    neural_dataset = TechniqueEventDataset(
        arguments.manifest,
        split=arguments.split,
        feature_config=neural_config,
        cache_directory=arguments.cqt_cache,
    )
    neural_logits = []
    references = []
    with torch.no_grad():
        for batch in DataLoader(
            neural_dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
        ):
            neural_logits.append(
                neural_model(
                    batch["features"].to(device),
                    batch["dynamics"].to(device),
                ).cpu()
            )
            references.append(batch["label"])
    references_array = torch.cat(references).numpy()
    neural_probabilities = torch.softmax(
        torch.cat(neural_logits)
        / float(neural_calibration["calibration"]["temperature"]),
        dim=-1,
    ).numpy()

    svm_model, feature_kind, svm_config, svm_metadata = load_svm_checkpoint(
        arguments.svm_checkpoint
    )
    if feature_kind != "descriptors":
        raise ValueError("Ensemble currently requires descriptor SVM features")
    svm_dataset = TechniqueDescriptorDataset(
        arguments.manifest,
        split=arguments.split,
        config=svm_config,
        cache_directory=arguments.descriptor_cache,
    )
    svm_features, svm_references = dataset_matrix(svm_dataset)
    if not np.array_equal(references_array, svm_references):
        raise ValueError("Ensemble datasets are not event-aligned")
    svm_probabilities = torch.softmax(
        torch.from_numpy(
            np.asarray(
                svm_model.decision_function(svm_features),
                dtype=np.float32,
            )
        )
        / float(svm_calibration["calibration"]["temperature"]),
        dim=-1,
    ).numpy()
    probabilities, agreement = agreement_probabilities(
        neural_probabilities,
        svm_probabilities,
    )
    domains = np.asarray(
        [
            str(
                neural_dataset.tracks[track_index].provenance.get(
                    "dataset",
                    "unknown",
                )
            )
            for track_index, _event_index in neural_dataset.events
        ]
    )
    ensemble_identity = {
        "neural": neural_identity,
        "svm": svm_identity,
    }
    if arguments.gate_report is None:
        if arguments.split != "validation":
            parser.error("A non-validation split requires --gate-report")
        gate_result = select_confidence_gates(
            probabilities,
            references_array,
            minimum_precision=arguments.minimum_precision,
            minimum_predictions=arguments.minimum_predictions,
            minimum_class_f1=arguments.minimum_class_f1,
            source="validation-ensemble-agreement",
            domains=domains,
            minimum_domain_predictions=arguments.minimum_domain_predictions,
        )
    else:
        gate_source = json.loads(
            arguments.gate_report.read_text(encoding="utf-8")
        )
        if gate_source.get("ensemble") != ensemble_identity:
            raise ValueError("Ensemble gate report checkpoint mismatch")
        gate_result = {
            "selection": {
                **dict(gate_source["selection"]),
                "applied_from": arguments.gate_report.name,
            },
            "gates": dict(gate_source["gates"]),
            "selective": evaluate_confidence_gates(
                probabilities,
                references_array,
                dict(gate_source["gates"]),
            ),
            "raw": classification_metrics(
                _confusion(
                    references_array,
                    probabilities.argmax(axis=1),
                )
            ),
        }
    gate_result["by_dataset"] = metrics_by_domain(
        probabilities,
        references_array,
        domains,
        dict(gate_result["gates"]),
    )
    report = {
        "split": arguments.split,
        "events": len(neural_dataset),
        "ensemble": ensemble_identity,
        "models": {
            "neural": neural_metadata,
            "svm": svm_metadata,
        },
        "component_calibration": {
            "neural_temperature": neural_calibration["calibration"][
                "temperature"
            ],
            "svm_temperature": svm_calibration["calibration"]["temperature"],
        },
        "agreement": {
            "events": int(agreement.sum()),
            "rate": float(agreement.mean()),
        },
        **gate_result,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
