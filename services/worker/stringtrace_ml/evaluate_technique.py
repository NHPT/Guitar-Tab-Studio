from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .technique_model import (
    TechniqueEventDataset,
    TechniqueFeatureConfig,
    TechniqueModelConfig,
    TechniqueNet,
)
from .technique_schema import TECHNIQUE_LABELS
from .train import choose_device
from .train_technique import classification_metrics


def expected_calibration_error(
    probabilities: np.ndarray,
    references: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == references
    error = 0.0
    boundaries = np.linspace(0, 1, bins + 1)
    for index, (lower, upper) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        included = (
            (confidences >= lower)
            & (
                confidences <= upper
                if index == bins - 1
                else confidences < upper
            )
        )
        if not np.any(included):
            continue
        error += float(included.mean()) * abs(
            float(correct[included].mean())
            - float(confidences[included].mean())
        )
    return error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_temperature(
    logits: torch.Tensor,
    references: torch.Tensor,
) -> float:
    cpu_logits = logits.detach().cpu().float()
    cpu_references = references.detach().cpu()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=80,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10)
        loss = torch.nn.functional.cross_entropy(
            cpu_logits / temperature,
            cpu_references,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10))


def _confusion(
    references: np.ndarray,
    predictions: np.ndarray,
) -> torch.Tensor:
    confusion = torch.zeros(
        len(TECHNIQUE_LABELS),
        len(TECHNIQUE_LABELS),
        dtype=torch.long,
    )
    for reference, prediction in zip(references, predictions):
        confusion[int(reference), int(prediction)] += 1
    return confusion


def select_confidence_gates(
    probabilities: np.ndarray,
    references: np.ndarray,
    *,
    minimum_precision: float = 0.90,
    minimum_predictions: int = 20,
    minimum_class_f1: float = 0.65,
    source: str = "validation",
    domains: np.ndarray | None = None,
    minimum_domain_predictions: int = 10,
) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    raw_metrics = classification_metrics(
        _confusion(references, predictions)
    )
    gates: dict[str, dict[str, object]] = {}
    for class_index, label in enumerate(TECHNIQUE_LABELS):
        raw_class = raw_metrics["per_class"][label]
        predicted_indices = np.flatnonzero(predictions == class_index)
        selected_threshold: float | None = None
        selected_count = 0
        selected_precision = 0.0
        selected_domains: dict[str, dict[str, int | float]] = {}
        if float(raw_class["f1"]) >= minimum_class_f1:
            thresholds = np.unique(confidences[predicted_indices])[::-1]
            for threshold in thresholds:
                accepted = predicted_indices[
                    confidences[predicted_indices] >= threshold
                ]
                if len(accepted) < minimum_predictions:
                    continue
                precision = float(
                    np.mean(references[accepted] == class_index)
                )
                domain_results: dict[str, dict[str, int | float]] = {}
                domains_pass = True
                if domains is not None:
                    for domain in sorted(set(domains)):
                        domain_support = int(
                            np.sum(
                                (domains == domain)
                                & (references == class_index)
                            )
                        )
                        if domain_support < minimum_domain_predictions:
                            continue
                        domain_accepted = accepted[
                            domains[accepted] == domain
                        ]
                        domain_precision = (
                            float(
                                np.mean(
                                    references[domain_accepted]
                                    == class_index
                                )
                            )
                            if len(domain_accepted)
                            else 0.0
                        )
                        domain_results[str(domain)] = {
                            "support": domain_support,
                            "accepted": len(domain_accepted),
                            "precision": domain_precision,
                        }
                        domains_pass = domains_pass and (
                            len(domain_accepted)
                            >= minimum_domain_predictions
                            and domain_precision >= minimum_precision
                        )
                if (
                    precision >= minimum_precision
                    and domains_pass
                    and len(accepted) > selected_count
                ):
                    selected_threshold = float(threshold)
                    selected_count = len(accepted)
                    selected_precision = precision
                    selected_domains = domain_results
        supported = selected_threshold is not None
        gates[label] = {
            "supported": supported,
            "threshold": selected_threshold,
            "validation_predictions": selected_count,
            "validation_precision": selected_precision,
            "raw_f1": float(raw_class["f1"]),
            "validation_domains": selected_domains,
        }

    return {
        "selection": {
            "minimum_precision": minimum_precision,
            "minimum_predictions": minimum_predictions,
            "minimum_raw_class_f1": minimum_class_f1,
            "minimum_domain_predictions": minimum_domain_predictions,
            "source": source,
        },
        "gates": gates,
        "selective": evaluate_confidence_gates(
            probabilities,
            references,
            gates,
        ),
        "raw": raw_metrics,
    }


def evaluate_confidence_gates(
    probabilities: np.ndarray,
    references: np.ndarray,
    gates: dict[str, dict[str, object]],
) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    accepted_predictions = np.full(references.shape, -1, dtype=np.int64)
    for class_index, label in enumerate(TECHNIQUE_LABELS):
        gate = gates.get(label, {})
        threshold = gate.get("threshold")
        if not gate.get("supported") or threshold is None:
            continue
        accepted = (
            (predictions == class_index)
            & (confidences >= float(threshold))
        )
        accepted_predictions[accepted] = class_index

    per_class: dict[str, dict[str, int | float | bool]] = {}
    for class_index, label in enumerate(TECHNIQUE_LABELS):
        accepted = accepted_predictions == class_index
        true_positive = int(np.sum(accepted & (references == class_index)))
        accepted_count = int(accepted.sum())
        reference_count = int(np.sum(references == class_index))
        precision = true_positive / max(1, accepted_count)
        recall = true_positive / max(1, reference_count)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "supported": bool(gates.get(label, {}).get("supported")),
            "accepted": accepted_count,
            "support": reference_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accepted": int(np.sum(accepted_predictions >= 0)),
        "unknown": int(np.sum(accepted_predictions < 0)),
        "coverage": float(np.mean(accepted_predictions >= 0)),
        "accuracy": float(
            np.mean(
                accepted_predictions[accepted_predictions >= 0]
                == references[accepted_predictions >= 0]
            )
        )
        if np.any(accepted_predictions >= 0)
        else 0.0,
        "per_class": per_class,
    }


def metrics_by_domain(
    probabilities: np.ndarray,
    references: np.ndarray,
    domains: np.ndarray,
    gates: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    predictions = probabilities.argmax(axis=1)
    return {
        str(domain): {
            "events": int(np.sum(domains == domain)),
            "raw": classification_metrics(
                _confusion(
                    references[domains == domain],
                    predictions[domains == domain],
                )
            ),
            "selective": evaluate_confidence_gates(
                probabilities[domains == domain],
                references[domains == domain],
                gates,
            ),
        }
        for domain in sorted(set(domains))
    }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TechniqueNet, TechniqueFeatureConfig, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("format_version", 0)) not in {2, 3}:
        if int(checkpoint.get("format_version", 0)) != 1:
            raise ValueError("Unsupported technique checkpoint format")
    metadata = dict(checkpoint.get("metadata", {}))
    version = str(metadata.get("version", ""))
    feature_values = dict(checkpoint["feature_config"])
    model_values = dict(checkpoint["model_config"])
    if "truncate_at_event_end" not in feature_values:
        feature_values["truncate_at_event_end"] = version.endswith("-v9")
    if "architecture" not in model_values:
        if version == "pitch-relative-cqt-cnn-v4":
            model_values["architecture"] = "global-average-v4"
        elif version in {
            "pitch-relative-cqt-summary-dynamics-v7",
            "pitch-relative-cqt-summary-dynamics-clipped-v8",
            "pitch-relative-cqt-domain-balanced-v9",
        }:
            model_values["architecture"] = "global-summary-dynamics-v1"
        else:
            raise ValueError(
                f"Checkpoint architecture is not reproducible: {version}"
            )
    feature_config = TechniqueFeatureConfig.from_dict(
        feature_values
    )
    model = TechniqueNet(
        feature_config,
        TechniqueModelConfig.from_dict(model_values),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, feature_config, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate a note-level technique model.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-predictions", type=int, default=20)
    parser.add_argument("--minimum-class-f1", type=float, default=0.65)
    parser.add_argument("--minimum-domain-predictions", type=int, default=10)
    parser.add_argument(
        "--gate-report",
        type=Path,
        help="Validation report supplying fixed temperature and class gates.",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    device = choose_device(arguments.device)
    model, feature_config, metadata = load_model(
        arguments.checkpoint,
        device,
    )
    dataset = TechniqueEventDataset(
        arguments.manifest,
        split=arguments.split,
        feature_config=feature_config,
        cache_directory=arguments.feature_cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    logits_batches = []
    reference_batches = []
    with torch.no_grad():
        for batch in loader:
            logits_batches.append(
                model(
                    batch["features"].to(device),
                    batch["dynamics"].to(device),
                ).cpu()
            )
            reference_batches.append(batch["label"].cpu())
    logits = torch.cat(logits_batches)
    references_tensor = torch.cat(reference_batches)
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
    references = references_tensor.numpy()
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
            "raw": classification_metrics(
                _confusion(references, probabilities.argmax(axis=1))
            ),
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
