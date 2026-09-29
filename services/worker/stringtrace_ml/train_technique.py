from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .technique_model import (
    TechniqueEventDataset,
    TechniqueFeatureConfig,
    TechniqueNet,
)
from .technique_schema import TECHNIQUE_LABELS
from .train import choose_device


def classification_metrics(
    confusion: torch.Tensor,
) -> dict[str, object]:
    per_class: dict[str, dict[str, int | float]] = {}
    f1_values = []
    for index, label in enumerate(TECHNIQUE_LABELS):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = int(confusion[index, :].sum()) - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "support": int(confusion[index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": float(confusion.diag().sum() / confusion.sum().clamp_min(1)),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def present_label_macro_f1(metrics: dict[str, object]) -> float:
    values = [
        float(class_metrics["f1"])
        for class_metrics in metrics["per_class"].values()
        if int(class_metrics["support"]) > 0
    ]
    return sum(values) / max(1, len(values))


def run_epoch(
    model: TechniqueNet,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros(
        len(TECHNIQUE_LABELS),
        len(TECHNIQUE_LABELS),
        dtype=torch.long,
    )
    loss_sum = 0.0
    batches = 0
    domain_confusions: dict[str, torch.Tensor] = {}
    for batch in loader:
        features = batch["features"].to(device)
        dynamics = batch["dynamics"].to(device)
        labels = batch["label"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features, dynamics)
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=class_weights,
            )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        predictions = logits.argmax(dim=-1).detach().cpu()
        references = labels.detach().cpu()
        domains = batch["domain"]
        for reference, prediction, domain in zip(
            references,
            predictions,
            domains,
        ):
            confusion[int(reference), int(prediction)] += 1
            domain_confusion = domain_confusions.setdefault(
                str(domain),
                torch.zeros_like(confusion),
            )
            domain_confusion[int(reference), int(prediction)] += 1
        loss_sum += float(loss.detach().cpu())
        batches += 1
    metrics = classification_metrics(confusion)
    return {
        "loss": loss_sum / max(1, batches),
        **metrics,
        "present_label_macro_f1": present_label_macro_f1(metrics),
        "by_dataset": {
            domain: {
                **classification_metrics(domain_confusion),
                "present_label_macro_f1": present_label_macro_f1(
                    classification_metrics(domain_confusion)
                ),
            }
            for domain, domain_confusion in sorted(
                domain_confusions.items()
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a note-level Guitar-TECHS technique classifier.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--post-onset-seconds", type=float, default=1.50)
    parser.add_argument("--balance-exponent", type=float, default=1.0)
    parser.add_argument("--balance-dataset", action="store_true")
    parser.add_argument(
        "--model-version",
        default="pitch-relative-cqt-domain-balanced-v9",
    )
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = choose_device(arguments.device)
    feature_config = TechniqueFeatureConfig(
        post_onset_seconds=arguments.post_onset_seconds,
    )
    cache_directory = (
        arguments.feature_cache.resolve()
        if arguments.feature_cache is not None
        else arguments.manifest.resolve().parent / ".technique-feature-cache"
    )
    train_dataset = TechniqueEventDataset(
        arguments.manifest,
        split="train",
        feature_config=feature_config,
        cache_directory=cache_directory,
        augment=True,
    )
    validation_dataset = TechniqueEventDataset(
        arguments.manifest,
        split="validation",
        feature_config=feature_config,
        cache_directory=cache_directory,
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Technique training requires train and validation events")
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

    class_weights = torch.ones(
        len(TECHNIQUE_LABELS),
        dtype=torch.float32,
        device=device,
    )
    sampler = WeightedRandomSampler(
        train_dataset.balanced_sample_weights(
            exponent=arguments.balance_exponent,
            balance_dataset=arguments.balance_dataset,
        ),
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(arguments.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = TechniqueNet(feature_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=1e-4,
    )
    best_selection_score = float("-inf")
    history: list[dict[str, object]] = []
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, arguments.epochs + 1):
        training = run_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer,
        )
        validation = run_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            None,
        )
        record = {
            "epoch": epoch,
            "train": training,
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        selection_score = min(
            float(metrics["present_label_macro_f1"])
            for metrics in validation["by_dataset"].values()
        )
        if selection_score <= best_selection_score:
            continue
        best_selection_score = selection_score
        torch.save(
            {
                "format_version": 3,
                "model_state": model.state_dict(),
                "model_config": model.model_config.to_dict(),
                "feature_config": feature_config.to_dict(),
                "metadata": {
                    "name": "stringtrace-technique-net",
                    "version": arguments.model_version,
                    "kind": "trained",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dataset": arguments.manifest.resolve().parent.name,
                    "manifest_name": arguments.manifest.name,
                    "labels": list(TECHNIQUE_LABELS),
                    "seed": arguments.seed,
                    "epoch": epoch,
                    "selection_metric": (
                        "minimum_present_label_macro_f1_by_dataset"
                    ),
                    "selection_metric_value": selection_score,
                    "balance_exponent": arguments.balance_exponent,
                    "balance_dataset": arguments.balance_dataset,
                    "training_groups": sorted(
                        {track.group_id for track in train_dataset.tracks}
                    ),
                    "validation_groups": sorted(
                        {
                            track.group_id
                            for track in validation_dataset.tracks
                        }
                    ),
                },
                "history": history,
            },
            arguments.output,
        )
    print(
        json.dumps(
            {
                "checkpoint": arguments.output.name,
                "best_validation_selection_score": best_selection_score,
                "train_events": len(train_dataset),
                "validation_events": len(validation_dataset),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
