from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch

from .evaluate_technique import load_model, sha256_file
from .evaluate_technique_svm import load_svm_checkpoint
from .technique_descriptors import (
    extract_track_descriptors,
    summarize_event_descriptors,
)
from .technique_model import extract_event_patch, extract_log_cqt
from .technique_schema import TECHNIQUE_LABELS
from .train_technique_svm import full_decision_scores


class TimedPitchEvent(Protocol):
    start: float
    end: float
    pitch: int


@dataclass(frozen=True)
class TechniqueDecision:
    technique: str
    confidence: float
    source: str
    evidence: list[str]
    candidates: list[tuple[str, float]]


def load_production_gate(
    path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("kind") != "technique-production-gate-v1":
        raise ValueError("Unsupported technique gate format")
    expected = {
        "name": checkpoint_path.name,
        "sha256": sha256_file(checkpoint_path),
    }
    if gate.get("checkpoint") != expected:
        raise ValueError("Technique gate does not match the checkpoint")
    return gate


def neural_probabilities(
    audio_path: Path,
    events: Sequence[TimedPitchEvent],
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, dict[str, object]]:
    model, feature_config, metadata = load_model(
        checkpoint_path,
        device,
    )
    track_features = extract_log_cqt(audio_path, feature_config)
    patches = []
    dynamics = []
    for index, event in enumerate(events):
        next_onset = next(
            (
                candidate.start
                for candidate in events[index + 1:]
                if candidate.start > event.start + 0.05
            ),
            None,
        )
        patch, event_dynamics = extract_event_patch(
            track_features,
            onset=event.start,
            offset=min(event.end, next_onset)
            if next_onset is not None
            else event.end,
            pitch=event.pitch,
            config=feature_config,
        )
        patches.append(patch)
        dynamics.append(event_dynamics)

    if not patches:
        return np.empty((0, len(TECHNIQUE_LABELS))), metadata

    logits = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            stop = start + batch_size
            logits.append(
                model(
                    torch.from_numpy(np.asarray(patches[start:stop])).to(
                        device
                    ),
                    torch.from_numpy(np.asarray(dynamics[start:stop])).to(
                        device
                    ),
                ).cpu()
            )
    return torch.cat(logits).numpy(), metadata


def decisions_from_probabilities(
    probabilities: np.ndarray,
    gate: dict[str, object],
    *,
    source: str,
) -> list[TechniqueDecision]:
    decisions = []
    for row in probabilities:
        order = np.argsort(row)[::-1]
        primary_index = int(order[0])
        primary_label = TECHNIQUE_LABELS[primary_index]
        primary_confidence = float(row[primary_index])
        class_gate = dict(gate["gates"][primary_label])
        threshold = class_gate.get("threshold")
        accepted = (
            bool(class_gate.get("supported"))
            and threshold is not None
            and primary_confidence >= float(threshold)
        )
        decisions.append(
            TechniqueDecision(
                technique=primary_label if accepted else "unknown",
                confidence=primary_confidence if accepted else 0.0,
                source=(
                    source
                    if accepted
                    else "technique-model-abstained"
                ),
                evidence=[
                    (
                        "fixed-validation-and-test-gate"
                        if accepted
                        else "unsupported-or-below-threshold"
                    )
                ],
                candidates=[
                    (TECHNIQUE_LABELS[int(candidate)], float(row[candidate]))
                    for candidate in order[:3]
                ],
            )
        )
    return decisions


def predict_techniques(
    audio_path: Path,
    events: Sequence[TimedPitchEvent],
    checkpoint_path: Path,
    gate_path: Path,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[list[TechniqueDecision], dict[str, object]]:
    logits, metadata = neural_probabilities(
        audio_path,
        events,
        checkpoint_path,
        device=device,
        batch_size=batch_size,
    )
    gate = load_production_gate(gate_path, checkpoint_path)
    temperature = float(gate["calibration"]["temperature"])
    probabilities = torch.softmax(
        torch.from_numpy(logits) / temperature,
        dim=-1,
    ).numpy()
    return decisions_from_probabilities(
        probabilities,
        gate,
        source=str(metadata.get("version", "technique-model")),
    ), {
        **metadata,
        "approved_labels": list(gate["approved_labels"]),
        "gate": gate_path.name,
    }


def svm_probabilities(
    audio_path: Path,
    events: Sequence[TimedPitchEvent],
    checkpoint_path: Path,
    *,
    temperature: float,
) -> tuple[np.ndarray, dict[str, object]]:
    model, feature_kind, config, metadata = load_svm_checkpoint(
        checkpoint_path
    )
    if feature_kind != "descriptors":
        raise ValueError("Technique bundle requires descriptor SVM features")
    descriptors = extract_track_descriptors(audio_path, config)
    features = np.asarray(
        [
            summarize_event_descriptors(
                descriptors,
                onset=event.start,
                config=config,
            )
            for event in events
        ],
        dtype=np.float32,
    )
    if not len(features):
        return np.empty((0, len(TECHNIQUE_LABELS))), metadata
    logits = full_decision_scores(model, features)
    probabilities = torch.softmax(
        torch.from_numpy(logits) / temperature,
        dim=-1,
    ).numpy()
    return probabilities, metadata


def predict_technique_bundle(
    audio_path: Path,
    events: Sequence[TimedPitchEvent],
    bundle_path: Path,
    *,
    device: torch.device,
) -> tuple[list[TechniqueDecision], dict[str, object]]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("kind") != "technique-selective-bundle-v1":
        raise ValueError("Unsupported technique bundle format")
    root = bundle_path.parent
    neural_path = (root / str(bundle["neural_checkpoint"])).resolve()
    svm_path = (root / str(bundle["svm_checkpoint"])).resolve()
    direct_gate_path = (root / str(bundle["direct_gate"])).resolve()
    ensemble_gate_path = (root / str(bundle["ensemble_gate"])).resolve()
    direct_gate = load_production_gate(direct_gate_path, neural_path)
    ensemble_gate = json.loads(
        ensemble_gate_path.read_text(encoding="utf-8")
    )
    expected_ensemble = {
        "neural": {
            "name": neural_path.name,
            "sha256": sha256_file(neural_path),
        },
        "svm": {
            "name": svm_path.name,
            "sha256": sha256_file(svm_path),
        },
    }
    if ensemble_gate.get("ensemble") != expected_ensemble:
        raise ValueError("Ensemble gate does not match bundle checkpoints")

    neural_logits, neural_metadata = neural_probabilities(
        audio_path,
        events,
        neural_path,
        device=device,
    )
    direct_neural_probabilities = torch.softmax(
        torch.from_numpy(neural_logits)
        / float(direct_gate["calibration"]["temperature"]),
        dim=-1,
    ).numpy()
    ensemble_neural_probabilities = torch.softmax(
        torch.from_numpy(neural_logits)
        / float(
            ensemble_gate["calibration"]["neural_temperature"]
        ),
        dim=-1,
    ).numpy()
    svm_probabilities_array, svm_metadata = svm_probabilities(
        audio_path,
        events,
        svm_path,
        temperature=float(
            ensemble_gate["calibration"]["svm_temperature"]
        ),
    )
    ensemble_scores = np.sqrt(
        ensemble_neural_probabilities * svm_probabilities_array
    )
    ensemble_scores /= np.maximum(
        1e-8,
        ensemble_scores.sum(axis=1, keepdims=True),
    )
    agreement = (
        ensemble_neural_probabilities.argmax(axis=1)
        == svm_probabilities_array.argmax(axis=1)
    )
    ensemble_scores[~agreement] = 0

    direct_decisions = decisions_from_probabilities(
        direct_neural_probabilities,
        direct_gate,
        source=str(neural_metadata.get("version", "technique-model")),
    )
    ensemble_decisions = decisions_from_probabilities(
        ensemble_scores,
        ensemble_gate,
        source="technique-ensemble-agreement-v1",
    )
    decisions = []
    for direct, ensemble in zip(direct_decisions, ensemble_decisions):
        accepted = [
            decision
            for decision in (direct, ensemble)
            if decision.technique != "unknown"
        ]
        labels = {decision.technique for decision in accepted}
        if len(labels) != 1:
            candidates = (
                ensemble.candidates
                if ensemble.candidates
                else direct.candidates
            )
            decisions.append(
                TechniqueDecision(
                    technique="unknown",
                    confidence=0.0,
                    source="technique-bundle-abstained",
                    evidence=["no-approved-consensus"],
                    candidates=candidates,
                )
            )
            continue
        selected = max(accepted, key=lambda decision: decision.confidence)
        decisions.append(selected)
    approved_labels = sorted(
        set(direct_gate["approved_labels"])
        | set(ensemble_gate["approved_labels"])
    )
    return decisions, {
        "name": "stringtrace-technique-selective-ensemble",
        "version": str(bundle.get("version", "unknown")),
        "kind": "trained",
        "labels": list(TECHNIQUE_LABELS),
        "approved_labels": approved_labels,
        "neural_model": neural_metadata.get("version"),
        "svm_model": svm_metadata.get("version"),
        "bundle": bundle_path.name,
    }
