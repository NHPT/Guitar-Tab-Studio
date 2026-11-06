from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .data_sources import (
    is_absolute_local_path,
    load_source_ledger,
    sha256_file,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and is_absolute_local_path(value)


def _training_identities(
    value: object,
    *,
    location: str = "checkpoint",
) -> list[tuple[str, str, dict[str, Any]]]:
    identities: list[tuple[str, str, dict[str, Any]]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_location = f"{location}.{key}"
            if key in {"training_data", "validation_data"} and isinstance(
                item,
                dict,
            ):
                identities.append((key, child_location, dict(item)))
            else:
                identities.extend(
                    _training_identities(
                        item,
                        location=child_location,
                    )
                )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            identities.extend(
                _training_identities(
                    item,
                    location=f"{location}[{index}]",
                )
            )
    return identities


def evaluate_checkpoint_governance(
    checkpoint: dict[str, Any],
    *,
    release_channel: str,
) -> dict[str, object]:
    if release_channel not in {"research", "production"}:
        raise ValueError("release_channel must be research or production")
    ledger = load_source_ledger()
    identities = _training_identities(checkpoint)
    training_identities = [
        identity
        for kind, _location, identity in identities
        if kind == "training_data"
    ]
    violations: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    if _contains_absolute_path(checkpoint):
        violations["absolute_local_path_embedded"] += 1
    if not training_identities:
        violations["missing_training_data_identity"] += 1

    for _kind, _location, identity in identities:
        manifest_sha256 = str(identity.get("sha256", ""))
        if not SHA256_PATTERN.fullmatch(manifest_sha256):
            violations["invalid_manifest_sha256"] += 1
        recorded_ledger = identity.get("source_ledger")
        if not isinstance(recorded_ledger, dict):
            violations["missing_source_ledger_identity"] += 1
        else:
            if recorded_ledger.get("sha256") != ledger.sha256:
                violations["stale_source_ledger_sha256"] += 1
            if recorded_ledger.get("version") != ledger.ledger_version:
                violations["stale_source_ledger_version"] += 1
        sources = identity.get("sources")
        if not isinstance(sources, dict) or not sources:
            violations["missing_source_counts"] += 1
            continue
        records = identity.get("records")
        records_valid = (
            isinstance(records, int)
            and not isinstance(records, bool)
            and records > 0
        )
        if not records_valid:
            violations["invalid_record_count"] += 1
        source_total = 0
        source_counts_valid = True
        for source_id, count in sources.items():
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
            ):
                violations["invalid_source_count"] += 1
                source_counts_valid = False
                continue
            source_counts[str(source_id)] += count
            source_total += count
            policy = ledger.policies.get(str(source_id))
            if policy is None:
                violations[f"unknown_source:{source_id}"] += 1
                continue
            if policy.verification_status not in {"verified", "internal"}:
                violations[
                    f"source_not_verified:{source_id}:"
                    f"{policy.verification_status}"
                ] += 1
            required_purpose = (
                "model-release"
                if release_channel == "production"
                else "training"
            )
            if required_purpose not in policy.allowed_purposes:
                violations[
                    f"source_not_allowed:{source_id}:{required_purpose}"
                ] += 1
        if (
            records_valid
            and source_counts_valid
            and source_total != records
        ):
            violations["source_count_total_mismatch"] += 1

    checks = {
        "no_embedded_absolute_paths": (
            "absolute_local_path_embedded" not in violations
        ),
        "training_data_identity_present": bool(training_identities),
        "data_record_counts_valid": not any(
            reason
            in {
                "invalid_record_count",
                "invalid_source_count",
                "source_count_total_mismatch",
            }
            for reason in violations
        ),
        "source_ledger_current": not any(
            reason.startswith("stale_source_ledger")
            or reason == "missing_source_ledger_identity"
            for reason in violations
        ),
        "sources_registered_and_verified": not any(
            reason.startswith("unknown_source:")
            or reason.startswith("source_not_verified:")
            for reason in violations
        ),
        "release_rights_allowed": not any(
            reason.startswith("source_not_allowed:")
            for reason in violations
        ),
        "manifest_hashes_valid": (
            "invalid_manifest_sha256" not in violations
        ),
    }
    return {
        "schema_version": 1,
        "release_channel": release_channel,
        "passed": not violations and all(checks.values()),
        "checks": checks,
        "violations": dict(sorted(violations.items())),
        "training_data_identities": len(training_identities),
        "validation_data_identities": sum(
            kind == "validation_data"
            for kind, _location, _identity in identities
        ),
        "sources": dict(sorted(source_counts.items())),
        "source_ledger": {
            "version": ledger.ledger_version,
            "sha256": ledger.sha256,
        },
    }


def load_checkpoint_file(checkpoint_path: Path) -> dict[str, Any]:
    with checkpoint_path.open("rb") as stream:
        prefix = stream.read(4)
    if prefix == b"PK\x03\x04":
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    else:
        with checkpoint_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Block model release when a checkpoint lacks current, verified "
            "training-source identities or embeds local paths."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--release-channel",
        choices=("research", "production"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    checkpoint_path = arguments.checkpoint.resolve()
    checkpoint = load_checkpoint_file(checkpoint_path)
    result = evaluate_checkpoint_governance(
        checkpoint,
        release_channel=arguments.release_channel,
    )
    result["checkpoint"] = {
        "name": checkpoint_path.name,
        "sha256": sha256_file(checkpoint_path),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
