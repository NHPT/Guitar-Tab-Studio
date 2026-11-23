from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .catalog import (
        CATALOG_KIND,
        canonical_json,
        validate_candidate,
    )
    from .seal import (
        IDENTITY_FIELDS,
        identity_digest,
        load_exclusion_index,
    )
except ImportError:
    from catalog import (
        CATALOG_KIND,
        canonical_json,
        validate_candidate,
    )
    from seal import (
        IDENTITY_FIELDS,
        identity_digest,
        load_exclusion_index,
    )


SCHEMA_VERSION = 1
PRIVATE_PLAN_KIND = "sealed-evaluation-blind-intake"
PUBLIC_COMMITMENT_KIND = "sealed-evaluation-blind-intake-commitment"
ACQUISITION_KIND = "sealed-evaluation-blind-acquisition"
LAYER_A_TARGET = 100
LAYER_B_TARGET = 100
LAYER_C_QUEUE_TARGET = 20
CREATOR_CAP_LAYER_A = 3
CREATOR_CAP_LAYER_B = 10
MAX_LAYER_A_ASSET_BYTES = 35 * 1024 * 1024
MAX_LAYER_A_TOTAL_BYTES = 1400 * 1024 * 1024
MINIMUM_LAYER_A_PER_PROVIDER = 40
PROVIDER_TARGET_LAYER_A = {
    "internet-archive": 50,
    "wikimedia-commons": 50,
}
CONTENT_EXCLUSION_TERMS = {
    "audiobook",
    "interview",
    "lesson",
    "podcast",
    "radio show",
    "sample pack",
    "sound effect",
    "spoken word",
    "talk show",
    "tutorial",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != CATALOG_KIND
        or not isinstance(value.get("records"), list)
    ):
        raise ValueError("Unsupported internet candidate catalog")
    candidate_ids = set()
    for candidate in value["records"]:
        if not isinstance(candidate, dict):
            raise ValueError("Catalog records must be objects")
        validate_candidate(candidate)
        candidate_id = candidate["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError(f"Duplicate candidate identity: {candidate_id}")
        candidate_ids.add(candidate_id)
    return value


def _blind_digest(salt: bytes, domain: str, value: str) -> str:
    return hmac.new(
        salt,
        f"{domain}\0{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _rank(salt: bytes, layer: str, candidate_id: str) -> str:
    return _blind_digest(salt, f"selection:{layer}", candidate_id)


def _creator_key(candidate: dict[str, Any]) -> str:
    return (
        candidate["work_identity"].get("normalized_creator")
        or "unknown"
    )


def _work_key(candidate: dict[str, Any]) -> str:
    return candidate["work_identity"]["identity_sha256"]


def _lineage_key(candidate: dict[str, Any]) -> str:
    evidence = candidate.get("content_evidence")
    if isinstance(evidence, dict):
        parent = evidence.get("parent_candidate_id")
        if isinstance(parent, str) and parent:
            return parent
    return candidate["candidate_id"]


def _content_text(candidate: dict[str, Any]) -> str:
    evidence = candidate.get("content_evidence")
    values = [
        candidate.get("title"),
        candidate.get("creator"),
        evidence,
    ]
    return json.dumps(
        values,
        sort_keys=True,
        ensure_ascii=False,
    ).casefold()


def _passes_content_screen(candidate: dict[str, Any]) -> bool:
    content = _content_text(candidate)
    if "guitar" not in content:
        return False
    return not any(term in content for term in CONTENT_EXCLUSION_TERMS)


def _asset_size(candidate: dict[str, Any]) -> int | None:
    evidence = candidate.get("content_evidence")
    if not isinstance(evidence, dict):
        return None
    value = evidence.get("file_size_bytes") or evidence.get("size_bytes")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value


def _eligible_layer_a(candidate: dict[str, Any]) -> bool:
    asset_size = _asset_size(candidate)
    return bool(
        candidate["rights_tier"] == "open-production"
        and candidate["media_type"] == "audio"
        and (candidate.get("duration_seconds") or 0) >= 30
        and candidate.get("asset_url")
        and asset_size is not None
        and asset_size <= MAX_LAYER_A_ASSET_BYTES
        and _passes_content_screen(candidate)
    )


def _eligible_layer_b(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate["rights_tier"] == "open-production"
        and candidate["media_type"] in {"score", "symbolic-score"}
        and candidate.get("score_assets")
    )


def _eligible_layer_c(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate["rights_tier"] == "open-production"
        and candidate["media_type"] == "audiovisual"
        and (candidate.get("duration_seconds") or 0) >= 30
        and candidate.get("asset_url")
        and _passes_content_screen(candidate)
    )


class _SelectionState:
    def __init__(self, *, creator_cap: int) -> None:
        self.creator_cap = creator_cap
        self.records: list[dict[str, Any]] = []
        self.candidate_ids: set[str] = set()
        self.work_keys: set[str] = set()
        self.lineage_keys: set[str] = set()
        self.creator_counts: Counter[str] = Counter()

    def add(self, candidate: dict[str, Any]) -> bool:
        candidate_id = candidate["candidate_id"]
        work_key = _work_key(candidate)
        lineage_key = _lineage_key(candidate)
        creator_key = _creator_key(candidate)
        if (
            candidate_id in self.candidate_ids
            or work_key in self.work_keys
            or lineage_key in self.lineage_keys
            or self.creator_counts[creator_key] >= self.creator_cap
        ):
            return False
        self.records.append(candidate)
        self.candidate_ids.add(candidate_id)
        self.work_keys.add(work_key)
        self.lineage_keys.add(lineage_key)
        self.creator_counts[creator_key] += 1
        return True


def _ranked(
    candidates: Iterable[dict[str, Any]],
    *,
    salt: bytes,
    layer: str,
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda value: (
            _rank(salt, layer, value["candidate_id"]),
            value["candidate_id"],
        ),
    )


def _fill(
    state: _SelectionState,
    candidates: Iterable[dict[str, Any]],
    *,
    target_count: int,
) -> None:
    for candidate in candidates:
        if len(state.records) >= target_count:
            return
        state.add(candidate)


def _select_layers(
    records: list[dict[str, Any]],
    *,
    salt: bytes,
) -> dict[str, list[dict[str, Any]]]:
    layer_c_pool = _ranked(
        (record for record in records if _eligible_layer_c(record)),
        salt=salt,
        layer="c",
    )
    layer_c_state = _SelectionState(creator_cap=CREATOR_CAP_LAYER_A)
    _fill(
        layer_c_state,
        layer_c_pool,
        target_count=LAYER_C_QUEUE_TARGET,
    )

    layer_a_state = _SelectionState(creator_cap=CREATOR_CAP_LAYER_A)
    layer_a_pool = [
        record for record in records if _eligible_layer_a(record)
    ]
    for provider, provider_target in PROVIDER_TARGET_LAYER_A.items():
        provider_records = _ranked(
            (
                record
                for record in layer_a_pool
                if record["provider"] == provider
            ),
            salt=salt,
            layer=f"a:{provider}",
        )
        current = sum(
            record["provider"] == provider
            for record in layer_a_state.records
        )
        _fill(
            layer_a_state,
            provider_records,
            target_count=len(layer_a_state.records)
            + max(0, provider_target - current),
        )
    _fill(
        layer_a_state,
        _ranked(layer_a_pool, salt=salt, layer="a:overflow"),
        target_count=LAYER_A_TARGET,
    )

    layer_b_state = _SelectionState(creator_cap=CREATOR_CAP_LAYER_B)
    layer_b_pool = _ranked(
        (record for record in records if _eligible_layer_b(record)),
        salt=salt,
        layer="b",
    )
    _fill(
        layer_b_state,
        layer_b_pool,
        target_count=LAYER_B_TARGET,
    )
    return {
        "layer_a": layer_a_state.records,
        "layer_b": layer_b_state.records,
        "layer_c_queue": layer_c_state.records,
    }


def _sealed_identities(
    candidate: dict[str, Any],
    *,
    salt: bytes,
) -> dict[str, str]:
    candidate_id = candidate["candidate_id"]
    lineage_key = _lineage_key(candidate)
    creator_key = _creator_key(candidate)
    work_key = _work_key(candidate)
    prefixes = {
        "track_id": ("track", candidate_id),
        "group_id": ("group", lineage_key),
        "performer_id": ("performer", creator_key),
        "work_id": ("work", work_key),
        "recording_id": ("recording", candidate_id),
        "session_id": ("session", lineage_key),
        "lineage_id": ("lineage", lineage_key),
    }
    return {
        field: f"blind-{prefix}-{_blind_digest(salt, field, value)[:24]}"
        for field, (prefix, value) in prefixes.items()
    }


def _private_layer_record(
    candidate: dict[str, Any],
    *,
    layer: str,
    salt: bytes,
) -> dict[str, Any]:
    blind_id = (
        f"blind-{layer}-"
        f"{_blind_digest(salt, f'blind-id:{layer}', candidate['candidate_id'])[:24]}"
    )
    result = {
        "blind_id": blind_id,
        "candidate_id": candidate["candidate_id"],
        "source_url": candidate["source_url"],
        "rights_evidence_url": candidate["rights_evidence_url"],
        "declared_license": candidate["declared_license"],
        "compliance_obligations": candidate["compliance_obligations"],
        "audit_status": "selected-pending-manual-audit",
        "qualified_for_sealed_evaluation": False,
    }
    if layer in {"a", "c"}:
        result.update(
            {
                "asset_url": candidate["asset_url"],
                "duration_seconds": candidate["duration_seconds"],
                "identities": _sealed_identities(candidate, salt=salt),
                "reference_status": "pending-onset-pitch-reference",
            }
        )
    if layer == "b":
        result.update(
            {
                "score_assets": candidate["score_assets"],
                "work_id": _sealed_identities(
                    candidate,
                    salt=salt,
                )["work_id"],
                "reference_status": "pending-score-lattice",
            }
        )
    if layer == "c":
        result["reference_status"] = "pending-physical-fingering-adjudication"
    return result


def _disposition(
    candidate: dict[str, Any],
    selected: dict[str, set[str]],
) -> list[str]:
    candidate_id = candidate["candidate_id"]
    dispositions = [
        f"selected-{layer.replace('_', '-')}"
        for layer, candidate_ids in selected.items()
        if candidate_id in candidate_ids
    ]
    if dispositions:
        return sorted(dispositions)
    if candidate["rights_tier"] == "restricted-research":
        return ["restricted-research-metadata"]
    if candidate["rights_tier"] == "discovery-only":
        return ["discovery-only-metadata"]
    if candidate["media_type"] == "audio-collection":
        return ["collection-metadata"]
    if candidate["media_type"] in {"audio", "audiovisual"}:
        if (candidate.get("duration_seconds") or 0) < 30:
            return ["ineligible-duration"]
        if not _passes_content_screen(candidate):
            return ["content-screen-rejected"]
        return ["open-audio-overflow"]
    if candidate["media_type"] in {"score", "symbolic-score"}:
        return ["open-score-overflow"]
    return ["unclassified-metadata"]


def _validate_layer_counts(
    layers: dict[str, list[dict[str, Any]]],
    *,
    unavailable_providers: set[str] | None = None,
) -> None:
    unavailable = unavailable_providers or set()
    requirements = {
        "layer_a": LAYER_A_TARGET,
        "layer_b": LAYER_B_TARGET,
        "layer_c_queue": LAYER_C_QUEUE_TARGET,
    }
    failures = {
        layer: {"actual": len(layers[layer]), "required": required}
        for layer, required in requirements.items()
        if len(layers[layer]) < required
    }
    if failures:
        detail = ", ".join(
            f"{layer}={value['actual']}<{value['required']}"
            for layer, value in sorted(failures.items())
        )
        raise ValueError(f"Blind intake coverage failed: {detail}")
    provider_counts = Counter(
        candidate["provider"] for candidate in layers["layer_a"]
    )
    missing_provider_diversity = {
        provider: provider_counts[provider]
        for provider in PROVIDER_TARGET_LAYER_A
        if provider not in unavailable
        if provider_counts[provider] < MINIMUM_LAYER_A_PER_PROVIDER
    }
    if missing_provider_diversity:
        detail = ", ".join(
            f"{provider}={count}<{MINIMUM_LAYER_A_PER_PROVIDER}"
            for provider, count in sorted(
                missing_provider_diversity.items()
            )
        )
        raise ValueError(f"Layer A provider diversity failed: {detail}")
    layer_a_bytes = sum(
        _asset_size(candidate) or 0
        for candidate in layers["layer_a"]
    )
    if layer_a_bytes > MAX_LAYER_A_TOTAL_BYTES:
        raise ValueError(
            "Layer A download budget exceeded: "
            f"{layer_a_bytes}>{MAX_LAYER_A_TOTAL_BYTES}"
        )


def _validate_exclusion(
    layers: dict[str, list[dict[str, Any]]],
    exclusion_index: dict[str, Any],
    *,
    salt: bytes,
) -> None:
    excluded = {
        field: set(exclusion_index["identity_digests"][field])
        for field in IDENTITY_FIELDS
    }
    collisions: Counter[str] = Counter()
    checked: set[str] = set()
    for candidate in layers["layer_a"]:
        if candidate["candidate_id"] in checked:
            continue
        checked.add(candidate["candidate_id"])
        identities = _sealed_identities(candidate, salt=salt)
        for field, value in identities.items():
            if identity_digest(field, value) in excluded[field]:
                collisions[field] += 1
    if collisions:
        detail = ", ".join(
            f"{field}={count}"
            for field, count in sorted(collisions.items())
        )
        raise ValueError(f"Blind intake overlaps prior data: {detail}")


def build_private_plan(
    catalog: dict[str, Any],
    exclusion_index: dict[str, Any],
    *,
    catalog_name: str,
    catalog_sha256: str,
    exclusion_name: str,
    exclusion_sha256: str,
    salt: bytes,
) -> dict[str, Any]:
    if len(salt) < 32:
        raise ValueError("Blind selection salt must contain at least 32 bytes")
    records = list(catalog["records"])
    layers = _select_layers(records, salt=salt)
    _validate_layer_counts(layers)
    _validate_exclusion(layers, exclusion_index, salt=salt)
    selected_ids = {
        layer: {candidate["candidate_id"] for candidate in candidates}
        for layer, candidates in layers.items()
    }
    assignments = [
        {
            "candidate_id": candidate["candidate_id"],
            "dispositions": _disposition(candidate, selected_ids),
        }
        for candidate in sorted(
            records,
            key=lambda value: value["candidate_id"],
        )
    ]
    private_layers = {
        "layer_a": [
            _private_layer_record(candidate, layer="a", salt=salt)
            for candidate in layers["layer_a"]
        ],
        "layer_b": [
            _private_layer_record(candidate, layer="b", salt=salt)
            for candidate in layers["layer_b"]
        ],
        "layer_c_queue": [
            _private_layer_record(candidate, layer="c", salt=salt)
            for candidate in layers["layer_c_queue"]
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_PLAN_KIND,
        "catalog": {
            "name": catalog_name,
            "sha256": catalog_sha256,
        },
        "exclusion_index": {
            "name": exclusion_name,
            "sha256": exclusion_sha256,
        },
        "selection_salt_hex": salt.hex(),
        "layers": private_layers,
        "assignments": assignments,
        "admission_status": "intake-only",
        "admitted_recordings": 0,
    }


def _load_private_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != PRIVATE_PLAN_KIND
        or not isinstance(value.get("layers"), dict)
    ):
        raise ValueError("Unsupported blind intake plan")
    return value


def _load_acquisition(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != ACQUISITION_KIND
        or not isinstance(value.get("layer_a"), dict)
        or not isinstance(value.get("layer_b"), dict)
    ):
        raise ValueError("Unsupported blind acquisition manifest")
    return value


def _successful_candidate_ids(
    acquisition: dict[str, Any],
    *,
    layer: str,
) -> set[str]:
    records = acquisition.get(layer, {}).get("records")
    if not isinstance(records, list):
        raise ValueError(f"Acquisition {layer} records are missing")
    candidate_ids = {
        record.get("candidate_id")
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("candidate_id"), str)
    }
    if len(candidate_ids) != len(records):
        raise ValueError(f"Acquisition {layer} identities are invalid")
    return candidate_ids


def _select_recovery_layers(
    records: list[dict[str, Any]],
    previous_plan: dict[str, Any],
    acquisition: dict[str, Any],
    *,
    unavailable_providers: set[str],
    salt: bytes,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    catalog_ids = {
        candidate["candidate_id"]: candidate for candidate in records
    }
    layer_specs = {
        "layer_a": (
            _SelectionState(creator_cap=CREATOR_CAP_LAYER_A),
            _eligible_layer_a,
            LAYER_A_TARGET,
            "a:recovery",
        ),
        "layer_b": (
            _SelectionState(creator_cap=CREATOR_CAP_LAYER_B),
            _eligible_layer_b,
            LAYER_B_TARGET,
            "b:recovery",
        ),
    }
    layers: dict[str, list[dict[str, Any]]] = {}
    retained: dict[str, int] = {}
    for layer, (
        state,
        eligibility,
        target,
        rank_domain,
    ) in layer_specs.items():
        successful_ids = _successful_candidate_ids(
            acquisition,
            layer=layer,
        )
        previous_records = previous_plan["layers"].get(layer)
        if not isinstance(previous_records, list):
            raise ValueError(f"Previous plan {layer} is missing")
        previous_ids = {
            record.get("candidate_id")
            for record in previous_records
            if isinstance(record, dict)
        }
        if not successful_ids.issubset(previous_ids):
            raise ValueError(
                f"Acquisition {layer} is not a subset of the previous plan"
            )
        rejected_ids = previous_ids - successful_ids
        for previous_record in previous_records:
            candidate_id = previous_record.get("candidate_id")
            if candidate_id not in successful_ids:
                continue
            candidate = catalog_ids.get(candidate_id)
            if candidate is None or not eligibility(candidate):
                raise ValueError(
                    f"Retained recovery candidate is no longer eligible: "
                    f"{candidate_id}"
                )
            if not state.add(candidate):
                raise ValueError(
                    f"Retained recovery candidate violates diversity: "
                    f"{candidate_id}"
                )
        retained[layer] = len(state.records)
        pool = _ranked(
            (
                candidate
                for candidate in records
                if eligibility(candidate)
                and candidate["candidate_id"] not in rejected_ids
                and candidate["provider"] not in unavailable_providers
            ),
            salt=salt,
            layer=rank_domain,
        )
        _fill(state, pool, target_count=target)
        layers[layer] = state.records

    previous_layer_c = previous_plan["layers"].get("layer_c_queue")
    if not isinstance(previous_layer_c, list):
        raise ValueError("Previous plan layer_c_queue is missing")
    layer_c = []
    for previous_record in previous_layer_c:
        candidate_id = previous_record.get("candidate_id")
        candidate = catalog_ids.get(candidate_id)
        if candidate is None or not _eligible_layer_c(candidate):
            raise ValueError(
                f"Previous Layer C candidate is no longer eligible: "
                f"{candidate_id}"
            )
        layer_c.append(candidate)
    layers["layer_c_queue"] = layer_c
    retained["layer_c_queue"] = len(layer_c)
    return layers, retained


def build_recovered_private_plan(
    catalog: dict[str, Any],
    exclusion_index: dict[str, Any],
    previous_plan: dict[str, Any],
    acquisition: dict[str, Any],
    *,
    catalog_name: str,
    catalog_sha256: str,
    exclusion_name: str,
    exclusion_sha256: str,
    previous_plan_name: str,
    previous_plan_sha256: str,
    acquisition_name: str,
    acquisition_sha256: str,
    unavailable_providers: set[str],
) -> dict[str, Any]:
    salt = bytes.fromhex(previous_plan["selection_salt_hex"])
    if len(salt) < 32:
        raise ValueError("Previous blind selection salt is invalid")
    records = list(catalog["records"])
    layers, retained = _select_recovery_layers(
        records,
        previous_plan,
        acquisition,
        unavailable_providers=unavailable_providers,
        salt=salt,
    )
    _validate_layer_counts(
        layers,
        unavailable_providers=unavailable_providers,
    )
    _validate_exclusion(layers, exclusion_index, salt=salt)
    selected_ids = {
        layer: {candidate["candidate_id"] for candidate in candidates}
        for layer, candidates in layers.items()
    }
    assignments = [
        {
            "candidate_id": candidate["candidate_id"],
            "dispositions": _disposition(candidate, selected_ids),
        }
        for candidate in sorted(
            records,
            key=lambda value: value["candidate_id"],
        )
    ]
    private_layers = {
        "layer_a": [
            _private_layer_record(candidate, layer="a", salt=salt)
            for candidate in layers["layer_a"]
        ],
        "layer_b": [
            _private_layer_record(candidate, layer="b", salt=salt)
            for candidate in layers["layer_b"]
        ],
        "layer_c_queue": [
            _private_layer_record(candidate, layer="c", salt=salt)
            for candidate in layers["layer_c_queue"]
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_PLAN_KIND,
        "catalog": {
            "name": catalog_name,
            "sha256": catalog_sha256,
        },
        "exclusion_index": {
            "name": exclusion_name,
            "sha256": exclusion_sha256,
        },
        "selection_salt_hex": salt.hex(),
        "layers": private_layers,
        "assignments": assignments,
        "recovery": {
            "previous_private_plan": {
                "name": previous_plan_name,
                "sha256": previous_plan_sha256,
            },
            "previous_acquisition": {
                "name": acquisition_name,
                "sha256": acquisition_sha256,
            },
            "unavailable_providers": sorted(unavailable_providers),
            "retained": retained,
            "replacements": {
                layer: len(private_layers[layer]) - retained[layer]
                for layer in private_layers
            },
            "reason": "external-host-acquisition-unavailable",
        },
        "admission_status": "intake-only",
        "admitted_recordings": 0,
    }


def build_public_commitment(
    private_plan: dict[str, Any],
    *,
    private_plan_sha256: str,
) -> dict[str, Any]:
    layers = private_plan["layers"]
    layer_a = layers["layer_a"]
    layer_b = layers["layer_b"]
    layer_c = layers["layer_c_queue"]
    identity_rows = sorted(
        {
            record["blind_id"]
            for records in layers.values()
            for record in records
        }
    )
    disposition_counts = Counter(
        disposition
        for assignment in private_plan["assignments"]
        for disposition in assignment["dispositions"]
    )
    commitment = {
        "schema_version": SCHEMA_VERSION,
        "kind": PUBLIC_COMMITMENT_KIND,
        "catalog": private_plan["catalog"],
        "exclusion_index": private_plan["exclusion_index"],
        "private_plan_sha256": private_plan_sha256,
        "selection_identity_commitment_sha256": hashlib.sha256(
            canonical_json(identity_rows)
        ).hexdigest(),
        "layers": {
            "layer_a": {
                "target": LAYER_A_TARGET,
                "selected": len(layer_a),
                "providers": dict(
                    sorted(
                        Counter(
                            record["candidate_id"].split(":", 1)[0]
                            for record in layer_a
                        ).items()
                    )
                ),
                "status": "selected-pending-manual-audit-and-reference",
            },
            "layer_b": {
                "target": LAYER_B_TARGET,
                "selected": len(layer_b),
                "status": "selected-pending-score-lattice",
            },
            "layer_c": {
                "queue_target": LAYER_C_QUEUE_TARGET,
                "queued": len(layer_c),
                "gold_admitted": 0,
                "status": "pending-physical-fingering-adjudication",
            },
        },
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "admission_status": "intake-only",
        "admitted_recordings": 0,
        "disclosure": "aggregate-only",
    }
    recovery = private_plan.get("recovery")
    if isinstance(recovery, dict):
        commitment["recovery"] = {
            "previous_private_plan": recovery["previous_private_plan"],
            "previous_acquisition": recovery["previous_acquisition"],
            "unavailable_providers": recovery["unavailable_providers"],
            "retained": recovery["retained"],
            "replacements": recovery["replacements"],
            "reason": recovery["reason"],
        }
    rendered = canonical_json(commitment)
    forbidden = [
        private_plan["selection_salt_hex"],
        *(
            record["candidate_id"]
            for records in layers.values()
            for record in records
        ),
    ]
    if any(value.encode("utf-8") in rendered for value in forbidden):
        raise ValueError("Public commitment discloses private selection data")
    return commitment


def _existing_or_new_salt(path: Path) -> bytes:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        rendered = value.get("selection_salt_hex")
        if isinstance(rendered, str):
            salt = bytes.fromhex(rendered)
            if len(salt) >= 32:
                return salt
    return secrets.token_bytes(32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select internet candidates into a private blind intake plan and "
            "publish an aggregate-only commitment."
        ),
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--exclusion-index", type=Path, required=True)
    parser.add_argument("--private-plan", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--previous-private-plan", type=Path)
    parser.add_argument("--previous-acquisition", type=Path)
    parser.add_argument(
        "--unavailable-provider",
        action="append",
        default=[],
    )
    arguments = parser.parse_args()

    catalog_path = arguments.catalog.resolve()
    exclusion_path = arguments.exclusion_index.resolve()
    private_path = arguments.private_plan.resolve()
    commitment_path = arguments.commitment.resolve()
    previous_arguments = (
        arguments.previous_private_plan,
        arguments.previous_acquisition,
    )
    if any(previous_arguments) and not all(previous_arguments):
        raise ValueError(
            "Recovery requires both --previous-private-plan and "
            "--previous-acquisition"
        )
    if all(previous_arguments):
        previous_path = arguments.previous_private_plan.resolve()
        acquisition_path = arguments.previous_acquisition.resolve()
        previous_plan = _load_private_plan(previous_path)
        acquisition = _load_acquisition(acquisition_path)
        if acquisition["private_plan"]["sha256"] != sha256_file(
            previous_path
        ):
            raise ValueError(
                "Recovery acquisition private plan SHA-256 mismatch"
            )
        private_plan = build_recovered_private_plan(
            load_catalog(catalog_path),
            load_exclusion_index(exclusion_path),
            previous_plan,
            acquisition,
            catalog_name=catalog_path.name,
            catalog_sha256=sha256_file(catalog_path),
            exclusion_name=exclusion_path.name,
            exclusion_sha256=sha256_file(exclusion_path),
            previous_plan_name=previous_path.name,
            previous_plan_sha256=sha256_file(previous_path),
            acquisition_name=acquisition_path.name,
            acquisition_sha256=sha256_file(acquisition_path),
            unavailable_providers=set(arguments.unavailable_provider),
        )
    else:
        salt = _existing_or_new_salt(private_path)
        private_plan = build_private_plan(
            load_catalog(catalog_path),
            load_exclusion_index(exclusion_path),
            catalog_name=catalog_path.name,
            catalog_sha256=sha256_file(catalog_path),
            exclusion_name=exclusion_path.name,
            exclusion_sha256=sha256_file(exclusion_path),
            salt=salt,
        )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(canonical_json(private_plan) + b"\n")
    commitment = build_public_commitment(
        private_plan,
        private_plan_sha256=sha256_file(private_path),
    )
    commitment_path.parent.mkdir(parents=True, exist_ok=True)
    commitment_path.write_bytes(canonical_json(commitment) + b"\n")
    print(
        json.dumps(
            {
                "private_plan": private_path.name,
                "private_plan_sha256": sha256_file(private_path),
                "commitment": commitment_path.name,
                "commitment_sha256": sha256_file(commitment_path),
                "layers": commitment["layers"],
                "admitted_recordings": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
