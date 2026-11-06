from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


DEFAULT_LEDGER_PATH = Path(__file__).with_name("source_ledger.json")
ALLOWED_VERIFICATION_STATUSES = {
    "verified",
    "internal",
    "conditional",
    "quarantined",
}
ALLOWED_PURPOSES = {
    "training",
    "evaluation",
    "model-release",
    "quarantine-audit",
}
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class ManifestRecord(Protocol):
    track_id: str
    audio: str
    provenance: dict[str, object]


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    name: str
    version: str
    aliases: tuple[str, ...]
    source_url: str
    license_spdx: str
    license_url: str | None
    license_required_in_manifest: bool
    verification_status: str
    allowed_purposes: frozenset[str]
    rights: dict[str, bool]
    hash_policy: dict[str, str]
    required_provenance: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class SourceLedger:
    schema_version: int
    ledger_version: str
    sha256: str
    policies: dict[str, SourcePolicy]
    aliases: dict[str, str]

    def resolve(self, provenance: dict[str, object]) -> SourcePolicy:
        declared_source_id = str(provenance.get("source_id", "")).strip()
        dataset = str(provenance.get("dataset", "")).strip()
        source_id = declared_source_id or self.aliases.get(dataset, "")
        if not source_id:
            raise ValueError(
                f"Unregistered dataset source: {dataset or 'missing'}"
            )
        policy = self.policies.get(source_id)
        if policy is None:
            raise ValueError(f"Unknown source_id: {source_id}")
        alias_source_id = self.aliases.get(dataset)
        if alias_source_id is not None and alias_source_id != policy.source_id:
            raise ValueError(
                f"Dataset alias {dataset!r} conflicts with source_id "
                f"{policy.source_id!r}"
            )
        return policy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _required_string(value: object, name: str) -> str:
    if value is None:
        raise ValueError(f"Source ledger field {name} is required")
    result = str(value).strip()
    if not result:
        raise ValueError(f"Source ledger field {name} is required")
    return result


def load_source_ledger(path: Path | None = None) -> SourceLedger:
    resolved_path = (path or DEFAULT_LEDGER_PATH).resolve()
    document = json.loads(resolved_path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported source ledger schema")
    raw_sources = document.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Source ledger must contain sources")

    policies: dict[str, SourcePolicy] = {}
    aliases: dict[str, str] = {}
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise ValueError("Source ledger entries must be objects")
        source_id = _required_string(raw_source.get("source_id"), "source_id")
        if source_id in policies:
            raise ValueError(f"Duplicate source_id: {source_id}")
        license_value = raw_source.get("license")
        verification = raw_source.get("verification")
        rights = raw_source.get("rights")
        hash_policy = raw_source.get("hash_policy")
        if not all(
            isinstance(value, dict)
            for value in (license_value, verification, rights, hash_policy)
        ):
            raise ValueError(
                f"Source {source_id} has invalid policy sections"
            )
        status = _required_string(
            verification.get("status"),
            f"{source_id}.verification.status",
        )
        if status not in ALLOWED_VERIFICATION_STATUSES:
            raise ValueError(
                f"Source {source_id} has unknown verification status {status}"
            )
        purposes = frozenset(
            str(value)
            for value in raw_source.get("allowed_purposes", [])
        )
        if not purposes <= ALLOWED_PURPOSES:
            unknown = ", ".join(sorted(purposes - ALLOWED_PURPOSES))
            raise ValueError(
                f"Source {source_id} has unknown purposes: {unknown}"
            )
        source_aliases = tuple(
            _required_string(value, f"{source_id}.aliases")
            for value in raw_source.get("aliases", [])
        )
        for alias in source_aliases:
            existing = aliases.setdefault(alias, source_id)
            if existing != source_id:
                raise ValueError(f"Duplicate dataset alias: {alias}")
        policy = SourcePolicy(
            source_id=source_id,
            name=_required_string(raw_source.get("name"), f"{source_id}.name"),
            version=_required_string(
                raw_source.get("version"),
                f"{source_id}.version",
            ),
            aliases=source_aliases,
            source_url=_required_string(
                raw_source.get("source_url"),
                f"{source_id}.source_url",
            ),
            license_spdx=_required_string(
                license_value.get("spdx"),
                f"{source_id}.license.spdx",
            ),
            license_url=(
                str(license_value["url"])
                if license_value.get("url") is not None
                else None
            ),
            license_required_in_manifest=bool(
                license_value.get("required_in_manifest", True)
            ),
            verification_status=status,
            allowed_purposes=purposes,
            rights={
                str(key): bool(value)
                for key, value in rights.items()
            },
            hash_policy={
                str(key): str(value)
                for key, value in hash_policy.items()
            },
            required_provenance={
                str(key): tuple(str(item) for item in value)
                for key, value in dict(
                    raw_source.get("required_provenance", {})
                ).items()
                if isinstance(value, list)
            },
        )
        policies[source_id] = policy

    return SourceLedger(
        schema_version=1,
        ledger_version=_required_string(
            document.get("ledger_version"),
            "ledger_version",
        ),
        sha256=sha256_file(resolved_path),
        policies=policies,
        aliases=aliases,
    )


def is_absolute_local_path(value: str) -> bool:
    return Path(value).is_absolute() or bool(
        WINDOWS_ABSOLUTE_PATH.match(value)
    )


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and is_absolute_local_path(value)


def audit_records(
    records: Iterable[ManifestRecord],
    *,
    purpose: str,
    ledger: SourceLedger | None = None,
    require_source_id: bool = True,
    require_portable_paths: bool = True,
) -> dict[str, object]:
    if purpose not in ALLOWED_PURPOSES:
        raise ValueError(f"Unknown audit purpose: {purpose}")
    source_ledger = ledger or load_source_ledger()
    source_counts: Counter[str] = Counter()
    violation_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    record_count = 0

    for record in records:
        record_count += 1
        provenance = record.provenance
        try:
            policy = source_ledger.resolve(provenance)
        except ValueError:
            violation_counts["unregistered_source"] += 1
            continue
        source_counts[policy.source_id] += 1
        if provenance.get("source_id") != policy.source_id:
            target = (
                violation_counts
                if require_source_id
                else warning_counts
            )
            target["missing_or_mismatched_source_id"] += 1
        if purpose not in policy.allowed_purposes:
            violation_counts[
                f"purpose_not_allowed:{policy.source_id}:{purpose}"
            ] += 1
        if (
            purpose != "quarantine-audit"
            and policy.verification_status not in {"verified", "internal"}
        ):
            violation_counts[
                f"source_not_verified:{policy.source_id}:"
                f"{policy.verification_status}"
            ] += 1
        declared_license = provenance.get("license")
        if (
            policy.license_required_in_manifest
            and declared_license != policy.license_spdx
        ):
            violation_counts[
                f"license_mismatch:{policy.source_id}"
            ] += 1
        for key, allowed_values in policy.required_provenance.items():
            if str(provenance.get(key, "")) not in allowed_values:
                violation_counts[
                    f"provenance_requirement:{policy.source_id}:{key}"
                ] += 1
        if require_portable_paths:
            if is_absolute_local_path(record.audio):
                violation_counts["absolute_audio_path"] += 1
            if _contains_absolute_path(provenance):
                violation_counts["absolute_provenance_path"] += 1

    return {
        "schema_version": 1,
        "purpose": purpose,
        "ledger": {
            "version": source_ledger.ledger_version,
            "sha256": source_ledger.sha256,
        },
        "records": record_count,
        "sources": dict(sorted(source_counts.items())),
        "violations": dict(sorted(violation_counts.items())),
        "warnings": dict(sorted(warning_counts.items())),
        "passed": not violation_counts,
    }


def enforce_records(
    records: Iterable[ManifestRecord],
    *,
    purpose: str,
    ledger: SourceLedger | None = None,
    require_source_id: bool = True,
    require_portable_paths: bool = True,
) -> dict[str, object]:
    report = audit_records(
        records,
        purpose=purpose,
        ledger=ledger,
        require_source_id=require_source_id,
        require_portable_paths=require_portable_paths,
    )
    if not report["passed"]:
        rendered = ", ".join(
            f"{reason}={count}"
            for reason, count in report["violations"].items()
        )
        raise ValueError(
            f"Data source policy rejected {purpose} manifest: {rendered}"
        )
    return report


def source_provenance(
    provenance: dict[str, object],
    policy: SourcePolicy,
) -> dict[str, object]:
    result = dict(provenance)
    result["source_id"] = policy.source_id
    result["source_version"] = policy.version
    result["license"] = policy.license_spdx
    if policy.license_url is not None:
        result["license_url"] = policy.license_url
    result["source_verification_status"] = policy.verification_status
    return result


def manifest_identity(
    path: Path,
    records: Iterable[ManifestRecord],
    *,
    purpose: str,
    ledger: SourceLedger | None = None,
    require_source_id: bool = True,
    require_portable_paths: bool = True,
) -> dict[str, object]:
    source_ledger = ledger or load_source_ledger()
    values = list(records)
    audit = enforce_records(
        values,
        purpose=purpose,
        ledger=source_ledger,
        require_source_id=require_source_id,
        require_portable_paths=require_portable_paths,
    )
    return {
        "name": path.name,
        "sha256": sha256_file(path),
        "records": len(values),
        "sources": audit["sources"],
        "source_ledger": audit["ledger"],
    }


def group_identity(group_ids: Iterable[str]) -> dict[str, object]:
    values = sorted(set(group_ids))
    return {
        "count": len(values),
        "sha256": sha256_values(values),
    }


def main() -> None:
    from .schema import load_manifest
    from .technique_schema import load_technique_manifest

    parser = argparse.ArgumentParser(
        description=(
            "Audit a StringTrace manifest against the tracked source-license "
            "ledger without exposing record paths."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("position", "technique"),
        default="position",
    )
    parser.add_argument(
        "--purpose",
        choices=tuple(sorted(ALLOWED_PURPOSES)),
        required=True,
    )
    parser.add_argument("--ledger", type=Path)
    parser.add_argument(
        "--allow-legacy-alias",
        action="store_true",
        help="Resolve legacy dataset aliases while reporting a warning.",
    )
    parser.add_argument(
        "--allow-absolute-paths",
        action="store_true",
        help="Audit source rights without enforcing portable manifest paths.",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    manifest_path = arguments.manifest.resolve()
    records = (
        load_manifest(manifest_path)
        if arguments.kind == "position"
        else load_technique_manifest(manifest_path)
    )
    report = audit_records(
        records,
        purpose=arguments.purpose,
        ledger=load_source_ledger(arguments.ledger),
        require_source_id=not arguments.allow_legacy_alias,
        require_portable_paths=not arguments.allow_absolute_paths,
    )
    report["manifest"] = {
        "name": manifest_path.name,
        "sha256": sha256_file(manifest_path),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
