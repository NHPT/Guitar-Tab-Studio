from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data_sources import sha256_file
from .schema import TrackAnnotation, load_manifest


SCOPED_PROVENANCE_FIELDS = {
    "performer": ("performer_id", "player_id", "player"),
    "work": ("work_id", "piece_id", "song_id"),
    "recording": ("recording_id",),
    "session": ("session_id",),
    "lineage": ("lineage_id",),
}


def _source_scope(annotation: TrackAnnotation) -> str:
    return str(
        annotation.provenance.get("source_id")
        or annotation.provenance.get("dataset")
        or "unknown"
    )


def manifest_identity_sets(
    annotations: list[TrackAnnotation],
) -> dict[str, set[str]]:
    identities: dict[str, set[str]] = defaultdict(set)
    for annotation in annotations:
        identities["track_id"].add(annotation.track_id)
        identities["group_id"].add(annotation.group_id)
        provenance = annotation.provenance
        for key, value in provenance.items():
            if (
                isinstance(value, str)
                and value
                and (
                    key == "audio_sha256"
                    or key.endswith("_audio_sha256")
                )
            ):
                identities["audio_sha256"].add(value)
        source_scope = _source_scope(annotation)
        for identity_name, fields in SCOPED_PROVENANCE_FIELDS.items():
            for field in fields:
                value = provenance.get(field)
                if value is not None and str(value).strip():
                    identities[identity_name].add(
                        f"{source_scope}:{str(value).strip()}"
                    )
    return dict(identities)


def audit_manifest_isolation(
    reference_annotations: list[TrackAnnotation],
    candidate_annotations: list[TrackAnnotation],
    *,
    example_limit: int = 20,
) -> dict[str, object]:
    if example_limit <= 0:
        raise ValueError("overlap example limit must be positive")
    reference = manifest_identity_sets(reference_annotations)
    candidate = manifest_identity_sets(candidate_annotations)
    identity_names = sorted(set(reference) | set(candidate))
    overlaps: dict[str, dict[str, object]] = {}
    for identity_name in identity_names:
        shared = sorted(
            reference.get(identity_name, set())
            & candidate.get(identity_name, set())
        )
        overlaps[identity_name] = {
            "count": len(shared),
            "examples": shared[:example_limit],
        }
    return {
        "passed": not any(
            int(result["count"]) > 0 for result in overlaps.values()
        ),
        "reference_records": len(reference_annotations),
        "candidate_records": len(candidate_annotations),
        "overlaps": overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed when evaluation manifests overlap a governed "
            "training or model-selection manifest."
        )
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.example_limit <= 0:
        parser.error("--example-limit must be positive")

    reference_path = arguments.reference_manifest.resolve()
    reference_annotations = load_manifest(reference_path)
    candidate_reports: list[dict[str, object]] = []
    for candidate_manifest in arguments.candidate_manifest:
        candidate_path = candidate_manifest.resolve()
        isolation = audit_manifest_isolation(
            reference_annotations,
            load_manifest(candidate_path),
            example_limit=arguments.example_limit,
        )
        candidate_reports.append(
            {
                "manifest": {
                    "name": candidate_path.name,
                    "sha256": sha256_file(candidate_path),
                },
                **isolation,
            }
        )

    report = {
        "schema_version": 1,
        "kind": "stringtrace-manifest-isolation-audit",
        "reference_manifest": {
            "name": reference_path.name,
            "sha256": sha256_file(reference_path),
            "records": len(reference_annotations),
        },
        "candidates": candidate_reports,
        "passed": all(
            bool(candidate["passed"]) for candidate in candidate_reports
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "passed": report["passed"],
                "candidates": {
                    candidate["manifest"]["name"]: candidate["passed"]
                    for candidate in candidate_reports
                },
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
