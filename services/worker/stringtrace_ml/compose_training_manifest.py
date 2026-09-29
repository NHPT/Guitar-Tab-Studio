from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from .prepare_guitarset_stems import sha256_file
from .schema import TrackAnnotation, load_manifest, write_manifest


def compose_training_manifests(
    manifest_paths: list[Path],
    output_path: Path,
) -> tuple[list[TrackAnnotation], dict[str, object]]:
    if not manifest_paths:
        raise ValueError("At least one source manifest is required")

    output_path = output_path.resolve()
    annotations: list[TrackAnnotation] = []
    seen_track_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    sources: list[dict[str, object]] = []

    for manifest_path in manifest_paths:
        manifest_path = manifest_path.resolve()
        source_annotations = load_manifest(manifest_path)
        source_splits = Counter(
            annotation.split for annotation in source_annotations
        )
        if source_splits.get("test", 0):
            raise ValueError(
                f"Training composition rejects test split: {manifest_path.name}"
            )
        sources.append(
            {
                "path": os.path.relpath(manifest_path, output_path.parent),
                "sha256": sha256_file(manifest_path),
                "tracks": len(source_annotations),
                "splits": dict(sorted(source_splits.items())),
            }
        )
        for annotation in source_annotations:
            if annotation.track_id in seen_track_ids:
                raise ValueError(
                    f"Duplicate track_id across manifests: {annotation.track_id}"
                )
            seen_track_ids.add(annotation.track_id)
            group_splits[annotation.group_id].add(annotation.split)
            relative_audio = os.path.relpath(
                annotation.resolve_audio(manifest_path),
                output_path.parent,
            )
            annotations.append(
                replace(annotation, audio=relative_audio)
            )

    leaking_groups = sorted(
        group_id
        for group_id, splits in group_splits.items()
        if len(splits) > 1
    )
    if leaking_groups:
        raise ValueError(
            "Groups cross dataset splits: " + ", ".join(leaking_groups[:5])
        )

    annotations.sort(
        key=lambda annotation: (annotation.split, annotation.track_id)
    )
    write_manifest(output_path, annotations)
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "training-manifest-composition",
        "sources": sources,
        "output_manifest": {
            "name": output_path.name,
            "sha256": sha256_file(output_path),
        },
        "tracks": len(annotations),
        "splits": dict(
            sorted(Counter(a.split for a in annotations).items())
        ),
        "groups": len(group_splits),
        "datasets": dict(
            sorted(
                Counter(
                    str(a.provenance.get("dataset", "unknown"))
                    for a in annotations
                ).items()
            )
        ),
        "audio_conditions": dict(
            sorted(
                Counter(
                    str(
                        a.provenance.get(
                            "audio_condition",
                            "source",
                        )
                    )
                    for a in annotations
                ).items()
            )
        ),
        "test_tracks": 0,
        "group_split_leaks": 0,
    }
    return annotations, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose training and validation manifests while rejecting test "
            "records and group-level split leakage."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    annotations, report = compose_training_manifests(
        arguments.manifest,
        arguments.output,
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tracks": len(annotations),
                "output": arguments.output.name,
                "report": arguments.report.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
