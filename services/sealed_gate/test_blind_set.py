from __future__ import annotations

import json
import unittest

from blind_set import (
    build_private_plan,
    build_public_commitment,
    build_recovered_private_plan,
)
from catalog import _candidate, build_catalog, canonical_json
from seal import IDENTITY_FIELDS


def empty_exclusion_index() -> dict:
    return {
        "schema_version": 1,
        "kind": "sealed-evaluation-exclusion-index",
        "manifests": [{"name": "prior.jsonl", "sha256": "a" * 64}],
        "identity_counts": {field: 0 for field in IDENTITY_FIELDS},
        "identity_digests": {field: [] for field in IDENTITY_FIELDS},
    }


def audio_candidate(
    index: int,
    *,
    audiovisual: bool = False,
    provider: str | None = None,
) -> dict:
    provider = provider or (
        "wikimedia-commons"
        if index % 2 == 0 or audiovisual
        else "internet-archive"
    )
    return _candidate(
        provider=provider,
        source_id=f"audio-{index:03d}",
        source_url=f"https://example.org/audio-{index:03d}",
        media_type="audiovisual" if audiovisual else "audio",
        title=f"Guitar Work {index:03d}",
        creator=f"Guitar Creator {index % 50:02d}",
        declared_license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        rights_evidence_url=f"https://example.org/audio-{index:03d}",
        duration_seconds=120,
        asset_url=f"https://cdn.example.org/audio-{index:03d}.ogg",
        content_evidence={
            "categories": ["Guitar music"],
            "file_size_bytes": 1024 * 1024,
            "parent_candidate_id": (
                f"internet-archive:parent-{index:03d}"
                if provider == "internet-archive"
                else ""
            ),
        },
    )


def score_candidate(index: int) -> dict:
    return _candidate(
        provider="mutopia",
        source_id=f"score-{index:03d}",
        source_url=f"https://example.org/score-{index:03d}",
        media_type="score",
        title=f"Score Work {index:03d}",
        creator=f"Score Creator {index % 20:02d}",
        declared_license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        rights_evidence_url=f"https://example.org/score-{index:03d}",
        score_formats=("midi",),
        score_assets=(
            {
                "format": "midi",
                "url": f"https://cdn.example.org/score-{index:03d}.mid",
            },
        ),
        content_evidence={"instrument": "Guitar"},
    )


class BlindSetTest(unittest.TestCase):
    def test_builds_complete_private_plan_and_aggregate_commitment(self) -> None:
        records = [
            *(audio_candidate(index) for index in range(130)),
            *(
                audio_candidate(1000 + index, audiovisual=True)
                for index in range(25)
            ),
            *(score_candidate(index) for index in range(120)),
            _candidate(
                provider="youtube",
                source_id="restricted-video",
                source_url="https://www.youtube.com/watch?v=restricted",
                media_type="audiovisual",
                title="Restricted Guitar",
                creator="Restricted Creator",
                declared_license="YouTube Standard License",
                license_url=None,
                rights_evidence_url=(
                    "https://www.youtube.com/watch?v=restricted"
                ),
                duration_seconds=120,
                content_evidence={"description": "Guitar performance"},
            ),
        ]
        catalog = build_catalog(records)

        private_plan = build_private_plan(
            catalog,
            empty_exclusion_index(),
            catalog_name="catalog.json",
            catalog_sha256="b" * 64,
            exclusion_name="exclusion.json",
            exclusion_sha256="c" * 64,
            salt=b"s" * 32,
        )
        private_sha256 = "d" * 64
        commitment = build_public_commitment(
            private_plan,
            private_plan_sha256=private_sha256,
        )

        self.assertEqual(len(private_plan["layers"]["layer_a"]), 100)
        self.assertEqual(len(private_plan["layers"]["layer_b"]), 100)
        self.assertEqual(len(private_plan["layers"]["layer_c_queue"]), 20)
        self.assertEqual(len(private_plan["assignments"]), len(records))
        self.assertEqual(commitment["admitted_recordings"], 0)
        self.assertEqual(
            commitment["layers"]["layer_a"]["selected"],
            100,
        )
        rendered = canonical_json(commitment)
        self.assertNotIn(b"selection_salt", rendered)
        self.assertNotIn(b"restricted-video", rendered)
        self.assertNotIn(b"Guitar Work", rendered)

    def test_selection_is_deterministic_for_fixed_salt(self) -> None:
        catalog = build_catalog(
            [
                *(audio_candidate(index) for index in range(130)),
                *(
                    audio_candidate(1000 + index, audiovisual=True)
                    for index in range(25)
                ),
                *(score_candidate(index) for index in range(120)),
            ]
        )
        arguments = {
            "catalog": catalog,
            "exclusion_index": empty_exclusion_index(),
            "catalog_name": "catalog.json",
            "catalog_sha256": "b" * 64,
            "exclusion_name": "exclusion.json",
            "exclusion_sha256": "c" * 64,
            "salt": b"s" * 32,
        }

        first = build_private_plan(**arguments)
        second = build_private_plan(**arguments)

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_recovery_retains_acquired_assets_and_replaces_blocked_source(
        self,
    ) -> None:
        original_records = [
            *(audio_candidate(index) for index in range(130)),
            *(
                audio_candidate(1000 + index, audiovisual=True)
                for index in range(25)
            ),
            *(score_candidate(index) for index in range(120)),
        ]
        original = build_private_plan(
            build_catalog(original_records),
            empty_exclusion_index(),
            catalog_name="catalog-v1.json",
            catalog_sha256="b" * 64,
            exclusion_name="exclusion.json",
            exclusion_sha256="c" * 64,
            salt=b"s" * 32,
        )
        retained_audio = [
            record
            for record in original["layers"]["layer_a"]
            if record["candidate_id"].startswith("internet-archive:")
        ]
        rejected_audio_id = retained_audio[0]["candidate_id"]
        retained_audio = retained_audio[1:]
        retained_scores = original["layers"]["layer_b"][:91]
        acquisition = {
            "layer_a": {
                "records": [
                    {"candidate_id": record["candidate_id"]}
                    for record in retained_audio
                ]
            },
            "layer_b": {
                "records": [
                    {"candidate_id": record["candidate_id"]}
                    for record in retained_scores
                ]
            },
        }
        expanded_records = [
            *original_records,
            *(
                audio_candidate(
                    2000 + index,
                    provider="internet-archive",
                )
                for index in range(120)
            ),
        ]

        recovered = build_recovered_private_plan(
            build_catalog(expanded_records),
            empty_exclusion_index(),
            original,
            acquisition,
            catalog_name="catalog-v2.json",
            catalog_sha256="d" * 64,
            exclusion_name="exclusion.json",
            exclusion_sha256="c" * 64,
            previous_plan_name="blind-v1.private.json",
            previous_plan_sha256="e" * 64,
            acquisition_name="acquisition-v1.json",
            acquisition_sha256="f" * 64,
            unavailable_providers={"wikimedia-commons"},
        )

        recovered_audio_ids = {
            record["candidate_id"]
            for record in recovered["layers"]["layer_a"]
        }
        self.assertEqual(len(recovered_audio_ids), 100)
        self.assertTrue(
            {
                record["candidate_id"] for record in retained_audio
            }.issubset(recovered_audio_ids)
        )
        self.assertTrue(
            all(
                candidate_id.startswith("internet-archive:")
                for candidate_id in recovered_audio_ids
            )
        )
        self.assertNotIn(rejected_audio_id, recovered_audio_ids)
        self.assertEqual(len(recovered["layers"]["layer_b"]), 100)
        self.assertEqual(
            recovered["recovery"]["unavailable_providers"],
            ["wikimedia-commons"],
        )
        commitment = build_public_commitment(
            recovered,
            private_plan_sha256="1" * 64,
        )
        self.assertEqual(commitment["recovery"]["retained"]["layer_b"], 91)


if __name__ == "__main__":
    unittest.main()
