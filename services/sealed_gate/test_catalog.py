from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from catalog import (
    build_catalog,
    canonical_json,
    classify_rights,
    expand_internet_archive_records,
    normalize_internet_archive,
    normalize_internet_archive_files,
    normalize_mutopia,
    normalize_openverse,
    normalize_wikimedia,
    normalize_youtube_metadata,
)


FIXTURES = Path(__file__).with_name("fixtures")


def json_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class InternetCandidateCatalogTest(unittest.TestCase):
    def test_classifies_open_restricted_and_unknown_rights(self) -> None:
        self.assertEqual(
            classify_rights(
                "CC BY 4.0",
                "https://creativecommons.org/licenses/by/4.0/",
                provider="example",
            )["tier"],
            "open-production",
        )
        self.assertEqual(
            classify_rights(
                "CC BY-NC-ND 4.0",
                "https://creativecommons.org/licenses/by-nc-nd/4.0/",
                provider="example",
            )["tier"],
            "restricted-research",
        )
        self.assertEqual(
            classify_rights(
                "unknown",
                None,
                provider="example",
            )["tier"],
            "discovery-only",
        )
        by_sa = classify_rights(
            "CC BY-SA 4.0",
            "https://creativecommons.org/licenses/by-sa/4.0/",
            provider="example",
        )
        self.assertEqual(by_sa["tier"], "open-production")
        self.assertIn(
            "share-alike-on-distributed-adaptations",
            by_sa["compliance_obligations"],
        )

    def test_normalizes_all_provider_fixtures(self) -> None:
        wikimedia = normalize_wikimedia(json_fixture("wikimedia.json"))
        openverse = normalize_openverse(json_fixture("openverse.json"))
        archive = normalize_internet_archive(
            json_fixture("internet_archive.json")
        )
        mutopia = normalize_mutopia(
            (FIXTURES / "mutopia.html").read_text(encoding="utf-8")
        )

        self.assertEqual(len(wikimedia), 2)
        self.assertEqual(len(openverse), 2)
        self.assertEqual(len(archive), 2)
        self.assertEqual(len(mutopia), 1)
        self.assertEqual(openverse[0]["duration_seconds"], 125.0)
        self.assertEqual(mutopia[0]["title"], "Open Guitar")
        self.assertEqual(mutopia[0]["media_type"], "score")
        self.assertEqual(mutopia[0]["rights_tier"], "open-production")
        self.assertEqual(
            mutopia[0]["available_score_formats"],
            ["lilypond", "midi", "pdf"],
        )
        self.assertEqual(len(mutopia[0]["score_assets"]), 3)

    def test_expands_archive_items_to_one_canonical_audio_file(self) -> None:
        parent = normalize_internet_archive(
            json_fixture("internet_archive.json")
        )[0]

        records = normalize_internet_archive_files(
            json_fixture("internet_archive_item.json"),
            parent,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["media_type"], "audio")
        self.assertEqual(records[0]["duration_seconds"], 125.0)
        self.assertTrue(records[0]["asset_url"].endswith("performance.ogg"))

    def test_incrementally_expands_only_unexpanded_archive_parents(self) -> None:
        parent = normalize_internet_archive(
            json_fixture("internet_archive.json")
        )[0]
        payload = json_fixture("internet_archive_item.json")

        with patch("catalog._request_json", return_value=payload) as request:
            expanded, summary = expand_internet_archive_records(
                [parent],
                timeout=1,
                workers=1,
            )

        self.assertEqual(summary["parents_requested"], 1)
        self.assertEqual(summary["parents_succeeded"], 1)
        self.assertEqual(summary["parents_failed"], 0)
        self.assertEqual(summary["records_added"], 1)
        request.assert_called_once()

        with patch("catalog._request_json") as request:
            repeated, repeated_summary = expand_internet_archive_records(
                [parent, *expanded],
                timeout=1,
                workers=1,
            )

        self.assertEqual(repeated, [])
        self.assertEqual(repeated_summary["parents_requested"], 0)
        request.assert_not_called()

    def test_restricted_unknown_and_youtube_never_expose_asset_url(self) -> None:
        records = [
            *normalize_wikimedia(json_fixture("wikimedia.json")),
            *normalize_openverse(json_fixture("openverse.json")),
            *normalize_youtube_metadata(
                [
                    {
                        "video_id": "video-1",
                        "title": "YouTube Guitar",
                        "channel": "Example Channel",
                        "declared_license": "YouTube Standard License",
                        "duration_seconds": 90,
                    }
                ]
            ),
        ]

        protected = [
            record
            for record in records
            if record["rights_tier"] != "open-production"
        ]
        self.assertTrue(protected)
        for record in protected:
            self.assertTrue(record["download_prohibited"])
            self.assertIsNone(record["asset_url"])
        youtube = next(
            record for record in records if record["provider"] == "youtube"
        )
        self.assertEqual(youtube["rights_tier"], "restricted-research")

    def test_build_is_deterministic_and_pairs_exact_score_identity(self) -> None:
        records = [
            *normalize_wikimedia(json_fixture("wikimedia.json")),
            *normalize_openverse(json_fixture("openverse.json")),
            *normalize_internet_archive(
                json_fixture("internet_archive.json")
            ),
            *normalize_mutopia(
                (FIXTURES / "mutopia.html").read_text(encoding="utf-8")
            ),
        ]

        first = build_catalog(records)
        second = build_catalog(reversed(records))

        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(
            first["summary"]["qualified_for_sealed_evaluation"],
            0,
        )
        matches = [
            record
            for record in first["records"]
            if record["score_pair"]["status"] == "candidate-match"
        ]
        self.assertEqual(
            {record["candidate_id"] for record in matches},
            {"openverse:ov-open", "wikimedia-commons:100"},
        )


if __name__ == "__main__":
    unittest.main()
