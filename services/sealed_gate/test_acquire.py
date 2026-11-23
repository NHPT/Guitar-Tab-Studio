from __future__ import annotations

import unittest

from acquire import _score_asset, _url_suffix, build_public_commitment
from catalog import canonical_json


class AcquireTest(unittest.TestCase):
    def test_prefers_midi_score_asset(self) -> None:
        record = {
            "blind_id": "blind-b-1",
            "score_assets": [
                {"format": "pdf", "url": "https://example.org/score.pdf"},
                {"format": "midi", "url": "https://example.org/score.mid"},
                {
                    "format": "lilypond",
                    "url": "https://example.org/score.ly",
                },
            ],
        }

        self.assertEqual(
            _score_asset(record),
            {"format": "midi", "url": "https://example.org/score.mid"},
        )

    def test_extracts_suffix_without_query(self) -> None:
        self.assertEqual(
            _url_suffix(
                "https://example.org/audio/file.ogg?download=1",
                fallback=".audio",
            ),
            ".ogg",
        )

    def test_public_commitment_does_not_disclose_private_ids(self) -> None:
        acquisition = {
            "catalog": {"name": "catalog.json", "sha256": "a" * 64},
            "private_plan": {"name": "plan.json", "sha256": "b" * 64},
            "layer_a": {
                "target": 1,
                "acquired": 1,
                "bytes": 10,
                "records": [
                    {
                        "blind_id": "blind-a-secret",
                        "candidate_id": "provider:secret-a",
                        "sha256": "c" * 64,
                    }
                ],
                "failures": [],
            },
            "layer_b": {
                "target": 1,
                "acquired": 1,
                "bytes": 5,
                "records": [
                    {
                        "blind_id": "blind-b-secret",
                        "candidate_id": "provider:secret-b",
                        "sha256": "d" * 64,
                    }
                ],
                "failures": [],
            },
            "layer_c": {"queued": 1, "acquired": 0},
            "complete": True,
            "admitted_recordings": 0,
        }

        commitment = build_public_commitment(
            acquisition,
            manifest_name="acquisition.json",
            manifest_sha256="e" * 64,
        )

        rendered = canonical_json(commitment)
        self.assertNotIn(b"blind-a-secret", rendered)
        self.assertNotIn(b"provider:secret-a", rendered)
        self.assertTrue(commitment["complete"])


if __name__ == "__main__":
    unittest.main()
