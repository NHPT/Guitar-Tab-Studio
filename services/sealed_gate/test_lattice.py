from __future__ import annotations

import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from catalog import canonical_json
from lattice import _midi_documents, build_public_commitment, playable_positions


class LatticeTest(unittest.TestCase):
    def test_zip_midi_members_are_sorted_by_filename(self) -> None:
        class FakeArchive:
            def __enter__(self) -> "FakeArchive":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def infolist(self) -> list[zipfile.ZipInfo]:
                return [
                    zipfile.ZipInfo("second.mid"),
                    zipfile.ZipInfo("first.midi"),
                ]

            def read(self, _member: zipfile.ZipInfo) -> bytes:
                return b"midi"

        with (
            patch("lattice.zipfile.ZipFile", return_value=FakeArchive()),
            patch(
                "lattice.pretty_midi.PrettyMIDI",
                side_effect=lambda _payload: object(),
            ),
        ):
            documents = _midi_documents(Path("score.zip"))

        self.assertEqual(
            [name for name, _midi in documents],
            ["first.midi", "second.mid"],
        )

    def test_enumerates_standard_tuning_positions(self) -> None:
        self.assertEqual(
            playable_positions(64),
            [
                {"string": 1, "fret": 0},
                {"string": 2, "fret": 5},
                {"string": 3, "fret": 9},
                {"string": 4, "fret": 14},
                {"string": 5, "fret": 19},
                {"string": 6, "fret": 24},
            ],
        )

    def test_public_commitment_hides_blind_ids(self) -> None:
        lattice = {
            "acquisition": {
                "name": "acquisition.json",
                "sha256": "a" * 64,
            },
            "scores": [
                {
                    "blind_id": "blind-b-secret",
                    "source_asset_sha256": "b" * 64,
                    "members": [
                        {
                            "event_count": 10,
                            "unplayable_event_count": 1,
                        }
                    ],
                }
            ],
            "failures": [],
            "complete": True,
            "reviewed_scores": 0,
        }

        commitment = build_public_commitment(
            lattice,
            lattice_name="lattice.json",
            lattice_sha256="c" * 64,
        )

        self.assertNotIn(b"blind-b-secret", canonical_json(commitment))
        self.assertEqual(commitment["events"], 10)
        self.assertEqual(commitment["reviewed_scores"], 0)


if __name__ == "__main__":
    unittest.main()
