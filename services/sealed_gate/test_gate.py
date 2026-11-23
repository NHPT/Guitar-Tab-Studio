from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from gate import SealedGateConfig, validate_sealed_inputs
from seal import (
    build_exclusion_index,
    canonical_json,
    identity_digest,
    seal_manifest,
    validate_intake_plan,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SealedGateTest(unittest.TestCase):
    def config(
        self,
        root: Path,
        manifest: Path,
        baseline: Path,
        exclusion_index: Path,
    ) -> SealedGateConfig:
        cache = root / "cache"
        cache.mkdir(exist_ok=True)
        return SealedGateConfig(
            api_url="http://127.0.0.1:8787",
            gate_id="sealed-a",
            gate_secret="sealed-a-secret-credential-0001",
            manifest=manifest,
            manifest_sha256=sha256_file(manifest),
            exclusion_index=exclusion_index,
            exclusion_index_sha256=sha256_file(exclusion_index),
            baseline=baseline,
            baseline_sha256=sha256_file(baseline),
            basic_pitch_cache=cache,
            cqt_cache=cache,
            work_root=root / "work",
            python=Path(sys.executable),
        )

    def write_manifest(self, root: Path, count: int = 100) -> Path:
        audio = root / "audio"
        audio.mkdir(exist_ok=True)
        manifest = root / "manifest.jsonl"
        records = []
        techniques = ("pick", "hammer-on", "slide")
        samples = np.zeros(30 * 8_000, dtype=np.float32)
        for index in range(count):
            audio_path = audio / f"sealed-{index:03d}.flac"
            samples[0] = (index + 1) / 200
            soundfile.write(audio_path, samples, 8_000, subtype="PCM_16")
            capture_mode = "microphone" if index % 2 == 0 else "pickup"
            provenance = {
                "dataset": "stringtrace-symbolic-synthetic",
                "source_id": "stringtrace-synthetic-v1",
                "rights_basis": "original",
                "performer_id": f"performer-{index % 5}",
                "work_id": f"work-{index:03d}",
                "recording_id": f"recording-{index:03d}",
                "session_id": f"session-{index % 5}",
                "lineage_id": f"lineage-{index:03d}",
                "audio_sha256": sha256_file(audio_path),
                "capture_mode": capture_mode,
                "guitar_id": f"guitar-{index % 3}",
            }
            if capture_mode == "microphone":
                provenance["microphone_id"] = f"microphone-{index % 2}"
            else:
                provenance["pickup_id"] = f"pickup-{(index // 2) % 2}"
            records.append(
                {
                    "track_id": f"sealed-{index:03d}",
                    "group_id": f"group-{index % 5}",
                    "audio": f"audio/{audio_path.name}",
                    "split": "test",
                    "duration": 30.0,
                    "events": [
                        {
                            "onset": 0.0,
                            "offset": 1.0,
                            "string": 1,
                            "fret": index % 13,
                            "technique": techniques[index % len(techniques)],
                        }
                    ],
                    "capo": 1 if index % 10 == 0 else 0,
                    "bpm": (80, 110, 140)[index % 3],
                    "provenance": provenance,
                }
            )
        manifest.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return manifest

    def write_exclusion_index(
        self,
        root: Path,
        *,
        audio_sha256: str | None = None,
    ) -> Path:
        prior = root / "prior.jsonl"
        provenance = {}
        if audio_sha256 is not None:
            provenance["audio_sha256"] = audio_sha256
        prior.write_text(
            json.dumps(
                {
                    "track_id": "prior-track",
                    "group_id": "prior-group",
                    "provenance": provenance,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        index = root / "exclusion-index.json"
        index.write_bytes(
            canonical_json(build_exclusion_index([prior])) + b"\n"
        )
        return index

    def test_requires_one_hundred_independent_diverse_works(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.write_manifest(root)
            baseline = root / "baseline.json"
            baseline.write_text("{}\n", encoding="utf-8")
            exclusion_index = self.write_exclusion_index(root)
            self.assertEqual(
                validate_sealed_inputs(
                    self.config(
                        root,
                        manifest,
                        baseline,
                        exclusion_index,
                    )
                ),
                {
                    "tracks": 100,
                    "groups": 5,
                    "works": 100,
                    "performers": 5,
                },
            )

            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["split"] = "validation"
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "Data leakage",
            ):
                validate_sealed_inputs(
                    self.config(
                        root,
                        manifest,
                        baseline,
                        exclusion_index,
                    )
                )

    def test_rejects_manifest_and_baseline_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.write_manifest(root)
            baseline = root / "baseline.json"
            baseline.write_text("{}\n", encoding="utf-8")
            exclusion_index = self.write_exclusion_index(root)
            config = self.config(
                root,
                manifest,
                baseline,
                exclusion_index,
            )
            mismatched = SealedGateConfig(
                **{
                    **config.__dict__,
                    "manifest_sha256": "a" * 64,
                }
            )
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                validate_sealed_inputs(mismatched)

            mismatched_index = SealedGateConfig(
                **{
                    **config.__dict__,
                    "exclusion_index_sha256": "b" * 64,
                }
            )
            with self.assertRaisesRegex(
                ValueError,
                "exclusion index SHA-256 mismatch",
            ):
                validate_sealed_inputs(mismatched_index)

    def test_rejects_prior_audio_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.write_manifest(root)
            first = json.loads(
                manifest.read_text(encoding="utf-8").splitlines()[0]
            )
            exclusion_index = self.write_exclusion_index(
                root,
                audio_sha256=first["provenance"]["audio_sha256"],
            )
            baseline = root / "baseline.json"
            baseline.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "overlaps prior data: audio_sha256=1",
            ):
                validate_sealed_inputs(
                    self.config(
                        root,
                        manifest,
                        baseline,
                        exclusion_index,
                    )
                )

    def test_commitment_discloses_aggregates_not_song_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.write_manifest(root)
            exclusion_index = self.write_exclusion_index(root)
            commitment_path = root / "commitment.json"

            commitment = seal_manifest(
                manifest,
                exclusion_index,
                commitment_path,
            )

            rendered = commitment_path.read_text(encoding="utf-8")
            self.assertEqual(commitment["summary"]["works"], 100)
            self.assertEqual(commitment["disclosure"], "aggregate-only")
            self.assertNotIn("sealed-000", rendered)
            self.assertNotIn("work-000", rendered)
            self.assertEqual(
                len(commitment["identity_commitment_sha256"]),
                64,
            )

    def test_identity_digest_is_domain_separated(self) -> None:
        self.assertNotEqual(
            identity_digest("track_id", "same"),
            identity_digest("work_id", "same"),
        )

    def test_checked_in_intake_plan_fills_one_hundred_work_slots(self) -> None:
        plan = Path(__file__).with_name("intake-plan-v1.json")

        self.assertEqual(
            validate_intake_plan(plan),
            {
                "target_works": 100,
                "performer_batches": 5,
                "microphone_slots": 50,
                "pickup_slots": 50,
                "nonzero_capo_minimum": 20,
            },
        )


if __name__ == "__main__":
    unittest.main()
