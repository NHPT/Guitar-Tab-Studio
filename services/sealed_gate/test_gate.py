from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from gate import SealedGateConfig, validate_sealed_inputs


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SealedGateTest(unittest.TestCase):
    def config(
        self,
        root: Path,
        manifest: Path,
        baseline: Path,
    ) -> SealedGateConfig:
        cache = root / "cache"
        cache.mkdir(exist_ok=True)
        return SealedGateConfig(
            api_url="http://127.0.0.1:8787",
            gate_id="sealed-a",
            gate_secret="sealed-a-secret-credential-0001",
            manifest=manifest,
            manifest_sha256=sha256_file(manifest),
            baseline=baseline,
            baseline_sha256=sha256_file(baseline),
            basic_pitch_cache=cache,
            cqt_cache=cache,
            work_root=root / "work",
            python=Path(sys.executable),
        )

    def test_requires_one_hundred_test_tracks_and_multiple_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            baseline = root / "baseline.json"
            baseline.write_text("{}\n", encoding="utf-8")
            records = [
                {
                    "track_id": f"sealed-{index:03d}",
                    "group_id": f"group-{index % 5}",
                    "split": "test",
                }
                for index in range(100)
            ]
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                validate_sealed_inputs(self.config(root, manifest, baseline)),
                {"tracks": 100, "groups": 5},
            )

            records[0]["split"] = "validation"
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not in the test split"):
                validate_sealed_inputs(self.config(root, manifest, baseline))

    def test_rejects_manifest_and_baseline_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            baseline = root / "baseline.json"
            manifest.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "track_id": f"sealed-{index:03d}",
                            "group_id": f"group-{index % 5}",
                            "split": "test",
                        }
                    )
                    for index in range(100)
                )
                + "\n",
                encoding="utf-8",
            )
            baseline.write_text("{}\n", encoding="utf-8")
            config = self.config(root, manifest, baseline)
            mismatched = SealedGateConfig(
                **{
                    **config.__dict__,
                    "manifest_sha256": "a" * 64,
                }
            )
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                validate_sealed_inputs(mismatched)


if __name__ == "__main__":
    unittest.main()
