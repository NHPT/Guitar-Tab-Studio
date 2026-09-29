from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from runner import (
    RunnerConfig,
    collect_public_validation_metrics,
    parse_training_output,
    podman_command,
    recipe_command,
    sandbox_command,
    sha256_file,
    validate_claim,
)


class TrainingRunnerTest(unittest.TestCase):
    def temporary_workspace(self) -> tempfile.TemporaryDirectory[str]:
        root = Path.cwd() / "apps" / "api" / "data"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(
            prefix=".test-training-runner-",
            dir=root,
        )

    def config(self, workspace: Path) -> RunnerConfig:
        return RunnerConfig(
            api_url="http://127.0.0.1:8787",
            runner_id="runner-a",
            runner_secret="runner-a-secret-credential-0001",
            workspace=workspace,
            work_root=workspace / "runner-work",
            cache_root=workspace / "runner-cache",
            isolation="sandbox-exec",
            poll_seconds=1,
        )

    def claim(
        self,
        manifest_sha256: str,
        *,
        recipe_id: str = "onset-position-smoke-v1",
    ) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "leaseToken": "lease-token",
            "heartbeatIntervalSeconds": 10,
            "leaseExpiresAt": "2026-09-29T00:05:00.000Z",
            "job": {
                "experimentId": "experiment-1",
                "recipe": {
                    "id": recipe_id,
                    "specVersion": 1,
                    "modelFamily": "onset-position-net",
                    "resourceClass": "cpu-small",
                    "timeoutSeconds": 900,
                    "promotionEligible": False,
                },
                "dataset": {
                    "id": "builtin:synthetic-smoke-v1",
                    "kind": "builtin",
                    "manifestSha256": manifest_sha256,
                },
                "parameters": {
                    "epochs": 1,
                    "learningRate": 0.0003,
                    "seed": 20260929,
                },
                "codeRevision": "workspace",
                "runtimeImage": "local-development",
                "lineageSha256": "a" * 64,
            },
        }

    def test_validates_manifest_hash_and_builds_only_a_fixed_command(self) -> None:
        with self.temporary_workspace() as temporary:
            workspace = Path(temporary)
            manifest = (
                workspace
                / "apps/api/data/training/synthetic-smoke/manifest.jsonl"
            )
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
            python = workspace / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            config = self.config(workspace)
            claim = self.claim(sha256_file(manifest))
            validate_claim(config, claim)
            work = workspace / "runner-work/experiment-1"
            work.mkdir(parents=True)
            command, artifact = recipe_command(config, claim, work)
            self.assertEqual(command[0], str(python))
            self.assertEqual(
                command[1:3],
                ["-m", "stringtrace_ml.train_onset_position"],
            )
            self.assertNotIn("shell", " ".join(command))
            self.assertEqual(artifact, work / "model.bin")

            claim["job"]["dataset"]["manifestSha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                validate_claim(config, claim)

    def test_extracts_bounded_numeric_metrics_from_training_output(self) -> None:
        output = b"\n".join(
            [
                json.dumps(
                    {
                        "epoch": 1,
                        "validation": {
                            "reference_position_accuracy": 0.7,
                            "tablature_f1": 0.6,
                            "pitch_f1": 0.8,
                        },
                    }
                ).encode(),
                json.dumps(
                    {
                        "best_validation_metric": 0.7,
                        "train_groups": 54,
                        "validation_groups": 18,
                    }
                ).encode(),
            ]
        )
        self.assertEqual(
            parse_training_output(output),
            {
                "validationPositionAccuracy": 0.7,
                "validationTablatureF1": 0.6,
                "validationPitchF1": 0.8,
                "bestValidationMetric": 0.7,
                "trainGroups": 54.0,
                "validationGroups": 18.0,
            },
        )

    def test_builds_a_rootless_networkless_podman_command(self) -> None:
        workspace = Path.cwd().resolve()
        work = workspace / "apps/api/data/training/runner-work/experiment-1"
        config = RunnerConfig(
            api_url="http://127.0.0.1:8787",
            runner_id="runner-a",
            runner_secret="runner-a-secret-credential-0001",
            workspace=workspace,
            work_root=work.parent,
            cache_root=workspace / "apps/api/data/training/runner-cache",
            isolation="podman",
            poll_seconds=1,
            container_image="localhost/gts-training@sha256:" + "a" * 64,
            gpu_device="nvidia.com/gpu=all",
        )
        invocation, isolation = podman_command(
            config,
            work,
            [str(workspace / ".venv/bin/python"), "-m", "approved.module"],
            resource_class="gpu-standard",
        )
        self.assertEqual(isolation, "podman")
        self.assertIn("--network=none", invocation)
        self.assertIn("--read-only", invocation)
        self.assertIn("--cap-drop=all", invocation)
        self.assertIn("--userns=keep-id", invocation)
        self.assertIn("--device=nvidia.com/gpu=all", invocation)
        self.assertIn(config.container_image, invocation)
        self.assertEqual(invocation[-3:], ["python", "-m", "approved.module"])

    def test_collects_public_validation_deltas_without_test_data(self) -> None:
        with self.temporary_workspace() as temporary:
            root = Path(temporary)
            reports = {}
            for index, domain in enumerate(
                ("clean", "isolated_stem", "mixture_stem")
            ):
                candidate = root / f"{domain}-candidate.json"
                baseline = root / f"{domain}-baseline.json"
                candidate.write_text(
                    json.dumps(
                        {
                            "track_count": 60,
                            "aggregate": {
                                "onset": {"f1": 0.81},
                                "tablature": {"f1": 0.71 + index * 0.01},
                                "repeated_tablature": {
                                    "recall": 0.72 + index * 0.01
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                baseline.write_text(
                    json.dumps(
                        {
                            "track_count": 60,
                            "aggregate": {
                                "onset": {"f1": 0.80},
                                "tablature": {"f1": 0.70 + index * 0.01},
                                "repeated_tablature": {
                                    "recall": 0.70 + index * 0.01
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                reports[domain] = (candidate, baseline)
            metrics = collect_public_validation_metrics(
                reports,
                evaluation_duration_ms=180_000,
                maximum_memory_mb=512,
            )
            self.assertAlmostEqual(metrics["minimumDomainTablatureDelta"], 0.01)
            self.assertAlmostEqual(
                metrics["minimumDomainRepeatedRecallDelta"],
                0.02,
            )
            self.assertEqual(metrics["publicValidationTracks"], 180)
            self.assertEqual(metrics["validationMeanLatencyMs"], 1000)

    @unittest.skipUnless(
        sys.platform == "darwin" and shutil.which("sandbox-exec"),
        "sandbox-exec is available only on macOS",
    )
    def test_sandbox_allows_work_output_and_blocks_other_workspace_writes(self) -> None:
        with self.temporary_workspace() as temporary:
            workspace = Path(temporary)
            work = workspace / "runner-work/experiment-1"
            cache = workspace / "runner-cache"
            work.mkdir(parents=True)
            cache.mkdir()
            config = self.config(workspace)
            allowed = work / "allowed.txt"
            blocked = workspace / "blocked.txt"
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path;"
                    f"Path({str(allowed)!r}).write_text('ok');"
                    f"Path({str(blocked)!r}).write_text('blocked')"
                ),
            ]
            isolated, name = sandbox_command(config, work, command)
            self.assertEqual(name, "sandbox-exec")
            completed = subprocess.run(isolated, capture_output=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(allowed.is_file())
            self.assertFalse(blocked.exists())


if __name__ == "__main__":
    unittest.main()
