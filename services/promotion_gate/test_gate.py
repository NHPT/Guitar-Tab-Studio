from __future__ import annotations

import unittest

from gate import (
    derive_checks,
    evidence_sha256,
    public_validation_check,
    reproduction_check,
)


def gate_job() -> dict[str, object]:
    return {
        "promotion": {
            "id": "promotion-1",
            "checks": {
                "lineage": {"passed": True},
            },
        },
        "experiment": {
            "id": "experiment-1",
            "lineageSha256": "a" * 64,
            "metrics": {
                "publicTablatureF1": 0.74,
                "publicBaselineTablatureF1": 0.73,
                "publicRepeatedRecall": 0.82,
                "publicBaselineRepeatedRecall": 0.80,
                "minimumDomainTablatureDelta": 0.001,
                "minimumDomainRepeatedRecallDelta": 0.011,
                "robustnessPassed": 1,
                "validationMeanLatencyMs": 80,
                "maximumMemoryMb": 512,
            },
        },
        "reproductions": [
            {
                "experimentId": "experiment-2",
                "artifactSha256": "b" * 64,
                "metrics": {
                    "publicTablatureF1": 0.7400001,
                    "publicBaselineTablatureF1": 0.73,
                    "publicRepeatedRecall": 0.82,
                    "publicBaselineRepeatedRecall": 0.80,
                    "minimumDomainTablatureDelta": 0.001,
                    "minimumDomainRepeatedRecallDelta": 0.011,
                    "robustnessPassed": 1,
                    "validationMeanLatencyMs": 80,
                    "maximumMemoryMb": 512,
                },
            }
        ],
    }


class PromotionGateTest(unittest.TestCase):
    def test_evidence_digest_is_canonical(self) -> None:
        self.assertEqual(
            evidence_sha256({"b": 2, "a": 1}),
            evidence_sha256({"a": 1, "b": 2}),
        )

    def test_public_validation_requires_no_regression_and_repeat_gain(self) -> None:
        passed = public_validation_check(gate_job())
        self.assertIsNotNone(passed)
        self.assertTrue(passed["passed"])

        failed_job = gate_job()
        failed_job["experiment"]["metrics"]["publicRepeatedRecall"] = 0.805
        failed = public_validation_check(failed_job)
        self.assertIsNotNone(failed)
        self.assertFalse(failed["passed"])

    def test_reproduction_uses_same_lineage_metrics_with_tolerance(self) -> None:
        passed = reproduction_check(gate_job(), 1e-6)
        self.assertIsNotNone(passed)
        self.assertTrue(passed["passed"])

        failed = reproduction_check(gate_job(), 1e-9)
        self.assertIsNotNone(failed)
        self.assertFalse(failed["passed"])

    def test_derives_only_pending_checks_with_available_evidence(self) -> None:
        checks = derive_checks(gate_job(), 1e-6)
        self.assertEqual(
            {check["check"] for check in checks},
            {"reproduction", "public-validation", "robustness"},
        )
        job = gate_job()
        job["promotion"]["checks"]["public-validation"] = {"passed": True}
        self.assertEqual(
            {check["check"] for check in derive_checks(job, 1e-6)},
            {"reproduction", "robustness"},
        )


if __name__ == "__main__":
    unittest.main()
