from __future__ import annotations

import argparse
import json
from pathlib import Path

from .technique_schema import TECHNIQUE_LABELS


def build_technique_gate(
    validation_report: dict[str, object],
    test_report: dict[str, object],
    *,
    minimum_test_support: int = 20,
    minimum_precision: float = 0.90,
    minimum_f1: float = 0.65,
) -> dict[str, object]:
    if validation_report.get("split") != "validation":
        raise ValueError("Gate selection report must use the validation split")
    if test_report.get("split") != "test":
        raise ValueError("Gate verification report must use the test split")
    identity_key = (
        "ensemble"
        if "ensemble" in validation_report or "ensemble" in test_report
        else "checkpoint"
    )
    if validation_report.get(identity_key) != test_report.get(identity_key):
        raise ValueError("Validation and test reports use different checkpoints")

    validation_gates = dict(validation_report["gates"])
    test_selective = dict(test_report["selective"]["per_class"])
    test_domains = dict(test_report.get("by_dataset", {}))
    approved_labels = []
    gates: dict[str, dict[str, object]] = {}
    for label in TECHNIQUE_LABELS:
        validation = dict(validation_gates[label])
        selective = dict(test_selective[label])
        domain_checks = {}
        for domain, domain_report in test_domains.items():
            domain_class = dict(
                domain_report["selective"]["per_class"][label]
            )
            if int(domain_class["support"]) < minimum_test_support:
                continue
            domain_checks[domain] = {
                "support": int(domain_class["support"]),
                "accepted": int(domain_class["accepted"]),
                "precision": float(domain_class["precision"]),
                "f1": float(domain_class["f1"]),
                "passed": (
                    int(domain_class["accepted"]) >= minimum_test_support
                    and float(domain_class["precision"]) >= minimum_precision
                    and float(domain_class["f1"]) >= minimum_f1
                ),
            }
        checks = {
            "validation_gate": bool(validation["supported"]),
            "test_support": int(selective["support"]) >= minimum_test_support,
            "test_precision": (
                float(selective["precision"]) >= minimum_precision
            ),
            "test_f1": float(selective["f1"]) >= minimum_f1,
            "test_domains": (
                all(result["passed"] for result in domain_checks.values())
                if test_domains
                else True
            ),
        }
        approved = all(checks.values())
        if approved:
            approved_labels.append(label)
        gates[label] = {
            "supported": approved,
            "threshold": validation["threshold"] if approved else None,
            "checks": checks,
            "validation_precision": validation["validation_precision"],
            "test_precision": selective["precision"],
            "test_support": selective["support"],
            "test_f1": selective["f1"],
            "test_domains": domain_checks,
        }

    return {
        "schema_version": 1,
        "kind": "technique-production-gate-v1",
        identity_key: validation_report[identity_key],
        "calibration": validation_report.get(
            "calibration",
            validation_report.get("component_calibration", {}),
        ),
        "criteria": {
            "minimum_test_support": minimum_test_support,
            "minimum_precision": minimum_precision,
            "minimum_f1": minimum_f1,
        },
        "approved_labels": approved_labels,
        "all_labels_approved": len(approved_labels) == len(TECHNIQUE_LABELS),
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote only technique labels that pass fixed test gates.",
    )
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--minimum-test-support", type=int, default=20)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-f1", type=float, default=0.65)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build_technique_gate(
        json.loads(arguments.validation_report.read_text(encoding="utf-8")),
        json.loads(arguments.test_report.read_text(encoding="utf-8")),
        minimum_test_support=arguments.minimum_test_support,
        minimum_precision=arguments.minimum_precision,
        minimum_f1=arguments.minimum_f1,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
