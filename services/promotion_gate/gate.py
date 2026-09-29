from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


PUBLIC_REQUIRED_METRICS = {
    "publicTablatureF1",
    "publicBaselineTablatureF1",
    "publicRepeatedRecall",
    "publicBaselineRepeatedRecall",
    "minimumDomainTablatureDelta",
    "minimumDomainRepeatedRecallDelta",
}


@dataclass(frozen=True)
class GateConfig:
    api_url: str
    gate_id: str
    gate_secret: str
    poll_seconds: float
    metric_tolerance: float

    @property
    def authorization(self) -> str:
        return f"Gate {self.gate_id}.{self.gate_secret}"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def evidence_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def request_json(
    config: GateConfig,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    payload = None if body is None else canonical_json(body)
    request = urllib.request.Request(
        f"{config.api_url.rstrip('/')}{path}",
        data=payload,
        method=method,
        headers={
            "Authorization": config.authorization,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API {method} {path} failed: {error.code} {detail}") from error


def metrics_match(
    candidate: dict[str, Any],
    reproduction: dict[str, Any],
    tolerance: float,
) -> bool:
    shared = set(candidate) & set(reproduction)
    if not shared:
        return False
    return all(
        abs(float(candidate[name]) - float(reproduction[name])) <= tolerance
        for name in shared
    )


def reproduction_check(
    job: dict[str, Any],
    tolerance: float,
) -> dict[str, Any] | None:
    metrics = dict(job["experiment"].get("metrics") or {})
    reproductions = list(job.get("reproductions") or [])
    if not metrics or not reproductions:
        return None
    matched = [
        reproduction
        for reproduction in reproductions
        if metrics_match(
            metrics,
            dict(reproduction.get("metrics") or {}),
            tolerance,
        )
    ]
    evidence = {
        "candidateExperimentId": job["experiment"]["id"],
        "candidateLineageSha256": job["experiment"]["lineageSha256"],
        "candidateMetrics": metrics,
        "reproductionExperimentIds": [
            item["experimentId"] for item in matched
        ],
        "metricTolerance": tolerance,
    }
    return {
        "check": "reproduction",
        "passed": bool(matched),
        "summary": (
            f"{len(matched)} 次同谱系运行在 {tolerance:g} 容差内复现"
            if matched
            else "同谱系重复运行超出指标容差"
        ),
        "evidenceSha256": evidence_sha256(evidence),
    }


def public_validation_check(job: dict[str, Any]) -> dict[str, Any] | None:
    metrics = dict(job["experiment"].get("metrics") or {})
    if not PUBLIC_REQUIRED_METRICS.issubset(metrics):
        return None
    tablature_f1 = float(metrics["publicTablatureF1"])
    baseline_f1 = float(metrics["publicBaselineTablatureF1"])
    repeated_recall = float(metrics["publicRepeatedRecall"])
    baseline_repeated = float(metrics["publicBaselineRepeatedRecall"])
    minimum_domain_delta = float(metrics["minimumDomainTablatureDelta"])
    checks = {
        "tablatureNoRegression": tablature_f1 >= baseline_f1,
        "repeatedRecallGain": repeated_recall - baseline_repeated >= 0.01,
        "allDomainsNoRegression": minimum_domain_delta >= 0,
        "allDomainsRepeatGain": (
            float(metrics["minimumDomainRepeatedRecallDelta"]) >= 0.01
        ),
    }
    evidence = {
        "experimentId": job["experiment"]["id"],
        "metrics": {
            name: metrics[name] for name in sorted(PUBLIC_REQUIRED_METRICS)
        },
        "checks": checks,
    }
    return {
        "check": "public-validation",
        "passed": all(checks.values()),
        "summary": (
            f"TAB F1 {tablature_f1:.4f}（基线 {baseline_f1:.4f}），"
            f"重复召回增益 {repeated_recall - baseline_repeated:+.4f}，"
            f"最差分域变化 {minimum_domain_delta:+.4f}"
        ),
        "evidenceSha256": evidence_sha256(evidence),
    }


def robustness_check(job: dict[str, Any]) -> dict[str, Any] | None:
    metrics = dict(job["experiment"].get("metrics") or {})
    required = {
        "robustnessPassed",
        "validationMeanLatencyMs",
        "maximumMemoryMb",
    }
    if not required.issubset(metrics):
        return None
    passed = float(metrics["robustnessPassed"]) == 1.0
    evidence = {
        "experimentId": job["experiment"]["id"],
        "robustnessPassed": metrics["robustnessPassed"],
        "validationMeanLatencyMs": metrics["validationMeanLatencyMs"],
        "maximumMemoryMb": metrics["maximumMemoryMb"],
    }
    return {
        "check": "robustness",
        "passed": passed,
        "summary": (
            f"稳健性 {'通过' if passed else '失败'}，"
            f"公开验证平均耗时 {float(metrics['validationMeanLatencyMs']):.1f} ms/轨，"
            f"最大内存 {float(metrics['maximumMemoryMb']):.1f} MB"
        ),
        "evidenceSha256": evidence_sha256(evidence),
    }


def derive_checks(
    job: dict[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    existing = dict(job["promotion"].get("checks") or {})
    checks: list[dict[str, Any]] = []
    for check in (
        reproduction_check(job, tolerance),
        public_validation_check(job),
        robustness_check(job),
    ):
        if check is not None and check["check"] not in existing:
            checks.append(check)
    return checks


def run_once(config: GateConfig) -> int:
    _status, response = request_json(
        config,
        "GET",
        "/api/community/gates/promotions",
    )
    jobs = list((response or {}).get("jobs") or [])
    recorded = 0
    for job in jobs:
        promotion_id = job["promotion"]["id"]
        for check in derive_checks(job, config.metric_tolerance):
            request_json(
                config,
                "POST",
                f"/api/community/gates/promotions/{promotion_id}/checks",
                check,
            )
            recorded += 1
            if not check["passed"]:
                break
    return recorded


def load_config(arguments: argparse.Namespace) -> GateConfig:
    gate_id = arguments.gate_id or os.environ.get("GTS_GATE_ID", "")
    gate_secret = arguments.gate_secret or os.environ.get("GTS_GATE_SECRET", "")
    if not gate_id or len(gate_secret) < 24:
        raise ValueError("Gate ID and a secret of at least 24 characters are required")
    return GateConfig(
        api_url=arguments.api_url,
        gate_id=gate_id,
        gate_secret=gate_secret,
        poll_seconds=arguments.poll_seconds,
        metric_tolerance=arguments.metric_tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record deterministic Guitar Tab Studio promotion checks.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8787")
    parser.add_argument("--gate-id")
    parser.add_argument("--gate-secret")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments)
    while True:
        recorded = run_once(config)
        if arguments.once:
            print(json.dumps({"recordedChecks": recorded}))
            return
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
