from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stringtrace_ml.gate import evaluate_gate


MINIMUM_TRACKS = 100
MINIMUM_ONSET_F1 = 0.80
MINIMUM_TABLATURE_F1 = 0.75
MINIMUM_BASELINE_IMPROVEMENT = 0.15
MINIMUM_REPEATED_RECALL = 0.85


@dataclass(frozen=True)
class SealedGateConfig:
    api_url: str
    gate_id: str
    gate_secret: str
    manifest: Path
    manifest_sha256: str
    baseline: Path
    baseline_sha256: str
    basic_pitch_cache: Path
    cqt_cache: Path
    work_root: Path
    python: Path

    @property
    def authorization(self) -> str:
        return f"SealedGate {self.gate_id}.{self.gate_secret}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def request_json(
    config: SealedGateConfig,
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


def download_artifact(
    config: SealedGateConfig,
    experiment_id: str,
    output: Path,
) -> None:
    request = urllib.request.Request(
        (
            f"{config.api_url.rstrip('/')}/api/community/sealed-gates/"
            f"experiments/{experiment_id}/artifact"
        ),
        headers={"Authorization": config.authorization},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with output.open("wb") as target:
            shutil.copyfileobj(response, target)


def validate_sealed_inputs(config: SealedGateConfig) -> dict[str, int]:
    if sha256_file(config.manifest) != config.manifest_sha256:
        raise ValueError("Sealed manifest SHA-256 mismatch")
    if sha256_file(config.baseline) != config.baseline_sha256:
        raise ValueError("Sealed baseline SHA-256 mismatch")
    tracks = []
    groups = set()
    for line_number, raw_line in enumerate(
        config.manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if record.get("split") != "test":
            raise ValueError(
                f"Sealed manifest line {line_number} is not in the test split"
            )
        tracks.append(str(record["track_id"]))
        groups.add(str(record["group_id"]))
    if len(set(tracks)) < MINIMUM_TRACKS:
        raise ValueError("Sealed evaluation requires at least 100 unique tracks")
    if len(groups) < 5:
        raise ValueError("Sealed evaluation requires at least five source groups")
    return {"tracks": len(set(tracks)), "groups": len(groups)}


def sandboxed_evaluation_command(
    config: SealedGateConfig,
    checkpoint: Path,
    report: Path,
) -> list[str]:
    command = [
        str(config.python),
        "-m",
        "stringtrace_ml.evaluate_onset_position",
        "--manifest",
        str(config.manifest),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "test",
        "--mode",
        "direct",
        "--minimum-direct-probability",
        "0.50",
        "--basic-pitch-cache",
        str(config.basic_pitch_cache),
        "--cqt-cache",
        str(config.cqt_cache),
        "--pitch-onset-threshold",
        "0.35",
        "--pitch-frame-threshold",
        "0.25",
        "--pitch-minimum-note-length",
        "70",
        "--device",
        "cpu",
        "--output",
        str(report),
    ]
    sandbox = shutil.which("sandbox-exec")
    if sandbox is None:
        raise RuntimeError("sealed evaluation requires sandbox-exec in this build")
    profile = (
        "(version 1)"
        "(allow default)"
        "(deny network*)"
        "(deny file-write*)"
        f'(allow file-write* (subpath "{config.work_root}") (subpath "/dev"))'
    )
    return [sandbox, "-p", profile, *command]


def evaluate_job(config: SealedGateConfig, job: dict[str, Any]) -> dict[str, Any]:
    promotion = job["promotion"]
    experiment = job["experiment"]
    work = config.work_root / promotion["id"]
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "model.bin"
    report_path = work / "sealed-report.json"
    download_artifact(config, experiment["id"], checkpoint)
    if sha256_file(checkpoint) != experiment["artifactSha256"]:
        raise ValueError("Downloaded model artifact SHA-256 mismatch")
    completed = subprocess.run(
        sandboxed_evaluation_command(config, checkpoint, report_path),
        capture_output=True,
        check=False,
        timeout=21_600,
        env={
            **os.environ,
            "HOME": str(work),
            "TMPDIR": str(work),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        },
    )
    (work / "stderr.log").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            "Sealed evaluator failed without releasing metrics: "
            f"{completed.stderr.decode('utf-8', errors='replace')[-300:]}"
        )
    candidate = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = json.loads(config.baseline.read_text(encoding="utf-8"))
    result = evaluate_gate(
        candidate,
        baseline,
        minimum_tracks=MINIMUM_TRACKS,
        minimum_onset_f1=MINIMUM_ONSET_F1,
        minimum_tablature_f1=MINIMUM_TABLATURE_F1,
        minimum_improvement=MINIMUM_BASELINE_IMPROVEMENT,
        minimum_repeated_recall=MINIMUM_REPEATED_RECALL,
    )
    evidence = {
        "schemaVersion": 1,
        "promotionId": promotion["id"],
        "experimentId": experiment["id"],
        "manifestSha256": config.manifest_sha256,
        "baselineSha256": config.baseline_sha256,
        "artifactSha256": experiment["artifactSha256"],
        "reportSha256": sha256_file(report_path),
        "gate": result,
    }
    evidence_path = work / "sealed-evidence.json"
    evidence_path.write_bytes(canonical_json(evidence) + b"\n")
    return {
        "passed": bool(result["passed"]),
        "summary": (
            "密封评测通过全部固定门槛"
            if result["passed"]
            else "密封评测未通过全部固定门槛"
        ),
        "evidenceSha256": sha256_file(evidence_path),
    }


def run_once(config: SealedGateConfig) -> int:
    validate_sealed_inputs(config)
    _status, response = request_json(
        config,
        "GET",
        "/api/community/sealed-gates/promotions",
    )
    jobs = list((response or {}).get("jobs") or [])
    for job in jobs:
        result = evaluate_job(config, job)
        request_json(
            config,
            "POST",
            (
                f"/api/community/sealed-gates/promotions/"
                f"{job['promotion']['id']}/checks"
            ),
            result,
        )
    return len(jobs)


def required_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"{label} is required")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def required_directory(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"{label} is required")
    path = Path(value).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate eligible candidates without exposing sealed metrics.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8787")
    parser.add_argument("--gate-id", default=os.environ.get("GTS_SEALED_GATE_ID"))
    parser.add_argument(
        "--gate-secret",
        default=os.environ.get("GTS_SEALED_GATE_SECRET"),
    )
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--basic-pitch-cache", required=True)
    parser.add_argument("--cqt-cache", required=True)
    parser.add_argument(
        "--work-root",
        default="apps/api/data/training/sealed-gate-work",
    )
    arguments = parser.parse_args()
    if not arguments.gate_id or len(arguments.gate_secret or "") < 24:
        raise ValueError("Sealed gate ID and secret are required")
    work_root = Path(arguments.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    config = SealedGateConfig(
        api_url=arguments.api_url,
        gate_id=arguments.gate_id,
        gate_secret=arguments.gate_secret,
        manifest=required_path(arguments.manifest, "manifest"),
        manifest_sha256=arguments.manifest_sha256,
        baseline=required_path(arguments.baseline, "baseline"),
        baseline_sha256=arguments.baseline_sha256,
        basic_pitch_cache=required_directory(
            arguments.basic_pitch_cache,
            "basic pitch cache",
        ),
        cqt_cache=required_directory(arguments.cqt_cache, "CQT cache"),
        work_root=work_root,
        python=Path(sys.executable).resolve(),
    )
    count = run_once(config)
    print(json.dumps({"evaluatedCandidates": count}))


if __name__ == "__main__":
    main()
