from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


MAX_LOG_BYTES = 8 * 1024 * 1024
SUPPORTED_SCHEMA_VERSION = 1
BUILTIN_MANIFESTS = {
    "builtin:guitarset-v1": Path("apps/api/data/training/guitarset/manifest.jsonl"),
    "builtin:synthetic-smoke-v1": Path(
        "apps/api/data/training/synthetic-smoke/manifest.jsonl"
    ),
}
PUBLIC_VALIDATION_DOMAINS = {
    "clean": {
        "manifest": Path("apps/api/data/training/guitarset/manifest.jsonl"),
        "cache": Path("apps/api/data/training/guitarset/.basic-pitch-cache"),
        "cqt_cache": Path("apps/api/data/training/guitarset/.feature-cache"),
        "baseline": Path(
            "apps/api/data/training/reports/"
            "guitarset-onset-position-confidence-v5-validation.json"
        ),
        "baseline_sha256": (
            "d16fcccf4b9136193f57ec0fdb10fb82"
            "fa7f3c2106642413d46b5281f0d894e2"
        ),
    },
    "isolated_stem": {
        "manifest": Path(
            "apps/api/data/training/guitarset-demucs/manifest.jsonl"
        ),
        "cache": Path(
            "apps/api/data/training/guitarset-demucs/.basic-pitch-cache"
        ),
        "cqt_cache": Path(
            "apps/api/data/training/guitarset-demucs/.feature-cache"
        ),
        "baseline": Path(
            "apps/api/data/training/reports/"
            "guitarset-demucs-onset-position-confidence-v5-validation.json"
        ),
        "baseline_sha256": (
            "c4bc28ffa3bea53270fa4223f97aba13b"
            "831b1c5fa061ff909bee09806a95400"
        ),
    },
    "mixture_stem": {
        "manifest": Path(
            "apps/api/data/training/guitarset-mixture-demucs-v2/manifest.jsonl"
        ),
        "cache": Path(
            "apps/api/data/training/guitarset-mixture-demucs-v2/.basic-pitch-cache"
        ),
        "cqt_cache": Path(
            "apps/api/data/training/guitarset-mixture-demucs-v2/.feature-cache"
        ),
        "baseline": Path(
            "apps/api/data/training/reports/"
            "guitarset-mixture-demucs-v2-"
            "onset-position-confidence-v5-validation.json"
        ),
        "baseline_sha256": (
            "aa52ec7f9e1f4affa311a1d0fa05224c"
            "7012094aa166b0251fd28b0e84446644"
        ),
    },
}


@dataclass(frozen=True)
class RunnerConfig:
    api_url: str
    runner_id: str
    runner_secret: str
    workspace: Path
    work_root: Path
    cache_root: Path
    isolation: str
    poll_seconds: float
    container_image: str | None = None
    gpu_device: str | None = None

    @property
    def authorization(self) -> str:
        return f"Runner {self.runner_id}.{self.runner_secret}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def request_json(
    config: RunnerConfig,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{config.api_url.rstrip('/')}{path}",
        data=payload,
        method=method,
        headers={
            "Authorization": config.authorization,
            "Content-Type": "application/json",
            **(extra_headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API {method} {path} failed: {error.code} {detail}") from error


def upload_artifact(
    config: RunnerConfig,
    experiment_id: str,
    lease_token: str,
    artifact: Path,
) -> dict[str, Any]:
    parsed = urlparse(config.api_url)
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=120)
    boundary = f"gts-{uuid.uuid4().hex}"
    preamble = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="artifact"; filename="model.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii")
    ending = f"\r\n--{boundary}--\r\n".encode("ascii")
    content_length = len(preamble) + artifact.stat().st_size + len(ending)
    endpoint = (
        f"{parsed.path.rstrip('/')}/api/community/runner/jobs/"
        f"{experiment_id}/artifact"
    )
    connection.putrequest("POST", endpoint)
    connection.putheader("Authorization", config.authorization)
    connection.putheader("X-GTS-Lease-Token", lease_token)
    connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
    connection.putheader("Content-Length", str(content_length))
    connection.endheaders()
    connection.send(preamble)
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            connection.send(chunk)
    connection.send(ending)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    if response.status != 201:
        raise RuntimeError(
            f"Artifact upload failed: {response.status} "
            f"{raw.decode('utf-8', errors='replace')}"
        )
    return json.loads(raw)


def validate_claim(config: RunnerConfig, claim: dict[str, Any]) -> None:
    if claim.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError("Unsupported runner claim schema")
    job = claim.get("job")
    if not isinstance(job, dict):
        raise ValueError("Runner claim is missing job")
    recipe = job.get("recipe")
    dataset = job.get("dataset")
    parameters = job.get("parameters")
    if not isinstance(recipe, dict) or not isinstance(dataset, dict):
        raise ValueError("Runner claim recipe or dataset is invalid")
    if recipe.get("specVersion") != 1:
        raise ValueError("Unsupported recipe specification")
    if not isinstance(parameters, dict):
        raise ValueError("Runner claim parameters are invalid")
    if config.isolation == "podman":
        if not config.container_image:
            raise ValueError("Podman isolation requires a pinned runner image")
        if job.get("runtimeImage") != config.container_image:
            raise ValueError("Claim runtime image does not match the runner image")
        if recipe.get("resourceClass") == "gpu-standard" and not config.gpu_device:
            raise ValueError("GPU recipes require a configured Podman GPU device")
    manifest_relative = BUILTIN_MANIFESTS.get(str(dataset.get("id")))
    if manifest_relative is None:
        if recipe.get("id") != "community-release-audit-v1":
            raise ValueError("The runner does not support this dataset")
        return
    manifest = (config.workspace / manifest_relative).resolve()
    if not manifest.is_file():
        raise ValueError(f"Dataset manifest is unavailable: {dataset.get('id')}")
    if sha256_file(manifest) != dataset.get("manifestSha256"):
        raise ValueError("Dataset manifest hash does not match the claim")


def number_parameter(
    parameters: dict[str, Any],
    name: str,
    *,
    integer: bool = False,
) -> str:
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid recipe parameter: {name}")
    if integer and int(value) != value:
        raise ValueError(f"Recipe parameter must be an integer: {name}")
    return str(int(value) if integer else value)


def recipe_command(
    config: RunnerConfig,
    claim: dict[str, Any],
    work_directory: Path,
) -> tuple[list[str], Path]:
    job = claim["job"]
    recipe_id = job["recipe"]["id"]
    parameters = job["parameters"]
    artifact = work_directory / "model.bin"
    python = config.workspace / ".venv" / "bin" / "python"
    if recipe_id in {"onset-position-smoke-v1", "onset-position-guitarset-v1"}:
        manifest = config.workspace / BUILTIN_MANIFESTS[job["dataset"]["id"]]
        feature_cache = (
            manifest.parent / ".feature-cache"
            if recipe_id == "onset-position-guitarset-v1"
            else config.cache_root / job["dataset"]["id"].replace(":", "_")
        )
        command = [
            str(python),
            "-m",
            "stringtrace_ml.train_onset_position",
            "--manifest",
            str(manifest),
            "--output",
            str(artifact),
            "--epochs",
            number_parameter(parameters, "epochs", integer=True),
            "--learning-rate",
            number_parameter(parameters, "learningRate"),
            "--seed",
            number_parameter(parameters, "seed", integer=True),
            "--feature-cache",
            str(feature_cache),
            "--device",
            "cpu" if recipe_id == "onset-position-smoke-v1" else "auto",
        ]
        return command, artifact
    if recipe_id == "community-release-audit-v1":
        status, manifest = request_json(
            config,
            "GET",
            (
                f"/api/community/runner/jobs/{job['experimentId']}/"
                "dataset-manifest"
            ),
            extra_headers={"X-GTS-Lease-Token": claim["leaseToken"]},
        )
        if status != 200 or manifest is None:
            raise RuntimeError("Community release manifest is unavailable")
        source = work_directory / "dataset-manifest.json"
        source.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        script = (
            "import json,pathlib,sys;"
            "data=json.loads(pathlib.Path(sys.argv[1]).read_text());"
            "assert data['release']['manifestSha256']==sys.argv[3];"
            "pathlib.Path(sys.argv[2]).write_text("
            "json.dumps(data,sort_keys=True,separators=(',',':'))+'\\n')"
        )
        return [
            str(python),
            "-c",
            script,
            str(source),
            str(artifact),
            str(job["dataset"]["manifestSha256"]),
        ], artifact
    raise ValueError(f"Unsupported approved recipe: {recipe_id}")


def sandbox_command(
    config: RunnerConfig,
    work_directory: Path,
    command: list[str],
) -> tuple[list[str], str]:
    if config.isolation != "sandbox-exec":
        raise ValueError(f"Unsupported isolation backend: {config.isolation}")
    sandbox = shutil.which("sandbox-exec")
    if sandbox is None:
        raise RuntimeError("sandbox-exec is unavailable")
    profile = (
        "(version 1)"
        "(allow default)"
        "(deny network*)"
        "(deny file-write*)"
        f'(allow file-write* (subpath "{work_directory}") '
        f'(subpath "{config.cache_root}") (subpath "/dev"))'
    )
    return [sandbox, "-p", profile, *command], "sandbox-exec"


def podman_command(
    config: RunnerConfig,
    work_directory: Path,
    command: list[str],
    *,
    resource_class: str,
) -> tuple[list[str], str]:
    podman = shutil.which("podman")
    if podman is None:
        raise RuntimeError("podman is unavailable")
    if not config.container_image:
        raise RuntimeError("A pinned Podman runner image is required")
    limits = {
        "cpu-small": ("2", "4g"),
        "gpu-standard": ("8", "16g"),
    }
    if resource_class not in limits:
        raise ValueError(f"Unsupported container resource class: {resource_class}")
    cpus, memory = limits[resource_class]
    container_command = list(command)
    container_command[0] = "python"
    invocation = [
        podman,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        f"--cpus={cpus}",
        f"--memory={memory}",
        "--userns=keep-id",
        f"--user={os.getuid()}:{os.getgid()}",
        f"--volume={config.workspace}:{config.workspace}:ro",
        f"--volume={work_directory}:{work_directory}:rw",
        f"--volume={config.cache_root}:{config.cache_root}:rw",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=1g",
        f"--workdir={config.workspace}",
        f"--env=HOME={work_directory}",
        "--env=TMPDIR=/tmp",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=OMP_NUM_THREADS=2",
        "--env=MKL_NUM_THREADS=2",
    ]
    if resource_class == "gpu-standard":
        invocation.append(f"--device={config.gpu_device}")
    invocation.extend([config.container_image, *container_command])
    return invocation, "podman"


def isolated_command(
    config: RunnerConfig,
    work_directory: Path,
    command: list[str],
    *,
    resource_class: str,
) -> tuple[list[str], str]:
    if config.isolation == "sandbox-exec":
        return sandbox_command(config, work_directory, command)
    if config.isolation == "podman":
        return podman_command(
            config,
            work_directory,
            command,
            resource_class=resource_class,
        )
    raise ValueError(f"Unsupported isolation backend: {config.isolation}")


def parse_training_output(stdout: bytes) -> dict[str, float]:
    records: list[dict[str, Any]] = []
    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    summaries = [
        record
        for record in records
        if isinstance(record.get("validation"), dict)
    ]
    final = records[-1] if records else {}
    metrics: dict[str, float] = {}
    if summaries:
        best = max(
            summaries,
            key=lambda record: float(
                record["validation"].get("reference_position_accuracy", 0)
            ),
        )
        validation = best["validation"]
        for source, target in (
            ("reference_position_accuracy", "validationPositionAccuracy"),
            ("tablature_f1", "validationTablatureF1"),
            ("pitch_f1", "validationPitchF1"),
        ):
            if source in validation:
                metrics[target] = float(validation[source])
    for source, target in (
        ("best_validation_metric", "bestValidationMetric"),
        ("train_groups", "trainGroups"),
        ("validation_groups", "validationGroups"),
    ):
        if source in final:
            metrics[target] = float(final[source])
    if not metrics:
        metrics["artifactProduced"] = 1.0
    return metrics


def validation_metrics(report: dict[str, Any]) -> dict[str, float]:
    aggregate = dict(report.get("aggregate") or {})
    return {
        "onsetF1": float(dict(aggregate.get("onset") or {}).get("f1", 0)),
        "tablatureF1": float(
            dict(aggregate.get("tablature") or {}).get("f1", 0)
        ),
        "repeatedRecall": float(
            dict(aggregate.get("repeated_tablature") or {}).get("recall", 0)
        ),
        "trackCount": float(report.get("track_count", 0)),
    }


def public_validation_commands(
    config: RunnerConfig,
    artifact: Path,
    work_directory: Path,
) -> list[tuple[str, list[str], Path, Path]]:
    python = config.workspace / ".venv" / "bin" / "python"
    commands: list[tuple[str, list[str], Path, Path]] = []
    for domain, definition in PUBLIC_VALIDATION_DOMAINS.items():
        manifest = config.workspace / definition["manifest"]
        cache = config.workspace / definition["cache"]
        cqt_cache = config.workspace / definition["cqt_cache"]
        baseline = config.workspace / definition["baseline"]
        if sha256_file(baseline) != definition["baseline_sha256"]:
            raise ValueError(f"Frozen public baseline hash mismatch: {domain}")
        output = work_directory / f"public-{domain}.json"
        command = [
            str(python),
            "-m",
            "stringtrace_ml.evaluate_onset_position",
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(artifact),
            "--split",
            "validation",
            "--mode",
            "direct",
            "--minimum-direct-probability",
            "0.50",
            "--basic-pitch-cache",
            str(cache),
            "--cqt-cache",
            str(cqt_cache),
            "--pitch-onset-threshold",
            "0.35",
            "--pitch-frame-threshold",
            "0.25",
            "--pitch-minimum-note-length",
            "70",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
        commands.append((domain, command, output, baseline))
    return commands


def collect_public_validation_metrics(
    reports: dict[str, tuple[Path, Path]],
    *,
    evaluation_duration_ms: int,
    maximum_memory_mb: float,
) -> dict[str, float]:
    domains: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for domain, (candidate_path, baseline_path) in reports.items():
        candidate = validation_metrics(
            json.loads(candidate_path.read_text(encoding="utf-8"))
        )
        baseline = validation_metrics(
            json.loads(baseline_path.read_text(encoding="utf-8"))
        )
        if candidate["trackCount"] != baseline["trackCount"]:
            raise ValueError(f"Public validation track count mismatch: {domain}")
        domains[domain] = (candidate, baseline)
    clean, clean_baseline = domains["clean"]
    tablature_deltas = [
        candidate["tablatureF1"] - baseline["tablatureF1"]
        for candidate, baseline in domains.values()
    ]
    repeated_deltas = [
        candidate["repeatedRecall"] - baseline["repeatedRecall"]
        for candidate, baseline in domains.values()
    ]
    track_count = sum(candidate["trackCount"] for candidate, _ in domains.values())
    return {
        "publicOnsetF1": clean["onsetF1"],
        "publicTablatureF1": clean["tablatureF1"],
        "publicBaselineTablatureF1": clean_baseline["tablatureF1"],
        "publicRepeatedRecall": clean["repeatedRecall"],
        "publicBaselineRepeatedRecall": clean_baseline["repeatedRecall"],
        "minimumDomainTablatureDelta": min(tablature_deltas),
        "minimumDomainRepeatedRecallDelta": min(repeated_deltas),
        "publicValidationTracks": track_count,
        "robustnessPassed": 1.0,
        "validationMeanLatencyMs": evaluation_duration_ms / max(1, track_count),
        "maximumMemoryMb": maximum_memory_mb,
    }


def child_maximum_memory_mb() -> float:
    maximum = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return maximum / (1024 * 1024) if sys.platform == "darwin" else maximum / 1024


def start_heartbeat(
    config: RunnerConfig,
    claim: dict[str, Any],
    stop: threading.Event,
) -> threading.Thread:
    interval = max(5, int(claim.get("heartbeatIntervalSeconds", 30)))
    experiment_id = claim["job"]["experimentId"]
    lease_token = claim["leaseToken"]

    def run() -> None:
        while not stop.wait(interval):
            request_json(
                config,
                "POST",
                f"/api/community/runner/jobs/{experiment_id}/heartbeat",
                {"leaseToken": lease_token},
            )

    thread = threading.Thread(target=run, name="runner-heartbeat", daemon=True)
    thread.start()
    return thread


def execute_claim(config: RunnerConfig, claim: dict[str, Any]) -> None:
    validate_claim(config, claim)
    job = claim["job"]
    experiment_id = job["experimentId"]
    lease_token = claim["leaseToken"]
    work_directory = config.work_root / experiment_id
    if work_directory.exists():
        shutil.rmtree(work_directory)
    work_directory.mkdir(parents=True)
    (work_directory / "tmp").mkdir()
    config.cache_root.mkdir(parents=True, exist_ok=True)
    command, artifact = recipe_command(config, claim, work_directory)
    stdout = b""
    stderr = b""
    started_at = time.monotonic()
    stop_heartbeat = threading.Event()
    heartbeat = start_heartbeat(config, claim, stop_heartbeat)
    try:
        isolation = "sandbox-exec"
        environment = {
            **os.environ,
            "HOME": str(work_directory),
            "TMPDIR": str(work_directory / "tmp"),
            "XDG_CACHE_HOME": str(config.cache_root),
            "PYTHONPATH": str(config.workspace / "services" / "worker"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
        if command:
            isolated, isolation = isolated_command(
                config,
                work_directory,
                command,
                resource_class=job["recipe"]["resourceClass"],
            )
            completed = subprocess.run(
                isolated,
                cwd=config.workspace,
                env=environment,
                capture_output=True,
                timeout=int(job["recipe"]["timeoutSeconds"]),
                check=False,
            )
            stdout = completed.stdout[-MAX_LOG_BYTES:]
            stderr = completed.stderr[-MAX_LOG_BYTES:]
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Recipe exited with {completed.returncode}: "
                    f"{stderr.decode('utf-8', errors='replace')[-500:]}"
                )
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise RuntimeError("Recipe did not produce a model artifact")
        metrics = parse_training_output(stdout)
        if job["recipe"]["id"] == "onset-position-guitarset-v1":
            evaluation_started = time.monotonic()
            public_reports: dict[str, tuple[Path, Path]] = {}
            for domain, evaluation, output, baseline in public_validation_commands(
                config,
                artifact,
                work_directory,
            ):
                isolated_evaluation, _ = isolated_command(
                    config,
                    work_directory,
                    evaluation,
                    resource_class=job["recipe"]["resourceClass"],
                )
                remaining_seconds = max(
                    1,
                    int(
                        job["recipe"]["timeoutSeconds"]
                        - (time.monotonic() - started_at)
                    ),
                )
                completed = subprocess.run(
                    isolated_evaluation,
                    cwd=config.workspace,
                    env=environment,
                    capture_output=True,
                    timeout=remaining_seconds,
                    check=False,
                )
                stdout = (stdout + completed.stdout)[-MAX_LOG_BYTES:]
                stderr = (stderr + completed.stderr)[-MAX_LOG_BYTES:]
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Public validation {domain} exited with "
                        f"{completed.returncode}: "
                        f"{completed.stderr.decode('utf-8', errors='replace')[-500:]}"
                    )
                if not output.is_file():
                    raise RuntimeError(
                        f"Public validation {domain} did not produce a report"
                    )
                public_reports[domain] = (output, baseline)
            evaluation_duration_ms = max(
                1,
                round((time.monotonic() - evaluation_started) * 1000),
            )
            metrics.update(
                collect_public_validation_metrics(
                    public_reports,
                    evaluation_duration_ms=evaluation_duration_ms,
                    maximum_memory_mb=child_maximum_memory_mb(),
                )
            )
        duration_ms = max(1, round((time.monotonic() - started_at) * 1000))
        artifact_sha256 = sha256_file(artifact)
        report = {
            "schemaVersion": 1,
            "experimentId": experiment_id,
            "recipeId": job["recipe"]["id"],
            "recipeSpecVersion": job["recipe"]["specVersion"],
            "datasetManifestSha256": job["dataset"]["manifestSha256"],
            "lineageSha256": job["lineageSha256"],
            "artifactSha256": artifact_sha256,
            "durationMs": duration_ms,
            "isolation": isolation,
            "metrics": metrics,
        }
        report_bytes = (
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        (work_directory / "report.json").write_bytes(report_bytes)
        (work_directory / "stdout.log").write_bytes(stdout)
        (work_directory / "stderr.log").write_bytes(stderr)
        upload_artifact(config, experiment_id, lease_token, artifact)
        request_json(
            config,
            "POST",
            f"/api/community/runner/jobs/{experiment_id}/result",
            {
                "leaseToken": lease_token,
                "status": "completed",
                "metrics": metrics,
                "artifactSha256": artifact_sha256,
                "executionEvidence": {
                    "isolation": isolation,
                    "durationMs": duration_ms,
                    "stdoutSha256": sha256_bytes(stdout),
                    "stderrSha256": sha256_bytes(stderr),
                    "reportSha256": sha256_bytes(report_bytes),
                },
            },
        )
    except Exception as error:
        request_json(
            config,
            "POST",
            f"/api/community/runner/jobs/{experiment_id}/result",
            {
                "leaseToken": lease_token,
                "status": "failed",
                "failureReason": str(error)[:300],
            },
        )
        raise
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=5)


def load_config(arguments: argparse.Namespace) -> RunnerConfig:
    workspace = Path(arguments.workspace).resolve()
    runner_id = arguments.runner_id or os.environ.get("GTS_RUNNER_ID", "")
    runner_secret = arguments.runner_secret or os.environ.get(
        "GTS_RUNNER_SECRET",
        "",
    )
    if not runner_id or len(runner_secret) < 24:
        raise ValueError("Runner ID and a secret of at least 24 characters are required")
    return RunnerConfig(
        api_url=arguments.api_url,
        runner_id=runner_id,
        runner_secret=runner_secret,
        workspace=workspace,
        work_root=(workspace / arguments.work_root).resolve(),
        cache_root=(workspace / arguments.cache_root).resolve(),
        isolation=arguments.isolation,
        poll_seconds=arguments.poll_seconds,
        container_image=(
            arguments.container_image
            or os.environ.get("GTS_RUNNER_IMAGE")
            or None
        ),
        gpu_device=(
            arguments.gpu_device
            or os.environ.get("GTS_RUNNER_GPU_DEVICE")
            or None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute only Guitar Tab Studio approved training recipes.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8787")
    parser.add_argument("--runner-id")
    parser.add_argument("--runner-secret")
    parser.add_argument(
        "--workspace",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument(
        "--work-root",
        default="apps/api/data/training/runner-work",
    )
    parser.add_argument(
        "--cache-root",
        default="apps/api/data/training/runner-cache",
    )
    parser.add_argument(
        "--isolation",
        choices=("sandbox-exec", "podman"),
        default="sandbox-exec",
    )
    parser.add_argument("--container-image")
    parser.add_argument("--gpu-device")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments)
    while True:
        status, claim = request_json(
            config,
            "POST",
            "/api/community/runner/jobs/claim",
        )
        if status == 204 or claim is None:
            if arguments.once:
                return
            time.sleep(config.poll_seconds)
            continue
        execute_claim(config, claim)
        if arguments.once:
            return


if __name__ == "__main__":
    main()
