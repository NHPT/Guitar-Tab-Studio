from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import soundfile

try:
    from .blind_set import (
        PRIVATE_PLAN_KIND,
        load_catalog,
        sha256_file,
    )
    from .catalog import USER_AGENT, canonical_json
except ImportError:
    from blind_set import (
        PRIVATE_PLAN_KIND,
        load_catalog,
        sha256_file,
    )
    from catalog import USER_AGENT, canonical_json


SCHEMA_VERSION = 1
ACQUISITION_KIND = "sealed-evaluation-blind-acquisition"
ACQUISITION_COMMITMENT_KIND = (
    "sealed-evaluation-blind-acquisition-commitment"
)
MAX_AUDIO_BYTES = 35 * 1024 * 1024
MAX_SCORE_BYTES = 20 * 1024 * 1024
AUDIO_SUFFIXES = {
    ".flac",
    ".mp3",
    ".oga",
    ".ogg",
    ".wav",
    ".wave",
}
SCORE_FORMAT_PRIORITY = {
    "midi": 0,
    "lilypond": 1,
    "pdf": 2,
}
HOST_MINIMUM_INTERVAL_SECONDS = {
    "upload.wikimedia.org": 2.0,
    "www.mutopiaproject.org": 0.25,
}
MAX_DOWNLOAD_ATTEMPTS = 5
_HOST_SCHEDULE_LOCK = threading.Lock()
_HOST_NEXT_REQUEST: dict[str, float] = {}


def load_private_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != PRIVATE_PLAN_KIND
        or not isinstance(value.get("layers"), dict)
    ):
        raise ValueError("Unsupported blind intake plan")
    return value


def _url_suffix(url: str, *, fallback: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.casefold()
    if not suffix or len(suffix) > 12:
        return fallback
    return suffix


def _score_asset(record: dict[str, Any]) -> dict[str, str]:
    assets = record.get("score_assets")
    if not isinstance(assets, list):
        raise ValueError(f"Blind score {record.get('blind_id')} has no assets")
    supported = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and asset.get("format") in SCORE_FORMAT_PRIORITY
        and isinstance(asset.get("url"), str)
    ]
    if not supported:
        raise ValueError(
            f"Blind score {record.get('blind_id')} has no supported asset"
        )
    return min(
        supported,
        key=lambda value: (
            SCORE_FORMAT_PRIORITY[value["format"]],
            value["url"],
        ),
    )


def _throttle(url: str) -> None:
    host = urllib.parse.urlparse(url).hostname or ""
    interval = HOST_MINIMUM_INTERVAL_SECONDS.get(host, 0.0)
    if interval <= 0:
        return
    with _HOST_SCHEDULE_LOCK:
        now = time.monotonic()
        scheduled = max(now, _HOST_NEXT_REQUEST.get(host, now))
        _HOST_NEXT_REQUEST[host] = scheduled + interval
    wait = scheduled - now
    if wait > 0:
        time.sleep(wait)


def _download_once(
    url: str,
    output: Path,
    *,
    maximum_bytes: int,
    timeout: float,
) -> tuple[str, int]:
    if output.is_file():
        size = output.stat().st_size
        if not 0 < size <= maximum_bytes:
            raise ValueError(f"Existing asset has invalid size: {output.name}")
        return sha256_file(output), size
    part = output.with_suffix(output.suffix + ".part")
    if part.exists():
        part.unlink()
    _throttle(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > maximum_bytes:
                raise ValueError(
                    f"Remote asset exceeds size limit: {output.name}"
                )
            with part.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > maximum_bytes:
                        raise ValueError(
                            f"Downloaded asset exceeds size limit: {output.name}"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if size <= 0:
            raise ValueError(f"Downloaded asset is empty: {output.name}")
        part.replace(output)
    except Exception:
        if part.exists():
            part.unlink()
        raise
    return digest.hexdigest(), size


def _download(
    url: str,
    output: Path,
    *,
    maximum_bytes: int,
    timeout: float,
) -> tuple[str, int]:
    last_error: Exception | None = None
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            return _download_once(
                url,
                output,
                maximum_bytes=maximum_bytes,
                timeout=timeout,
            )
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code != 429 or attempt + 1 >= MAX_DOWNLOAD_ATTEMPTS:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                delay = 0.0
            time.sleep(max(delay, 15.0 * (attempt + 1)))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Download failed without an error: {output.name}")


def _verify_audio(
    path: Path,
    *,
    expected_duration: float,
) -> dict[str, object]:
    try:
        info = soundfile.info(path)
    except RuntimeError as error:
        raise ValueError(f"Audio cannot be decoded: {path.name}") from error
    duration = float(info.duration)
    if abs(duration - expected_duration) > max(2.0, expected_duration * 0.02):
        raise ValueError(
            f"Audio duration mismatch: {path.name} "
            f"{duration:.3f}!={expected_duration:.3f}"
        )
    if info.samplerate < 8_000:
        raise ValueError(f"Audio sample rate is too low: {path.name}")
    if info.channels not in {1, 2}:
        raise ValueError(f"Audio must be mono or stereo: {path.name}")
    return {
        "duration_seconds": duration,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def _audio_task(
    record: dict[str, Any],
    output_directory: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    url = record.get("asset_url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError(f"Blind audio {record.get('blind_id')} has no URL")
    suffix = _url_suffix(url, fallback=".audio")
    if suffix not in AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported audio suffix: {suffix}")
    output = output_directory / f"{record['blind_id']}{suffix}"
    sha256, size = _download(
        url,
        output,
        maximum_bytes=MAX_AUDIO_BYTES,
        timeout=timeout,
    )
    media = _verify_audio(
        output,
        expected_duration=float(record["duration_seconds"]),
    )
    return {
        "blind_id": record["blind_id"],
        "candidate_id": record["candidate_id"],
        "path": output.name,
        "sha256": sha256,
        "bytes": size,
        "declared_license": record["declared_license"],
        "rights_evidence_url": record["rights_evidence_url"],
        "compliance_obligations": record["compliance_obligations"],
        "media": media,
        "status": "acquired-verified",
    }


def _score_task(
    record: dict[str, Any],
    output_directory: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    asset = _score_asset(record)
    suffix = _url_suffix(
        asset["url"],
        fallback=f".{asset['format']}",
    )
    output = output_directory / f"{record['blind_id']}{suffix}"
    sha256, size = _download(
        asset["url"],
        output,
        maximum_bytes=MAX_SCORE_BYTES,
        timeout=timeout,
    )
    return {
        "blind_id": record["blind_id"],
        "candidate_id": record["candidate_id"],
        "path": output.name,
        "sha256": sha256,
        "bytes": size,
        "asset_format": asset["format"],
        "declared_license": record["declared_license"],
        "rights_evidence_url": record["rights_evidence_url"],
        "compliance_obligations": record["compliance_obligations"],
        "status": "acquired-verified",
    }


def _run_tasks(
    records: list[dict[str, Any]],
    function: Any,
    output_directory: Path,
    *,
    workers: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                function,
                record,
                output_directory,
                timeout=timeout,
            ): record
            for record in records
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "blind_id": record["blind_id"],
                        "candidate_id": record["candidate_id"],
                        "error_type": type(error).__name__,
                        "error": str(error)[:300],
                    }
                )
    return (
        sorted(completed, key=lambda value: value["blind_id"]),
        sorted(failures, key=lambda value: value["blind_id"]),
    )


def acquire(
    catalog_path: Path,
    private_plan_path: Path,
    output_root: Path,
    manifest_path: Path,
    *,
    workers: int,
    timeout: float,
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    plan = load_private_plan(private_plan_path)
    if plan["catalog"]["sha256"] != sha256_file(catalog_path):
        raise ValueError("Private plan catalog SHA-256 mismatch")
    catalog_ids = {
        record["candidate_id"]: record for record in catalog["records"]
    }
    for layer in ("layer_a", "layer_b", "layer_c_queue"):
        for record in plan["layers"][layer]:
            candidate = catalog_ids.get(record["candidate_id"])
            if candidate is None:
                raise ValueError(
                    f"Private plan references missing {record['candidate_id']}"
                )
            if candidate["rights_tier"] != "open-production":
                raise ValueError(
                    f"Private plan selected restricted {record['candidate_id']}"
                )
    audio_directory = output_root / "layer-a"
    score_directory = output_root / "layer-b"
    audio_directory.mkdir(parents=True, exist_ok=True)
    score_directory.mkdir(parents=True, exist_ok=True)
    audio, audio_failures = _run_tasks(
        plan["layers"]["layer_a"],
        _audio_task,
        audio_directory,
        workers=workers,
        timeout=timeout,
    )
    scores, score_failures = _run_tasks(
        plan["layers"]["layer_b"],
        _score_task,
        score_directory,
        workers=workers,
        timeout=timeout,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": ACQUISITION_KIND,
        "catalog": {
            "name": catalog_path.name,
            "sha256": sha256_file(catalog_path),
        },
        "private_plan": {
            "name": private_plan_path.name,
            "sha256": sha256_file(private_plan_path),
        },
        "layer_a": {
            "target": len(plan["layers"]["layer_a"]),
            "acquired": len(audio),
            "bytes": sum(record["bytes"] for record in audio),
            "records": audio,
            "failures": audio_failures,
        },
        "layer_b": {
            "target": len(plan["layers"]["layer_b"]),
            "acquired": len(scores),
            "bytes": sum(record["bytes"] for record in scores),
            "records": scores,
            "failures": score_failures,
        },
        "layer_c": {
            "queued": len(plan["layers"]["layer_c_queue"]),
            "acquired": 0,
            "status": "metadata-only-pending-physical-review",
        },
        "complete": not audio_failures and not score_failures,
        "admitted_recordings": 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json(result) + b"\n")
    return result


def build_public_commitment(
    acquisition: dict[str, Any],
    *,
    manifest_name: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    audio = acquisition["layer_a"]["records"]
    scores = acquisition["layer_b"]["records"]
    asset_hashes = sorted(
        record["sha256"] for record in [*audio, *scores]
    )
    commitment = {
        "schema_version": SCHEMA_VERSION,
        "kind": ACQUISITION_COMMITMENT_KIND,
        "catalog": acquisition["catalog"],
        "private_plan": acquisition["private_plan"],
        "acquisition_manifest": {
            "name": manifest_name,
            "sha256": manifest_sha256,
        },
        "asset_commitment_sha256": hashlib.sha256(
            canonical_json(asset_hashes)
        ).hexdigest(),
        "layer_a": {
            "target": acquisition["layer_a"]["target"],
            "acquired": acquisition["layer_a"]["acquired"],
            "bytes": acquisition["layer_a"]["bytes"],
            "failures": len(acquisition["layer_a"]["failures"]),
        },
        "layer_b": {
            "target": acquisition["layer_b"]["target"],
            "acquired": acquisition["layer_b"]["acquired"],
            "bytes": acquisition["layer_b"]["bytes"],
            "failures": len(acquisition["layer_b"]["failures"]),
        },
        "layer_c": acquisition["layer_c"],
        "complete": acquisition["complete"],
        "admitted_recordings": acquisition["admitted_recordings"],
        "disclosure": "aggregate-only",
    }
    rendered = canonical_json(commitment)
    forbidden = [
        record[field]
        for record in [*audio, *scores]
        for field in ("blind_id", "candidate_id")
    ]
    if any(value.encode("utf-8") in rendered for value in forbidden):
        raise ValueError(
            "Acquisition commitment discloses private identities"
        )
    return commitment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire and verify open-production assets selected by the "
            "private blind intake plan."
        ),
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--private-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--commitment", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 16:
        raise ValueError("workers must be in 1..16")
    result = acquire(
        arguments.catalog.resolve(),
        arguments.private_plan.resolve(),
        arguments.output_root.resolve(),
        arguments.manifest.resolve(),
        workers=arguments.workers,
        timeout=arguments.timeout,
    )
    manifest_path = arguments.manifest.resolve()
    if arguments.commitment is not None:
        commitment_path = arguments.commitment.resolve()
        commitment = build_public_commitment(
            result,
            manifest_name=manifest_path.name,
            manifest_sha256=sha256_file(manifest_path),
        )
        commitment_path.parent.mkdir(parents=True, exist_ok=True)
        commitment_path.write_bytes(canonical_json(commitment) + b"\n")
    summary = {
        "manifest": arguments.manifest.name,
        "manifest_sha256": sha256_file(manifest_path),
        "layer_a": {
            "target": result["layer_a"]["target"],
            "acquired": result["layer_a"]["acquired"],
            "bytes": result["layer_a"]["bytes"],
            "failures": len(result["layer_a"]["failures"]),
        },
        "layer_b": {
            "target": result["layer_b"]["target"],
            "acquired": result["layer_b"]["acquired"],
            "bytes": result["layer_b"]["bytes"],
            "failures": len(result["layer_b"]["failures"]),
        },
        "complete": result["complete"],
        "admitted_recordings": 0,
    }
    if arguments.commitment is not None:
        summary["commitment"] = arguments.commitment.name
        summary["commitment_sha256"] = sha256_file(
            arguments.commitment.resolve()
        )
    print(json.dumps(summary, sort_keys=True))
    if arguments.require_complete and not result["complete"]:
        raise RuntimeError("Blind acquisition is incomplete")


if __name__ == "__main__":
    main()
