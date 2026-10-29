#!/usr/bin/env python3
"""Offline media separation and guitar-note transcription worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from bisect import bisect_left
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable


STANDARD_TUNING = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
GUITAR_PITCH_RANGE = (STANDARD_TUNING[6], STANDARD_TUNING[1] + 24)
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
TECHNIQUE_LABELS = (
    "unknown",
    "pick",
    "hammer-on",
    "pull-off",
    "slide",
    "harmonic",
    "palm-mute",
    "dead-note",
    "slap",
    "body-tap",
    "tremolo",
    "bend",
    "pinch-harmonic",
    "vibrato",
)
MIN_ANALYSIS_FREE_BYTES = 1024 * 1024 * 1024
MIN_SEPARATION_FREE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXTERNAL_ERROR_BYTES = 16 * 1024
NOISE2FRET_ARCHIVE_URL = (
    "https://zenodo.org/api/records/23037260/files/"
    "CheckpointsGOAT.zip/content"
)
NOISE2FRET_ARCHIVE_SHA256 = (
    "82e4fe8f0a4e99816a875980c391d5e552841da80d6cb09d61939e8efbeb52d6"
)
NOISE2FRET_ARCHIVE_BYTES = 111_317_004
NOISE2FRET_MODEL_SHA256 = (
    "1f454f75e9e6f44d80286e2c13188264b10d09c08fa8bdd791b1c809217bab95"
)
MR_MT3_MODEL_URL = (
    "https://huggingface.co/gudgud1014/MR-MT3/resolve/main/mt3.pth"
)
MR_MT3_MODEL_SHA256 = (
    "b8a3807ed265059abd25ad7f68142c06c35e8f6144dcaa45bd55946a3745398f"
)
MR_MT3_MODEL_BYTES = 183_672_643
YOURMT3_MODEL_URL = (
    "https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/"
    "amt/logs/2024/"
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/"
    "checkpoints/last.ckpt"
)
YOURMT3_MODEL_SHA256 = (
    "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25"
)
YOURMT3_MODEL_BYTES = 561_544_628
_MODEL_PROGRESS_RANGE = (0.0, 100.0, "正在下载模型")
_CURRENT_STAGE_LABEL = "正在启动媒体分析"


def emit_model_event(event_type: str, **payload: object) -> None:
    print(
        json.dumps(
            {"type": event_type, **payload},
            ensure_ascii=False,
        ),
        flush=True,
    )


def emit_model_progress(progress: float, stage_label: str) -> None:
    emit_model_event(
        "gts-model-progress",
        progress=round(max(0.0, min(100.0, progress)), 2),
        stageLabel=stage_label,
    )


def configure_model_progress(
    start: float,
    end: float,
    stage_label: str,
) -> None:
    global _MODEL_PROGRESS_RANGE
    _MODEL_PROGRESS_RANGE = (start, end, stage_label)


def model_download_tqdm():
    from tqdm.auto import tqdm

    class GTSModelDownloadTqdm(tqdm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["file"] = io.StringIO()
            kwargs["mininterval"] = 0.2
            super().__init__(*args, **kwargs)
            self._last_emitted = -1

        def update(self, amount: float = 1) -> bool | None:
            changed = super().update(amount)
            if self.total:
                start, end, label = _MODEL_PROGRESS_RANGE
                percent = max(0.0, min(1.0, self.n / self.total))
                mapped = round(start + (end - start) * percent, 2)
                whole = int(mapped)
                if whole != self._last_emitted:
                    self._last_emitted = whole
                    emit_model_progress(mapped, label)
            return changed

    return GTSModelDownloadTqdm


@dataclass
class TechniqueCandidate:
    technique: str
    confidence: float
    evidence: list[str]


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int
    velocity: int
    string: int
    fret: int
    technique: str
    confidence: float
    technique_confidence: float = 0.0
    technique_source: str = "acoustic-heuristic-v2"
    technique_evidence: list[str] = field(default_factory=list)
    technique_candidates: list[TechniqueCandidate] = field(default_factory=list)
    related_note_index: int | None = None
    harmonic_type: str | None = None
    harmonic_touch_fret: int | None = None
    position_confidence: float = 0.0
    position_source: str = "playable-optimizer-v2"


@dataclass
class AudioFeature:
    onset: float
    centroid: float
    flatness: float
    zero_crossing: float


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        if completed.returncode in {28, 228}:
            raise RuntimeError("可用磁盘空间不足，外部媒体处理命令未能完成")
        output = f"{getattr(completed, 'stderr', '')}\n{getattr(completed, 'stdout', '')}"
        detail = next(
            (
                line.removeprefix("ERROR: ").strip()
                for line in reversed(output.splitlines())
                if line.strip().startswith("ERROR:")
            ),
            "",
        )
        executable = Path(command[0]).name if command else "external command"
        raise RuntimeError(
            f"{executable} failed ({completed.returncode})"
            + (f": {detail}" if detail else "")
        )


def emit_progress(progress: float, status: str, stage_label: str) -> None:
    global _CURRENT_STAGE_LABEL
    _CURRENT_STAGE_LABEL = stage_label
    print(
        json.dumps(
            {
                "type": "gts-progress",
                "progress": round(max(0.0, min(99.0, progress)), 2),
                "status": status,
                "stageLabel": stage_label,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def emit_muscriptor_progress(completed: int, total: int) -> None:
    if total <= 0:
        return
    bounded_completed = max(0, min(total, completed))
    progress = 65 + 15 * bounded_completed / total
    emit_progress(
        progress,
        "transcribing",
        f"正在识别吉他音符（{bounded_completed}/{total} 个片段）",
    )


def release_torch_accelerator_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def hardware_probe() -> dict[str, object]:
    def available(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            return False

    accelerator = "cpu"
    accelerator_name = "CPU"
    accelerator_memory_bytes = None
    if sys.platform == "darwin" and platform.machine() == "arm64":
        accelerator = "mps"
        accelerator_name = "Apple Metal"
    else:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            first_gpu = (
                result.stdout.splitlines()[0]
                if result.returncode == 0 and result.stdout
                else ""
            )
            match = re.match(r"^(.+),\s*(\d+(?:\.\d+)?)$", first_gpu)
            if match:
                name, memory_mib = match.groups()
                accelerator = "cuda"
                accelerator_name = name.strip()
                accelerator_memory_bytes = round(float(memory_mib) * 1024**2)
    providers = {
        "basic-pitch": True,
        "muscriptor": available("muscriptor"),
        "mr-mt3": available("mt3_infer.models.mr_mt3.t5"),
        "guitar-specialist": available(
            "stringtrace_ml.onset_position_bundle"
        ),
        "multitrack-research": available(
            "mt3_infer.adapters.yourmt3"
        ),
    }
    return {
        "accelerator": accelerator,
        "acceleratorName": accelerator_name,
        **(
            {"acceleratorMemoryBytes": accelerator_memory_bytes}
            if accelerator_memory_bytes is not None
            else {}
        ),
        "providers": providers,
    }


def runtime_self_test() -> dict[str, object]:
    from importlib.metadata import version
    from importlib.resources import files

    import basic_pitch
    import demucs
    import soundfile
    import torch

    demucs_remote = files("demucs").joinpath("remote")
    required_demucs_data = ("files.txt", "htdemucs_6s.yaml")
    missing = [
        name
        for name in required_demucs_data
        if not demucs_remote.joinpath(name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Demucs 运行数据缺失：" + ", ".join(missing)
        )
    return {
        "basicPitch": version("basic-pitch"),
        "demucs": getattr(demucs, "__version__", version("demucs")),
        "ffmpeg": ffmpeg_command(),
        "soundfile": soundfile.__version__,
        "torch": torch.__version__,
    }


def prepare_transcription_provider(
    provider: str,
    model_reference: str | None,
    model_variant: str | None = None,
) -> None:
    if provider == "basic-pitch":
        print(
            json.dumps(
                {
                    "type": "gts-model-ready",
                    "provider": provider,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    if provider == "multitrack-research":
        prepare_yourmt3_model()
        prepare_noise2fret_model()
        emit_model_event(
            "gts-model-ready",
            provider=provider,
            model="yourmt3",
            progress=100,
            stageLabel="多轨研究模型已准备完成",
        )
        return
    if provider == "guitar-specialist":
        emit_model_event(
            "gts-model-ready",
            provider=provider,
            model="stringtrace",
            progress=100,
            stageLabel="吉他专用模型已准备完成",
        )
        return
    if provider != "muscriptor":
        raise ValueError(f"Unknown transcription provider: {provider}")

    model = model_reference or "medium"
    if model not in {"small", "medium", "large"}:
        path = Path(model)
        if not path.is_file():
            raise RuntimeError("所选 MuScriptor 权重文件不存在")
        model_label = "local"
    else:
        try:
            from huggingface_hub import get_token, hf_hub_download
            from huggingface_hub._login import _save_oauth_token
            from huggingface_hub.errors import (
                DeviceCodeError,
                GatedRepoError,
                HfHubHTTPError,
                RepositoryNotFoundError,
            )
            from huggingface_hub.utils._oauth_device import (
                poll_device_token,
                request_device_code,
            )

            repository = f"MuScriptor/muscriptor-{model}"

            authorization_completed = False

            def authorize() -> None:
                nonlocal authorization_completed
                device_info = request_device_code()
                def emit_auth_required() -> None:
                    emit_model_event(
                        "gts-auth-required",
                        progress=3,
                        stageLabel="等待 Hugging Face 登录完成",
                        authorizationUrl=device_info[
                            "verification_uri_complete"
                        ],
                        userCode=device_info["user_code"],
                    )

                emit_auth_required()
                try:
                    response = poll_device_token(
                        device_info,
                        on_pending=emit_auth_required,
                    )
                except DeviceCodeError as error:
                    error_code = getattr(error, "error_code", "")
                    error_value = getattr(error_code, "value", str(error_code))
                    if error_value == "access_denied":
                        raise RuntimeError(
                            "Hugging Face 登录已取消，请重新尝试"
                        ) from error
                    raise RuntimeError(
                        "Hugging Face 登录等待超时，请重新尝试"
                    ) from error
                _save_oauth_token(response)
                authorization_completed = True
                emit_model_progress(
                    5,
                    "Hugging Face 登录成功，正在检查模型授权",
                )

            emit_model_progress(2, "正在检查 Hugging Face 登录状态")
            if get_token() is None:
                authorize()

            def download(filename: str, start: float, end: float) -> None:
                configure_model_progress(
                    start,
                    end,
                    f"正在下载 MuScriptor {model} 权重",
                )
                license_url = f"https://huggingface.co/{repository}"
                deadline = time.monotonic() + 15 * 60
                license_prompted = False
                while True:
                    try:
                        hf_hub_download(
                            repo_id=repository,
                            filename=filename,
                            token=True,
                            tqdm_class=model_download_tqdm(),
                        )
                        return
                    except (GatedRepoError, RepositoryNotFoundError) as error:
                        if not license_prompted:
                            emit_model_event(
                                "gts-license-required",
                                progress=start,
                                stageLabel=(
                                    "请在浏览器确认 MuScriptor 官方权重许可，"
                                    "完成后将自动继续"
                                ),
                                authorizationUrl=license_url,
                            )
                            license_prompted = True
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "MuScriptor 官方权重许可确认超时，请重试"
                            ) from error
                        time.sleep(5)
                    except HfHubHTTPError as error:
                        status = getattr(error.response, "status_code", None)
                        if status == 401:
                            if authorization_completed:
                                raise RuntimeError(
                                    "Hugging Face 登录失效，请重试"
                                ) from error
                            authorize()
                            continue
                        if status != 403:
                            raise
                        if not license_prompted:
                            emit_model_event(
                                "gts-license-required",
                                progress=start,
                                stageLabel=(
                                    "请在浏览器确认 MuScriptor 官方权重许可，"
                                    "完成后将自动继续"
                                ),
                                authorizationUrl=license_url,
                            )
                            license_prompted = True
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "Hugging Face 授权确认超时，请重试"
                            ) from error
                        time.sleep(5)

            emit_model_progress(6, f"正在检查 MuScriptor {model} 权重")
            download("model.safetensors", 6, 68)
            emit_model_progress(69, "正在验证 MuScriptor 模型配置")
            download("config.json", 69, 72)
        except HfHubHTTPError as error:
            raise RuntimeError(
                "MuScriptor 权重下载失败，请检查网络后重试"
            ) from error
        model_label = model
    prepare_noise2fret_model()
    selected_variant = (
        model_variant
        if model_variant in {"small", "medium", "large"}
        else model if model in {"small", "medium", "large"} else "medium"
    )
    probe = hardware_probe()
    if (
        selected_variant != "small"
        and probe["accelerator"] == "cuda"
        and bool(probe["providers"].get("mr-mt3"))
    ):
        prepare_mr_mt3_model()
    emit_model_event(
        "gts-model-ready",
        provider=provider,
        model=model_label,
        progress=100,
        stageLabel="高精度模型已准备完成",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_resume(
    url: str,
    destination: Path,
    expected_size: int,
    label: str,
    progress_start: float,
    progress_end: float,
) -> None:
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size > expected_size:
        destination.unlink()
        existing_size = 0
    request = urllib.request.Request(url)
    if existing_size:
        request.add_header("Range", f"bytes={existing_size}-")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            append = existing_size > 0 and response.status == 206
            total = existing_size if append else 0
            with destination.open("ab" if append else "wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > expected_size:
                        raise RuntimeError(f"{label} 下载大小超过预期")
                    output.write(chunk)
                    fraction = total / expected_size
                    emit_model_progress(
                        progress_start
                        + (progress_end - progress_start) * fraction,
                        f"正在下载 {label}",
                    )
    except Exception as error:
        raise RuntimeError(f"{label}下载失败，请检查网络后重试") from error
    if destination.stat().st_size != expected_size:
        raise RuntimeError(f"{label}下载不完整，请重试")


def prepare_noise2fret_model() -> Path:
    configured_root = os.environ.get("GTS_MODEL_PROVIDER_ROOT")
    if not configured_root:
        raise RuntimeError("高精度模型存储目录未配置")
    root = Path(configured_root).resolve() / "noise2fret" / "1.0.0"
    destination = root / "model.pth"
    manifest = root / "manifest.json"
    if (
        destination.is_file()
        and _sha256(destination) == NOISE2FRET_MODEL_SHA256
    ):
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "provider": "noise2fret",
                    "version": "1.0.0",
                    "sizeBytes": destination.stat().st_size,
                    "sha256": NOISE2FRET_MODEL_SHA256,
                    "license": "CC-BY-4.0",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "CheckpointsGOAT.zip.part"
    emit_model_progress(73, "正在下载 Noise2Fret 弦品模型")
    _download_with_resume(
        NOISE2FRET_ARCHIVE_URL,
        archive,
        NOISE2FRET_ARCHIVE_BYTES,
        "Noise2Fret 权重",
        73,
        86,
    )
    emit_model_progress(87, "正在验证 Noise2Fret 弦品模型")
    if _sha256(archive) != NOISE2FRET_ARCHIVE_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeError("Noise2Fret 权重校验失败")
    member = (
        "GOAT_N2F_Laux/my_checkpoints/best_checkpoint.pth"
    )
    temporary = destination.with_suffix(".pth.part")
    try:
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        if _sha256(temporary) != NOISE2FRET_MODEL_SHA256:
            raise RuntimeError("Noise2Fret 模型文件校验失败")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "provider": "noise2fret",
                    "version": "1.0.0",
                    "sizeBytes": destination.stat().st_size,
                    "sha256": NOISE2FRET_MODEL_SHA256,
                    "license": "CC-BY-4.0",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(manifest, 0o600)
        archive.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_mr_mt3_model() -> Path:
    configured_root = os.environ.get("GTS_MODEL_PROVIDER_ROOT")
    if not configured_root:
        raise RuntimeError("高精度模型存储目录未配置")
    root = Path(configured_root).resolve() / "mr-mt3" / "1.0.0"
    destination = root / "model.pth"
    manifest = root / "manifest.json"
    if destination.is_file() and _sha256(destination) == MR_MT3_MODEL_SHA256:
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "provider": "mr-mt3",
                    "version": "1.0.0",
                    "sizeBytes": destination.stat().st_size,
                    "sha256": MR_MT3_MODEL_SHA256,
                    "license": "MIT",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pth.part")
    emit_model_progress(89, "正在下载 MR-MT3 复核模型")
    _download_with_resume(
        MR_MT3_MODEL_URL,
        temporary,
        MR_MT3_MODEL_BYTES,
        "MR-MT3 权重",
        89,
        97,
    )
    emit_model_progress(98, "正在验证 MR-MT3 复核模型")
    if _sha256(temporary) != MR_MT3_MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("MR-MT3 权重校验失败")
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "provider": "mr-mt3",
                "version": "1.0.0",
                "sizeBytes": destination.stat().st_size,
                "sha256": MR_MT3_MODEL_SHA256,
                "license": "MIT",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest, 0o600)
    return destination


def prepare_yourmt3_model() -> Path:
    configured_root = os.environ.get("GTS_MODEL_PROVIDER_ROOT")
    if not configured_root:
        raise RuntimeError("研究模型存储目录未配置")
    root = Path(configured_root).resolve() / "yourmt3" / "1.0.0"
    destination = root / "last.ckpt"
    manifest = root / "manifest.json"
    if destination.is_file() and _sha256(destination) == YOURMT3_MODEL_SHA256:
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "provider": "yourmt3",
                    "version": "1.0.0",
                    "sizeBytes": destination.stat().st_size,
                    "sha256": YOURMT3_MODEL_SHA256,
                    "license": "research-only",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".ckpt.part")
    emit_model_progress(6, "正在下载 YourMT3+ 多轨研究模型")
    _download_with_resume(
        YOURMT3_MODEL_URL,
        temporary,
        YOURMT3_MODEL_BYTES,
        "YourMT3+ 权重",
        6,
        70,
    )
    emit_model_progress(71, "正在验证 YourMT3+ 多轨研究模型")
    if _sha256(temporary) != YOURMT3_MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("YourMT3+ 权重校验失败")
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "provider": "yourmt3",
                "version": "1.0.0",
                "sizeBytes": destination.stat().st_size,
                "sha256": YOURMT3_MODEL_SHA256,
                "license": "research-only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest, 0o600)
    return destination


def run_with_percent_progress(
    command: list[str],
    *,
    start: float,
    end: float,
    status: str,
    stage_label: str,
) -> None:
    process = subprocess.Popen(
        command,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
    )
    assert process.stderr is not None
    buffer = ""
    error_tail: deque[str] = deque(maxlen=MAX_EXTERNAL_ERROR_BYTES)
    last_percent = -1
    while True:
        character = process.stderr.read(1)
        if character == "":
            break
        sys.stderr.write(character)
        error_tail.append(character)
        buffer += character
        if character not in "\r\n":
            continue
        match = re.search(r"(\d{1,3})%\|", buffer)
        buffer = ""
        if not match:
            continue
        percent = min(100, int(match.group(1)))
        if percent < last_percent + 2 and percent != 100:
            continue
        last_percent = percent
        emit_progress(
            start + (end - start) * percent / 100,
            status,
            f"{stage_label} {percent}%",
        )
    return_code = process.wait()
    if return_code != 0:
        error_text = "".join(error_tail)
        if return_code in {28, 228} or re.search(
            r"no space left on device|errno\s*28",
            error_text,
            re.IGNORECASE,
        ):
            raise RuntimeError("可用磁盘空间不足，外部媒体处理命令未能完成")
        if any(
            entry in command
            for entry in ("demucs.separate", "--internal-demucs")
        ):
            if return_code in {-9, 137, 3221225495} or re.search(
                r"out of memory|cannot allocate memory|bad_alloc|killed",
                error_text,
                re.IGNORECASE,
            ):
                raise RuntimeError(
                    "音轨分离内存不足，请关闭其他大型程序后重试"
                )
            if re.search(
                r"http error|url error|connection|timed out|ssl|download",
                error_text,
                re.IGNORECASE,
            ):
                raise RuntimeError(
                    "Demucs 模型下载或加载失败，请检查网络和系统代理后重试"
                )
            if re.search(
                r"no module named|modulenotfounderror|dll load failed|"
                r"winerror\s*126",
                error_text,
                re.IGNORECASE,
            ):
                raise RuntimeError(
                    "音轨分离组件缺失或损坏，请重新安装当前版本"
                )
            raise RuntimeError(
                f"音轨分离进程异常退出（代码 {return_code}）"
            )
        raise RuntimeError(
            f"Command failed ({return_code}): {' '.join(command)}"
        )


def require_command(command: str) -> str:
    resolved = shutil.which(command)
    if not resolved:
        raise RuntimeError(f"Required executable is unavailable: {command}")
    return resolved


def python_module_command(module: str) -> list[str]:
    if not getattr(sys, "frozen", False):
        return [sys.executable, "-m", module]
    internal_entrypoints = {
        "demucs.separate": "--internal-demucs",
        "yt_dlp": "--internal-yt-dlp",
    }
    entrypoint = internal_entrypoints.get(module)
    if not entrypoint:
        raise RuntimeError(f"Packaged module entrypoint is unavailable: {module}")
    return [sys.executable, entrypoint]


def run_internal_module() -> bool:
    if len(sys.argv) < 2:
        return False
    if sys.argv[1] == "--internal-demucs":
        from demucs.separate import main as demucs_main

        demucs_main(sys.argv[2:])
        return True
    if sys.argv[1] == "--internal-yt-dlp":
        import yt_dlp

        yt_dlp.main(sys.argv[2:])
        return True
    if sys.argv[1] == "--internal-qqmusic-credential":
        from qqmusic_windows import print_qqmusic_windows_credential

        print_qqmusic_windows_credential()
        return True
    if sys.argv[1] == "--internal-qqmusic-runtime-self-test":
        from qqmusic_windows import print_qqmusic_windows_runtime_self_test

        print_qqmusic_windows_runtime_self_test()
        return True
    return False


def ffmpeg_command() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    if configured and Path(configured).is_file():
        return configured
    return require_command("ffmpeg")


def acquire_url(
    url: str,
    destination: Path,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    ffmpeg = ffmpeg_command()
    output = destination / "source.%(ext)s"
    javascript_runtime = (
        ["--js-runtimes", "node"]
        if shutil.which("node")
        else []
    )
    session_options: list[str] = []
    configured_cookies_file = (
        cookies_file
        or (
            Path(os.environ["GTS_MEDIA_COOKIES_FILE"])
            if os.environ.get("GTS_MEDIA_COOKIES_FILE")
            else None
        )
    )
    if configured_cookies_file:
        cookies_path = configured_cookies_file.resolve()
        if (
            not cookies_path.is_file()
            or cookies_path.stat().st_size <= 0
            or cookies_path.stat().st_size > 2 * 1024 * 1024
        ):
            raise RuntimeError("Configured media cookies file is unavailable")
        session_options.extend(["--cookies", str(cookies_path)])
    elif cookies_from_browser:
        supported_browsers = {
            "brave",
            "chrome",
            "chromium",
            "edge",
            "firefox",
            "opera",
            "safari",
            "vivaldi",
        }
        if cookies_from_browser not in supported_browsers:
            raise RuntimeError(
                "Configured browser cookie source is unsupported"
            )
        session_options.extend(
            ["--cookies-from-browser", cookies_from_browser]
        )
    proxy = os.environ.get("GTS_MEDIA_PROXY")
    if proxy:
        session_options.extend(["--proxy", proxy])
    user_agent = os.environ.get("GTS_MEDIA_USER_AGENT")
    if user_agent:
        session_options.extend(["--user-agent", user_agent])
    run(
        [
            *python_module_command("yt_dlp"),
            "--no-config",
            "--no-playlist",
            "--no-cache-dir",
            "--extractor-retries",
            "3",
            "--fragment-retries",
            "3",
            *javascript_runtime,
            *session_options,
            "--write-info-json",
            "--format",
            "bestaudio[acodec!=none]/best[acodec!=none]/bestaudio/best",
            "--ffmpeg-location",
            ffmpeg,
            "--output",
            str(output),
            url,
        ]
    )
    matches = sorted(
        path
        for path in destination.glob("source.*")
        if path.suffix not in {".json", ".part", ".ytdl"}
    )
    if not matches:
        raise RuntimeError("The media adapter did not produce an audio file")
    info_path = destination / "source.info.json"
    metadata: dict[str, object] = {"title": matches[0].stem, "artist": "在线媒体"}
    embedded_lyrics: list[dict[str, object]] = []
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        creators = info.get("creators")
        creator = (
            creators[0]
            if isinstance(creators, list) and creators
            else info.get("creator")
        )
        metadata = {
            "title": info.get("title") or matches[0].stem,
            "artist": (
                info.get("uploader")
                or info.get("artist")
                or info.get("album_artist")
                or creator
                or "在线媒体"
            ),
            "duration": info.get("duration"),
            "webpage_url": info.get("webpage_url") or url,
        }
        description = info.get("description")
        if isinstance(description, str):
            embedded_lyrics = parse_timed_lyric_text(description)
    timed_lyrics = fetch_timed_lyric_lines(url) or embedded_lyrics
    if timed_lyrics:
        metadata["timed_lyrics"] = timed_lyrics
    return matches[0], metadata


def parse_timed_lyric_text(raw: str) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []
    for line in raw.splitlines():
        timestamp = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
        if not timestamp:
            continue
        text = timestamp.group(3).strip()
        if (
            not text
            or re.match(r"^(作词|作曲|编曲|制作|混音|母带)\s*[:：]", text)
        ):
            continue
        lines.append(
            {
                "at": int(timestamp.group(1)) * 60 + float(timestamp.group(2)),
                "text": text,
            }
        )
    return lines


def fetch_timed_lyric_lines(url: str) -> list[dict[str, object]]:
    """Fetch public timestamped lyrics when the source exposes them."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    if "music.163.com" not in url:
        return []
    matched = re.search(r"(?:id=|song/)(\d+)", url)
    if not matched:
        return []
    endpoint = "https://music.163.com/api/song/lyric?" + urlencode(
        {
            "id": matched.group(1),
            "lv": 1,
            "kv": 1,
            "tv": -1,
        }
    )
    try:
        request = Request(
            endpoint,
            headers={
                "Referer": "https://music.163.com/",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urlopen(request, timeout=8) as response:
            payload = json.load(response)
        raw = payload.get("lrc", {}).get("lyric", "")
    except Exception:
        return []
    return parse_timed_lyric_text(str(raw))


def normalize_audio(source: Path, destination: Path) -> Path:
    ffmpeg = ffmpeg_command()
    normalized = destination / "normalized.wav"
    normalized.unlink(missing_ok=True)
    available_bytes = shutil.disk_usage(destination).free
    required_bytes = max(MIN_ANALYSIS_FREE_BYTES, source.stat().st_size * 4)
    if available_bytes < required_bytes:
        raise RuntimeError(
            "可用磁盘空间不足，无法安全完成音频标准化与分轨"
        )
    try:
        run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(normalized),
            ]
        )
    except Exception:
        normalized.unlink(missing_ok=True)
        raise
    return normalized


def decoded_audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(path)
    return float(info.frames / info.samplerate)


def extract_vocal_melody(path: Path) -> list[dict[str, object]]:
    """Extract a conservative monophonic melody track from the vocal stem."""
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter

    audio, sample_rate = librosa.load(path, sr=16000, mono=True)
    if audio.size == 0:
        return []
    hop_length = 256
    f0, voiced, probability = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sample_rate,
        frame_length=2048,
        hop_length=hop_length,
    )
    probability = np.nan_to_num(probability, nan=0.0)
    midi = np.nan_to_num(librosa.hz_to_midi(f0), nan=-1.0)
    midi = median_filter(midi, size=5, mode="nearest")
    pitches = np.rint(midi).astype(np.int16)
    active = voiced & (probability >= 0.35) & (pitches >= 36) & (pitches <= 84)
    frame_times = librosa.frames_to_time(
        np.arange(len(pitches)),
        sr=sample_rate,
        hop_length=hop_length,
    )
    notes: list[dict[str, object]] = []
    start = 0
    while start < len(pitches):
        if not active[start]:
            start += 1
            continue
        pitch = int(pitches[start])
        end = start + 1
        while (
            end < len(pitches)
            and active[end]
            and int(pitches[end]) == pitch
        ):
            end += 1
        duration = (end - start) * hop_length / sample_rate
        if duration >= 0.064:
            segment = pitches[start:end][active[start:end]]
            resolved_pitch = int(np.median(segment)) if segment.size else pitch
            note = {
                "start": round(float(frame_times[start]), 4),
                "end": round(float(frame_times[min(end, len(frame_times) - 1)]), 4),
                "pitch": resolved_pitch,
                "confidence": round(float(np.mean(probability[start:end])), 4),
            }
            previous = notes[-1] if notes else None
            if (
                previous
                and previous["pitch"] == note["pitch"]
                and float(note["start"]) - float(previous["end"]) <= 0.1
            ):
                previous["end"] = note["end"]
                previous["confidence"] = round(
                    (
                        float(previous["confidence"])
                        + float(note["confidence"])
                    )
                    / 2,
                    4,
                )
            else:
                notes.append(note)
        start = end
    previous: dict[str, object] | None = None
    for note in notes:
        if previous and float(note["start"]) - float(previous["end"]) <= 0.9:
            candidates = [
                int(note["pitch"]) + octave
                for octave in (-12, 0, 12)
                if 36 <= int(note["pitch"]) + octave <= 84
            ]
            note["pitch"] = min(
                candidates,
                key=lambda candidate: abs(candidate - int(previous["pitch"])),
            )
        previous = note
    return notes


def align_lyrics_to_melody(
    lyric_lines: list[dict[str, object]],
    melody: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Place lyric words on nearby vocal-note attacks."""
    tokens: list[dict[str, object]] = []
    for index, line in enumerate(lyric_lines):
        start = float(line["at"])
        end = (
            float(lyric_lines[index + 1]["at"])
            if index + 1 < len(lyric_lines)
            else start + 6
        )
        text = str(line["text"]).strip()
        if re.search(r"[\u3400-\u9fff]", text):
            words = [
                character
                for character in text
                if not character.isspace() and character not in "，。！？、；："
            ]
        else:
            words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?", text)
        if not words:
            continue
        attacks = [
            float(note["start"])
            for note in melody
            if start <= float(note["start"]) < end
        ]
        for word_index, word in enumerate(words):
            target = start + (end - start) * (
                (word_index + 0.35) / max(1, len(words))
            )
            if len(attacks) >= len(words):
                attack_index = min(
                    len(attacks) - 1,
                    round(
                        word_index
                        * (len(attacks) - 1)
                        / max(1, len(words) - 1)
                    ),
                )
                at = attacks[attack_index]
            else:
                at = target
            tokens.append({"at": round(at, 4), "text": word})
    return tokens


def complete_vocal_melody_for_lyrics(
    melody: list[dict[str, object]],
    lyrics: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Ensure every timed lyric has a readable numbered-note anchor."""
    if not melody or not lyrics:
        return melody
    completed = [dict(note) for note in melody]
    original = [dict(note) for note in melody]
    for lyric_index, lyric in enumerate(lyrics):
        at = float(lyric["at"])
        if any(abs(float(note["start"]) - at) <= 0.025 for note in completed):
            continue
        nearest = min(
            original,
            key=lambda note: min(
                abs(float(note["start"]) - at),
                abs(float(note["end"]) - at),
            ),
        )
        distance = min(
            abs(float(nearest["start"]) - at),
            abs(float(nearest["end"]) - at),
        )
        next_at = (
            float(lyrics[lyric_index + 1]["at"])
            if lyric_index + 1 < len(lyrics)
            else at + 0.5
        )
        completed.append(
            {
                "start": round(at, 4),
                "end": round(
                    at + max(0.08, min(0.5, next_at - at)),
                    4,
                ),
                "pitch": int(nearest["pitch"]),
                "confidence": round(
                    max(
                        0.15 if distance > 0.85 else 0.2,
                        float(nearest["confidence"])
                        * (0.5 if distance > 0.85 else 0.72),
                    ),
                    4,
                ),
            }
        )
    completed.sort(key=lambda note: (float(note["start"]), int(note["pitch"])))
    return completed


def separate_audio(source: Path, destination: Path) -> dict[str, Path]:
    stems_root = destination / "stems"
    available_bytes = shutil.disk_usage(destination).free
    required_bytes = max(
        MIN_SEPARATION_FREE_BYTES,
        source.stat().st_size * 10,
    )
    if available_bytes < required_bytes:
        raise RuntimeError(
            "可用磁盘空间不足，无法安全完成六轨音频分离"
        )
    command = [
        *python_module_command("demucs.separate"),
        "-n",
        "htdemucs_6s",
        "--out",
        str(stems_root),
    ]
    model_repository = os.environ.get("DEMUCS_MODEL_REPOSITORY")
    if model_repository:
        repository = Path(model_repository)
        if not repository.is_dir():
            raise RuntimeError("Configured Demucs model repository is unavailable")
        command.extend(["--repo", str(repository)])
    command.append(str(source))
    run_with_percent_progress(
        command,
        start=22,
        end=50,
        status="separating",
        stage_label="正在分离人声与乐器",
    )
    result_root = stems_root / "htdemucs_6s" / source.stem
    expected = ("vocals", "guitar", "bass", "drums", "piano", "other")
    stems = {name: result_root / f"{name}.wav" for name in expected}
    missing = [name for name, path in stems.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Demucs output is incomplete: {', '.join(missing)}")
    return stems


def suppress_guitar_stem_leakage(
    guitar: Path,
    references: dict[str, Path],
    destination: Path,
) -> tuple[Path, dict[str, float]]:
    """Remove only components coherent with independently separated stems."""
    import librosa
    import numpy as np
    import soundfile as sf
    from scipy.ndimage import uniform_filter

    audio, sample_rate = librosa.load(guitar, sr=22050, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    if audio.size == 0:
        return guitar, {
            "removed_energy_ratio": 0.0,
            "retained_rms_ratio": 1.0,
        }

    original_rms = float(np.sqrt(np.mean(np.square(audio))))
    cleaned_channels: list[np.ndarray] = []
    for channel_index in range(audio.shape[0]):
        source_channel = audio[channel_index]
        source_spectrum = librosa.stft(
            source_channel,
            n_fft=2048,
            hop_length=512,
        )
        residual = source_spectrum.copy()
        reference_peak = np.zeros_like(np.abs(source_spectrum))
        for reference in references.values():
            reference_audio, _ = librosa.load(
                reference,
                sr=sample_rate,
                mono=False,
            )
            if reference_audio.ndim == 1:
                reference_channel = reference_audio
            else:
                reference_channel = reference_audio[
                    min(channel_index, reference_audio.shape[0] - 1)
                ]
            aligned = np.zeros_like(source_channel)
            sample_count = min(len(aligned), len(reference_channel))
            aligned[:sample_count] = reference_channel[:sample_count]
            if float(np.sqrt(np.mean(np.square(aligned)))) < original_rms * 0.01:
                continue
            reference_spectrum = librosa.stft(
                aligned,
                n_fft=2048,
                hop_length=512,
            )
            frame_count = min(
                residual.shape[1],
                reference_spectrum.shape[1],
            )
            current = residual[:, :frame_count]
            reference_slice = reference_spectrum[:, :frame_count]
            reference_peak[:, :frame_count] = np.maximum(
                reference_peak[:, :frame_count],
                np.abs(reference_slice),
            )
            cross_power = np.sum(
                current * np.conjugate(reference_slice),
                axis=1,
            )
            current_power = np.sum(np.square(np.abs(current)), axis=1)
            reference_power = np.sum(
                np.square(np.abs(reference_slice)),
                axis=1,
            )
            coherence = np.square(np.abs(cross_power)) / np.maximum(
                current_power * reference_power,
                1e-12,
            )
            suppression = np.clip((coherence - 0.35) / 0.45, 0.0, 1.0)
            transfer = cross_power / np.maximum(reference_power, 1e-12)
            transfer_magnitude = np.abs(transfer)
            transfer *= np.minimum(
                1.0,
                0.8 / np.maximum(transfer_magnitude, 1e-8),
            )
            residual[:, :frame_count] = (
                current
                - transfer[:, None]
                * suppression[:, None]
                * reference_slice
            )
        dominance = np.abs(source_spectrum) / np.maximum(
            reference_peak,
            1e-8,
        )
        competition_mask = 0.18 + 0.82 * np.clip(
            (dominance - 0.5) / 1.5,
            0.0,
            1.0,
        )
        competition_mask = uniform_filter(
            competition_mask,
            size=(3, 5),
            mode="nearest",
        )
        residual *= competition_mask
        cleaned_channels.append(
            librosa.istft(
                residual,
                hop_length=512,
                length=len(source_channel),
            )
        )

    cleaned = np.stack(cleaned_channels)
    cleaned_rms = float(np.sqrt(np.mean(np.square(cleaned))))
    retained_rms_ratio = cleaned_rms / max(original_rms, 1e-8)
    # Keep the processor conservative when references are musically correlated.
    if retained_rms_ratio < 0.72:
        original_share = (0.72 - retained_rms_ratio) / max(
            1e-8,
            1.0 - retained_rms_ratio,
        )
        cleaned = cleaned * (1.0 - original_share) + audio * original_share
        cleaned_rms = float(np.sqrt(np.mean(np.square(cleaned))))
        retained_rms_ratio = cleaned_rms / max(original_rms, 1e-8)

    removed = audio - cleaned
    removed_energy_ratio = float(
        np.sum(np.square(removed)) /
        max(float(np.sum(np.square(audio))), 1e-12)
    )
    cleaned_root = destination / "cleaned-stems"
    cleaned_root.mkdir(parents=True, exist_ok=True)
    path = cleaned_root / "guitar.wav"
    sf.write(
        path,
        np.clip(cleaned.T, -1, 1),
        sample_rate,
        subtype="PCM_16",
    )
    return path, {
        "removed_energy_ratio": round(removed_energy_ratio, 6),
        "retained_rms_ratio": round(retained_rms_ratio, 6),
    }


def separate_guitar_parts(
    guitar: Path,
    destination: Path,
) -> list[dict[str, object]]:
    """Split spatially distinct guitars without inventing duplicate tracks."""
    import librosa
    import numpy as np
    import soundfile as sf

    audio, sample_rate = librosa.load(guitar, sr=22050, mono=False)
    if audio.ndim != 2 or audio.shape[0] < 2 or audio.shape[1] < sample_rate * 4:
        return []
    left = librosa.stft(audio[0], n_fft=2048, hop_length=512)
    right = librosa.stft(audio[1], n_fft=2048, hop_length=512)
    left_magnitude = np.abs(left)
    right_magnitude = np.abs(right)
    energy = left_magnitude + right_magnitude
    active = energy >= np.percentile(energy, 72)
    if not np.any(active):
        return []
    balance = (right_magnitude - left_magnitude) / np.maximum(energy, 1e-8)
    values = balance[active]
    weights = energy[active]
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= max(float(cumulative[-1]), 1e-8)
    centers = np.array(
        [
            sorted_values[np.searchsorted(cumulative, 0.28)],
            sorted_values[np.searchsorted(cumulative, 0.72)],
        ],
        dtype=np.float32,
    )
    for _iteration in range(8):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        updated = []
        for index in range(2):
            selected = labels == index
            if not np.any(selected):
                return []
            updated.append(
                float(np.average(values[selected], weights=weights[selected]))
            )
        centers = np.asarray(updated, dtype=np.float32)
    if centers[0] > centers[1]:
        centers = centers[::-1]
    center_distance = float(centers[1] - centers[0])
    distances = np.abs(values[:, None] - centers[None, :])
    labels = np.argmin(distances, axis=1)
    shares = np.array(
        [
            float(np.sum(weights[labels == index]) / np.sum(weights))
            for index in range(2)
        ]
    )
    # A narrow stereo field is one aggregate guitar, not evidence of two parts.
    if center_distance < 0.16 or float(np.min(shares)) < 0.14:
        return []

    part_root = destination / "guitar-parts"
    part_root.mkdir(parents=True, exist_ok=True)
    try:
        import warnings

        from sklearn.decomposition import FastICA
        from sklearn.exceptions import ConvergenceWarning

        observations = audio.T.astype(np.float64)
        stride = max(1, len(observations) // 300_000)
        separator = FastICA(
            n_components=2,
            whiten="unit-variance",
            max_iter=1000,
            tol=1e-5,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            separator.fit(observations[::stride])
        sources = separator.transform(observations)
        contributions = [
            np.outer(sources[:, index], separator.mixing_[:, index])
            + separator.mean_ / 2
            for index in range(2)
        ]

        def spatial_balance(part: np.ndarray) -> float:
            left_level = float(np.sqrt(np.mean(np.square(part[:, 0]))))
            right_level = float(np.sqrt(np.mean(np.square(part[:, 1]))))
            return (right_level - left_level) / max(
                right_level + left_level,
                1e-8,
            )

        contributions.sort(key=spatial_balance)
        component_levels = [
            float(np.sqrt(np.mean(np.square(part))))
            for part in contributions
        ]
        component_correlation = float(
            np.corrcoef(
                np.mean(contributions[0], axis=1),
                np.mean(contributions[1], axis=1),
            )[0, 1]
        )
        component_balance_distance = (
            spatial_balance(contributions[1])
            - spatial_balance(contributions[0])
        )
        if (
            math.isfinite(component_correlation)
            and abs(component_correlation) <= 0.2
            and component_balance_distance >= 0.16
            and min(component_levels) >= max(component_levels) * 0.08
        ):
            parts: list[dict[str, object]] = []
            for index, part in enumerate(contributions):
                path = part_root / f"guitar_{index + 1}.wav"
                sf.write(
                    path,
                    np.clip(part, -1, 1),
                    sample_rate,
                    subtype="PCM_16",
                )
                parts.append(
                    {
                        "id": f"guitar-{index + 1}",
                        "name": f"吉他 {index + 1}",
                        "role": "guitar",
                        "path": path,
                        "separation_confidence": round(
                            min(0.97, 0.68 + component_balance_distance * 0.2),
                            4,
                        ),
                        "separation_method": "fastica-stereo-unmix-v1",
                    }
                )
            return parts
    except Exception:
        pass

    scale = max(0.06, center_distance * 0.42)
    log_scores = np.stack(
        [
            -8.0 * np.square((balance - centers[index]) / scale)
            for index in range(2)
        ]
    )
    # Sharpen the partition without breaking reconstruction. The previous
    # broad masks left an audible copy of each spatial part in the other.
    scores = np.exp(log_scores - np.max(log_scores, axis=0, keepdims=True))
    masks = scores / np.sum(scores, axis=0, keepdims=True)
    parts: list[dict[str, object]] = []
    reconstructed: list[np.ndarray] = []
    for index in range(2):
        channels = [
            librosa.istft(
                channel * masks[index],
                hop_length=512,
                length=audio.shape[1],
            )
            for channel in (left, right)
        ]
        part = np.stack(channels, axis=1)
        reconstructed.append(np.mean(part, axis=1))
        path = part_root / f"guitar_{index + 1}.wav"
        sf.write(path, np.clip(part, -1, 1), sample_rate, subtype="PCM_16")
        parts.append(
            {
                "id": f"guitar-{index + 1}",
                "name": f"吉他 {index + 1}",
                "role": "guitar",
                "path": path,
                "separation_confidence": round(
                    min(0.95, 0.55 + center_distance * 0.8), 4
                ),
                "separation_method": "coherent-leakage-suppression+stereo-spatial-mask-v3",
            }
        )
    correlation = float(np.corrcoef(reconstructed)[0, 1])
    if not math.isfinite(correlation) or abs(correlation) > 0.985:
        for part in parts:
            Path(part["path"]).unlink(missing_ok=True)
        return []
    return parts


def suppress_vocal_leakage_in_guitar_part(
    guitar_part: Path,
    vocals: Path,
    destination: Path,
) -> tuple[Path, dict[str, float]]:
    """Attenuate vocal-shaped bins after spatial guitar separation."""
    import librosa
    import numpy as np
    import soundfile as sf
    from scipy.ndimage import uniform_filter

    sample_rate = 22050
    part_audio, _ = librosa.load(
        guitar_part,
        sr=sample_rate,
        mono=False,
    )
    vocal_audio, _ = librosa.load(
        vocals,
        sr=sample_rate,
        mono=False,
    )
    if part_audio.ndim == 1:
        part_audio = np.stack([part_audio, part_audio])
    if vocal_audio.ndim == 1:
        vocal_audio = np.stack([vocal_audio, vocal_audio])
    sample_count = min(part_audio.shape[1], vocal_audio.shape[1])
    if sample_count == 0:
        return guitar_part, {
            "retained_rms_ratio": 1.0,
            "vocal_similarity_reduction": 0.0,
        }
    part_audio = part_audio[:, :sample_count]
    vocal_audio = vocal_audio[:, :sample_count]
    part_rms = float(np.sqrt(np.mean(np.square(part_audio))))
    vocal_rms = float(np.sqrt(np.mean(np.square(vocal_audio))))
    reference_scale = min(
        1.5,
        max(0.12, part_rms / max(vocal_rms, 1e-8) * 3.4),
    )

    cleaned_channels: list[np.ndarray] = []
    before_similarity: list[np.ndarray] = []
    after_similarity: list[np.ndarray] = []
    for channel_index in range(part_audio.shape[0]):
        part_spectrum = librosa.stft(
            part_audio[channel_index],
            n_fft=2048,
            hop_length=256,
        )
        vocal_spectrum = librosa.stft(
            vocal_audio[min(channel_index, vocal_audio.shape[0] - 1)],
            n_fft=2048,
            hop_length=256,
        )
        frame_count = min(part_spectrum.shape[1], vocal_spectrum.shape[1])
        part_spectrum = part_spectrum[:, :frame_count]
        vocal_spectrum = vocal_spectrum[:, :frame_count]
        part_magnitude = np.abs(part_spectrum)
        vocal_magnitude = np.abs(vocal_spectrum)
        vocal_frame_rms = np.sqrt(np.mean(np.square(vocal_magnitude), axis=0))
        vocal_activity = np.clip(
            (
                librosa.amplitude_to_db(
                    vocal_frame_rms,
                    ref=np.max,
                )
                + 45
            )
            / 15,
            0,
            1,
        )
        estimated_leakage = reference_scale * vocal_magnitude
        mask = np.square(part_magnitude) / np.maximum(
            np.square(part_magnitude) + np.square(estimated_leakage),
            1e-12,
        )
        mask = np.maximum(
            0.06,
            uniform_filter(mask, size=(3, 7), mode="nearest"),
        )
        mask = 1 - vocal_activity[None, :] * (1 - mask)
        cleaned_spectrum = part_spectrum * mask
        cleaned_channels.append(
            librosa.istft(
                cleaned_spectrum,
                hop_length=256,
                length=sample_count,
            )
        )

        active = vocal_activity >= 0.25
        if np.any(active):
            denominator = (
                np.linalg.norm(part_magnitude, axis=0)
                * np.linalg.norm(vocal_magnitude, axis=0)
                + 1e-12
            )
            before_similarity.append(
                (
                    np.sum(part_magnitude * vocal_magnitude, axis=0)
                    / denominator
                )[active]
            )
            cleaned_magnitude = np.abs(cleaned_spectrum)
            cleaned_denominator = (
                np.linalg.norm(cleaned_magnitude, axis=0)
                * np.linalg.norm(vocal_magnitude, axis=0)
                + 1e-12
            )
            after_similarity.append(
                (
                    np.sum(cleaned_magnitude * vocal_magnitude, axis=0)
                    / cleaned_denominator
                )[active]
            )

    cleaned = np.stack(cleaned_channels)
    cleaned_rms = float(np.sqrt(np.mean(np.square(cleaned))))
    before = (
        float(np.median(np.concatenate(before_similarity)))
        if before_similarity
        else 0.0
    )
    after = (
        float(np.median(np.concatenate(after_similarity)))
        if after_similarity
        else before
    )
    cleaned_root = destination / "debleeded-guitar-parts"
    cleaned_root.mkdir(parents=True, exist_ok=True)
    path = cleaned_root / f"{guitar_part.stem}.wav"
    sf.write(
        path,
        np.clip(cleaned.T, -1, 1),
        sample_rate,
        subtype="PCM_16",
    )
    return path, {
        "retained_rms_ratio": round(
            cleaned_rms / max(part_rms, 1e-8),
            6,
        ),
        "vocal_similarity_reduction": round(
            max(0.0, 1.0 - after / max(before, 1e-8)),
            6,
        ),
    }


def stabilize_beat_times(
    beat_times: Iterable[float],
    *,
    maximum_correction: float = 0.025,
    passes: int = 2,
) -> list[float]:
    stabilized = [float(time) for time in beat_times]
    if len(stabilized) < 3:
        return stabilized
    for _pass in range(passes):
        previous_pass = stabilized
        stabilized = list(previous_pass)
        for index in range(1, len(previous_pass) - 1):
            midpoint = (previous_pass[index - 1] + previous_pass[index + 1]) / 2
            correction = midpoint - previous_pass[index]
            if abs(correction) <= maximum_correction:
                stabilized[index] += correction * 0.5
    return stabilized


def regularize_beat_times(
    beat_times: Iterable[float],
    *,
    maximum_window: int = 49,
) -> list[float]:
    """Reduce local detector jitter without flattening real tempo movement."""
    import numpy as np

    times = np.asarray([float(time) for time in beat_times], dtype=np.float64)
    if len(times) < 3:
        return times.tolist()
    intervals = np.diff(times)
    valid = intervals[np.isfinite(intervals) & (intervals > 0.08)]
    if valid.size == 0:
        return times.tolist()
    median_interval = float(np.median(valid))
    maximum_correction = min(0.05, median_interval * 0.1)
    passes = max(2, min(4, maximum_window // 16))
    return stabilize_beat_times(
        times.tolist(),
        maximum_correction=maximum_correction,
        passes=passes,
    )


def tempo_map_from_beats(
    beat_times: Iterable[float],
    fallback_bpm: float = 120.0,
) -> dict[str, object]:
    import numpy as np

    beats = [
        float(beat)
        for beat in beat_times
        if math.isfinite(float(beat)) and float(beat) >= 0
    ]
    beats = [
        beat
        for index, beat in enumerate(beats)
        if index == 0 or beat - beats[index - 1] > 0.05
    ]
    safe_fallback = (
        float(fallback_bpm)
        if math.isfinite(fallback_bpm) and fallback_bpm > 0
        else 120.0
    )
    if len(beats) < 2:
        return {
            "mode": "steady",
            "median_bpm": round(safe_fallback, 4),
            "variation": 0.0,
            "points": [
                {
                    "at": round(beats[0] if beats else 0.0, 4),
                    "beat": 0,
                    "bpm": round(safe_fallback, 4),
                    "confidence": 0.0,
                }
            ],
        }

    intervals = np.diff(np.asarray(beats, dtype=np.float64))
    points: list[dict[str, float | int]] = []
    for index, at in enumerate(beats):
        start = max(0, index - 2)
        end = min(len(intervals), index + 3)
        local = intervals[start:end]
        local = local[
            np.isfinite(local) & (local >= 0.15) & (local <= 3.0)
        ]
        local_median = (
            float(np.median(local))
            if local.size
            else 60.0 / safe_fallback
        )
        relative_deviation = (
            float(np.median(np.abs(local - local_median)))
            / max(local_median, 1e-6)
            if local.size
            else 1.0
        )
        points.append(
            {
                "at": round(at, 4),
                "beat": index,
                "bpm": round(60.0 / local_median, 4),
                "confidence": round(
                    max(0.0, min(1.0, 1.0 - relative_deviation * 6.0)),
                    4,
                ),
            }
        )
    tempos = np.asarray(
        [
            float(point["bpm"])
            for point in points
            if 20 <= float(point["bpm"]) <= 320
        ],
        dtype=np.float64,
    )
    median_bpm = (
        float(np.median(tempos))
        if tempos.size
        else safe_fallback
    )
    variation = (
        float(
            (np.quantile(tempos, 0.9) - np.quantile(tempos, 0.1))
            / max(median_bpm, 1.0)
        )
        if tempos.size > 1
        else 0.0
    )
    return {
        "mode": "variable" if variation >= 0.04 else "steady",
        "median_bpm": round(median_bpm, 4),
        "variation": round(variation, 6),
        "points": points,
    }


def beat_grid_irregularity(beat_times: Iterable[float]) -> float:
    """Score detector jitter and isolated edge beats; lower is better."""
    import numpy as np

    beats = np.asarray(
        [
            float(value)
            for value in beat_times
            if math.isfinite(float(value)) and float(value) >= 0
        ],
        dtype=np.float64,
    )
    if beats.size < 8:
        return math.inf
    raw_intervals = np.diff(beats)
    valid_mask = (
        np.isfinite(raw_intervals)
        & (raw_intervals >= 0.15)
        & (raw_intervals <= 3.0)
    )
    intervals = raw_intervals[valid_mask]
    if intervals.size < 7:
        return math.inf
    median_interval = float(np.median(intervals))
    relative_errors = np.abs(intervals / median_interval - 1.0)
    opening_error = float(np.mean(relative_errors[: min(2, len(relative_errors))]))
    invalid_interval_ratio = 1.0 - intervals.size / max(1, raw_intervals.size)
    return float(
        np.median(relative_errors)
        + np.quantile(relative_errors, 0.9)
        + opening_error * 0.5
        + invalid_interval_ratio * 2.0
    )


def _grid_alignment(
    reference: Iterable[float],
    candidate: Iterable[float],
) -> tuple[float, float | None]:
    import numpy as np

    reference_beats = sorted(float(value) for value in reference)
    candidate_beats = sorted(float(value) for value in candidate)
    if len(reference_beats) < 2 or not candidate_beats:
        return 0.0, None
    intervals = np.diff(np.asarray(reference_beats, dtype=np.float64))
    intervals = intervals[
        np.isfinite(intervals) & (intervals >= 0.15) & (intervals <= 3.0)
    ]
    if intervals.size == 0:
        return 0.0, None
    tolerance = min(0.07, max(0.025, float(np.median(intervals)) * 0.2))
    errors: list[float] = []
    for beat in candidate_beats:
        position = bisect_left(reference_beats, beat)
        neighbours = reference_beats[
            max(0, position - 1) : min(len(reference_beats), position + 1)
        ]
        if neighbours:
            errors.append(min(abs(beat - value) for value in neighbours))
    if not errors:
        return 0.0, None
    aligned = [error for error in errors if error <= tolerance]
    coverage = len(aligned) / len(errors)
    median_error = float(np.median(aligned)) if aligned else None
    return coverage, median_error


def timing_grid_diagnostics(
    mix_bpm: float,
    mix_beats: Iterable[float],
    percussion_bpm: float,
    percussion_beats: Iterable[float],
) -> dict[str, object]:
    mix = [float(value) for value in mix_beats]
    percussion = [float(value) for value in percussion_beats]
    ratio = percussion_bpm / max(mix_bpm, 1e-8)
    if 0.9 <= ratio <= 1.1:
        relationship = "same-tempo"
        alignment_reference = mix
        alignment_candidate = percussion
    elif 0.45 <= ratio <= 0.55:
        relationship = "percussion-half-tempo"
        alignment_reference = mix
        alignment_candidate = percussion
    elif 1.8 <= ratio <= 2.2:
        relationship = "percussion-double-tempo"
        alignment_reference = percussion
        alignment_candidate = mix
    else:
        relationship = "tempo-mismatch"
        alignment_reference = mix
        alignment_candidate = percussion
    alignment_coverage, median_alignment_error = _grid_alignment(
        alignment_reference,
        alignment_candidate,
    )

    def finite_score(values: list[float]) -> float | None:
        score = beat_grid_irregularity(values)
        return round(score, 6) if math.isfinite(score) else None

    return {
        "mix_bpm": round(float(mix_bpm), 4),
        "percussion_bpm": round(float(percussion_bpm), 4),
        "bpm_ratio": round(float(ratio), 6),
        "relationship": relationship,
        "mix_beat_count": len(mix),
        "percussion_beat_count": len(percussion),
        "mix_irregularity": finite_score(mix),
        "percussion_irregularity": finite_score(percussion),
        "alignment_coverage": round(alignment_coverage, 6),
        "median_alignment_error_ms": (
            round(median_alignment_error * 1000, 3)
            if median_alignment_error is not None
            else None
        ),
    }


def select_timing_grid(
    mix_bpm: float,
    mix_beats: Iterable[float],
    percussion_bpm: float,
    percussion_beats: Iterable[float],
) -> tuple[list[float], str]:
    """Prefer a coherent percussion grid without accepting tempo mismatches."""
    mix = [float(value) for value in mix_beats]
    percussion = [float(value) for value in percussion_beats]
    diagnostics = timing_grid_diagnostics(
        mix_bpm,
        mix,
        percussion_bpm,
        percussion,
    )
    if len(percussion) < 16:
        return mix, "mix"
    mix_error = beat_grid_irregularity(mix)
    percussion_error = beat_grid_irregularity(percussion)
    relationship = diagnostics["relationship"]
    if (
        relationship == "same-tempo"
        and len(percussion) >= len(mix) * 0.75
        and percussion_error + 0.005 < mix_error
    ):
        return percussion, "drums"
    if (
        relationship == "percussion-half-tempo"
        and 0.4 <= len(percussion) / max(1, len(mix)) <= 0.6
        and float(diagnostics["alignment_coverage"]) >= 0.9
        and diagnostics["median_alignment_error_ms"] is not None
        and float(diagnostics["median_alignment_error_ms"]) <= 35
        and percussion_error <= mix_error + 0.005
    ):
        return percussion, "drums"
    return mix, "mix"


def timing_grid_selection_reason(
    diagnostics: dict[str, object],
    selected_source: str,
) -> str:
    relationship = diagnostics["relationship"]
    if selected_source == "drums":
        return (
            "aligned-half-tempo"
            if relationship == "percussion-half-tempo"
            else "lower-grid-irregularity"
        )
    if relationship == "percussion-half-tempo":
        return "half-tempo-alignment-insufficient"
    if relationship == "same-tempo":
        return "mix-grid-not-worse"
    return "tempo-relationship-rejected"


def _beat_meter_features(
    beat_times: Iterable[float],
    notes: Iterable[NoteEvent],
) -> tuple[list[float], list[float]]:
    beats = [float(value) for value in beat_times]
    note_list = list(notes)
    intervals = [
        current - previous
        for previous, current in zip(beats, beats[1:])
        if 0.15 <= current - previous <= 3.0
    ]
    fallback_interval = (
        sorted(intervals)[len(intervals) // 2]
        if intervals
        else 0.5
    )
    vectors: list[list[float]] = []
    attack_strengths: list[float] = []
    for index, start in enumerate(beats):
        end = (
            beats[index + 1]
            if index + 1 < len(beats)
            else start + fallback_interval
        )
        vector = [0.0] * 12
        attack_strength = 0.0
        for note in note_list:
            weight = (
                max(1, note.velocity)
                / 127.0
                * max(0.05, note.confidence)
            )
            overlap = max(
                0.0,
                min(end, note.end) - max(start, note.start),
            )
            if overlap > 0:
                vector[note.pitch % 12] += overlap * weight
            if start <= note.start < end:
                attack_strength += weight
        vectors.append(vector)
        attack_strengths.append(attack_strength)

    novelty: list[float] = []
    for index, current in enumerate(vectors):
        previous = vectors[index - 1] if index > 0 else [0.0] * 12
        current_norm = math.sqrt(sum(value * value for value in current))
        previous_norm = math.sqrt(sum(value * value for value in previous))
        if current_norm == 0 and previous_norm == 0:
            novelty.append(0.0)
        elif current_norm == 0 or previous_norm == 0:
            novelty.append(1.0)
        else:
            similarity = sum(
                left * right
                for left, right in zip(previous, current)
            ) / (previous_norm * current_norm)
            novelty.append(max(0.0, min(1.0, 1.0 - similarity)))

    positive_attacks = sorted(
        value for value in attack_strengths if value > 0
    )
    if positive_attacks:
        scale_index = round((len(positive_attacks) - 1) * 0.9)
        attack_scale = max(positive_attacks[scale_index], 1e-8)
        attacks = [
            max(0.0, min(1.0, value / attack_scale))
            for value in attack_strengths
        ]
    else:
        attacks = [0.0] * len(attack_strengths)
    return novelty, attacks


def _average(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _meter_candidate_score(
    novelty: list[float],
    attacks: list[float],
    signature: tuple[int, int],
) -> dict[str, object]:
    numerator, _denominator = signature
    margin = numerator
    usable = list(range(margin, max(margin, len(novelty) - margin)))
    novelty_profiles = [
        _average(
            novelty[index]
            for index in usable
            if index % numerator == phase
        )
        for phase in range(numerator)
    ]
    attack_profiles = [
        _average(
            attacks[index]
            for index in usable
            if index % numerator == phase
        )
        for phase in range(numerator)
    ]
    best: dict[str, object] | None = None
    for phase in range(numerator):
        other_phases = [
            candidate
            for candidate in range(numerator)
            if candidate != phase
        ]
        harmonic_contrast = max(
            0.0,
            novelty_profiles[phase]
            - _average(novelty_profiles[index] for index in other_phases),
        )
        accent_contrast = max(
            0.0,
            attack_profiles[phase]
            - _average(attack_profiles[index] for index in other_phases),
        )
        primary_values = [
            attacks[index]
            for index in usable
            if index % numerator == phase
        ]
        alternating_imbalance = abs(
            _average(primary_values[::2])
            - _average(primary_values[1::2])
        )
        compound_support = 0.0
        compound_hierarchy = 0.0
        score = (
            harmonic_contrast * 0.7
            + accent_contrast * 0.3
            - (
                alternating_imbalance * 0.4
                if numerator == 3
                else 0.0
            )
        )
        if signature == (6, 8):
            secondary = (phase + 3) % numerator
            unaccented = [
                candidate
                for candidate in range(numerator)
                if candidate not in {phase, secondary}
            ]
            compound_support = max(
                0.0,
                attack_profiles[secondary]
                - _average(attack_profiles[index] for index in unaccented),
            )
            compound_hierarchy = max(
                0.0,
                attack_profiles[phase] - attack_profiles[secondary],
            )
            score = (
                harmonic_contrast * 0.55
                + accent_contrast * 0.25
                + compound_support * 0.15
                + compound_hierarchy * 0.05
            )
        candidate_result: dict[str, object] = {
            "time_signature": list(signature),
            "score": max(0.0, score),
            "phase": phase,
            "measure_count": len(primary_values),
            "harmonic_contrast": harmonic_contrast,
            "accent_contrast": accent_contrast,
            "alternating_imbalance": alternating_imbalance,
            "compound_support": compound_support,
            "compound_hierarchy": compound_hierarchy,
        }
        if (
            best is None
            or float(candidate_result["score"]) > float(best["score"])
        ):
            best = candidate_result
    assert best is not None
    return best


def select_time_signature(
    beat_times: Iterable[float],
    notes: Iterable[NoteEvent],
) -> tuple[tuple[int, int], dict[str, object]]:
    """Select a stable 3/4, 4/4, or 6/8 meter, otherwise keep 4/4."""
    beats = [float(value) for value in beat_times]
    novelty, attacks = _beat_meter_features(beats, notes)
    signatures = ((3, 4), (4, 4), (6, 8))
    candidates = [
        _meter_candidate_score(novelty, attacks, signature)
        for signature in signatures
    ]

    def eligible(candidate: dict[str, object]) -> bool:
        signature = tuple(candidate["time_signature"])
        if int(candidate["measure_count"]) < 6:
            return False
        if float(candidate["score"]) < 0.14:
            return False
        if signature == (3, 4):
            return float(candidate["alternating_imbalance"]) <= 0.12
        if signature == (6, 8):
            return (
                float(candidate["compound_support"]) >= 0.08
                and float(candidate["compound_hierarchy"]) >= 0.04
            )
        return True

    ranked = sorted(
        (candidate for candidate in candidates if eligible(candidate)),
        key=lambda candidate: float(candidate["score"]),
        reverse=True,
    )
    selected = (4, 4)
    reason = "default-4-4-insufficient-evidence"
    confidence = 0.0
    if ranked:
        best = ranked[0]
        second_score = (
            float(ranked[1]["score"])
            if len(ranked) > 1
            else 0.0
        )
        confidence = max(0.0, float(best["score"]) - second_score)
        best_signature = tuple(int(value) for value in best["time_signature"])
        if best_signature == (4, 4) or confidence >= 0.04:
            selected = best_signature
            reason = "recurring-meter-evidence"
        else:
            confidence = 0.0

    rounded_candidates = [
        {
            **candidate,
            **{
                key: round(float(candidate[key]), 6)
                for key in (
                    "score",
                    "harmonic_contrast",
                    "accent_contrast",
                    "alternating_imbalance",
                    "compound_support",
                    "compound_hierarchy",
                )
            },
        }
        for candidate in candidates
    ]
    return selected, {
        "selected_time_signature": list(selected),
        "confidence": round(confidence, 6),
        "reason": reason,
        "candidates": rounded_candidates,
    }


def align_beats_to_downbeats(
    beat_times: Iterable[float],
    notes: Iterable[NoteEvent],
    beats_per_measure: int = 4,
) -> tuple[list[float], int, float]:
    """Align beat zero to the strongest recurring harmonic-change phase."""
    beats = [float(value) for value in beat_times]
    note_list = list(notes)
    if len(beats) < beats_per_measure * 6 or not note_list:
        return beats, 0, 0.0
    novelty, _attacks = _beat_meter_features(beats, note_list)

    margin = beats_per_measure * 2
    usable = range(margin, max(margin, len(beats) - margin))
    phase_scores = [
        [
            novelty[index]
            for index in usable
            if index % beats_per_measure == phase
        ]
        for phase in range(beats_per_measure)
    ]
    averages = [
        sum(values) / len(values) if values else 0.0
        for values in phase_scores
    ]
    ranked = sorted(
        range(beats_per_measure),
        key=lambda phase: averages[phase],
        reverse=True,
    )
    best = ranked[0]
    second = averages[ranked[1]] if len(ranked) > 1 else 0.0
    confidence = max(0.0, averages[best] - second)
    if (
        averages[best] < 0.12
        or confidence < 0.04
        or averages[best] < second * 1.2
    ):
        return beats, 0, confidence
    return beats[best:], best, confidence


def prepend_pickup_beats(
    beat_times: Iterable[float],
    notes: Iterable[NoteEvent],
    *,
    maximum_pickup_beats: int = 3,
) -> tuple[list[float], int]:
    """Reconstruct a short anacrusis before the first detected downbeat."""
    beats = [float(value) for value in beat_times]
    note_list = [
        note
        for note in notes
        if note.confidence >= 0.6 and note.start >= 0
    ]
    if len(beats) < 2 or not note_list:
        return beats, 0
    intervals = sorted(
        current - previous
        for previous, current in zip(beats, beats[1:])
        if 0.15 <= current - previous <= 3.0
    )
    if not intervals:
        return beats, 0
    interval = intervals[len(intervals) // 2]
    first_downbeat = beats[0]
    early_notes = [
        note.start
        for note in note_list
        if max(0.0, first_downbeat - maximum_pickup_beats * interval - 0.1)
        <= note.start
        < first_downbeat - min(0.06, interval * 0.1)
    ]
    if not early_notes:
        return beats, 0
    earliest = min(early_notes)
    pickup_beats = min(
        maximum_pickup_beats,
        max(
            1,
            math.ceil(
                (first_downbeat - earliest - interval * 0.15)
                / interval
            ),
        ),
    )
    while (
        pickup_beats > 0
        and first_downbeat - pickup_beats * interval < 0
    ):
        pickup_beats -= 1
    if pickup_beats == 0:
        return beats, 0
    pickup = [
        round(first_downbeat - index * interval, 4)
        for index in range(pickup_beats, 0, -1)
    ]
    return [*pickup, *beats], pickup_beats


def select_quarter_note_grid(
    onset_envelope: object,
    fast_tempo: float,
    fast_frames: Iterable[int],
    slow_tempo: float,
    slow_frames: Iterable[int],
) -> tuple[float, list[int]]:
    """Resolve a confident eighth-note/quarter-note octave ambiguity."""
    import numpy as np

    fast = [int(frame) for frame in fast_frames]
    slow = [int(frame) for frame in slow_frames]
    if (
        not math.isfinite(fast_tempo)
        or not math.isfinite(slow_tempo)
        or fast_tempo < 155
        or not 65 <= slow_tempo <= 100
        or not 1.85 <= fast_tempo / max(slow_tempo, 1e-8) <= 2.15
        or not 0.45 <= len(slow) / max(len(fast), 1) <= 0.55
    ):
        return fast_tempo, fast

    envelope = np.asarray(onset_envelope)

    def accent_contrast(frames: list[int]) -> float:
        strengths = np.asarray(
            [
                float(
                    np.max(
                        envelope[
                            max(0, frame - 1) :
                            min(len(envelope), frame + 2)
                        ],
                        initial=0.0,
                    )
                )
                for frame in frames
            ]
        )
        if len(strengths) < 8:
            return 0.0
        even = float(np.mean(strengths[::2]))
        odd = float(np.mean(strengths[1::2]))
        return abs(even - odd) / max(even, odd, 1e-8)

    fast_contrast = accent_contrast(fast)
    slow_contrast = accent_contrast(slow)
    if slow_contrast >= 0.06 and slow_contrast - fast_contrast >= 0.045:
        return slow_tempo, slow
    return fast_tempo, fast


def estimate_tempo(
    path: Path,
) -> tuple[float, list[float], list[float], dict[str, object]]:
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    onset_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
    )
    fast_tempo, fast_beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        start_bpm=120,
        trim=False,
    )
    slow_tempo, slow_beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        start_bpm=80,
        trim=False,
    )
    value, beat_frames = select_quarter_note_grid(
        onset_envelope,
        float(np.atleast_1d(fast_tempo)[0]),
        fast_beat_frames,
        float(np.atleast_1d(slow_tempo)[0]),
        slow_beat_frames,
    )
    fallback_bpm = value if math.isfinite(value) and value > 0 else 120.0
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        units="frames",
        backtrack=False,
    )
    beat_times = stabilize_beat_times(
        librosa.frames_to_time(beat_frames, sr=sample_rate),
    )
    if onset_frames.size:
        first_onset = float(librosa.frames_to_time(onset_frames[0], sr=sample_rate))
        beat_times = [
            time for time in beat_times if time >= first_onset - 0.08
        ]
    beat_times = regularize_beat_times(beat_times)
    usable_beats = [
        round(float(time), 4)
        for time in beat_times
        if np.isfinite(time) and time >= 0
    ]
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    usable_onsets = [
        round(float(time), 4)
        for time in onset_times
        if np.isfinite(time) and time >= 0
    ]
    tempo_map = tempo_map_from_beats(usable_beats, fallback_bpm)
    bpm = round(float(tempo_map["median_bpm"]), 2)
    return bpm, usable_beats, usable_onsets, tempo_map


def waveform(path: Path, bar_count: int = 120) -> list[float]:
    import librosa
    import numpy as np

    audio, _sample_rate = librosa.load(path, sr=11025, mono=True)
    if audio.size == 0:
        return [0.0] * bar_count
    chunks = np.array_split(np.abs(audio), bar_count)
    peak = max(float(chunk.max(initial=0.0)) for chunk in chunks) or 1.0
    return [round(float(chunk.max(initial=0.0)) / peak, 4) for chunk in chunks]


def visualization_timeline(path: Path, point_count: int = 720) -> list[dict[str, float]]:
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    if audio.size == 0:
        return [
            {
                "energy": 0.0,
                "pitch": 0.5,
                "low": 0.0,
                "mid": 0.0,
                "high": 0.0,
                "impact": 0.0,
            }
            for _ in range(point_count)
        ]

    hop_length = 512
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=hop_length,
        n_mels=96,
        fmin=40,
        fmax=8000,
        power=2.0,
    )
    mel_amplitude = np.sqrt(np.maximum(mel, 0))
    mel_frequencies = librosa.mel_frequencies(
        n_mels=mel.shape[0],
        fmin=40,
        fmax=8000,
    )
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )
    frame_count = min(mel.shape[1], len(rms), len(onset))
    mel_amplitude = mel_amplitude[:, :frame_count]
    rms = rms[:frame_count]
    onset = onset[:frame_count]
    energy_scale = max(float(np.percentile(rms, 95)), 1e-6)
    onset_scale = max(float(np.percentile(onset, 95)), 1e-6)
    band_scale = max(float(np.percentile(mel_amplitude, 95)), 1e-6)
    low_mask = mel_frequencies < 250
    mid_mask = (mel_frequencies >= 250) & (mel_frequencies < 2000)
    high_mask = mel_frequencies >= 2000

    timeline: list[dict[str, float]] = []
    for point_index in range(point_count):
        start = min(frame_count - 1, int(point_index * frame_count / point_count))
        end = min(
            frame_count,
            max(start + 1, int((point_index + 1) * frame_count / point_count)),
        )
        frame_slice = mel_amplitude[:, start:end]
        frame_energy = min(1.0, float(np.max(rms[start:end])) / energy_scale)
        frame_impact = min(1.0, float(np.max(onset[start:end])) / onset_scale)
        mean_spectrum = np.mean(frame_slice, axis=1)
        dominant_frequency = float(mel_frequencies[int(np.argmax(mean_spectrum))])
        pitch = min(
            1.0,
            max(0.0, math.log2(max(70.0, dominant_frequency) / 70.0) / math.log2(2200 / 70)),
        )

        def band_level(mask: object) -> float:
            selected = frame_slice[mask]
            return min(1.0, float(np.max(selected)) / band_scale) if selected.size else 0.0

        timeline.append(
            {
                "energy": round(frame_energy, 4),
                "pitch": round(pitch, 4),
                "low": round(band_level(low_mask), 4),
                "mid": round(band_level(mid_mask), 4),
                "high": round(band_level(high_mask), 4),
                "impact": round(frame_impact, 4),
            }
        )
    return timeline


def audio_rms(path: Path) -> float:
    import librosa
    import numpy as np

    audio, _sample_rate = librosa.load(path, sr=11025, mono=True)
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def transcription_source(
    normalized: Path,
    stems: dict[str, Path],
) -> tuple[Path, str | None]:
    original_level = audio_rms(normalized)
    guitar_level = audio_rms(stems["guitar"])
    if guitar_level >= max(0.0001, original_level * 0.04):
        return stems["guitar"], None

    other_level = audio_rms(stems["other"])
    if other_level >= original_level * 0.08:
        return stems["other"], "吉他分轨能量过低，已使用伴奏残余轨进行音符识别。"
    return normalized, "吉他分轨能量过低，已使用原始混音进行音符识别。"


def compress_stems(stems: dict[str, Path], destination: Path) -> dict[str, Path]:
    ffmpeg = ffmpeg_command()
    media_root = destination / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    compressed: dict[str, Path] = {}
    for name, source in stems.items():
        target = media_root / f"{name}.m4a"
        run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                str(target),
            ]
        )
        compressed[name] = target
        source.unlink(missing_ok=True)
    return compressed


def candidate_positions(pitch: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for string, open_pitch in STANDARD_TUNING.items():
        fret = pitch - open_pitch
        if 0 <= fret <= 24:
            candidates.append((string, fret))
    return candidates


def select_fingering(
    pitch: int,
    previous: tuple[int, int] | None,
) -> tuple[int, int]:
    candidates = candidate_positions(pitch)
    if not candidates:
        clamped_pitch = min(max(pitch, STANDARD_TUNING[6]), STANDARD_TUNING[1] + 24)
        candidates = candidate_positions(clamped_pitch)
    if not candidates:
        return (1, 0)
    if previous is None:
        return min(candidates, key=lambda item: (item[1] > 12, item[1], item[0]))

    previous_string, previous_fret = previous
    return min(
        candidates,
        key=lambda item: (
            abs(item[1] - previous_fret) * 1.6
            + abs(item[0] - previous_string) * 0.7
            + max(0, item[1] - 12) * 0.4
        ),
    )


def fingering_group_cost(
    pitches: list[int],
    positions: list[tuple[int, int]],
    previous_positions: list[tuple[int, int]],
) -> float:
    cost = 0.0
    fretted = [fret for _string, fret in positions if fret > 0]
    if fretted:
        fret_span = max(fretted) - min(fretted)
        cost += fret_span * 0.18 + max(0, fret_span - 4) * 1.4

    for _string, fret in positions:
        cost += fret * 0.045 + max(0, fret - 12) * 0.9
        if fret == 0:
            cost -= 0.35

    ordered = sorted(zip(pitches, positions), key=lambda item: item[0])
    for (_left_pitch, left), (_right_pitch, right) in zip(
        ordered,
        ordered[1:],
    ):
        if right[0] >= left[0]:
            cost += 2.4

    if previous_positions:
        previous_fret = sum(fret for _string, fret in previous_positions) / len(
            previous_positions
        )
        previous_string = sum(
            string for string, _fret in previous_positions
        ) / len(previous_positions)
        current_fret = sum(fret for _string, fret in positions) / len(positions)
        current_string = sum(string for string, _fret in positions) / len(positions)
        cost += abs(current_fret - previous_fret) * 0.42
        cost += abs(current_string - previous_string) * 0.16
    return cost


def assign_fingering_group(
    pitches: list[int],
    previous_positions: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], float]:
    if not pitches:
        return [], 0.0
    previous_positions = previous_positions or []
    if len(pitches) > 6:
        fallback: list[tuple[int, int]] = []
        previous = previous_positions[-1] if previous_positions else None
        for pitch in pitches:
            position = select_fingering(pitch, previous)
            fallback.append(position)
            previous = position
        return fallback, 0.32
    candidate_sets = [candidate_positions(pitch) for pitch in pitches]
    if any(not candidates for candidates in candidate_sets):
        fallback: list[tuple[int, int]] = []
        previous = previous_positions[-1] if previous_positions else None
        for pitch in pitches:
            position = select_fingering(pitch, previous)
            fallback.append(position)
            previous = position
        return fallback, 0.42

    scored = fingering_group_options(
        pitches,
        previous_positions,
    )
    if not scored:
        return [
            select_fingering(
                pitch,
                previous_positions[-1] if previous_positions else None,
            )
            for pitch in pitches
        ], 0.4

    best_score, best_positions = scored[0]
    if len(scored) == 1:
        confidence = 0.94
    else:
        margin = max(0.0, scored[1][0] - best_score)
        confidence = min(0.92, 0.56 + margin * 0.1)
    return best_positions, round(confidence, 4)


def fingering_group_options(
    pitches: list[int],
    previous_positions: list[tuple[int, int]] | None = None,
    limit: int = 12,
) -> list[tuple[float, list[tuple[int, int]]]]:
    if not pitches or len(pitches) > 6:
        return []
    previous_positions = previous_positions or []
    candidate_sets = [candidate_positions(pitch) for pitch in pitches]
    if any(not candidates for candidates in candidate_sets):
        return []
    scored: list[tuple[float, list[tuple[int, int]]]] = []

    def visit(
        index: int,
        used_strings: set[int],
        positions: list[tuple[int, int]],
    ) -> None:
        if index == len(candidate_sets):
            scored.append(
                (
                    fingering_group_cost(
                        pitches,
                        positions,
                        previous_positions,
                    ),
                    list(positions),
                )
            )
            return

        available = [
            position
            for position in candidate_sets[index]
            if position[0] not in used_strings
        ]
        if not available:
            available = candidate_sets[index]
        for position in available:
            duplicate_penalty = 9.0 if position[0] in used_strings else 0.0
            if duplicate_penalty and len(pitches) <= 6:
                continue
            positions.append(position)
            visit(index + 1, used_strings | {position[0]}, positions)
            positions.pop()

    visit(0, set(), [])
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[: max(1, limit)]


def assign_fingering_sequence(
    pitch_groups: list[list[int]],
    beam_width: int = 8,
    window_size: int = 32,
) -> list[tuple[list[tuple[int, int]], float]]:
    if not pitch_groups:
        return []
    resolved: list[tuple[list[tuple[int, int]], float]] = []
    previous_window_positions: list[tuple[int, int]] = []
    for window_start in range(0, len(pitch_groups), max(1, window_size)):
        window = pitch_groups[window_start : window_start + window_size]
        beams: list[
            tuple[
                float,
                list[tuple[int, int]],
                list[tuple[list[tuple[int, int]], float]],
            ]
        ] = [(0.0, previous_window_positions, [])]
        for pitches in window:
            candidates: list[
                tuple[
                    float,
                    list[tuple[int, int]],
                    list[tuple[list[tuple[int, int]], float]],
                ]
            ] = []
            for path_cost, previous_positions, assignments in beams:
                options = fingering_group_options(
                    pitches,
                    previous_positions,
                    limit=max(beam_width, 12),
                )
                if not options:
                    fallback, confidence = assign_fingering_group(
                        pitches,
                        previous_positions,
                    )
                    options = [(12.0, fallback)]
                    option_confidences = [confidence]
                else:
                    option_confidences = [
                        (
                            0.94
                            if len(options) == 1
                            else min(
                                0.92,
                                0.56
                                + max(
                                    0.0,
                                    (
                                        options[index + 1][0]
                                        if index + 1 < len(options)
                                        else options[index][0] + 2.0
                                    )
                                    - option[0],
                                )
                                * 0.1,
                            )
                        )
                        for index, option in enumerate(options)
                    ]
                for option_index, (cost, positions) in enumerate(options):
                    candidates.append(
                        (
                            path_cost + cost,
                            positions,
                            [
                                *assignments,
                                (
                                    positions,
                                    round(
                                        option_confidences[option_index],
                                        4,
                                    ),
                                ),
                            ],
                        )
                    )
            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )
            beams = candidates[: max(1, beam_width)]
        selected = beams[0][2]
        resolved.extend(selected)
        if selected:
            previous_window_positions = selected[-1][0]
    return resolved


NATURAL_HARMONIC_INTERVALS = {
    12: 12,
    19: 7,
    24: 5,
}


def natural_harmonic_positions(pitch: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for string, open_pitch in STANDARD_TUNING.items():
        interval = pitch - open_pitch
        if interval in NATURAL_HARMONIC_INTERVALS:
            positions.append((string, NATURAL_HARMONIC_INTERVALS[interval]))
    return positions


def has_natural_harmonic_candidate(pitch: int) -> bool:
    return bool(natural_harmonic_positions(pitch))


def select_harmonic_position(
    pitch: int,
    previous: tuple[int, int] | None,
) -> tuple[int, int, str, int | None, float] | None:
    natural = natural_harmonic_positions(pitch)
    if natural:
        selected = min(
            natural,
            key=lambda position: (
                abs(position[1] - previous[1]) * 0.5
                + abs(position[0] - previous[0]) * 0.3
                if previous
                else position[1]
            ),
        )
        confidence = 0.92 if len(natural) == 1 else 0.84
        return selected[0], selected[1], "natural", None, confidence

    artificial: list[tuple[int, int, int]] = []
    for string, open_pitch in STANDARD_TUNING.items():
        base_fret = pitch - open_pitch - 12
        if 0 <= base_fret <= 12:
            artificial.append((string, base_fret, base_fret + 12))
    if not artificial:
        return None
    selected = min(
        artificial,
        key=lambda position: (
            abs(position[1] - previous[1]) * 0.6
            + abs(position[0] - previous[0]) * 0.3
            if previous
            else position[1]
        ),
    )
    confidence = 0.8 if len(artificial) == 1 else 0.68
    return selected[0], selected[1], "artificial", selected[2], confidence


def collapse_harmonic_stacks(
    events: list[NoteEvent],
    onset_tolerance: float = 0.025,
) -> tuple[list[NoteEvent], set[int]]:
    groups: list[list[NoteEvent]] = []
    for event in sorted(events, key=lambda item: (item.start, item.pitch)):
        if (
            not groups
            or event.start - groups[-1][0].start > onset_tolerance
        ):
            groups.append([event])
        else:
            groups[-1].append(event)

    retained: list[NoteEvent] = []
    forced_event_ids: set[int] = set()
    for group in groups:
        removed_ids: set[int] = set()
        by_pitch = {event.pitch: event for event in group}
        for base in sorted(group, key=lambda event: event.pitch):
            if id(base) in removed_ids or not has_natural_harmonic_candidate(base.pitch):
                continue
            partials = [
                by_pitch.get(base.pitch + 12),
                by_pitch.get(base.pitch + 24),
            ]
            if any(partial is None for partial in partials):
                continue
            cluster = [base, *[partial for partial in partials if partial]]
            durations = [event.end - event.start for event in cluster]
            if max(durations) - min(durations) > max(0.12, durations[0] * 0.12):
                continue
            forced_event_ids.add(id(base))
            removed_ids.update(id(event) for event in cluster[1:])

        retained.extend(event for event in group if id(event) not in removed_ids)

    retained.sort(key=lambda event: (event.start, event.pitch))
    forced_indices = {
        index
        for index, event in enumerate(retained)
        if id(event) in forced_event_ids
    }
    return retained, forced_indices


def collapse_note_fragments(
    events: list[NoteEvent],
    maximum_gap: float = 0.03,
) -> list[NoteEvent]:
    retained: list[NoteEvent] = []
    latest_by_pitch: dict[int, NoteEvent] = {}
    for event in sorted(events, key=lambda item: (item.start, item.pitch)):
        previous = latest_by_pitch.get(event.pitch)
        if previous:
            gap = event.start - previous.end
            onset_gap = event.start - previous.start
            previous_duration = previous.end - previous.start
            event_duration = event.end - event.start
            if (
                0 <= gap <= maximum_gap
                and onset_gap <= 0.25
                and previous_duration <= 0.18
                and previous.velocity <= 45
                and event_duration >= 0.18
                and event.velocity >= previous.velocity + 12
            ):
                retained.remove(previous)
        latest_by_pitch[event.pitch] = event
        retained.append(event)
    return retained


def collapse_inaudible_rearticulations(
    events: list[NoteEvent],
    features: list[AudioFeature],
    maximum_gap: float = 0.03,
    maximum_onset_strength: float = 0.28,
) -> tuple[list[NoteEvent], list[AudioFeature]]:
    if len(events) != len(features):
        raise ValueError("Events and audio features must have equal lengths")
    retained: list[NoteEvent] = []
    retained_features: list[AudioFeature] = []
    latest_by_pitch: dict[int, int] = {}
    for event, feature in zip(events, features, strict=True):
        previous_index = latest_by_pitch.get(event.pitch)
        if previous_index is not None:
            previous = retained[previous_index]
            gap = event.start - previous.end
            if (
                0 <= gap <= maximum_gap
                and feature.onset < maximum_onset_strength
            ):
                retained[previous_index] = replace(
                    previous,
                    end=max(previous.end, event.end),
                    velocity=max(previous.velocity, event.velocity),
                    confidence=max(previous.confidence, event.confidence),
                    position_confidence=max(
                        previous.position_confidence,
                        event.position_confidence,
                    ),
                )
                continue
        latest_by_pitch[event.pitch] = len(retained)
        retained.append(event)
        retained_features.append(feature)
    return retained, retained_features


def extract_note_features(path: Path, notes: list[NoteEvent]) -> list[AudioFeature]:
    import librosa
    import numpy as np

    if not notes:
        return []
    audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    hop_length = 256
    onset = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        hop_length=hop_length,
    )[0]
    zero_crossing = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=hop_length,
    )[0]
    onset_scale = max(float(np.percentile(onset, 95)), 1e-6)
    frame_count = min(
        len(onset),
        len(rms),
        len(centroid),
        len(flatness),
        len(zero_crossing),
    )

    features: list[AudioFeature] = []
    for note in notes:
        frame = int(
            librosa.time_to_frames(
                note.start,
                sr=sample_rate,
                hop_length=hop_length,
            )
        )
        start = max(0, min(frame_count - 1, frame - 1))
        end = max(start + 1, min(frame_count, frame + 3))
        features.append(
            AudioFeature(
                onset=min(1.0, float(np.max(onset[start:end])) / onset_scale),
                centroid=float(np.mean(centroid[start:end])),
                flatness=float(np.mean(flatness[start:end])),
                zero_crossing=float(np.mean(zero_crossing[start:end])),
            )
        )
    return features


def confidence_score(value: float) -> float:
    return round(min(0.98, max(0.01, value)), 4)


def infer_technique_candidates(
    previous: NoteEvent | None,
    current: NoteEvent,
    feature: AudioFeature,
) -> list[TechniqueCandidate]:
    duration = current.end - current.start
    natural_harmonic_position = has_natural_harmonic_candidate(current.pitch)
    candidates: list[TechniqueCandidate] = []

    pick_evidence = ["stable-pitched-event"]
    if feature.onset >= 0.38:
        pick_evidence.append("audible-onset")
    candidates.append(
        TechniqueCandidate(
            "pick",
            confidence_score(
                0.38
                + feature.onset * 0.38
                + (0.12 if duration >= 0.2 else 0)
                + min(1.0, current.velocity / 127) * 0.08
            ),
            pick_evidence,
        )
    )

    harmonic_evidence: list[str] = []
    harmonic_score = 0.08
    if natural_harmonic_position:
        harmonic_score += 0.2
        harmonic_evidence.append("natural-harmonic-fret")
    if feature.centroid >= 1800:
        harmonic_score += min(0.28, (feature.centroid - 1800) / 4000)
        harmonic_evidence.append("bright-spectrum")
    if feature.flatness <= 0.12:
        harmonic_score += 0.2
        harmonic_evidence.append("tonal-spectrum")
    if feature.onset <= 0.58:
        harmonic_score += 0.14
        harmonic_evidence.append("soft-onset")
    if duration >= 0.34:
        harmonic_score += 0.16
        harmonic_evidence.append("sustained-note")
    if (
        duration >= 0.34
        and current.velocity <= 96
        and feature.flatness <= 0.12
        and feature.onset <= 0.58
        and (
            (natural_harmonic_position and feature.centroid >= 1800)
            or feature.centroid >= 2600
        )
    ):
        candidates.append(
            TechniqueCandidate(
                "harmonic",
                confidence_score(harmonic_score),
                harmonic_evidence,
            )
        )

    if (
        duration <= 0.16
        and feature.onset >= 0.82
        and feature.flatness >= 0.24
        and feature.zero_crossing >= 0.12
    ):
        candidates.append(
            TechniqueCandidate(
                "body-tap",
                confidence_score(
                    0.7
                    + feature.onset * 0.2
                    + feature.flatness * 0.18
                    + feature.zero_crossing * 0.18
                ),
                ["hard-transient", "noise-dominant", "very-short-event"],
            )
        )

    if (
        duration <= 0.2
        and feature.onset >= 0.72
        and (feature.zero_crossing >= 0.08 or feature.centroid >= 2800)
    ):
        candidates.append(
            TechniqueCandidate(
                "slap",
                confidence_score(
                    0.58
                    + feature.onset * 0.22
                    + min(1.0, feature.centroid / 4000) * 0.12
                    + min(1.0, feature.zero_crossing / 0.2) * 0.08
                ),
                ["hard-onset", "bright-or-noisy-transient", "short-event"],
            )
        )

    if (
        current.string >= 5
        and duration <= 0.22
        and feature.onset >= 0.34
        and feature.centroid <= 2400
    ):
        candidates.append(
            TechniqueCandidate(
                "palm-mute",
                confidence_score(
                    0.5
                    + feature.onset * 0.16
                    + (1 - min(1.0, feature.centroid / 2400)) * 0.2
                    + (0.12 if current.string >= 5 else 0)
                ),
                ["short-low-string-note", "damped-spectrum"],
            )
        )

    if (
        duration < 0.13
        and current.velocity < 58
        and feature.flatness >= 0.18
    ):
        candidates.append(
            TechniqueCandidate(
                "dead-note",
                confidence_score(
                    0.54
                    + feature.flatness * 0.22
                    + (1 - current.velocity / 127) * 0.16
                ),
                ["very-short-event", "weak-pitched-content", "noise-component"],
            )
        )

    if previous is None:
        return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)[:3]

    gap = current.start - previous.end
    onset_gap = current.start - previous.start
    fret_delta = current.fret - previous.fret
    same_string = current.string == previous.string
    connected = -0.08 <= gap <= 0.16 and 0.06 <= onset_gap <= 0.55
    weak_onset = feature.onset <= 0.34

    if same_string and onset_gap < 0.18 and abs(fret_delta) <= 1 and feature.onset >= 0.42:
        candidates.append(
            TechniqueCandidate(
                "tremolo",
                confidence_score(0.76 + feature.onset * 0.18),
                ["same-string-repeat", "short-onset-gap", "rearticulated-onset"],
            )
        )
    if same_string and connected and weak_onset and 0 < fret_delta <= 4:
        candidates.append(
            TechniqueCandidate(
                "hammer-on",
                confidence_score(0.7 + (0.34 - feature.onset) * 0.45 + 0.12),
                ["same-string", "connected-notes", "weak-second-onset", "ascending-fret"],
            )
        )
    if same_string and connected and weak_onset and -4 <= fret_delta < 0:
        candidates.append(
            TechniqueCandidate(
                "pull-off",
                confidence_score(0.7 + (0.34 - feature.onset) * 0.45 + 0.12),
                ["same-string", "connected-notes", "weak-second-onset", "descending-fret"],
            )
        )
    if same_string and connected and weak_onset and 3 <= abs(fret_delta) <= 7:
        candidates.append(
            TechniqueCandidate(
                "slide",
                confidence_score(
                    0.72
                    + min(0.16, abs(fret_delta) * 0.025)
                    + (0.34 - feature.onset) * 0.28
                    + (
                        0.1
                        if feature.onset <= 0.22
                        and gap <= 0.08
                        and abs(fret_delta) <= 7
                        else 0
                    )
                ),
                ["same-string", "connected-notes", "weak-second-onset", "large-fret-change"],
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)[:3]


def infer_technique(
    previous: NoteEvent | None,
    current: NoteEvent,
    feature: AudioFeature,
) -> str:
    return infer_technique_candidates(previous, current, feature)[0].technique


def select_legato_predecessor(
    events: list[NoteEvent],
    current_index: int,
    feature: AudioFeature,
) -> tuple[int, NoteEvent, int] | None:
    if feature.onset > 0.34:
        return None
    current = events[current_index]
    candidates: list[tuple[float, int, NoteEvent, int]] = []
    for previous_index in range(current_index - 1, max(-1, current_index - 16), -1):
        previous = events[previous_index]
        onset_gap = current.start - previous.start
        if onset_gap > 0.5:
            break
        if onset_gap < 0.06:
            continue
        gap = current.start - previous.end
        if not -0.08 <= gap <= 0.16:
            continue
        target_fret = current.pitch - STANDARD_TUNING[previous.string]
        if not 0 <= target_fret <= 24:
            continue
        fret_delta = target_fret - previous.fret
        if fret_delta == 0:
            continue
        hammer_or_pull = abs(fret_delta) <= 4
        slide = (
            3 <= abs(fret_delta) <= 7
            and feature.onset <= 0.22
            and gap <= 0.08
        )
        if not hammer_or_pull and not slide:
            continue
        score = (
            abs(gap) * 2
            + onset_gap * 0.15
            + abs(fret_delta) * 0.04
            + (0 if previous.string == current.string else 0.18)
        )
        candidates.append((score, previous_index, previous, target_fret))
    if not candidates:
        return None
    _score, previous_index, previous, target_fret = min(
        candidates,
        key=lambda candidate: candidate[0],
    )
    return previous_index, previous, target_fret


def annotate_techniques(
    events: list[NoteEvent],
    features: list[AudioFeature],
    forced_harmonic_indices: set[int] | None = None,
    preserve_positions: bool = False,
) -> list[NoteEvent]:
    forced_harmonic_indices = forced_harmonic_indices or set()
    previous_by_string: dict[int, tuple[int, NoteEvent]] = {}
    relational_techniques = {"hammer-on", "pull-off", "slide"}
    transition_techniques = relational_techniques | {"tremolo"}
    for event_index, (event, feature) in enumerate(zip(events, features, strict=True)):
        original_position = (event.string, event.fret)
        previous_entry = previous_by_string.get(event.string)
        legato_entry = (
            None
            if preserve_positions
            else select_legato_predecessor(events, event_index, feature)
        )
        if legato_entry:
            previous_index, previous_event, target_fret = legato_entry
            event.string = previous_event.string
            event.fret = target_fret
            previous_entry = (previous_index, previous_event)
        previous_event = previous_entry[1] if previous_entry else None
        candidates = infer_technique_candidates(previous_event, event, feature)
        if event_index in forced_harmonic_indices:
            candidates = [
                TechniqueCandidate(
                    "harmonic",
                    0.94,
                    [
                        "harmonic-overtone-stack",
                        "coincident-partials",
                        "sustained-note",
                    ],
                ),
                *[
                    candidate
                    for candidate in candidates
                    if candidate.technique != "harmonic"
                ],
            ][:3]
        primary = candidates[0]
        if legato_entry and primary.technique in relational_techniques:
            event.position_confidence = max(event.position_confidence, 0.74)
            event.position_source = "legato-optimizer-v1"
        elif legato_entry:
            event.string, event.fret = original_position
            previous_entry = previous_by_string.get(event.string)
        event.technique = primary.technique
        event.technique_confidence = primary.confidence
        event.technique_candidates = candidates
        event.technique_evidence = primary.evidence
        event.technique_source = (
            "transition-heuristic-v2"
            if primary.technique in transition_techniques
            else "acoustic-heuristic-v2"
        )
        if primary.technique in relational_techniques and previous_entry:
            event.related_note_index = previous_entry[0]
        if primary.technique == "harmonic" and not preserve_positions:
            harmonic_position = select_harmonic_position(
                event.pitch,
                (event.string, event.fret),
            )
            if harmonic_position:
                (
                    event.string,
                    event.fret,
                    event.harmonic_type,
                    event.harmonic_touch_fret,
                    event.position_confidence,
                ) = harmonic_position
                event.position_source = "harmonic-optimizer-v1"
        previous_by_string[event.string] = (event_index, event)
    return events


def transcribe_basic_pitch(
    path: Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.7,
) -> tuple[list[NoteEvent], int]:
    from basic_pitch.inference import predict

    _model_output, midi_data, _raw_events = predict(
        str(path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
    )
    all_midi_notes = sorted(
        (
            note
            for instrument in midi_data.instruments
            for note in instrument.notes
        ),
        key=lambda note: (note.start, note.pitch),
    )
    midi_notes = [
        note
        for note in all_midi_notes
        if GUITAR_PITCH_RANGE[0] <= int(note.pitch) <= GUITAR_PITCH_RANGE[1]
    ]
    dropped_note_count = len(all_midi_notes) - len(midi_notes)

    note_groups: list[list[object]] = []
    for midi_note in midi_notes:
        if (
            not note_groups
            or float(midi_note.start) - float(note_groups[-1][0].start) > 0.045
        ):
            note_groups.append([midi_note])
        else:
            note_groups[-1].append(midi_note)

    events: list[NoteEvent] = []
    assignments = assign_fingering_sequence(
        [
            [int(note.pitch) for note in group]
            for group in note_groups
        ]
    )
    for group, (positions, position_confidence) in zip(
        note_groups,
        assignments,
        strict=True,
    ):
        for midi_note, (string, fret) in zip(group, positions, strict=True):
            events.append(
                NoteEvent(
                    start=round(float(midi_note.start), 4),
                    end=round(float(midi_note.end), 4),
                    pitch=int(midi_note.pitch),
                    velocity=int(midi_note.velocity),
                    string=string,
                    fret=fret,
                    technique="pick",
                    confidence=min(0.98, 0.54 + midi_note.velocity / 220),
                    position_confidence=position_confidence,
                    position_source="sequence-optimizer-v3",
                )
            )
    events = collapse_note_fragments(events)
    features = extract_note_features(path, events)
    events, features = collapse_inaudible_rearticulations(events, features)
    feature_by_event = {
        id(event): feature
        for event, feature in zip(events, features, strict=True)
    }
    events, forced_harmonic_indices = collapse_harmonic_stacks(events)
    features = [feature_by_event[id(event)] for event in events]
    annotated = annotate_techniques(
        events,
        features,
        forced_harmonic_indices,
    )
    return (
        annotated,
        dropped_note_count,
    )


def transcribe_string_model(
    path: Path,
    checkpoint_path: Path,
) -> tuple[list[NoteEvent], dict[str, object]]:
    from stringtrace_ml.inference import transcribe_audio

    predictions, metadata = transcribe_audio(
        path,
        checkpoint_path,
        device_name=os.environ.get("STRINGTRACE_MODEL_DEVICE", "cpu"),
    )
    events = [
        NoteEvent(
            start=prediction.onset,
            end=prediction.offset,
            pitch=STANDARD_TUNING[prediction.string] + prediction.fret,
            velocity=round(40 + prediction.confidence * 80),
            string=prediction.string,
            fret=prediction.fret,
            technique="pick",
            confidence=prediction.confidence,
            position_confidence=prediction.confidence,
            position_source="string-fret-model-v1",
        )
        for prediction in predictions
    ]
    features = extract_note_features(path, events)
    return (
        annotate_techniques(
            events,
            features,
            preserve_positions=True,
        ),
        {
            "name": metadata.get("name", "stringtrace-string-fret-net"),
            "version": metadata.get("version", "unknown"),
            "kind": "trained",
            "checkpoint": checkpoint_path.name,
        },
    )


def transcribe_muscriptor(
    path: Path,
    model_reference: str,
) -> tuple[list[NoteEvent], int, dict[str, object]]:
    from importlib.metadata import version

    from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
    from muscriptor.transcription_model import TranscriptionModel

    model = TranscriptionModel.load_model(weights_path=model_reference)
    starts: dict[int, NoteStartEvent] = {}
    decoded: list[tuple[float, float, int]] = []
    for event in model.transcribe(
        path,
        instruments=[
            "acoustic_guitar",
            "clean_electric_guitar",
            "distorted_electric_guitar",
        ],
    ):
        if isinstance(event, ProgressEvent):
            emit_muscriptor_progress(event.completed, event.total)
            continue
        if isinstance(event, NoteStartEvent):
            starts[event.index] = event
            continue
        if not isinstance(event, NoteEndEvent):
            continue
        start = starts.pop(event.start_event_index, None)
        if start is None:
            continue
        decoded.append(
            (
                max(0.0, float(start.start_time)),
                max(float(start.start_time) + 0.03, float(event.end_time)),
                int(start.pitch),
            )
        )
    decoded.sort(key=lambda item: (item[0], item[2]))
    guitar_notes = [
        item
        for item in decoded
        if GUITAR_PITCH_RANGE[0] <= item[2] <= GUITAR_PITCH_RANGE[1]
    ]
    dropped_note_count = len(decoded) - len(guitar_notes)
    groups: list[list[tuple[float, float, int]]] = []
    for item in guitar_notes:
        if not groups or item[0] - groups[-1][0][0] > 0.045:
            groups.append([item])
        else:
            groups[-1].append(item)

    events: list[NoteEvent] = []
    assignments = assign_fingering_sequence(
        [[item[2] for item in group] for group in groups]
    )
    for group, (positions, position_confidence) in zip(
        groups,
        assignments,
        strict=True,
    ):
        for (start, end, pitch), (string, fret) in zip(
            group,
            positions,
            strict=True,
        ):
            events.append(
                NoteEvent(
                    start=round(start, 4),
                    end=round(end, 4),
                    pitch=pitch,
                    velocity=96,
                    string=string,
                    fret=fret,
                    technique="pick",
                    confidence=0.82,
                    position_confidence=position_confidence,
                    position_source="muscriptor-pitch-optimizer-v1",
                )
            )

    events = collapse_note_fragments(events)
    events, forced_harmonic_indices = collapse_harmonic_stacks(events)
    features = extract_note_features(path, events)
    return (
        annotate_techniques(
            events,
            features,
            forced_harmonic_indices,
        ),
        dropped_note_count,
        {
            "name": "muscriptor",
            "version": version("muscriptor"),
            "kind": "trained",
            "variant": (
                model_reference
                if model_reference in {"small", "medium", "large"}
                else "local"
            ),
            "license": "CC-BY-NC-4.0",
        },
    )


def transcribe_mr_mt3(
    path: Path,
    checkpoint_path: Path,
) -> tuple[list[NoteEvent], int]:
    import mido
    import numpy as np
    from mt3_infer.adapters.mr_mt3 import MRMT3Adapter

    import librosa

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    audio, _ = librosa.load(path, sr=16000, mono=True)
    adapter = MRMT3Adapter()
    adapter.load_model(
        str(checkpoint_path),
        device=device,
    )
    midi = adapter.transcribe(audio.astype(np.float32), sr=16000)
    tempo = 500_000
    elapsed = 0.0
    active: dict[tuple[int, int], tuple[float, int]] = {}
    decoded: list[tuple[float, float, int, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if not hasattr(message, "channel") or message.channel == 9:
            continue
        key = (message.channel, message.note)
        if message.type == "note_on" and message.velocity > 0:
            active[key] = (elapsed, int(message.velocity))
        elif message.type in {"note_off", "note_on"} and key in active:
            start, velocity = active.pop(key)
            decoded.append(
                (
                    start,
                    max(start + 0.03, elapsed),
                    int(message.note),
                    velocity,
                )
            )

    decoded.sort(key=lambda item: (item[0], item[2]))
    guitar_notes = [
        item
        for item in decoded
        if GUITAR_PITCH_RANGE[0] <= item[2] <= GUITAR_PITCH_RANGE[1]
    ]
    groups: list[list[tuple[float, float, int, int]]] = []
    for item in guitar_notes:
        if not groups or item[0] - groups[-1][0][0] > 0.045:
            groups.append([item])
        else:
            groups[-1].append(item)
    events: list[NoteEvent] = []
    assignments = assign_fingering_sequence(
        [[item[2] for item in group] for group in groups]
    )
    for group, (positions, position_confidence) in zip(
        groups,
        assignments,
        strict=True,
    ):
        for (start, end, pitch, velocity), (string, fret) in zip(
            group,
            positions,
            strict=True,
        ):
            events.append(
                NoteEvent(
                    start=round(start, 4),
                    end=round(end, 4),
                    pitch=pitch,
                    velocity=velocity,
                    string=string,
                    fret=fret,
                    technique="pick",
                    confidence=min(0.9, 0.58 + velocity / 256),
                    position_confidence=position_confidence,
                    position_source="mr-mt3-pitch-optimizer-v1",
                )
            )
    return events, len(decoded) - len(guitar_notes)


def transcribe_yourmt3(
    path: Path,
    checkpoint_path: Path,
) -> tuple[list[NoteEvent], int]:
    import mido
    import numpy as np
    import torch
    from mt3_infer.adapters.yourmt3 import YourMT3Adapter

    import librosa

    audio, _ = librosa.load(path, sr=16000, mono=True)
    adapter = YourMT3Adapter("yptf_moe_nops")
    adapter.load_model(
        str(checkpoint_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    midi = adapter.transcribe(audio.astype(np.float32), sr=16000)
    tempo = 500_000
    elapsed = 0.0
    active: dict[tuple[int, int], tuple[float, int]] = {}
    decoded: list[tuple[float, float, int, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if not hasattr(message, "channel") or message.channel == 9:
            continue
        key = (message.channel, message.note)
        if message.type == "note_on" and message.velocity > 0:
            active[key] = (elapsed, int(message.velocity))
        elif message.type in {"note_off", "note_on"} and key in active:
            start, velocity = active.pop(key)
            decoded.append(
                (
                    start,
                    max(start + 0.03, elapsed),
                    int(message.note),
                    velocity,
                )
            )

    decoded.sort(key=lambda item: (item[0], item[2]))
    guitar_notes = [
        item
        for item in decoded
        if GUITAR_PITCH_RANGE[0] <= item[2] <= GUITAR_PITCH_RANGE[1]
    ]
    groups: list[list[tuple[float, float, int, int]]] = []
    for item in guitar_notes:
        if not groups or item[0] - groups[-1][0][0] > 0.045:
            groups.append([item])
        else:
            groups[-1].append(item)
    events: list[NoteEvent] = []
    assignments = assign_fingering_sequence(
        [[item[2] for item in group] for group in groups]
    )
    for group, (positions, position_confidence) in zip(
        groups,
        assignments,
        strict=True,
    ):
        for (start, end, pitch, velocity), (string, fret) in zip(
            group,
            positions,
            strict=True,
        ):
            events.append(
                NoteEvent(
                    start=round(start, 4),
                    end=round(end, 4),
                    pitch=pitch,
                    velocity=velocity,
                    string=string,
                    fret=fret,
                    technique="pick",
                    confidence=min(0.9, 0.58 + velocity / 256),
                    position_confidence=position_confidence,
                    position_source="yourmt3-playability-v1",
                )
            )
    return events, len(decoded) - len(guitar_notes)


def fuse_transcription_candidates(
    primary: list[NoteEvent],
    secondary: list[NoteEvent],
    onset_tolerance: float = 0.06,
) -> list[NoteEvent]:
    used_secondary: set[int] = set()
    fused: list[NoteEvent] = []
    for primary_note in primary:
        matches = [
            (index, candidate)
            for index, candidate in enumerate(secondary)
            if index not in used_secondary
            and candidate.pitch == primary_note.pitch
            and abs(candidate.start - primary_note.start) <= onset_tolerance
        ]
        if not matches:
            fused.append(primary_note)
            continue
        index, corroborating = min(
            matches,
            key=lambda item: abs(item[1].start - primary_note.start),
        )
        used_secondary.add(index)
        fused.append(
            replace(
                primary_note,
                start=round(
                    primary_note.start * 0.7
                    + corroborating.start * 0.3,
                    4,
                ),
                end=round(
                    primary_note.end * 0.7
                    + corroborating.end * 0.3,
                    4,
                ),
                confidence=min(
                    0.99,
                    max(primary_note.confidence, corroborating.confidence)
                    + 0.05,
                ),
                position_confidence=max(
                    primary_note.position_confidence,
                    corroborating.position_confidence,
                ),
                position_source="multi-model-consensus-v1",
            )
        )
    fused.extend(
        replace(
            candidate,
            position_source="basic-pitch-residual-v1",
        )
        for index, candidate in enumerate(secondary)
        if index not in used_secondary
        and candidate.confidence >= 0.86
        and candidate.end - candidate.start >= 0.08
    )
    return collapse_note_fragments(
        sorted(fused, key=lambda note: (note.start, note.pitch))
    )


def refine_positions_with_noise2fret(
    audio_path: Path,
    events: list[NoteEvent],
    checkpoint_path: Path,
) -> tuple[list[NoteEvent], int]:
    from stringtrace_ml.noise2fret import predict_position_priors

    windows = sorted({max(0, int(event.start)) for event in events})
    priors = predict_position_priors(
        audio_path,
        checkpoint_path,
        windows,
    )
    refined = list(events)
    changed = 0
    for window in windows:
        event_indices = [
            index
            for index, event in enumerate(refined)
            if int(event.start) == window
        ]
        groups: list[list[int]] = []
        for index in event_indices:
            if (
                not groups
                or refined[index].start - refined[groups[-1][0]].start > 0.045
            ):
                groups.append([index])
            else:
                groups[-1].append(index)
        remaining_slots = list(priors.get(window, []))
        for group in groups:
            pitches = {refined[index].pitch for index in group}
            if not remaining_slots:
                break
            slot_index, slot = max(
                enumerate(remaining_slots),
                key=lambda item: sum(
                    1 for _string, _fret, pitch in item[1] if pitch in pitches
                ),
            )
            if not any(pitch in pitches for _string, _fret, pitch in slot):
                continue
            remaining_slots.pop(slot_index)
            available = list(slot)
            used_strings: set[int] = set()
            for index in group:
                note = refined[index]
                matches = [
                    position
                    for position in available
                    if position[2] == note.pitch
                    and position[0] not in used_strings
                ]
                if not matches:
                    continue
                string, fret, _pitch = min(
                    matches,
                    key=lambda position: abs(position[1] - note.fret),
                )
                available.remove((string, fret, _pitch))
                used_strings.add(string)
                if (note.string, note.fret) != (string, fret):
                    changed += 1
                refined[index] = replace(
                    note,
                    string=string,
                    fret=fret,
                    position_confidence=max(note.position_confidence, 0.86),
                    position_source="noise2fret-consensus-v1",
                )
    return refined, changed


def transcribe_hybrid_model(
    path: Path,
    checkpoint_path: Path,
) -> tuple[list[NoteEvent], int, dict[str, object]]:
    import torch

    from stringtrace_ml.features import extract_file_cqt
    from stringtrace_ml.hybrid import (
        PitchEvent,
        PositionAssignmentConfig,
        assign_pitch_events,
    )
    from stringtrace_ml.inference import (
        load_checkpoint,
        predict_probabilities_with_position,
    )

    notes, dropped_note_count = transcribe_basic_pitch(path)
    pitch_events = [
        PitchEvent(
            onset=note.start,
            offset=note.end,
            pitch=note.pitch,
            confidence=note.confidence,
        )
        for note in notes
    ]
    device = torch.device(os.environ.get("STRINGTRACE_MODEL_DEVICE", "cpu"))
    loaded = load_checkpoint(checkpoint_path, device)
    if not loaded.model.config.pitch_conditioned_position:
        raise ValueError(
            "Hybrid transcription requires a pitch-conditioned "
            "StringTrace checkpoint"
        )
    features = extract_file_cqt(path, loaded.feature_config)
    (
        fret_probabilities,
        onset_probabilities,
        position_probabilities,
    ) = predict_probabilities_with_position(
        loaded,
        features,
        device=device,
    )
    assigned = assign_pitch_events(
        pitch_events,
        fret_probabilities,
        onset_probabilities,
        loaded.feature_config,
        position_probabilities=position_probabilities,
        config=PositionAssignmentConfig(
            position_head_weight=4.0,
            continuity_weight=0.75,
        ),
    )
    assigned_by_key: dict[tuple[float, int], list[object]] = {}
    for event in assigned:
        key = (
            event.onset,
            STANDARD_TUNING[event.string] + event.fret,
        )
        assigned_by_key.setdefault(key, []).append(event)

    positioned: list[NoteEvent] = []
    for note in notes:
        key = (note.start, note.pitch)
        candidates = assigned_by_key.get(key, [])
        if not candidates:
            continue
        assignment = candidates.pop(0)
        positioned.append(
            replace(
                note,
                string=assignment.string,
                fret=assignment.fret,
                technique="pick",
                technique_confidence=0.0,
                technique_source="acoustic-heuristic-v4",
                technique_evidence=[],
                technique_candidates=[],
                related_note_index=None,
                harmonic_type=None,
                harmonic_touch_fret=None,
                position_confidence=assignment.confidence,
                position_source="hybrid-pitch-conditioned-v1",
            )
        )
    positioned_features = extract_note_features(path, positioned)
    return (
        annotate_techniques(
            positioned,
            positioned_features,
            preserve_positions=True,
        ),
        dropped_note_count,
        {
            "name": "basic-pitch-stringtrace-hybrid",
            "version": loaded.metadata.get(
                "version",
                "tabcnn-gru-v3-pitch-conditioned",
            ),
            "kind": "hybrid",
            "pitch_model": "spotify-basic-pitch-0.4",
            "position_model": loaded.metadata.get(
                "name",
                "stringtrace-string-fret-net",
            ),
            "checkpoint": checkpoint_path.name,
        },
    )


def transcribe_onset_position_research_model(
    path: Path,
    bundle_path: Path,
) -> tuple[list[NoteEvent], int, dict[str, object]]:
    import torch

    from stringtrace_ml.onset_position_bundle import (
        load_research_bundle,
        transcribe_with_research_bundle,
    )

    device = torch.device(os.environ.get("STRINGTRACE_MODEL_DEVICE", "cpu"))
    loaded = load_research_bundle(bundle_path, device)
    primary_frontend = dict(
        loaded.inference_config["primary_frontend"]
    )
    supplemental_frontend = dict(
        loaded.inference_config["supplemental_frontend"]
    )
    primary_notes, primary_dropped = transcribe_basic_pitch(
        path,
        onset_threshold=float(primary_frontend["onset_threshold"]),
        frame_threshold=float(primary_frontend["frame_threshold"]),
        minimum_note_length=float(
            primary_frontend["minimum_note_length"]
        ),
    )
    supplemental_notes, supplemental_dropped = transcribe_basic_pitch(
        path,
        onset_threshold=float(supplemental_frontend["onset_threshold"]),
        frame_threshold=float(supplemental_frontend["frame_threshold"]),
        minimum_note_length=float(
            supplemental_frontend["minimum_note_length"]
        ),
    )

    def candidates(notes: list[NoteEvent]) -> list[dict[str, float | int]]:
        return [
            {
                "onset": note.start,
                "offset": note.end,
                "pitch": note.pitch,
                "confidence": note.confidence,
            }
            for note in notes
        ]

    predictions = transcribe_with_research_bundle(
        path,
        candidates(primary_notes),
        candidates(supplemental_notes),
        loaded,
        device,
    )
    events = [
        NoteEvent(
            start=prediction.onset,
            end=prediction.offset,
            pitch=STANDARD_TUNING[prediction.string] + prediction.fret,
            velocity=round(40 + prediction.confidence * 80),
            string=prediction.string,
            fret=prediction.fret,
            technique="pick",
            confidence=prediction.confidence,
            position_confidence=prediction.confidence,
            position_source="onset-position-research-v1",
        )
        for prediction in predictions
    ]
    features = extract_note_features(path, events)
    return (
        annotate_techniques(
            events,
            features,
            preserve_positions=True,
        ),
        max(primary_dropped, supplemental_dropped),
        {
            "name": loaded.metadata.get(
                "name",
                "stringtrace-onset-position-research",
            ),
            "version": loaded.metadata.get("version", "unknown"),
            "kind": "research",
            "release_channel": "research",
            "bundle": bundle_path.name,
            "source_checkpoints": loaded.metadata.get(
                "source_checkpoints",
                {},
            ),
        },
    )


def transcribe_guitar(
    path: Path,
    checkpoint_path: Path | None = None,
    model_mode: str = "direct",
    transcription_provider: str = "basic-pitch",
    transcription_model: str | None = None,
    position_model: Path | None = None,
    verification_model: Path | None = None,
) -> tuple[list[NoteEvent], int, dict[str, object]]:
    if checkpoint_path:
        if model_mode == "hybrid":
            return transcribe_hybrid_model(path, checkpoint_path)
        if model_mode == "onset-position-research":
            return transcribe_onset_position_research_model(
                path,
                checkpoint_path,
            )
        if model_mode != "direct":
            raise ValueError(f"Unknown trained model mode: {model_mode}")
        notes, model_info = transcribe_string_model(path, checkpoint_path)
        return notes, 0, model_info
    if transcription_provider == "multitrack-research":
        try:
            if transcription_model is None:
                raise FileNotFoundError("YourMT3+ checkpoint is unavailable")
            primary, primary_dropped = transcribe_yourmt3(
                path,
                Path(transcription_model),
            )
            secondary, secondary_dropped = transcribe_basic_pitch(path)
            fused = fuse_transcription_candidates(primary, secondary)
            refined_count = 0
            if position_model is not None:
                fused, refined_count = refine_positions_with_noise2fret(
                    path,
                    fused,
                    position_model,
                )
            model_names = ["yourmt3", "basic-pitch"]
            if position_model is not None:
                model_names.append("noise2fret")
            return (
                fused,
                max(primary_dropped, secondary_dropped),
                {
                    "name": "-".join(model_names),
                    "version": "yourmt3-0.2.0+basic-pitch-0.4",
                    "kind": "research",
                    "release_channel": "research",
                    "position_refinements": refined_count,
                },
            )
        except Exception as error:
            notes, dropped_note_count = transcribe_basic_pitch(path)
            return (
                notes,
                dropped_note_count,
                {
                    "name": "spotify-basic-pitch",
                    "version": "0.4",
                    "kind": "baseline",
                    "requested_provider": "multitrack-research",
                    "fallback_reason": (
                        "YourMT3+ 研究模型不可用"
                        if isinstance(error, FileNotFoundError)
                        else "YourMT3+ 本次运行失败"
                    ),
                },
            )
    if transcription_provider == "muscriptor":
        try:
            primary, primary_dropped, primary_model = transcribe_muscriptor(
                path,
                transcription_model or "medium",
            )
            release_torch_accelerator_cache()
            emit_progress(81, "transcribing", "正在运行 Basic Pitch 复核")
            secondary, secondary_dropped = transcribe_basic_pitch(path)
            fused = fuse_transcription_candidates(primary, secondary)
            verification_warning = None
            verification_used = False
            verification_dropped = 0
            if verification_model is not None:
                try:
                    emit_progress(
                        81.5,
                        "transcribing",
                        "正在运行 MR-MT3 复核",
                    )
                    verification, verification_dropped = transcribe_mr_mt3(
                        path,
                        verification_model,
                    )
                    fused = fuse_transcription_candidates(
                        fused,
                        verification,
                        onset_tolerance=0.07,
                    )
                    verification_used = True
                except Exception:
                    verification_warning = (
                        "MR-MT3 复核模型运行失败，已使用 MuScriptor "
                        "与 Basic Pitch 结果继续"
                    )
            position_warning = None
            if position_model is not None:
                try:
                    emit_progress(
                        81.8,
                        "transcribing",
                        "正在优化吉他弦品",
                    )
                    fused, refined_count = refine_positions_with_noise2fret(
                        path,
                        fused,
                        position_model,
                    )
                except Exception:
                    refined_count = 0
                    position_warning = (
                        "Noise2Fret 弦品模型运行失败，已保留原有指板优化结果"
                    )
            else:
                refined_count = 0
            model_names = ["muscriptor"]
            if verification_used:
                model_names.append("mr-mt3")
            model_names.append("basic-pitch")
            if position_model is not None and not position_warning:
                model_names.append("noise2fret")
            return (
                fused,
                max(
                    primary_dropped,
                    secondary_dropped,
                    verification_dropped,
                ),
                {
                    "name": "-".join(model_names),
                    "version": (
                        f"muscriptor-{primary_model['version']}"
                        "+basic-pitch-0.4"
                        + ("+mr-mt3-0.2.0" if verification_used else "")
                    ),
                    "kind": "trained",
                    **(
                        {"position_refinements": refined_count}
                        if position_model is not None
                        else {}
                    ),
                    **(
                        {"position_model_warning": position_warning}
                        if position_warning
                        else {}
                    ),
                    **(
                        {"verification_model_warning": verification_warning}
                        if verification_warning
                        else {}
                    ),
                },
            )
        except Exception as error:
            notes, dropped_note_count = transcribe_basic_pitch(path)
            if isinstance(error, ModuleNotFoundError):
                reason = "MuScriptor 运行组件未安装"
            elif isinstance(error, FileNotFoundError):
                reason = "所选 MuScriptor 权重不可用"
            elif error.__class__.__name__ == "ModelDownloadError":
                reason = "MuScriptor 权重尚未完成 Hugging Face 授权或下载"
            else:
                reason = "MuScriptor 本次运行失败"
            return (
                notes,
                dropped_note_count,
                {
                    "name": "spotify-basic-pitch",
                    "version": "0.4",
                    "kind": "baseline",
                    "requested_provider": "muscriptor",
                    "fallback_reason": reason,
                },
            )
    if transcription_provider not in {
        "basic-pitch",
        "guitar-specialist",
    }:
        raise ValueError(
            f"Unknown transcription provider: {transcription_provider}"
        )
    notes, dropped_note_count = transcribe_basic_pitch(path)
    return (
        notes,
        dropped_note_count,
        {
            "name": "spotify-basic-pitch",
            "version": "0.4",
            "kind": "baseline",
        },
    )


def annotate_with_technique_model(
    path: Path,
    events: list[NoteEvent],
    checkpoint_path: Path,
    gate_path: Path,
) -> tuple[list[NoteEvent], dict[str, object]]:
    import torch

    from stringtrace_ml.technique_inference import predict_techniques

    device = torch.device(
        os.environ.get("STRINGTRACE_TECHNIQUE_DEVICE", "cpu")
    )
    decisions, model_info = predict_techniques(
        path,
        events,
        checkpoint_path,
        gate_path,
        device=device,
    )
    for event, decision in zip(events, decisions):
        event.technique = decision.technique
        event.technique_confidence = decision.confidence
        event.technique_source = decision.source
        event.technique_evidence = decision.evidence
        event.technique_candidates = [
            TechniqueCandidate(
                technique=label,
                confidence=confidence,
                evidence=["model-candidate"],
            )
            for label, confidence in decision.candidates
        ]
        event.related_note_index = None
        event.harmonic_type = None
        event.harmonic_touch_fret = None
    return events, model_info


def annotate_with_technique_bundle(
    path: Path,
    events: list[NoteEvent],
    bundle_path: Path,
) -> tuple[list[NoteEvent], dict[str, object]]:
    import torch

    from stringtrace_ml.technique_inference import predict_technique_bundle

    device = torch.device(
        os.environ.get("STRINGTRACE_TECHNIQUE_DEVICE", "cpu")
    )
    decisions, model_info = predict_technique_bundle(
        path,
        events,
        bundle_path,
        device=device,
    )
    for event, decision in zip(events, decisions):
        event.technique = decision.technique
        event.technique_confidence = decision.confidence
        event.technique_source = decision.source
        event.technique_evidence = decision.evidence
        event.technique_candidates = [
            TechniqueCandidate(
                technique=label,
                confidence=confidence,
                evidence=["model-candidate"],
            )
            for label, confidence in decision.candidates
        ]
        event.related_note_index = None
        event.harmonic_type = (
            "natural" if decision.technique == "harmonic" else None
        )
        event.harmonic_touch_fret = None
    return events, model_info


def finalize_techniques(
    path: Path,
    notes: list[NoteEvent],
    technique_model_path: Path | None = None,
    technique_gate_path: Path | None = None,
    technique_bundle_path: Path | None = None,
) -> tuple[list[NoteEvent], dict[str, object]]:
    if technique_bundle_path is not None:
        return annotate_with_technique_bundle(
            path,
            notes,
            technique_bundle_path,
        )
    if technique_model_path is not None and technique_gate_path is not None:
        return annotate_with_technique_model(
            path,
            notes,
            technique_model_path,
            technique_gate_path,
        )

    for note in notes:
        if note.harmonic_type is not None:
            note.string, note.fret = select_fingering(note.pitch, None)
            note.position_source = "pitch-preserving-fallback-v1"
        note.technique = "unknown"
        note.technique_confidence = 0.0
        note.technique_source = "unreviewed-heuristic"
        note.technique_evidence = ["heuristic-candidate-withheld"]
        note.related_note_index = None
        note.harmonic_type = None
        note.harmonic_touch_fret = None
    return notes, {
        "name": "unreviewed-technique-candidates",
        "version": "v1",
        "kind": "heuristic",
        "labels": ["unknown"],
    }


def chord_score(pitch_classes: set[int], root: int, intervals: tuple[int, ...]) -> float:
    chord = {(root + interval) % 12 for interval in intervals}
    matched = len(chord & pitch_classes)
    foreign = len(pitch_classes - chord)
    return matched * 2.0 - foreign * 0.35


def infer_chords(
    notes: Iterable[NoteEvent],
    bpm: float,
    beat_times: Iterable[float] | None = None,
    beats_per_measure: int = 4,
    pickup_beats: int = 0,
) -> list[dict[str, object]]:
    note_list = list(notes)
    if not note_list:
        return []
    beat = 60 / bpm
    duration = max(note.end for note in note_list)
    result: list[dict[str, object]] = []
    detected_beats = [
        float(value)
        for value in (beat_times or [])
        if math.isfinite(float(value)) and float(value) >= 0
    ]
    detected_beats = [
        value
        for index, value in enumerate(detected_beats)
        if index == 0 or value - detected_beats[index - 1] > 0.05
    ]
    windows: list[tuple[float, float]] = []
    if len(detected_beats) >= beats_per_measure + 1:
        safe_pickup_beats = max(
            0,
            min(int(pickup_beats), beats_per_measure - 1),
        )
        intervals = [
            current - previous
            for previous, current in zip(
                detected_beats,
                detected_beats[1:],
            )
            if 0.15 <= current - previous <= 3.0
        ]
        fallback_interval = (
            sorted(intervals)[len(intervals) // 2]
            if intervals
            else beat
        )
        if safe_pickup_beats > 0:
            windows.append(
                (
                    detected_beats[0],
                    detected_beats[safe_pickup_beats],
                )
            )
        for index in range(
            safe_pickup_beats,
            len(detected_beats),
            beats_per_measure,
        ):
            start = detected_beats[index]
            if start >= duration:
                break
            end_index = index + beats_per_measure
            end = (
                detected_beats[end_index]
                if end_index < len(detected_beats)
                else start + fallback_interval * beats_per_measure
            )
            windows.append((start, max(end, start + 0.1)))
    if not windows:
        window = beat * beats_per_measure
        start = 0.0
        while start < duration:
            windows.append((start, min(start + window, duration)))
            start += window

    qualities = (("", (0, 4, 7)), ("m", (0, 3, 7)), ("7", (0, 4, 7, 10)))
    for start, end in windows:
        active = {
            note.pitch % 12
            for note in note_list
            if note.start < end and note.end > start
        }
        if len(active) < 2:
            result.append(
                {
                    "start": round(start, 4),
                    "duration": round(end - start, 4),
                    "chord": "N.C.",
                }
            )
            continue
        best_name = "N.C."
        best_score = -100.0
        for root in range(12):
            for suffix, intervals in qualities:
                score = chord_score(active, root, intervals)
                if score > best_score:
                    best_score = score
                    best_name = f"{PITCH_NAMES[root]}{suffix}"
        result.append(
            {
                "start": round(start, 4),
                "duration": round(end - start, 4),
                "chord": best_name,
            }
        )
    return result


def classify_guitar_part_roles(
    parts: list[tuple[str, list[NoteEvent]]],
) -> dict[str, tuple[str, float]]:
    if len(parts) <= 1:
        return {
            part_id: ("unknown", 0.5)
            for part_id, _events in parts
        }

    def profile(events: list[NoteEvent]) -> tuple[float, float]:
        if not events:
            return 0.5, 0.0
        ordered = sorted(events, key=lambda event: (event.start, event.pitch))
        groups: list[list[NoteEvent]] = []
        for event in ordered:
            if not groups or event.start - groups[-1][0].start > 0.055:
                groups.append([event])
            else:
                groups[-1].append(event)
        monophonic_ratio = sum(len(group) == 1 for group in groups) / len(groups)
        chord_ratio = sum(len(group) >= 3 for group in groups) / len(groups)
        pitches = sorted(event.pitch for event in ordered)
        median_pitch = pitches[len(pitches) // 2]
        duration = max(event.end for event in ordered)
        note_rate = len(ordered) / max(1.0, duration)
        high_register = max(0.0, min(1.0, (median_pitch - 52) / 20))
        sparse_motion = 1.0 - max(0.0, min(1.0, note_rate / 14))
        lead_signal = (
            monophonic_ratio * 0.48
            + high_register * 0.34
            + sparse_motion * 0.18
            - chord_ratio * 0.2
        )
        return lead_signal, chord_ratio

    profiles = {
        part_id: profile(events)
        for part_id, events in parts
    }
    ordered_ids = sorted(
        profiles,
        key=lambda part_id: profiles[part_id][0],
        reverse=True,
    )
    lead_id = ordered_ids[0]
    rhythm_id = ordered_ids[-1]
    contrast = profiles[lead_id][0] - profiles[rhythm_id][0]
    if contrast < 0.12:
        return {
            part_id: ("unknown", round(0.5 + contrast, 4))
            for part_id, _events in parts
        }

    confidence = round(min(0.94, 0.6 + contrast * 0.55), 4)
    roles: dict[str, tuple[str, float]] = {
        part_id: ("mixed", round(max(0.5, confidence - 0.12), 4))
        for part_id, _events in parts
    }
    roles[lead_id] = ("lead", confidence)
    roles[rhythm_id] = ("rhythm", confidence)
    return roles


def confirm_independent_guitar_parts(
    parts: list[tuple[str, list[NoteEvent]]],
    roles: dict[str, tuple[str, float]],
    separation_confidences: dict[str, float],
) -> bool:
    """Require musical evidence before exposing spatial splits as instruments."""
    if len(parts) != 2:
        return False
    part_ids = [part_id for part_id, _events in parts]
    if {
        roles.get(part_id, ("unknown", 0.0))[0]
        for part_id in part_ids
    } != {"lead", "rhythm"}:
        return False
    if any(
        roles.get(part_id, ("unknown", 0.0))[1] < 0.66
        or separation_confidences.get(part_id, 0.0) < 0.72
        for part_id in part_ids
    ):
        return False

    event_sets = [events for _part_id, events in parts]
    if any(len(events) < 8 for events in event_sets):
        return False
    if any(
        max(event.end for event in events)
        - min(event.start for event in events)
        < 3.0
        for events in event_sets
    ):
        return False

    smaller, larger = sorted(event_sets, key=len)
    starts_by_pitch: dict[int, list[float]] = {}
    for event in larger:
        starts_by_pitch.setdefault(event.pitch, []).append(event.start)
    for starts in starts_by_pitch.values():
        starts.sort()

    matched = 0
    consumed: dict[int, set[int]] = {}
    for event in sorted(smaller, key=lambda item: (item.pitch, item.start)):
        starts = starts_by_pitch.get(event.pitch, [])
        index = bisect_left(starts, event.start - 0.07)
        used = consumed.setdefault(event.pitch, set())
        while index < len(starts) and starts[index] <= event.start + 0.07:
            if index not in used:
                used.add(index)
                matched += 1
                break
            index += 1

    duplicate_ratio = matched / max(1, len(smaller))
    return duplicate_ratio <= 0.4


def estimate_key(notes: Iterable[NoteEvent]) -> str:
    pitch_weights = [0.0] * 12
    for note in notes:
        pitch_weights[note.pitch % 12] += (
            max(0.04, note.end - note.start)
            * max(1, note.velocity)
            * max(0.1, note.confidence)
        )
    total = sum(pitch_weights)
    if total <= 0:
        return "Unknown"
    pitch_weights = [weight / total for weight in pitch_weights]
    profiles = {
        "major": (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
        "minor": (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
    }
    best = ("Unknown", float("-inf"))
    for quality, profile in profiles.items():
        for root in range(12):
            score = sum(
                pitch_weights[pitch_class] * profile[(pitch_class - root) % 12]
                for pitch_class in range(12)
            )
            if score > best[1]:
                best = (f"{PITCH_NAMES[root]} {quality}", score)
    return best[0]


def json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    configure_utf8_stdio()
    if run_internal_module():
        return
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware-probe",
        action="store_true",
        help="Report local accelerator and optional provider availability.",
    )
    parser.add_argument(
        "--runtime-self-test",
        action="store_true",
        help="Verify packaged runtime imports and required model registries.",
    )
    parser.add_argument(
        "--media-probe",
        action="store_true",
        help="Normalize one local media file and report its decoded duration.",
    )
    parser.add_argument(
        "--prepare-transcription-provider",
        action="store_true",
        help="Download and verify the selected transcription provider.",
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--input", type=Path)
    source_group.add_argument("--url")
    parser.add_argument(
        "--cookies-from-browser",
        choices=(
            "brave",
            "chrome",
            "chromium",
            "edge",
            "firefox",
            "opera",
            "safari",
            "vivaldi",
        ),
        help="Read cookies from this browser for this URL acquisition only.",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        help="Read an ephemeral Netscape cookie file for this acquisition only.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--transcription-provider",
        choices=(
            "basic-pitch",
            "muscriptor",
            "guitar-specialist",
            "multitrack-research",
        ),
        default=os.environ.get(
            "GTS_TRANSCRIPTION_PROVIDER",
            "basic-pitch",
        ),
    )
    parser.add_argument(
        "--transcription-model",
        default=os.environ.get("GTS_TRANSCRIPTION_MODEL"),
        help="Provider model variant or local weights path.",
    )
    parser.add_argument(
        "--transcription-variant",
        choices=("small", "medium", "large"),
        default=os.environ.get("GTS_TRANSCRIPTION_VARIANT"),
        help="Resolved hardware tier for auxiliary model selection.",
    )
    parser.add_argument(
        "--tab-model",
        type=Path,
        help="Optional trained StringTrace string/fret checkpoint.",
    )
    parser.add_argument(
        "--tab-model-mode",
        choices=("direct", "hybrid", "onset-position-research"),
        default=os.environ.get("STRINGTRACE_TAB_MODEL_MODE", "direct"),
        help=(
            "Use direct StringTrace decoding, Basic Pitch plus StringTrace, "
            "or the explicit onset-position research bundle."
        ),
    )
    parser.add_argument(
        "--technique-model",
        type=Path,
        help="Optional calibrated note-technique checkpoint.",
    )
    parser.add_argument(
        "--technique-gate",
        type=Path,
        help="Required fixed validation/test gate for --technique-model.",
    )
    parser.add_argument(
        "--technique-bundle",
        type=Path,
        help="Optional selective multi-model technique bundle.",
    )
    parser.add_argument(
        "--acquire-only",
        action="store_true",
        help="Download an online source into the output directory and stop.",
    )
    arguments = parser.parse_args()

    if arguments.hardware_probe:
        print(json.dumps(hardware_probe(), ensure_ascii=False))
        return
    if arguments.runtime_self_test:
        print(json.dumps(runtime_self_test(), ensure_ascii=False))
        return
    if arguments.prepare_transcription_provider:
        prepare_transcription_provider(
            arguments.transcription_provider,
            arguments.transcription_model,
            arguments.transcription_variant,
        )
        return
    if arguments.output is None:
        parser.error("--output is required")
    if arguments.input is None and arguments.url is None:
        parser.error("one of --input or --url is required")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments.media_probe:
        if arguments.input is None:
            parser.error("--media-probe requires --input")
        normalized = normalize_audio(arguments.input.resolve(), output)
        print(
            json.dumps(
                {
                    "duration": decoded_audio_duration(normalized),
                    "normalized": str(normalized),
                },
                ensure_ascii=False,
            )
        )
        return

    if arguments.url:
        emit_progress(3, "acquiring", "正在获取在线媒体")
        source, metadata = acquire_url(
            arguments.url,
            output,
            arguments.cookies_from_browser,
            arguments.cookies_file,
        )
        emit_progress(10, "acquiring", "在线媒体获取完成")
    else:
        source = arguments.input.resolve()
        metadata = {"title": source.stem, "artist": "本地上传"}
        emit_progress(10, "acquiring", "本地媒体读取完成")
    if not source.exists():
        raise FileNotFoundError(source)
    if arguments.acquire_only:
        acquired_metadata = {
            key: metadata[key]
            for key in ("title", "artist", "webpage_url")
            if metadata.get(key) is not None
        }
        (output / "source.metadata.json").write_text(
            json.dumps(
                acquired_metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        emit_progress(100, "acquiring", "在线媒体已保存到本机临时目录")
        return

    emit_progress(13, "separating", "正在标准化音频")
    normalized = normalize_audio(source, output)
    decoded_duration = decoded_audio_duration(normalized)
    catalog_duration = metadata.get("duration")
    preview_warning = None
    if (
        isinstance(catalog_duration, (int, float))
        and catalog_duration > decoded_duration + 2
    ):
        preview_warning = (
            f"媒体源仅提供 {decoded_duration:.1f} 秒可播放音频，"
            f"短于页面标注的 {float(catalog_duration):.1f} 秒；"
            "项目已按实际音频截断。"
        )
    metadata["duration"] = decoded_duration
    emit_progress(20, "separating", "音频标准化完成")
    emit_progress(22, "separating", "正在分离人声与乐器")
    stems = separate_audio(normalized, output)
    emit_progress(50, "separating", "基础分轨完成，正在抑制吉他串音")
    cleaned_guitar, guitar_leakage_report = suppress_guitar_stem_leakage(
        stems["guitar"],
        {
            name: path
            for name, path in stems.items()
            if name != "guitar"
        },
        output,
    )
    stems["guitar"] = cleaned_guitar
    emit_progress(53, "separating", "吉他串音抑制完成，正在检测多吉他声部")
    guitar_parts = separate_guitar_parts(stems["guitar"], output)
    vocal_leakage_reports: list[dict[str, float]] = []
    for part in guitar_parts:
        cleaned_part, leakage_report = suppress_vocal_leakage_in_guitar_part(
            Path(part["path"]),
            stems["vocals"],
            output,
        )
        part["path"] = cleaned_part
        part["vocal_leakage_report"] = leakage_report
        vocal_leakage_reports.append(leakage_report)
    emit_progress(
        55,
        "separating",
        (
            f"检测到 {len(guitar_parts)} 个可分离吉他声部"
            if len(guitar_parts) > 1
            else "吉他声部检测完成"
        ),
    )
    emit_progress(57, "transcribing", "正在检测速度与拍点")
    mix_bpm, mix_beats, onset_times, mix_tempo_map = estimate_tempo(source)
    timing_source = "mix"
    bpm = mix_bpm
    beat_times = mix_beats
    tempo_map = mix_tempo_map
    timing_diagnostics: dict[str, object] | None = None
    try:
        drum_bpm, drum_beats, _drum_onsets, drum_tempo_map = (
            estimate_tempo(stems["drums"])
        )
        timing_diagnostics = timing_grid_diagnostics(
            mix_bpm,
            mix_beats,
            drum_bpm,
            drum_beats,
        )
        beat_times, timing_source = select_timing_grid(
            mix_bpm,
            mix_beats,
            drum_bpm,
            drum_beats,
        )
        timing_diagnostics["selected_source"] = timing_source
        timing_diagnostics["selection_reason"] = timing_grid_selection_reason(
            timing_diagnostics,
            timing_source,
        )
        if timing_source == "drums":
            bpm = drum_bpm
            tempo_map = drum_tempo_map
    except Exception:
        timing_source = "mix"
    emit_progress(63, "transcribing", "拍点检测完成")
    note_source, source_warning = transcription_source(normalized, stems)
    configured_model = arguments.tab_model
    if configured_model is None and os.environ.get("STRINGTRACE_TAB_MODEL"):
        configured_model = Path(os.environ["STRINGTRACE_TAB_MODEL"])
    if configured_model is not None:
        configured_model = configured_model.resolve()
        if not configured_model.is_file():
            raise FileNotFoundError(configured_model)
    configured_position_model = (
        Path(os.environ["NOISE2FRET_MODEL"]).resolve()
        if os.environ.get("NOISE2FRET_MODEL")
        else None
    )
    configured_verification_model = (
        Path(os.environ["MR_MT3_MODEL"]).resolve()
        if os.environ.get("MR_MT3_MODEL")
        else None
    )
    transcription_targets = (
        guitar_parts
        if len(guitar_parts) > 1
        else [
            {
                "id": "guitar-1",
                "track_id": "guitar",
                "name": "吉他",
                "path": note_source,
                "separation_confidence": 1.0,
                "separation_method": "aggregate-guitar-stem",
            }
        ]
    )
    guitar_part_candidates_rejected = False
    part_transcriptions: list[dict[str, object]] = []
    for index, part in enumerate(transcription_targets):
        part_path = Path(str(part["path"]))
        emit_progress(
            65 + 16 * index / len(transcription_targets),
            "transcribing",
            (
                f"正在识别{part['name']}音符"
                if len(transcription_targets) > 1
                else "正在识别吉他音符"
            ),
        )
        part_notes, part_dropped, part_model = transcribe_guitar(
            part_path,
            configured_model,
            model_mode=arguments.tab_model_mode,
            transcription_provider=arguments.transcription_provider,
            transcription_model=arguments.transcription_model,
            position_model=configured_position_model,
            verification_model=configured_verification_model,
        )
        part_transcriptions.append(
            {
                "id": str(part["id"]),
                "track_id": str(part.get("track_id", part["id"])),
                "name": str(part["name"]),
                "path": part_path,
                "notes": part_notes,
                "dropped": part_dropped,
                "model": part_model,
                "level": audio_rms(part_path),
            }
        )
        release_torch_accelerator_cache()
    part_roles = classify_guitar_part_roles(
        [
            (str(part["id"]), part["notes"])
            for part in part_transcriptions
        ]
    )
    if len(part_transcriptions) > 1 and not confirm_independent_guitar_parts(
        [
            (str(part["id"]), part["notes"])
            for part in part_transcriptions
        ],
        part_roles,
        {
            str(part["id"]): float(part.get("separation_confidence", 0.0))
            for part in guitar_parts
        },
    ):
        guitar_part_candidates_rejected = True
        emit_progress(
            79,
            "transcribing",
            "多吉他证据不足，正在按单吉他重新识别",
        )
        aggregate_notes, aggregate_dropped, aggregate_model = transcribe_guitar(
            note_source,
            configured_model,
            model_mode=arguments.tab_model_mode,
            transcription_provider=arguments.transcription_provider,
            transcription_model=arguments.transcription_model,
            position_model=configured_position_model,
            verification_model=configured_verification_model,
        )
        release_torch_accelerator_cache()
        guitar_parts = []
        vocal_leakage_reports = []
        part_transcriptions = [
            {
                "id": "guitar-1",
                "track_id": "guitar",
                "name": "吉他",
                "path": note_source,
                "notes": aggregate_notes,
                "dropped": aggregate_dropped,
                "model": aggregate_model,
                "level": audio_rms(note_source),
            }
        ]
        part_roles = classify_guitar_part_roles(
            [("guitar-1", aggregate_notes)]
        )
    for part in part_transcriptions:
        role, role_confidence = part_roles[str(part["id"])]
        part["role"] = role
        part["role_confidence"] = role_confidence
        if role_confidence >= 0.6 and role in {"lead", "rhythm"}:
            part["name"] = "主音吉他" if role == "lead" else "节奏吉他"
        source_part = next(
            (
                candidate
                for candidate in guitar_parts
                if str(candidate["id"]) == str(part["id"])
            ),
            None,
        )
        if source_part is not None:
            source_part["name"] = part["name"]
            source_part["performance_role"] = role
            source_part["role_confidence"] = role_confidence
    primary_transcription = next(
        (
            part
            for part in part_transcriptions
            if part["role"] == "lead"
            and float(part["role_confidence"]) >= 0.6
        ),
        max(
            part_transcriptions,
            key=lambda part: float(part["level"]),
        ),
    )
    notes = primary_transcription["notes"]
    dropped_note_count = sum(
        int(part["dropped"])
        for part in part_transcriptions
    )
    transcription_model = primary_transcription["model"]
    emit_progress(82, "transcribing", "吉他音符识别完成")
    configured_technique_model = arguments.technique_model
    configured_technique_gate = arguments.technique_gate
    configured_technique_bundle = arguments.technique_bundle
    if (
        configured_technique_model is None
        and os.environ.get("STRINGTRACE_TECHNIQUE_MODEL")
    ):
        configured_technique_model = Path(
            os.environ["STRINGTRACE_TECHNIQUE_MODEL"]
        )
    if (
        configured_technique_gate is None
        and os.environ.get("STRINGTRACE_TECHNIQUE_GATE")
    ):
        configured_technique_gate = Path(
            os.environ["STRINGTRACE_TECHNIQUE_GATE"]
        )
    if (
        configured_technique_bundle is None
        and os.environ.get("STRINGTRACE_TECHNIQUE_BUNDLE")
    ):
        configured_technique_bundle = Path(
            os.environ["STRINGTRACE_TECHNIQUE_BUNDLE"]
        )
    if configured_technique_bundle is not None and (
        configured_technique_model is not None
        or configured_technique_gate is not None
    ):
        raise ValueError(
            "Technique bundle cannot be combined with a single model/gate"
        )
    if (configured_technique_model is None) != (
        configured_technique_gate is None
    ):
        raise ValueError(
            "Technique model and gate must be configured together"
        )
    if configured_technique_bundle is not None:
        configured_technique_bundle = configured_technique_bundle.resolve()
        if not configured_technique_bundle.is_file():
            raise FileNotFoundError(configured_technique_bundle)
    if configured_technique_model is not None:
        configured_technique_model = configured_technique_model.resolve()
        assert configured_technique_gate is not None
        configured_technique_gate = configured_technique_gate.resolve()
        if not configured_technique_model.is_file():
            raise FileNotFoundError(configured_technique_model)
        if not configured_technique_gate.is_file():
            raise FileNotFoundError(configured_technique_gate)
    emit_progress(84, "transcribing", "正在分析演奏技法")
    for part in part_transcriptions:
        part_notes, part_technique_model = finalize_techniques(
            Path(str(part["path"])),
            part["notes"],
            configured_technique_model,
            configured_technique_gate,
            configured_technique_bundle,
        )
        part["notes"] = part_notes
        part["technique_model"] = part_technique_model
    notes = primary_transcription["notes"]
    technique_model = primary_transcription["technique_model"]
    timing_notes = sorted(
        (
            note
            for part in part_transcriptions
            for note in part["notes"]
        ),
        key=lambda note: (note.start, note.pitch, note.string),
    )
    time_signature, meter_diagnostics = select_time_signature(
        beat_times,
        timing_notes,
    )
    beats_per_measure = time_signature[0]
    beat_times, downbeat_shift, downbeat_confidence = (
        align_beats_to_downbeats(
            beat_times,
            timing_notes,
            beats_per_measure=beats_per_measure,
        )
    )
    beat_times, pickup_beats = prepend_pickup_beats(
        beat_times,
        timing_notes,
        maximum_pickup_beats=beats_per_measure - 1,
    )
    tempo_map = tempo_map_from_beats(beat_times, bpm)
    bpm = round(float(tempo_map["median_bpm"]), 2)
    transcription_model["timing_model"] = "meter-aware-downbeat-grid-v3"
    emit_progress(89, "transcribing", "演奏技法分析完成")
    emit_progress(91, "mapping", "正在提取人声旋律与歌词")
    vocal_melody = extract_vocal_melody(stems["vocals"])
    lyric_lines = metadata.get("timed_lyrics", [])
    lyrics = align_lyrics_to_melody(
        lyric_lines if isinstance(lyric_lines, list) else [],
        vocal_melody,
    )
    vocal_melody = complete_vocal_melody_for_lyrics(
        vocal_melody,
        lyrics,
    )
    emit_progress(93, "mapping", "正在生成波形与可视化")
    levels = {name: audio_rms(path) for name, path in stems.items()}
    waveforms = {name: waveform(path) for name, path in stems.items()}
    for part in guitar_parts:
        part["level"] = audio_rms(Path(part["path"]))
        part["waveform"] = waveform(Path(part["path"]))
    visualization = visualization_timeline(normalized)
    emit_progress(96, "mapping", "波形与可视化生成完成")
    emit_progress(97, "mapping", "正在编码播放音轨")
    playable_stems = compress_stems(
        {
            **stems,
            **{
                str(part["id"]): Path(part["path"])
                for part in guitar_parts
            },
        },
        output,
    )
    for part in guitar_parts:
        part["path"] = str(playable_stems[str(part["id"])].resolve())
    emit_progress(99, "mapping", "正在写入分析结果")
    normalized.unlink(missing_ok=True)
    warnings = []
    if (
        configured_technique_model is None
        and configured_technique_bundle is None
    ):
        warnings.append(
            "Technique labels are experimental acoustic heuristics, "
            "not reliable classifications."
        )
    else:
        warnings.append(
            "Only independently gated technique labels are applied; "
            "all other predictions remain unknown."
        )
    if transcription_model.get("fallback_reason"):
        warnings.append(
            f"{transcription_model['fallback_reason']}，已自动回退到 Basic Pitch。"
        )
    elif transcription_model.get("name") == "spotify-basic-pitch":
        warnings.append(
            "String and fret positions use the Basic Pitch baseline and require review."
        )
    else:
        warnings.append(
            "String and fret positions use a trained model; publish only after benchmark gates pass."
        )
    if transcription_model.get("position_model_warning"):
        warnings.append(str(transcription_model["position_model_warning"]))
    if transcription_model.get("verification_model_warning"):
        warnings.append(
            str(transcription_model["verification_model_warning"])
        )
    if source_warning:
        warnings.append(source_warning)
    if preview_warning:
        warnings.append(preview_warning)
    if len(guitar_parts) > 1:
        warnings.append(
            "Spatially distinct guitars were transcribed into independent "
            "parts; centered or mono layers can still remain combined."
        )
        if vocal_leakage_reports:
            median_reduction = sum(
                report["vocal_similarity_reduction"]
                for report in vocal_leakage_reports
            ) / len(vocal_leakage_reports)
            warnings.append(
                "Post-separation vocal-reference masking reduced median "
                f"vocal spectral similarity by {median_reduction:.1%}."
            )
    else:
        warnings.append(
            "No independently separable guitar layers were found; "
            "centered or mono multiple guitars remain in the aggregate guitar track."
        )
        if guitar_part_candidates_rejected:
            warnings.append(
                "Stereo spatial candidates did not pass the independent-part "
                "gate, so the song was kept as one aggregate guitar part."
            )
    warnings.append(
        "Coherent leakage shared with non-guitar stems was suppressed before "
        f"guitar-part separation ({guitar_leakage_report['removed_energy_ratio']:.1%} "
        "of estimated stem energy removed)."
    )
    warnings.append(
        f"Timing uses the {timing_source} grid, "
        f"{time_signature[0]}/{time_signature[1]} meter "
        f"(confidence {float(meter_diagnostics['confidence']):.3f}), "
        f"harmonic downbeat phase {downbeat_shift} "
        f"(confidence {downbeat_confidence:.3f}), and "
        f"{pickup_beats} reconstructed pickup beat(s)."
    )
    if dropped_note_count:
        warnings.append(
            f"{dropped_note_count} note events outside the 24-fret guitar range were discarded."
        )
    result = {
        "version": 10,
        "transcription_model": transcription_model,
        "technique_model": technique_model,
        "metadata": metadata,
        "bpm": bpm,
        "time_signature": list(time_signature),
        "key": estimate_key(notes),
        "capo": 0,
        "beats": beat_times,
        "pickup_beats": pickup_beats,
        "tempo_map": tempo_map,
        "meter_diagnostics": meter_diagnostics,
        **(
            {"timing_diagnostics": timing_diagnostics}
            if timing_diagnostics is not None
            else {}
        ),
        "onsets": onset_times,
        "vocal_melody": vocal_melody,
        "lyrics": lyrics,
        "stems": {name: str(path.resolve()) for name, path in playable_stems.items()},
        "stem_tracks": [
            {
                **part,
                "path": str(part["path"]),
            }
            for part in guitar_parts
        ],
        "guitar_parts": [
            {
                "id": str(part["id"]),
                "track_id": str(part["track_id"]),
                "name": str(part["name"]),
                "role": str(part["role"]),
                "role_confidence": float(part["role_confidence"]),
                "transcription_model": part["model"],
                "notes": [
                    asdict(note)
                    for note in part["notes"]
                ],
                "chords": infer_chords(
                    part["notes"],
                    bpm,
                    beat_times,
                    beats_per_measure=beats_per_measure,
                    pickup_beats=pickup_beats,
                ),
            }
            for part in part_transcriptions
        ],
        "levels": levels,
        "waveforms": waveforms,
        "visualization": visualization,
        "notes": [asdict(note) for note in notes],
        "chords": infer_chords(
            notes,
            bpm,
            beat_times,
            beats_per_measure=beats_per_measure,
            pickup_beats=pickup_beats,
        ),
        "warnings": warnings,
    }
    result_path = output / "analysis.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps({"analysis": str(result_path)}, ensure_ascii=False))


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        import multiprocessing

        multiprocessing.freeze_support()
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        sys.stderr.write(
            json.dumps(
                {
                    "type": "gts-error",
                    "stageLabel": _CURRENT_STAGE_LABEL,
                    "errorType": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stderr.flush()
        raise SystemExit(1) from None
