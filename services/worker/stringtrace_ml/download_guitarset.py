from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ZENODO_RECORD = "3371780"
CONTENT_RANGE_PATTERN = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)")


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    size: int
    md5: str

    @property
    def url(self) -> str:
        return (
            f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/"
            f"{self.name}/content"
        )


DOWNLOADS = {
    spec.name: spec
    for spec in (
        DownloadSpec(
            name="annotation.zip",
            size=39_132_574,
            md5="b39b78e63d3446f2e54ddb7a54df9b10",
        ),
        DownloadSpec(
            name="audio_mono-mic.zip",
            size=656_927_981,
            md5="275966d6610ac34999b58426beb119c3",
        ),
    )
}


def byte_ranges(size: int, chunk_size: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("Download size must be positive")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    return [
        (start, min(start + chunk_size, size) - 1)
        for start in range(0, size, chunk_size)
    ]


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_complete(path: Path, spec: DownloadSpec) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == spec.size
        and file_md5(path) == spec.md5
    )


def part_path(parts_directory: Path, start: int, end: int) -> Path:
    return parts_directory / f"{start:012d}-{end:012d}.part"


def validate_content_range(
    value: str,
    *,
    start: int,
    end: int,
    total: int,
) -> None:
    match = CONTENT_RANGE_PATTERN.fullmatch(value.strip())
    if not match:
        raise RuntimeError(f"Missing or invalid Content-Range: {value!r}")
    actual = tuple(int(group) for group in match.groups())
    expected = (start, end, total)
    if actual != expected:
        raise RuntimeError(
            f"Unexpected Content-Range {actual}; expected {expected}"
        )


def download_part(
    spec: DownloadSpec,
    destination: Path,
    *,
    start: int,
    end: int,
) -> None:
    expected_size = end - start + 1
    temporary = destination.with_suffix(".download")
    command = [
        "curl",
        "--http2",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--max-time",
        "300",
        "--range",
        f"{start}-{end}",
        "--output",
        str(temporary),
        "--write-out",
        "%{http_code}\t%{size_download}\t%header{content-range}",
        spec.url,
    ]
    errors: list[str] = []
    for attempt in range(1, 13):
        temporary.unlink(missing_ok=True)
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            fields = result.stdout.strip().split("\t", 2)
            if len(fields) != 3:
                raise RuntimeError(
                    f"Unexpected curl status output: {result.stdout!r}"
                )
            status, downloaded, content_range = fields
            if status != "206":
                raise RuntimeError(
                    f"Range request returned HTTP {status}, not 206"
                )
            if int(float(downloaded)) != expected_size:
                raise RuntimeError(
                    f"Range request wrote {downloaded} bytes; "
                    f"expected {expected_size}"
                )
            validate_content_range(
                content_range,
                start=start,
                end=end,
                total=spec.size,
            )
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Part file is {temporary.stat().st_size} bytes; "
                    f"expected {expected_size}"
                )
            os.replace(temporary, destination)
            return
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if isinstance(error, subprocess.CalledProcessError):
                detail = error.stderr.strip() or str(error)
            else:
                detail = str(error)
            errors.append(f"attempt {attempt}: {detail}")
            if attempt < 12:
                time.sleep(min(2 * attempt, 20))
    raise RuntimeError(
        f"Could not download bytes {start}-{end} of {spec.name}: "
        + "; ".join(errors)
    )


def assemble_download(
    spec: DownloadSpec,
    destination: Path,
    parts_directory: Path,
    ranges: list[tuple[int, int]],
) -> None:
    temporary = destination.with_suffix(".assembling")
    digest = hashlib.md5()
    written = 0
    with temporary.open("wb") as output:
        for start, end in ranges:
            source_path = part_path(parts_directory, start, end)
            expected_size = end - start + 1
            if source_path.stat().st_size != expected_size:
                raise RuntimeError(f"Invalid part size: {source_path}")
            with source_path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    written += len(block)
        output.flush()
        os.fsync(output.fileno())
    if written != spec.size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Assembled {written} bytes for {spec.name}; expected {spec.size}"
        )
    actual_md5 = digest.hexdigest()
    if actual_md5 != spec.md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {spec.name}: {actual_md5}; expected {spec.md5}. "
            "Retain the parts and rerun with --discard-parts."
        )
    os.replace(temporary, destination)


def download_file(
    spec: DownloadSpec,
    output_directory: Path,
    *,
    workers: int,
    chunk_size: int,
    discard_parts: bool,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / spec.name
    if is_complete(destination, spec):
        print(f"{spec.name}: already complete and verified")
        return destination

    ranges = byte_ranges(spec.size, chunk_size)
    parts_directory = output_directory / ".parts" / spec.name
    parts_directory.mkdir(parents=True, exist_ok=True)
    expected_parts = {
        part_path(parts_directory, start, end)
        for start, end in ranges
    }
    for existing in parts_directory.glob("*.download"):
        existing.unlink()
    for existing in parts_directory.glob("*.part"):
        if discard_parts or existing not in expected_parts:
            existing.unlink()

    missing: list[tuple[int, int]] = []
    completed_bytes = 0
    for start, end in ranges:
        path = part_path(parts_directory, start, end)
        expected_size = end - start + 1
        if path.is_file() and path.stat().st_size == expected_size:
            completed_bytes += expected_size
        else:
            path.unlink(missing_ok=True)
            missing.append((start, end))

    lock = threading.Lock()

    def fetch(item: tuple[int, int]) -> int:
        start, end = item
        download_part(
            spec,
            part_path(parts_directory, start, end),
            start=start,
            end=end,
        )
        return end - start + 1

    if missing:
        print(
            f"{spec.name}: {len(missing)} ranges remaining, "
            f"{completed_bytes / spec.size:.1%} restored",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch, item) for item in missing]
            for future in as_completed(futures):
                size = future.result()
                with lock:
                    completed_bytes += size
                    print(
                        f"{spec.name}: {completed_bytes / spec.size:.1%} "
                        f"({completed_bytes}/{spec.size} bytes)",
                        flush=True,
                    )

    assemble_download(spec, destination, parts_directory, ranges)
    for source_path in expected_parts:
        source_path.unlink()
    parts_directory.rmdir()
    parts_root = parts_directory.parent
    if not any(parts_root.iterdir()):
        parts_root.rmdir()
    print(f"{spec.name}: verified md5:{spec.md5}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the GuitarSet files used by StringTrace.",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--file",
        action="append",
        choices=tuple(DOWNLOADS),
        dest="files",
        help="Download one archive. Repeat to select multiple archives.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=1)
    parser.add_argument(
        "--discard-parts",
        action="store_true",
        help="Discard completed range parts and download them again.",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if not 1 <= arguments.chunk_mib <= 64:
        parser.error("--chunk-mib must be between 1 and 64")

    selected = arguments.files or list(DOWNLOADS)
    for name in selected:
        download_file(
            DOWNLOADS[name],
            arguments.output_directory.resolve(),
            workers=arguments.workers,
            chunk_size=arguments.chunk_mib * 1024 * 1024,
            discard_parts=arguments.discard_parts,
        )


if __name__ == "__main__":
    main()
