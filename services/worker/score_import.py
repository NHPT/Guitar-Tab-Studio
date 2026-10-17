#!/usr/bin/env python3
"""Import clean guitar TAB images into an editable score document."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import label


@dataclass(frozen=True)
class Staff:
    page: int
    lines: tuple[int, int, int, int, int, int]
    left: int
    right: int

    @property
    def gap(self) -> float:
        return float(np.median(np.diff(self.lines)))


@dataclass(frozen=True)
class Symbol:
    x: float
    string: int
    fret: int
    confidence: float
    source: str


def longest_run(row: np.ndarray) -> tuple[int, int]:
    padded = np.pad(row.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if starts.size == 0:
        return 0, 0
    lengths = ends - starts
    index = int(np.argmax(lengths))
    return int(starts[index]), int(ends[index])


def grouped_centers(values: list[int]) -> list[int]:
    groups: list[list[int]] = []
    for value in values:
        if not groups or value > groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(sum(group) / len(group)) for group in groups]


def detect_staffs(gray: np.ndarray, page: int) -> list[Staff]:
    dark = gray < 190
    height, width = dark.shape
    row_runs = [longest_run(row) for row in dark]
    candidates = grouped_centers(
        [
            y
            for y, (start, end) in enumerate(row_runs)
            if end - start >= max(42, width * 0.24)
        ]
    )
    staffs: list[Staff] = []
    index = 0
    while index + 5 < len(candidates):
        group = candidates[index : index + 6]
        gaps = np.diff(group)
        median_gap = float(np.median(gaps))
        if (
            4 <= median_gap <= max(80, height * 0.08)
            and float(np.max(np.abs(gaps - median_gap))) <= max(2.5, median_gap * 0.34)
        ):
            spans = [row_runs[y] for y in group]
            left = int(np.median([span[0] for span in spans]))
            right = int(np.median([span[1] for span in spans]))
            if right - left >= max(42, width * 0.22):
                staffs.append(
                    Staff(
                        page=page,
                        lines=tuple(group),  # type: ignore[arg-type]
                        left=left,
                        right=right,
                    )
                )
                index += 6
                continue
        index += 1
    return staffs


def tesseract_tsv(
    image: Image.Image,
    *,
    languages: str,
    page_segmentation: int,
    whitelist: str | None = None,
) -> list[dict[str, object]]:
    executable = shutil.which("tesseract")
    if not executable:
        return []
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    command = [
        executable,
        "stdin",
        "stdout",
        "--psm",
        str(page_segmentation),
        "-l",
        languages,
    ]
    if whitelist:
        command.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
    command.append("tsv")
    completed = subprocess.run(
        command,
        input=stream.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return []
    rows: list[dict[str, object]] = []
    lines = completed.stdout.decode("utf-8", errors="replace").splitlines()
    for line in lines[1:]:
        columns = line.split("\t")
        if len(columns) < 12:
            continue
        text = columns[11].strip()
        if not text:
            continue
        try:
            rows.append(
                {
                    "left": int(columns[6]),
                    "top": int(columns[7]),
                    "width": int(columns[8]),
                    "height": int(columns[9]),
                    "confidence": max(0.0, float(columns[10])) / 100,
                    "text": text,
                }
            )
        except ValueError:
            continue
    return rows


def cleaned_staff_image(
    gray: np.ndarray,
    staff: Staff,
    *,
    padding: float = 0.55,
) -> tuple[Image.Image, int, int, int]:
    gap = staff.gap
    top = max(0, round(staff.lines[0] - gap * padding))
    bottom = min(gray.shape[0], round(staff.lines[-1] + gap * padding) + 1)
    left = max(0, round(staff.left - gap))
    right = min(gray.shape[1], round(staff.right + gap) + 1)
    crop = gray[top:bottom, left:right].copy()
    dark = crop < 190
    for row_index, row in enumerate(dark):
        start, end = longest_run(row)
        if end - start >= max(30, crop.shape[1] * 0.2):
            crop[row_index, start:end] = 255
    scale = max(3, min(8, math.ceil(38 / max(gap, 1))))
    image = Image.fromarray(crop).resize(
        (crop.shape[1] * scale, crop.shape[0] * scale),
        Image.Resampling.LANCZOS,
    )
    image = ImageOps.autocontrast(image).point(
        lambda value: 0 if value < 185 else 255
    )
    return image, left, top, scale


def parse_fret_tokens(
    gray: np.ndarray,
    staff: Staff,
) -> list[Symbol]:
    image, offset_x, offset_y, scale = cleaned_staff_image(gray, staff)
    tokens = tesseract_tsv(
        image,
        languages="eng",
        page_segmentation=11,
        whitelist="0123456789xX",
    )
    symbols: list[Symbol] = []
    for token in tokens:
        raw = str(token["text"])
        matches = list(re.finditer(r"\d{1,2}|[xX]", raw))
        if not matches:
            continue
        token_left = float(token["left"]) / scale + offset_x
        token_width = float(token["width"]) / scale
        token_top = float(token["top"]) / scale + offset_y
        token_height = float(token["height"]) / scale
        center_y = token_top + token_height / 2
        string = int(
            np.argmin([abs(center_y - line) for line in staff.lines])
        ) + 1
        if abs(center_y - staff.lines[string - 1]) > staff.gap * 0.72:
            continue
        for match in matches:
            center_x = token_left + token_width * (
                (match.start() + match.end()) / 2 / max(1, len(raw))
            )
            value = match.group(0)
            fret = 0 if value.lower() == "x" else min(24, int(value))
            symbols.append(
                Symbol(
                    x=center_x,
                    string=string,
                    fret=fret,
                    confidence=max(0.45, float(token["confidence"])),
                    source="ocr",
                )
            )
    return symbols


def component_fallback_symbols(
    gray: np.ndarray,
    staff: Staff,
    existing: list[Symbol],
) -> list[Symbol]:
    image, offset_x, offset_y, scale = cleaned_staff_image(gray, staff)
    pixels = np.asarray(image) < 128
    components, count = label(pixels)
    candidates: list[tuple[Symbol, tuple[int, int, int, int]]] = []
    for component in range(1, count + 1):
        ys, xs = np.where(components == component)
        if xs.size < 5:
            continue
        left = float(xs.min()) / scale + offset_x
        right = float(xs.max() + 1) / scale + offset_x
        top = float(ys.min()) / scale + offset_y
        bottom = float(ys.max() + 1) / scale + offset_y
        width = right - left
        height = bottom - top
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        if not staff.left + staff.gap < center_x < staff.right - staff.gap:
            continue
        if not 0.3 * staff.gap <= width <= 2.3 * staff.gap:
            continue
        if not 0.2 * staff.gap <= height <= 1.65 * staff.gap:
            continue
        string = int(
            np.argmin([abs(center_y - line) for line in staff.lines])
        ) + 1
        if abs(center_y - staff.lines[string - 1]) > staff.gap * 0.65:
            continue
        if any(
            abs(symbol.x - center_x) <= staff.gap * 0.7
            and symbol.string == string
            for symbol in existing
        ):
            continue
        candidates.append(
            (
                Symbol(
                    x=center_x,
                    string=string,
                    fret=0,
                    confidence=0.3,
                    source="visual-marker",
                ),
                (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
            )
        )
    if not candidates:
        return []

    cell_width = 84
    cell_height = 64
    sheet = Image.new("L", (cell_width, cell_height * len(candidates)), 255)
    for index, (_symbol, (left, top, right, bottom)) in enumerate(candidates):
        glyph = Image.fromarray(
            np.where(pixels[top:bottom, left:right], 0, 255).astype(np.uint8)
        )
        glyph.thumbnail((48, 48), Image.Resampling.NEAREST)
        sheet.paste(
            glyph,
            (
                (cell_width - glyph.width) // 2,
                index * cell_height + (cell_height - glyph.height) // 2,
            ),
        )
    recognized: dict[int, tuple[int, float]] = {}
    for token in tesseract_tsv(
        sheet,
        languages="eng",
        page_segmentation=6,
        whitelist="0123456789xX",
    ):
        match = re.search(r"\d{1,2}|[xX]", str(token["text"]))
        if not match:
            continue
        index = min(
            len(candidates) - 1,
            max(
                0,
                int(
                    (float(token["top"]) + float(token["height"]) / 2)
                    // cell_height
                ),
            ),
        )
        value = match.group(0)
        recognized[index] = (
            0 if value.lower() == "x" else min(24, int(value)),
            max(0.45, float(token["confidence"])),
        )
    return [
        Symbol(
            x=symbol.x,
            string=symbol.string,
            fret=recognized.get(index, (symbol.fret, symbol.confidence))[0],
            confidence=recognized.get(
                index,
                (symbol.fret, symbol.confidence),
            )[1],
            source="ocr" if index in recognized else symbol.source,
        )
        for index, (symbol, _bounds) in enumerate(candidates)
    ]


def detect_barlines(gray: np.ndarray, staff: Staff) -> list[float]:
    dark = gray < 190
    top = max(0, staff.lines[0] - 2)
    bottom = min(gray.shape[0], staff.lines[-1] + 3)
    band = dark[top:bottom, staff.left : staff.right + 1]
    occupancy = np.mean(band, axis=0)
    candidates = grouped_centers(
        [
            staff.left + index
            for index, value in enumerate(occupancy)
            if value >= 0.72
        ]
    )
    boundaries = [
        float(value)
        for value in candidates
        if staff.left - 2 <= value <= staff.right + 2
    ]
    if not boundaries or boundaries[0] > staff.left + staff.gap * 2:
        boundaries.insert(0, float(staff.left))
    if boundaries[-1] < staff.right - staff.gap * 2:
        boundaries.append(float(staff.right))
    boundaries = sorted(set(boundaries))
    if len(boundaries) < 2:
        return [float(staff.left), float(staff.right)]
    return boundaries


def chord_tokens(gray: np.ndarray, staff: Staff) -> list[tuple[float, str]]:
    top = max(0, round(staff.lines[0] - staff.gap * 7.2))
    bottom = max(top + 1, round(staff.lines[0] - staff.gap * 1.3))
    left = max(0, round(staff.left - staff.gap * 3))
    right = min(gray.shape[1], round(staff.right + staff.gap * 3) + 1)
    crop = Image.fromarray(gray[top:bottom, left:right])
    scale = max(2, min(6, math.ceil(30 / max(staff.gap, 1))))
    crop = ImageOps.autocontrast(
        crop.resize(
            (crop.width * scale, crop.height * scale),
            Image.Resampling.LANCZOS,
        )
    )
    rows = tesseract_tsv(
        crop,
        languages="eng",
        page_segmentation=6,
        whitelist="ABCDEFG#bmsuajdM234679/.",
    )
    chords: list[tuple[float, str]] = []
    pattern = re.compile(r"^[A-G](?:#|b)?(?:m|maj|sus|dim|aug)?[234679]?(?:/[A-G](?:#|b)?)?$")
    for row in rows:
        text = str(row["text"]).strip()
        if pattern.fullmatch(text):
            chords.append(
                (
                    left
                    + (float(row["left"]) + float(row["width"]) / 2) / scale,
                    text,
                )
            )
    return chords


def lyric_text(gray: np.ndarray, staff: Staff, next_top: int) -> str:
    top = min(gray.shape[0], round(staff.lines[-1] + staff.gap * 7))
    bottom = min(
        gray.shape[0],
        max(top + 1, min(next_top, round(top + staff.gap * 5))),
    )
    if bottom - top < 5:
        return ""
    crop = Image.fromarray(gray[top:bottom, staff.left : staff.right + 1])
    scale = max(2, min(5, math.ceil(26 / max(staff.gap, 1))))
    crop = ImageOps.autocontrast(
        crop.resize(
            (crop.width * scale, crop.height * scale),
            Image.Resampling.LANCZOS,
        )
    )
    rows = tesseract_tsv(
        crop,
        languages="chi_sim+eng",
        page_segmentation=6,
    )
    words = [
        str(row["text"]).strip()
        for row in rows
        if float(row["confidence"]) >= 0.25
    ]
    return " ".join(word for word in words if word)


def lyric_tokens(text: str) -> list[str]:
    if re.search(r"[\u3400-\u9fff]", text):
        return re.findall(r"[\u3400-\u9fff]", text)
    return re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?", text)


def score_document(paths: list[Path], original_name: str) -> dict[str, object]:
    pages = [
        np.asarray(Image.open(path).convert("L"))
        for path in paths
    ]
    staffs = [
        staff
        for page_index, gray in enumerate(pages)
        for staff in detect_staffs(gray, page_index)
    ]
    if not staffs:
        raise ValueError("未检测到六线谱，请上传清晰、正向且包含完整六条谱线的图片")

    measures: list[dict[str, object]] = []
    lyrics: list[dict[str, object]] = []
    uncertain_count = 0
    recognized_count = 0
    for staff_index, staff in enumerate(staffs):
        gray = pages[staff.page]
        symbols = component_fallback_symbols(gray, staff, [])
        symbols.sort(key=lambda symbol: (symbol.x, symbol.string))
        deduplicated: list[Symbol] = []
        for symbol in sorted(
            symbols,
            key=lambda item: (-item.confidence, item.x, item.string),
        ):
            duplicate = next(
                (
                    existing
                    for existing in deduplicated
                    if existing.string == symbol.string
                    and abs(existing.x - symbol.x) <= staff.gap * 0.78
                ),
                None,
            )
            if duplicate is None:
                deduplicated.append(symbol)
        symbols = sorted(deduplicated, key=lambda symbol: (symbol.x, symbol.string))
        recognized_count += sum(symbol.source == "ocr" for symbol in symbols)
        uncertain_count += sum(
            symbol.source == "visual-marker"
            for symbol in symbols
        )
        boundaries = detect_barlines(gray, staff)
        chords = chord_tokens(gray, staff)
        next_staff = next(
            (
                candidate.lines[0]
                for candidate in staffs[staff_index + 1 :]
                if candidate.page == staff.page
            ),
            gray.shape[0],
        )
        text = lyric_text(gray, staff, next_staff)
        staff_measure_start = len(measures)
        for left, right in zip(boundaries, boundaries[1:]):
            if right - left < staff.gap * 3:
                continue
            measure_symbols = [
                symbol
                for symbol in symbols
                if left + 1 < symbol.x < right - 1
            ]
            attacks: list[list[Symbol]] = []
            for symbol in measure_symbols:
                if (
                    not attacks
                    or symbol.x - attacks[-1][0].x > max(3, staff.gap * 0.72)
                ):
                    attacks.append([symbol])
                else:
                    attacks[-1].append(symbol)
            chord = next(
                (
                    name
                    for x, name in reversed(chords)
                    if x <= (left + right) / 2
                ),
                "N.C.",
            )
            events: list[dict[str, object]] = []
            for attack_index, attack in enumerate(attacks):
                center = float(np.mean([symbol.x for symbol in attack]))
                raw_position = (center - left) / max(1, right - left) * 4
                position = min(3.75, max(0, round(raw_position * 4) / 4))
                next_position = (
                    min(
                        4,
                        max(
                            position + 0.25,
                            round(
                                (
                                    float(
                                        np.mean(
                                            [
                                                symbol.x
                                                for symbol in attacks[attack_index + 1]
                                            ]
                                        )
                                    )
                                    - left
                                )
                                / max(1, right - left)
                                * 16
                            )
                            / 4,
                        ),
                    )
                    if attack_index + 1 < len(attacks)
                    else min(4, position + 1)
                )
                for symbol in attack:
                    events.append(
                        {
                            "position": position,
                            "duration": max(0.25, next_position - position),
                            "string": symbol.string,
                            "fret": symbol.fret,
                            "confidence": symbol.confidence,
                            "source": symbol.source,
                        }
                    )
            measures.append(
                {
                    "chord": chord,
                    "events": events,
                    "source": {
                        "page": staff.page + 1,
                        "staff": staff_index + 1,
                    },
                }
            )
        words = lyric_tokens(text)
        staff_measure_count = len(measures) - staff_measure_start
        if words and staff_measure_count > 0:
            span = staff_measure_count * 4
            for index, word in enumerate(words):
                lyrics.append(
                    {
                        "position": staff_measure_start * 4
                        + span * (index + 0.5) / len(words),
                        "text": word,
                    }
                )

    if not measures:
        raise ValueError("检测到六线谱，但未找到可用的小节范围")
    warnings: list[str] = []
    if uncertain_count:
        warnings.append(
            f"{uncertain_count} 个符号仅检测到位置、未可靠读出品位，已按 0 品保留待校对。"
        )
    if recognized_count == 0:
        warnings.append("未可靠识别到品位数字，请在编辑器中逐音校对。")
    confidence = recognized_count / max(1, recognized_count + uncertain_count)
    return {
        "version": 1,
        "title": Path(original_name).stem or "导入曲谱",
        "pageCount": len(pages),
        "staffCount": len(staffs),
        "timeSignature": [4, 4],
        "bpm": 90,
        "confidence": round(confidence, 4),
        "measures": measures,
        "lyrics": lyrics,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--name", required=True)
    arguments = parser.parse_args()
    document = score_document(arguments.input, arguments.name)
    print(json.dumps(document, ensure_ascii=False))


if __name__ == "__main__":
    main()
