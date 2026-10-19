#!/usr/bin/env python3
"""Generate the six-string sunrise pick desktop icon assets."""

from __future__ import annotations

import io
import struct
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "apps" / "desktop" / "build-resources"
WINDOWS_SIZES = (16, 20, 24, 30, 32, 36, 40, 48, 60, 64, 72, 96, 128, 256)


def cubic(
    start: tuple[float, float],
    control_a: tuple[float, float],
    control_b: tuple[float, float],
    end: tuple[float, float],
    steps: int = 36,
) -> list[tuple[float, float]]:
    points = []
    for index in range(1, steps + 1):
        progress = index / steps
        inverse = 1 - progress
        points.append(
            (
                inverse**3 * start[0]
                + 3 * inverse**2 * progress * control_a[0]
                + 3 * inverse * progress**2 * control_b[0]
                + progress**3 * end[0],
                inverse**3 * start[1]
                + 3 * inverse**2 * progress * control_a[1]
                + 3 * inverse * progress**2 * control_b[1]
                + progress**3 * end[1],
            )
        )
    return points


def pick_polygon(size: int) -> list[tuple[float, float]]:
    points = [(0.5, 0.18)]
    points += cubic(points[-1], (0.31, 0.16), (0.18, 0.27), (0.21, 0.43))
    points += cubic(points[-1], (0.24, 0.61), (0.39, 0.79), (0.5, 0.87))
    points += cubic(points[-1], (0.61, 0.79), (0.76, 0.61), (0.79, 0.43))
    points += cubic(points[-1], (0.82, 0.27), (0.69, 0.16), (0.5, 0.18))
    return [(x * size, y * size) for x, y in points]


def vertical_gradient(
    size: int,
    top: tuple[int, int, int, int],
    bottom: tuple[int, int, int, int],
) -> Image.Image:
    image = Image.new("RGBA", (size, size))
    pixels = image.load()
    for y in range(size):
        progress = y / max(1, size - 1)
        color = tuple(
            round(top[channel] * (1 - progress) + bottom[channel] * progress)
            for channel in range(4)
        )
        for x in range(size):
            pixels[x, y] = color
    return image


def render_icon(size: int, platform: str) -> Image.Image:
    scale = 4
    canvas_size = size * scale
    margin_ratio = 0.09 if platform == "mac" else 0.12
    margin = round(canvas_size * margin_ratio)
    radius = round(canvas_size * (0.19 if platform == "mac" else 0.17))

    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    tile = (margin, margin, canvas_size - margin, canvas_size - margin)
    draw.rounded_rectangle(tile, radius=radius, fill=(24, 26, 32, 255))
    draw.rounded_rectangle(
        tile,
        radius=radius,
        outline=(255, 255, 255, 22),
        width=max(scale, round(canvas_size * 0.004)),
    )

    pick_mask = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(pick_mask).polygon(pick_polygon(canvas_size), fill=255)
    gradient = vertical_gradient(
        canvas_size,
        (246, 175, 82, 255),
        (203, 71, 55, 255),
    )
    image.paste(gradient, (0, 0), pick_mask)

    symbol = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    symbol_draw = ImageDraw.Draw(symbol)
    ink = (24, 26, 32, 255)
    ring_radius = canvas_size * (0.09 if size <= 48 else 0.115)
    ring_center = (canvas_size * 0.5, canvas_size * 0.43)
    ring_box = (
        ring_center[0] - ring_radius,
        ring_center[1] - ring_radius,
        ring_center[0] + ring_radius,
        ring_center[1] + ring_radius,
    )
    if size > 24:
        symbol_draw.ellipse(
            ring_box,
            outline=ink,
            width=max(scale, round(canvas_size * 0.025)),
        )

    string_positions = (
        (0.29, 0.374, 0.458, 0.542, 0.626, 0.71)
        if size <= 24
        else (0.33, 0.398, 0.466, 0.534, 0.602, 0.67)
    )
    string_widths = (
        (0.01,) * 6
        if size <= 48
        else (0.012, 0.011, 0.010, 0.009, 0.008, 0.007)
    )
    for position, width in zip(string_positions, string_widths, strict=True):
        x = round(canvas_size * position)
        symbol_draw.line(
            (
                x,
                round(canvas_size * 0.245),
                x,
                round(canvas_size * 0.695),
            ),
            fill=ink,
            width=max(scale, round(canvas_size * width)),
        )
    symbol_draw.rounded_rectangle(
        (
            round(canvas_size * 0.355),
            round(canvas_size * 0.61),
            round(canvas_size * 0.645),
            round(canvas_size * 0.642),
        ),
        radius=max(scale, round(canvas_size * 0.01)),
        fill=ink,
    )
    symbol.putalpha(ImageChops.multiply(symbol.getchannel("A"), pick_mask))
    image.alpha_composite(symbol)

    result = image.resize((size, size), Image.Resampling.LANCZOS)
    if size <= 48:
        crisp = ImageDraw.Draw(result)
        positions = (0.27, 0.36, 0.44, 0.56, 0.64, 0.73)
        width = 1 if size < 40 else 2
        for position in positions:
            x = round(size * position)
            crisp.line(
                (x, round(size * 0.25), x, round(size * 0.69)),
                fill=ink,
                width=width,
            )
        if size > 24:
            radius = round(size * 0.09)
            center = (round(size * 0.5), round(size * 0.43))
            crisp.ellipse(
                (
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                ),
                outline=ink,
                width=width,
            )
        crisp.line(
            (
                round(size * 0.3),
                round(size * 0.62),
                round(size * 0.7),
                round(size * 0.62),
            ),
            fill=ink,
            width=max(1, width + 1),
        )
    return result


def build_ico() -> None:
    frames: list[bytes] = []
    for size in WINDOWS_SIZES:
        payload = io.BytesIO()
        render_icon(size, "windows").save(payload, format="PNG")
        frames.append(payload.getvalue())

    directory_size = 6 + len(frames) * 16
    offset = directory_size
    entries = []
    for size, frame in zip(WINDOWS_SIZES, frames, strict=True):
        entries.append(
            struct.pack(
                "<BBBBHHII",
                0 if size == 256 else size,
                0 if size == 256 else size,
                0,
                0,
                1,
                32,
                len(frame),
                offset,
            )
        )
        offset += len(frame)

    with (RESOURCES / "icon.ico").open("wb") as output:
        output.write(struct.pack("<HHH", 0, 1, len(frames)))
        output.writelines(entries)
        output.writelines(frames)


def build_icns() -> None:
    with tempfile.TemporaryDirectory(
        prefix=".gts-icon-",
        dir=RESOURCES,
    ) as temporary:
        iconset = Path(temporary) / "icon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            render_icon(base, "mac").save(iconset / f"icon_{base}x{base}.png")
            render_icon(base * 2, "mac").save(
                iconset / f"icon_{base}x{base}@2x.png"
            )
        subprocess.run(
            [
                "iconutil",
                "-c",
                "icns",
                str(iconset),
                "-o",
                str(RESOURCES / "icon.icns"),
            ],
            check=True,
        )


def main() -> None:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    render_icon(1024, "mac").save(RESOURCES / "icon.png")
    render_icon(1024, "windows").save(RESOURCES / "icon-windows.png")
    build_ico()
    build_icns()


if __name__ == "__main__":
    main()
