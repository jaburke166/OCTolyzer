"""Generate the OCTolyzer placeholder app icon in every format the installers need.

This is a one-time (or occasional) developer utility, not a build step: the
generated files below are committed to the repository and consumed directly
by ``build/build_gui.py`` and ``gui/app.py``. Re-run this script only when the
source artwork changes.

Usage:
    python -m gui.assets.generate_icons

Requires Pillow (already a transitive dependency via requirements.txt).
Building ``icon.icns`` additionally requires macOS's bundled ``iconutil`` and
``sips`` tools; on other platforms that step is skipped with a warning and the
existing committed ``icon.icns`` is left untouched.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ASSETS_DIR = Path(__file__).resolve().parent
SOURCE_SIZE = 1024

# A simple, recognizable placeholder: a rounded-square badge evoking an OCT
# B-scan of an eye -- a pale "eye" lens shape with an amber pupil, crossed by
# thin horizontal "scan lines" -- on a dark teal-to-blue gradient background.
BACKGROUND_TOP = (12, 74, 110)      # deep teal
BACKGROUND_BOTTOM = (7, 41, 74)     # deep blue
EYE_FILL = (235, 244, 248)          # near-white
PUPIL_FILL = (223, 140, 44)         # amber
SCAN_LINE = (110, 200, 214, 130)    # translucent light cyan


def _rounded_square_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def _gradient_background(size: int) -> Image.Image:
    gradient = Image.new("RGB", (1, size), color=0)
    for y in range(size):
        t = y / max(size - 1, 1)
        colour = tuple(
            round(BACKGROUND_TOP[channel] + (BACKGROUND_BOTTOM[channel] - BACKGROUND_TOP[channel]) * t)
            for channel in range(3)
        )
        gradient.putpixel((0, y), colour)
    return gradient.resize((size, size))


def build_source_icon() -> Image.Image:
    size = SOURCE_SIZE
    background = _gradient_background(size).convert("RGBA")
    mask = _rounded_square_mask(size, radius=round(size * 0.22))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(background, (0, 0), mask)

    draw = ImageDraw.Draw(canvas, "RGBA")

    # Almond-shaped "eye": intersection of two circles.
    centre = size / 2
    eye_half_width = size * 0.34
    eye_radius = size * 0.30
    left_centre = (centre - eye_half_width * 0.55, centre)
    right_centre = (centre + eye_half_width * 0.55, centre)
    eye_mask = Image.new("L", (size, size), 0)
    eye_draw = ImageDraw.Draw(eye_mask)
    eye_draw.ellipse(
        [left_centre[0] - eye_radius, left_centre[1] - eye_radius,
         left_centre[0] + eye_radius, left_centre[1] + eye_radius],
        fill=255,
    )
    right_mask = Image.new("L", (size, size), 0)
    right_draw = ImageDraw.Draw(right_mask)
    right_draw.ellipse(
        [right_centre[0] - eye_radius, right_centre[1] - eye_radius,
         right_centre[0] + eye_radius, right_centre[1] + eye_radius],
        fill=255,
    )
    # Intersection of the two ellipse masks gives the almond/lens shape.
    eye_shape = ImageChops.logical_and(
        eye_mask.point(lambda p: 255 if p else 0).convert("1"),
        right_mask.point(lambda p: 255 if p else 0).convert("1"),
    ).convert("L")

    eye_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    eye_layer.paste(EYE_FILL, (0, 0, size, size), eye_shape)
    canvas.alpha_composite(eye_layer)

    # Pupil.
    pupil_radius = size * 0.10
    draw.ellipse(
        [centre - pupil_radius, centre - pupil_radius, centre + pupil_radius, centre + pupil_radius],
        fill=PUPIL_FILL,
    )

    # Thin horizontal "OCT scan line" accents across the badge.
    for offset in (-0.16, -0.02, 0.12):
        y = centre + size * offset
        draw.line([(size * 0.12, y), (size * 0.88, y)], fill=SCAN_LINE, width=max(2, round(size * 0.006)))

    return canvas


def save_png_variants(source: Image.Image) -> None:
    for name, edge in (("icon-256.png", 256), ("icon-512.png", 512)):
        resized = source.resize((edge, edge), Image.LANCZOS)
        resized.save(ASSETS_DIR / name)


def save_ico(source: Image.Image) -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    source.save(ASSETS_DIR / "icon.ico", sizes=[(size, size) for size in sizes])


def save_icns(source: Image.Image) -> None:
    if platform.system() != "Darwin" or shutil.which("iconutil") is None:
        print("Skipping icon.icns: iconutil is only available on macOS.")
        return
    with tempfile.TemporaryDirectory() as staging:
        iconset = Path(staging) / "icon.iconset"
        iconset.mkdir()
        # iconutil expects this exact filename set (base + @2x retina variants).
        specs = [
            ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
            ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
            ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
            ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
            ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
        ]
        for filename, edge in specs:
            source.resize((edge, edge), Image.LANCZOS).save(iconset / filename)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(ASSETS_DIR / "icon.icns")],
            check=True,
        )


def main() -> None:
    source = build_source_icon()
    source.save(ASSETS_DIR / "icon-source.png")
    save_png_variants(source)
    save_ico(source)
    save_icns(source)
    print(f"Icons written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
