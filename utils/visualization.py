"""Annotated-image rendering.

Takes label **strings** directly.  The old signature took label *indices* plus a
``coco_labels`` list, which forced callers into a name -> index -> name
round-trip through the very hardcoded list that caused F1.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Box = tuple[int, int, int, int]

_BOX_COLOR = (220, 38, 38)
_TEXT_COLOR = (255, 255, 255)


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def visualize_detections(
    image: Image.Image | np.ndarray,
    boxes: Sequence[Box],
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
) -> Image.Image:
    """Draw boxes and labels onto a copy of the image.

    Boxes are in original-image pixels (see utils/compose coordinate contract).
    Returns a new image; the input is never mutated.
    """
    base = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    canvas = base.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _load_font()
    width, height = canvas.size

    for index, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = (int(v) for v in box)
        draw.rectangle([x1, y1, x2, y2], outline=_BOX_COLOR, width=3)

        caption = label
        if scores is not None and index < len(scores):
            caption = f"{label} {scores[index]:.0%}"

        left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = right - left, bottom - top
        pad = 3

        # Prefer above the box; drop inside it when there is no room.
        text_x = min(max(0, x1), max(0, width - text_w - 2 * pad))
        text_y = y1 - text_h - 2 * pad
        if text_y < 0:
            text_y = min(y1 + pad, max(0, height - text_h - 2 * pad))

        draw.rectangle(
            [text_x, text_y, text_x + text_w + 2 * pad, text_y + text_h + 2 * pad],
            fill=_BOX_COLOR,
        )
        draw.text((text_x + pad - left, text_y + pad - top), caption, fill=_TEXT_COLOR, font=font)

    return canvas


def add_provenance_mark(image: Image.Image, text: str) -> Image.Image:
    """Composite a subtle provenance label onto a copy of ``image``.

    Applied to EXPORTED images only - never to the arrays or crops fed to a
    model, which would corrupt inference inputs.

    Wording note: the source photograph is user-provided, not synthetic; only
    the annotations and captions are model output.  Callers pass an
    "AI-annotated" style string, never a bare "AI-generated" claim about the
    image itself.
    """
    canvas = image.convert("RGB").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(14)

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    pad = 6
    x = max(0, canvas.width - text_w - 3 * pad)
    y = max(0, canvas.height - text_h - 3 * pad)

    draw.rectangle(
        [x, y, x + text_w + 2 * pad, y + text_h + 2 * pad], fill=(0, 0, 0, 110)
    )
    draw.text((x + pad - left, y + pad - top), text, fill=(255, 255, 255, 210), font=font)

    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def crops_to_images(crops: Iterable[np.ndarray]) -> list[Image.Image]:
    return [Image.fromarray(c) for c in crops]
