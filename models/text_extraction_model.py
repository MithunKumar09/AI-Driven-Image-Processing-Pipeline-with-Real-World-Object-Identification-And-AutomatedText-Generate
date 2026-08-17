"""OCR via EasyOCR, returning structured spans in original-image coordinates.

Fixes F18.  The previous version ran on the full-resolution original (which
dominates latency on large photos) and used only ``result[1]`` - discarding the
geometry and confidence, so recovered text could never be tied to the object it
sat on.  A grounding signal that was already being paid for was thrown away.

Coordinate contract: OCR downscaling is this module's own concern.  Spans are
converted back to **original-image pixels at this wrapper's boundary**, so no
consumer ever needs to know the OCR scale ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from config import settings
from utils.compose import Box
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OcrSpan:
    """One recognized text span, in ORIGINAL-image pixel coordinates."""

    text: str
    confidence: float
    box: Box


class TextExtractionModel:
    def __init__(self, *, use_gpu: bool = False) -> None:
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    def extract(self, image: Image.Image | np.ndarray) -> list[OcrSpan]:
        """Recognize text, returning spans in original-image coordinates."""
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        if pil.mode != "RGB":
            pil = pil.convert("RGB")

        original_w, original_h = pil.size

        # Downscale for speed; remember the ratio so we can invert it here.
        max_dim = max(original_w, original_h)
        scale = min(1.0, settings.ocr_max_dim / max_dim) if max_dim else 1.0
        if scale < 1.0:
            work = pil.resize(
                (max(1, int(original_w * scale)), max(1, int(original_h * scale))),
                Image.LANCZOS,
            )
        else:
            work = pil

        results = self.reader.readtext(np.asarray(work))

        spans: list[OcrSpan] = []
        for raw_box, text, confidence in results:
            cleaned = (text or "").strip()
            if not cleaned or float(confidence) < settings.ocr_min_confidence:
                continue

            box = self._to_original_coords(raw_box, scale, original_w, original_h)
            if box is None:
                continue
            spans.append(
                OcrSpan(text=cleaned, confidence=round(float(confidence), 4), box=box)
            )

        # Stable, intrinsic ordering: reading order, then text.
        spans.sort(key=lambda s: (s.box[1], s.box[0], s.text))
        return spans

    @staticmethod
    def _to_original_coords(
        raw_box, scale: float, width: int, height: int
    ) -> Box | None:
        """Convert EasyOCR's 4-point polygon to an axis-aligned original-space box."""
        try:
            xs = [float(point[0]) / scale for point in raw_box]
            ys = [float(point[1]) / scale for point in raw_box]
        except (TypeError, IndexError, ZeroDivisionError):
            return None

        x1 = max(0, min(int(round(min(xs))), width))
        y1 = max(0, min(int(round(min(ys))), height))
        x2 = max(0, min(int(round(max(xs))), width))
        y2 = max(0, min(int(round(max(ys))), height))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


def document_text(spans: list[OcrSpan]) -> list[str]:
    """Flatten spans to plain strings, preserving their stable order."""
    return [span.text for span in spans]
