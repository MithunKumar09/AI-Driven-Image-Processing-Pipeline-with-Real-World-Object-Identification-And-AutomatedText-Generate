"""Pure, side-effect-free composition helpers.

Everything here is deterministic and free of I/O, model calls and Streamlit
state, which is what makes it the highest-value unit-test surface in the repo.

Coordinate contract
-------------------
**Every box crossing a module boundary is in original-image pixels.**
``DetrImageProcessor`` owns DETR's own resizing (see ``models/detection_model``)
and the OCR wrapper converts its spans back to original coordinates at its own
boundary, so association here needs no live rescaling.

Determinism contract
--------------------
Association and caption selection rank on **intrinsic fields only** - never on
list position.  ``st.cache_data`` caches results, so an order-dependent result
would make cached and fresh runs disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from models.detection_model import Detection
    from models.text_extraction_model import OcrSpan

Box = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def clamp_box(box: Sequence[float], width: int, height: int) -> Box | None:
    """Clamp a box to image bounds, returning ``None`` if degenerate.

    Guards the defect in the original code, where ``int()`` of a negative
    coordinate became a *Python negative index* so ``image_np[-5:300]`` silently
    produced an empty or wrong-region crop.

    Returns ``(x1, y1, x2, y2)`` with ``0 <= x1 < x2 <= width`` and
    ``0 <= y1 < y2 <= height``, or ``None`` when the box has no area.
    """
    if len(box) != 4:
        return None

    x1, y1, x2, y2 = (float(v) for v in box)

    # Tolerate inverted boxes rather than silently producing an empty slice.
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    xi1 = max(0, min(int(round(x1)), width))
    yi1 = max(0, min(int(round(y1)), height))
    xi2 = max(0, min(int(round(x2)), width))
    yi2 = max(0, min(int(round(y2)), height))

    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return xi1, yi1, xi2, yi2


def pad_box(box: Box, padding: int, width: int, height: int) -> Box:
    """Grow a box by ``padding`` px on each side, staying inside the image."""
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a: Box, b: Box) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = min(ax2, bx2) - max(ax1, bx1)
    dy = min(ay2, by2) - max(ay1, by1)
    return dx * dy if dx > 0 and dy > 0 else 0


def is_captionable(box: Box, min_crop_pixels: int) -> bool:
    """BOTH width and height must be >= ``min_crop_pixels``.

    One knob, one meaning - there is deliberately no separate ``MIN_CROP_AREA``.
    A 12x9 crop yields a meaningless caption at full model cost.
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_crop_pixels and (y2 - y1) >= min_crop_pixels


# ---------------------------------------------------------------------------
# Caption budgeting
# ---------------------------------------------------------------------------
def select_for_captioning(
    detections: Sequence["Detection"],
    *,
    min_crop_pixels: int,
    max_descriptions: int,
) -> tuple[set[int], dict[int, str]]:
    """Choose which detections get a caption.

    BLIP is one generation per object, so cost is linear in detection count and
    unbounded in principle (a crowd scene can yield 30+ boxes).  Returns the set
    of selected indices plus a ``{index: reason}`` map for everything skipped.

    Selection ranks on intrinsic fields only, so it is invariant under any
    permutation of ``detections``:
    ``score DESC -> (x1, y1, x2, y2) -> label``.
    """
    skipped: dict[int, str] = {}
    eligible: list[int] = []

    for index, det in enumerate(detections):
        if not is_captionable(det.box, min_crop_pixels):
            skipped[index] = f"crop smaller than {min_crop_pixels}px"
        else:
            eligible.append(index)

    ranked = sorted(
        eligible,
        key=lambda i: (
            -detections[i].score,
            detections[i].box,
            detections[i].label,
        ),
    )

    selected = set(ranked[:max_descriptions])
    for index in ranked[max_descriptions:]:
        skipped[index] = "description skipped (cap reached)"
    return selected, skipped


# ---------------------------------------------------------------------------
# OCR <-> object association
# ---------------------------------------------------------------------------
def associate_spans(
    spans: Sequence["OcrSpan"],
    detections: Sequence["Detection"],
    *,
    min_overlap_ratio: float,
) -> tuple[dict[int, list["OcrSpan"]], list["OcrSpan"]]:
    """Attach each OCR span to at most one detection.

    Centroid containment alone is not sufficient: detections routinely overlap
    (a person holding a bottle; nested fruit), so one centroid can fall inside
    several boxes and be double-counted.  Instead, score
    ``intersection_area(span, box) / span_area`` and rank candidates by
    **intrinsic fields only**::

        overlap_ratio DESC -> box_area ASC -> score DESC -> (x1,y1,x2,y2) -> label

    Never use list position as a tie-break - it is the input order, which would
    contradict the shuffle-invariance this function guarantees.  If two
    detections tie on every intrinsic field they are duplicates of the same box,
    so either choice is by definition equivalent.

    Returns ``({detection_index: [spans]}, unassociated_spans)``.
    """
    attached: dict[int, list["OcrSpan"]] = {}
    loose: list["OcrSpan"] = []

    for span in spans:
        span_box = span.box
        area = box_area(span_box)
        if area <= 0:
            loose.append(span)
            continue

        candidates = []
        for index, det in enumerate(detections):
            ratio = intersection_area(span_box, det.box) / area
            if ratio >= min_overlap_ratio:
                candidates.append((index, det, ratio))

        if not candidates:
            loose.append(span)
            continue

        best = min(
            candidates,
            key=lambda c: (-c[2], box_area(c[1].box), -c[1].score, c[1].box, c[1].label),
        )
        attached.setdefault(best[0], []).append(span)

    return attached, loose


# ---------------------------------------------------------------------------
# Description composition
# ---------------------------------------------------------------------------
def _clean_caption(caption: str) -> str:
    text = " ".join(caption.split()).strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text.rstrip(" .")


def captions_agree(label: str, caption: str) -> bool:
    """Cross-model agreement heuristic - a DIAGNOSTIC, not a correctness check.

    BLIP captions the crop *unconditionally*, so its caption is independent of
    the DETR label; checking whether the caption mentions the label is therefore
    a free signal.  It flags likely-unreliable rows for the UI.  It does not
    verify anything, and must never be presented as verification.
    """
    if not caption:
        return False
    haystack = caption.lower()
    return any(part in haystack for part in label.lower().split() if len(part) > 2)


def compose_description(
    *,
    label: str,
    score: float,
    caption: str | None,
    region_text: Iterable[str] = (),
    skipped_reason: str | None = None,
) -> str:
    """Build the object description from measured fields plus the caption.

    Deterministic string assembly - no language model involved, so the summary
    itself has no hallucination surface.  The caption stays quoted and attributed
    so the UI can distinguish *measured* fields (label, confidence, box, OCR)
    from *generated* prose.
    """
    parts = [f"{label.capitalize()} — detected with {score:.0%} confidence."]

    if skipped_reason:
        parts.append(f"Visual description not generated: {skipped_reason}.")
    elif caption:
        parts.append(f'Visual description: "{_clean_caption(caption)}".')

    text = [t.strip() for t in region_text if t and t.strip()]
    if text:
        parts.append(f'Text found in this region: "{" ".join(text)}".')

    return " ".join(parts)


def build_object_record(
    *,
    detection: "Detection",
    caption: str | None,
    region_text: Sequence[str],
    skipped_reason: str | None,
) -> dict[str, Any]:
    """Serializable per-object record (plain Python types only)."""
    agreement = (
        None if (skipped_reason or not caption) else captions_agree(detection.label, caption)
    )
    return {
        "label": detection.label,
        "score": round(float(detection.score), 4),
        "box": list(detection.box),
        "caption": caption,
        "region_text": list(region_text),
        # Diagnostic heuristic only - see captions_agree().
        "caption_mentions_label": agreement,
        "description_skipped": skipped_reason,
        "description": compose_description(
            label=detection.label,
            score=detection.score,
            caption=caption,
            region_text=region_text,
            skipped_reason=skipped_reason,
        ),
    }
