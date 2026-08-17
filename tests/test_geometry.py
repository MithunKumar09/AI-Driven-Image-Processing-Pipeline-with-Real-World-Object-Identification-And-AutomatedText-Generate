"""Tier 1: box clamping, caption budgeting and OCR association."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from utils.compose import (
    associate_spans,
    box_area,
    clamp_box,
    intersection_area,
    is_captionable,
    pad_box,
    select_for_captioning,
)


@dataclass(frozen=True)
class FakeDetection:
    label: str
    score: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class FakeSpan:
    text: str
    confidence: float
    box: tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# clamp_box - guards the negative-index defect (F9)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "box,expected",
    [
        ((10, 10, 50, 40), (10, 10, 50, 40)),          # already valid
        ((-20, -5, 50, 40), (0, 0, 50, 40)),           # negative -> clamped, not wrapped
        ((10, 10, 999, 999), (10, 10, 100, 80)),       # beyond bounds
        ((50, 40, 10, 10), (10, 10, 50, 40)),          # inverted -> normalized
        ((10, 10, 10, 40), None),                      # zero width
        ((10, 10, 50, 10), None),                      # zero height
        ((-50, -50, -10, -10), None),                  # entirely off-image
        ((200, 200, 300, 300), None),                  # entirely past bounds
    ],
)
def test_clamp_box(box, expected):
    assert clamp_box(box, 100, 80) == expected


def test_clamp_box_never_yields_negative_indices():
    """A negative coord must never survive as a Python negative index."""
    result = clamp_box((-10, -10, 30, 30), 100, 80)
    assert result is not None
    assert all(v >= 0 for v in result)


def test_clamp_box_rejects_malformed():
    assert clamp_box((1, 2, 3), 100, 80) is None


def test_pad_box_stays_in_bounds():
    assert pad_box((0, 0, 10, 10), 5, 100, 80) == (0, 0, 15, 15)
    assert pad_box((95, 75, 100, 80), 5, 100, 80) == (90, 70, 100, 80)


def test_area_and_intersection():
    assert box_area((0, 0, 10, 5)) == 50
    assert intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25
    assert intersection_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0


# ---------------------------------------------------------------------------
# Caption budgeting - MIN_CROP_PIXELS applies to BOTH dimensions
# ---------------------------------------------------------------------------
def test_is_captionable_requires_both_dimensions():
    assert is_captionable((0, 0, 40, 40), 32)
    assert not is_captionable((0, 0, 200, 8), 32)   # wide but short
    assert not is_captionable((0, 0, 8, 200), 32)   # tall but narrow


def test_select_skips_small_crops_and_caps_count():
    detections = [
        FakeDetection("a", 0.9, (0, 0, 100, 100)),
        FakeDetection("b", 0.8, (0, 0, 10, 10)),      # too small
        FakeDetection("c", 0.7, (0, 0, 100, 100)),
        FakeDetection("d", 0.95, (0, 0, 100, 100)),
    ]
    selected, skipped = select_for_captioning(
        detections, min_crop_pixels=32, max_descriptions=2
    )
    # Highest confidence first: d (0.95), a (0.9)
    assert selected == {3, 0}
    assert "smaller" in skipped[1]
    assert "cap reached" in skipped[2]


def test_selection_is_order_invariant():
    """Selection must depend on intrinsic fields, never list position."""
    detections = [
        FakeDetection("a", 0.91, (0, 0, 100, 100)),
        FakeDetection("b", 0.85, (10, 10, 90, 90)),
        FakeDetection("c", 0.77, (20, 20, 80, 80)),
        FakeDetection("d", 0.60, (30, 30, 70, 70)),
    ]
    baseline = {
        detections[i] for i in select_for_captioning(
            detections, min_crop_pixels=32, max_descriptions=2
        )[0]
    }

    for seed in range(8):
        shuffled = detections[:]
        random.Random(seed).shuffle(shuffled)
        chosen = {
            shuffled[i] for i in select_for_captioning(
                shuffled, min_crop_pixels=32, max_descriptions=2
            )[0]
        }
        # Compare by VALUE, not by index - an index comparison would pass
        # trivially under shuffling and hide the bug this guards.
        assert chosen == baseline


# ---------------------------------------------------------------------------
# OCR association - overlap ranking with intrinsic tie-breaks
# ---------------------------------------------------------------------------
def test_span_assigned_to_tighter_of_nested_boxes():
    outer = FakeDetection("person", 0.9, (0, 0, 200, 200))
    inner = FakeDetection("bottle", 0.8, (40, 40, 120, 120))
    span = FakeSpan("Fresh", 0.9, (50, 50, 90, 70))

    attached, loose = associate_spans([span], [outer, inner], min_overlap_ratio=0.5)

    # Both fully contain the span (ratio 1.0), so box_area ASC picks the tighter.
    assert loose == []
    assert attached == {1: [span]}


def test_span_assigned_at_most_once():
    a = FakeDetection("a", 0.9, (0, 0, 100, 100))
    b = FakeDetection("b", 0.9, (0, 0, 100, 100))
    span = FakeSpan("x", 0.9, (10, 10, 20, 20))

    attached, loose = associate_spans([span], [a, b], min_overlap_ratio=0.5)

    assert loose == []
    assert sum(len(v) for v in attached.values()) == 1


def test_span_below_overlap_threshold_is_document_level():
    det = FakeDetection("a", 0.9, (0, 0, 20, 20))
    span = FakeSpan("far away", 0.9, (500, 500, 560, 520))

    attached, loose = associate_spans([span], [det], min_overlap_ratio=0.5)

    assert attached == {}
    assert loose == [span]


def test_association_is_order_invariant():
    detections = [
        FakeDetection("person", 0.90, (0, 0, 200, 200)),
        FakeDetection("bottle", 0.85, (40, 40, 120, 120)),
        FakeDetection("cup", 0.80, (45, 45, 110, 110)),
    ]
    spans = [
        FakeSpan("one", 0.9, (50, 50, 90, 70)),
        FakeSpan("two", 0.9, (150, 150, 190, 170)),
    ]

    def assign_by_value(dets):
        attached, loose = associate_spans(spans, dets, min_overlap_ratio=0.5)
        # Map span text -> the DETECTION VALUE it landed on, never an index.
        return (
            {s.text: dets[i] for i, group in attached.items() for s in group},
            {s.text for s in loose},
        )

    baseline = assign_by_value(detections)
    for seed in range(8):
        shuffled = detections[:]
        random.Random(seed).shuffle(shuffled)
        assert assign_by_value(shuffled) == baseline


def test_zero_area_span_is_loose():
    det = FakeDetection("a", 0.9, (0, 0, 100, 100))
    span = FakeSpan("x", 0.9, (10, 10, 10, 10))
    attached, loose = associate_spans([span], [det], min_overlap_ratio=0.5)
    assert attached == {}
    assert loose == [span]
