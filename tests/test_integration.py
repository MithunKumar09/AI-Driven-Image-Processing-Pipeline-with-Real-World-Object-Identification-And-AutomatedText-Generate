"""Tier 2: integration tests that load real models.

Run with `pytest -m slow`. First run downloads ~1.2 GB of weights.
"""

from __future__ import annotations

import json

import pytest

from config import settings
from utils.data_mapping import serialize_payload
from utils.upload import load_validated_bytes

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def apples_result(apples_bytes):
    from pipeline import run_inference_uncached

    validated = load_validated_bytes(apples_bytes)
    return validated, run_inference_uncached(validated)


# ---------------------------------------------------------------------------
# THE headline regression gate (F1)
# ---------------------------------------------------------------------------
def test_apples_are_labeled_apple(apples_result):
    """Two_Apples.jpg must yield >=1 high-confidence "apple" and zero "hot dog".

    Against the pre-fix code this FAILS: DETR's sparse 91-space category id 53
    (apple) was indexed into a contiguous 81-entry list, landing on "hot dog" -
    which is exactly what data/output/729c67a9-..._data_mapping.json recorded at
    0.99 confidence.

    Deliberately >= 1, not == 2. The filename and the two saved crops describe
    the OLD broken run, produced under both the mislabeling bug and the manual
    coordinate round-trip that P0.1 removed; detection count is also
    threshold-dependent. Never gate on a count without measurement behind it.
    """
    _, result = apples_result
    labels = [d.label for d in result.detections]

    assert "hot dog" not in labels, f"the F1 mislabeling is back: {labels}"

    apples = [d for d in result.detections if d.label == "apple" and d.score > 0.9]
    assert len(apples) >= 1, f"expected >=1 high-confidence apple, got {labels}"


def test_labels_come_from_model_config(apples_result):
    """No hardcoded COCO mapping survives; id2label is the source of truth."""
    from models.registry import get_detection_model

    id2label = get_detection_model().id2label
    assert len(id2label) == 91, "DETR uses the sparse 91-entry COCO id space"
    assert id2label[53] == "apple"

    _, result = apples_result
    known = set(id2label.values())
    for detection in result.detections:
        assert detection.label in known


def test_boxes_are_in_original_image_space(apples_result):
    """P0.1: one canonical coordinate system, no scale_factor round-trip."""
    validated, result = apples_result
    for detection in result.detections:
        x1, y1, x2, y2 = detection.box
        assert 0 <= x1 < x2 <= validated.width
        assert 0 <= y1 < y2 <= validated.height


def test_no_scale_factor_in_detection_path():
    """P0.1 removed the manual resize round-trip.

    Checks executable identifiers via AST, not raw source: the module docstring
    legitimately mentions ``scale_factor`` while explaining that it was removed.
    """
    import ast
    from pathlib import Path

    import models.detection_model as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "scale_factor" not in identifiers
    assert "resize" not in identifiers, "DetrImageProcessor owns DETR resizing"


# ---------------------------------------------------------------------------
# Grounded captioning (F6)
# ---------------------------------------------------------------------------
def test_captions_are_grounded_and_clean(apples_result):
    _, result = apples_result
    described = [o for o in result.payload["objects"] if o["caption"]]
    assert described, "expected at least one captioned object"

    for obj in described:
        description = obj["description"]
        # F6 defect 2: the prompt used to be echoed into the output verbatim.
        assert "Define" not in description
        assert "in the real world" not in description
        # F6 defect 3: max_length truncation then a blindly appended period.
        assert not description.endswith(",.")
        assert description.endswith(".")
        assert len(description) < 300
        assert obj["label"].split()[0].lower() in description.lower()


def test_caption_mentions_apple(apples_result):
    _, result = apples_result
    captions = " ".join(o["caption"] or "" for o in result.payload["objects"]).lower()
    assert "apple" in captions


# ---------------------------------------------------------------------------
# Determinism - the P1.5 cache precondition
# ---------------------------------------------------------------------------
def test_determinism_same_process(apples_bytes):
    """Same environment, same process: byte-identical descriptions.

    This is only meaningful in-process. Pinned revisions give model identity and
    version reproducibility, NOT cross-machine numerical determinism - identical
    weights still run through different BLAS/oneDNN kernels and thread counts.
    """
    from pipeline import run_inference_uncached

    validated = load_validated_bytes(apples_bytes)
    first = run_inference_uncached(validated)
    second = run_inference_uncached(validated)

    assert serialize_payload(first.payload, canonical=True) == serialize_payload(
        second.payload, canonical=True
    )


# ---------------------------------------------------------------------------
# End-to-end - returned values, not files on disk
# ---------------------------------------------------------------------------
def test_end_to_end_returns_bytes_and_writes_nothing(apples_result, tmp_path):
    """PERSIST_ARTIFACTS is off by default, so nothing should touch the disk."""
    from utils.artifacts import build_export_bundle

    validated, result = apples_result
    assert not settings.persist_artifacts

    bundle = build_export_bundle(
        payload_json=serialize_payload(result.payload),
        annotated=result.annotated,
        crops=list(result.crops),
    )

    assert bundle.annotated_png.startswith(b"\x89PNG")
    payload = json.loads(bundle.payload_json)
    assert payload["schema_version"]
    assert payload["object_count"] == len(result.detections)
    assert settings.detr_revision in payload["models"]["detection"]
    assert settings.blip_revision in payload["models"]["captioning"]

    # No run directory should have been created by the default path.
    assert not (settings.runs_dir).exists() or not any(settings.runs_dir.iterdir())


def test_input_tensors_land_on_the_resolved_device():
    """Model .to(device) is not enough; inputs must move too (P0.2)."""
    import torch
    from PIL import Image

    from models.registry import get_detection_model, get_device

    device = get_device()
    detector = get_detection_model()

    captured = {}
    original = detector._model.forward

    def spy(**kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                captured[key] = value.device
        return original(**kwargs)

    detector._model.forward = spy
    try:
        detector.detect(Image.new("RGB", (64, 64), (30, 30, 30)), threshold=0.9)
    finally:
        detector._model.forward = original

    assert captured, "no tensors were passed to the model"
    for name, tensor_device in captured.items():
        assert tensor_device.type == device.type, f"{name} on {tensor_device}"


def test_zero_detection_path_is_valid(monkeypatch):
    """A clean image with no COCO objects must be a valid result, not a crash."""
    from PIL import Image

    from pipeline import run_inference_uncached

    import io

    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), (245, 245, 245)).save(buffer, format="PNG")
    validated = load_validated_bytes(buffer.getvalue())

    result = run_inference_uncached(validated, detection_threshold=0.99)

    assert result.payload["object_count"] == 0
    assert result.payload["objects"] == []
    json.loads(serialize_payload(result.payload))  # must still serialize
