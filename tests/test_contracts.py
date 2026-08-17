"""Tier 1: config, serialization, cache identity, logging hygiene, rate limiting.

These encode the invariants that the plan's corrections were about - the ones
that would silently rot without a test.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

import pipeline
from config import PROJECT_ROOT, settings
from utils import rate_limit
from utils.compose import build_object_record, captions_agree, compose_description
from utils.data_mapping import build_payload, serialize_payload


@dataclass(frozen=True)
class FakeDetection:
    label: str
    score: float
    box: tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_project_root_is_the_repository():
    """`.parent`, not `parents[1]` - config.py sits AT the repo root."""
    assert (PROJECT_ROOT / "requirements.txt").is_file()
    assert (PROJECT_ROOT / "config.py").is_file()


def test_no_hardcoded_coco_mapping_survives():
    """F1's root cause was a hardcoded label list. Labels now come from
    ``model.config.id2label`` only - no list may be reintroduced.

    Detected structurally: a COCO list necessarily contains these adjacent
    class names, and that is what a copy-paste reintroduction would look like.
    """
    import ast

    signatures = [("hot dog", "pizza"), ("teddy bear", "hair drier")]
    offenders = []

    for path in PROJECT_ROOT.rglob("*.py"):
        if "newenv" in path.parts or "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Only literal string collections count; prose in docstrings does not.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                continue
            values = {
                e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
            if any(all(name in values for name in pair) for pair in signatures):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, f"hardcoded COCO label list reintroduced: {offenders}"


def test_no_dead_detr_resize_knob():
    """DetrImageProcessor owns DETR resizing, so a MAX_IMAGE_DIM would do nothing."""
    assert not hasattr(settings, "max_image_dim")


def test_no_separate_min_crop_area_knob():
    assert not hasattr(settings, "min_crop_area")
    assert hasattr(settings, "min_crop_pixels")


def test_model_revisions_are_pinned_shas():
    for revision in (settings.detr_revision, settings.blip_revision):
        assert len(revision) == 40, f"{revision!r} is not a commit SHA"
        int(revision, 16)  # raises if not hex


@pytest.mark.parametrize(
    "variable,value,message",
    [
        ("DETECTION_THRESHOLD", "5.0", "out of range"),
        ("DETECTION_THRESHOLD", "banana", "not a number"),
        ("MAX_UPLOAD_MB", "0", "out of range"),
        ("MIN_CROP_PIXELS", "-1", "out of range"),
        ("LOG_LEVEL", "CHATTY", "must be one of"),
        ("PERSIST_ARTIFACTS", "maybe", "not a valid boolean"),
    ],
)
def test_invalid_setting_fails_fast(monkeypatch, variable, value, message):
    """Exercises the real import-time path via _load().

    Deliberately not importlib.reload(): reload redefines ConfigError, so a
    captured class reference no longer matches the raised instance, and a failed
    reload leaves a half-initialized module behind for every later test.
    """
    import config

    monkeypatch.setenv(variable, value)
    with pytest.raises(config.ConfigError, match=message):
        config._load()


# ---------------------------------------------------------------------------
# Cache identity  (P1.5)
# ---------------------------------------------------------------------------
def test_cache_identity_tracks_threshold():
    assert pipeline.cache_identity(0.5) != pipeline.cache_identity(0.6)


def test_cache_identity_includes_model_revisions():
    identity = pipeline.cache_identity(0.5)
    assert settings.detr_ref in identity
    assert settings.blip_ref in identity
    # ids alone are not identity - the revision must be in the ref string
    assert settings.detr_revision in settings.detr_ref


def test_cache_identity_includes_pipeline_version_and_generation_config():
    identity = pipeline.cache_identity(0.5)
    for expected in (
        settings.pipeline_version,
        settings.ocr_max_dim,
        settings.ocr_min_confidence,
        settings.min_overlap_ratio,
        settings.min_crop_pixels,
        settings.max_descriptions_per_image,
        settings.blip_max_new_tokens,
        settings.blip_num_beams,
    ):
        assert expected in identity


def test_run_id_never_enters_the_cache():
    """run_id is execution identity; caching it would replay a stale id.

    A cache hit must never reuse another execution's correlation id.
    """
    params = inspect.signature(pipeline._run_inference_cached).parameters
    assert "run_id" not in params
    assert not any("run_id" in name for name in params)

    fields = pipeline.PipelineResult.__dataclass_fields__
    assert "run_id" not in fields
    assert "compute_id" not in fields

    source = inspect.getsource(pipeline._execute)
    assert "run_id" not in source


def test_payload_carries_no_request_identity():
    payload = build_payload(
        objects=[],
        document_text=[],
        image_meta={"width": 10, "height": 10},
        model_refs={"detection": settings.detr_ref},
        pipeline_version=settings.pipeline_version,
        detection_threshold=0.5,
    )
    serialized = serialize_payload(payload).decode()
    for forbidden in ("run_id", "compute_id", "content_hash"):
        assert forbidden not in serialized


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def test_payload_is_json_serializable_with_numpy_and_torch():
    import numpy as np
    import torch

    payload = build_payload(
        objects=[{"score": np.float32(0.87), "box": torch.tensor([1, 2, 3, 4])}],
        document_text=["hello"],
        image_meta={"width": np.int64(640), "height": np.int64(480)},
        model_refs={"detection": settings.detr_ref, "captioning": settings.blip_ref},
        pipeline_version=settings.pipeline_version,
        detection_threshold=0.5,
    )
    parsed = json.loads(serialize_payload(payload))
    assert parsed["objects"][0]["box"] == [1, 2, 3, 4]
    assert isinstance(parsed["objects"][0]["score"], float)
    assert parsed["image"]["width"] == 640


def test_payload_records_model_provenance():
    payload = build_payload(
        objects=[],
        document_text=[],
        image_meta={},
        model_refs={"detection": settings.detr_ref, "captioning": settings.blip_ref},
        pipeline_version=settings.pipeline_version,
        detection_threshold=0.5,
    )
    assert "@" in payload["models"]["detection"]
    assert settings.detr_revision in payload["models"]["detection"]


def test_canonical_serialization_is_stable_for_snapshots():
    payload = build_payload(
        objects=[{"b": 1, "a": 2}],
        document_text=[],
        image_meta={},
        model_refs={},
        pipeline_version="1.0.0",
        detection_threshold=0.5,
    )
    assert serialize_payload(payload, canonical=True) == serialize_payload(
        payload, canonical=True
    )


# ---------------------------------------------------------------------------
# Description composition
# ---------------------------------------------------------------------------
def test_description_has_no_prompt_echo_and_terminal_punctuation():
    text = compose_description(
        label="apple", score=0.99, caption="a red apple on a wooden table"
    )
    assert "Define" not in text                 # F6 defect 2
    assert "real world" not in text
    assert text.endswith(".")
    assert not text.endswith(",.")              # F6 defect 3
    assert "99%" in text
    assert len(text) < 300


def test_description_includes_region_text():
    text = compose_description(
        label="apple", score=0.9, caption="an apple", region_text=["Fresh"]
    )
    assert "Fresh" in text


def test_skipped_crop_still_reports_measured_fields():
    record = build_object_record(
        detection=FakeDetection("apple", 0.91, (0, 0, 10, 10)),
        caption=None,
        region_text=[],
        skipped_reason="crop smaller than 32px",
    )
    assert record["label"] == "apple"
    assert record["caption"] is None
    assert "not generated" in record["description"]
    assert record["caption_mentions_label"] is None


def test_agreement_flag_is_a_heuristic():
    assert captions_agree("apple", "a red apple on a table") is True
    assert captions_agree("apple", "a wooden surface") is False
    assert captions_agree("apple", "") is False


# ---------------------------------------------------------------------------
# Logging hygiene
# ---------------------------------------------------------------------------
def test_content_hash_never_logged(caplog):
    from utils.logging_setup import get_logger, stage

    logger = get_logger("hygiene_test")
    logger.propagate = True
    with caplog.at_level(logging.INFO):
        with stage(logger, "detect", compute_id="abc123", content_hash="SECRETHASH"):
            pass

    output = "\n".join(
        [r.getMessage() + str(r.__dict__) for r in caplog.records]
    )
    assert "SECRETHASH" not in output


def test_run_id_and_compute_id_are_distinct():
    from utils.logging_setup import new_compute_id, new_run_id

    assert new_run_id() != new_run_id()
    assert new_run_id() != new_compute_id()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def test_rate_limit_blocks_after_budget():
    state: dict = {}
    for _ in range(3):
        assert rate_limit.check(state, max_runs_per_hour=3).allowed
        rate_limit.record_run(state)

    blocked = rate_limit.check(state, max_runs_per_hour=3)
    assert not blocked.allowed
    assert blocked.retry_after_seconds > 0


def test_rate_limit_window_rolls_off():
    state: dict = {}
    rate_limit.record_run(state, now=1000.0)
    # 2 hours later the old run has aged out of the 1-hour window.
    assert rate_limit.check(state, max_runs_per_hour=1, now=1000.0 + 7200).allowed
