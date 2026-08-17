"""Tier 1: export bundling, watermark non-destructiveness, retention policy."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image

from utils.artifacts import build_export_bundle, image_to_png_bytes
from utils.visualization import add_provenance_mark, visualize_detections


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.fromarray(
        np.linspace(0, 255, 64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)
    )


# ---------------------------------------------------------------------------
# Watermark must never touch inference inputs
# ---------------------------------------------------------------------------
def test_provenance_mark_is_non_destructive(sample_image):
    """Watermarking an inference input would corrupt results.

    The mark is composited onto the EXPORTED image only; the source object must
    be byte-identical afterwards.
    """
    before = image_to_png_bytes(sample_image)
    marked = add_provenance_mark(sample_image, "AI-annotated · test")
    after = image_to_png_bytes(sample_image)

    assert before == after, "the source image was mutated"
    assert marked is not sample_image
    assert image_to_png_bytes(marked) != before, "the mark had no visible effect"


def test_crops_are_untouched_by_export(sample_image):
    """Crops fed to BLIP must be byte-identical before and after export."""
    crop = sample_image.crop((0, 0, 32, 24))
    before = image_to_png_bytes(crop)

    bundle = build_export_bundle(
        payload_json=b"{}",
        annotated=add_provenance_mark(sample_image, "AI-annotated · test"),
        crops=[("crop_0.png", crop)],
    )

    assert image_to_png_bytes(crop) == before
    assert bundle.crop_pngs[0][1] == before


def test_provenance_wording_does_not_claim_the_photo_is_synthetic():
    """The source photograph is user-provided; only annotations are generated.

    Labeling the whole export "AI-generated" would misdescribe someone's own
    photo as synthetic media - the opposite of what provenance marking is for.

    Reads the source rather than importing it: importing app.py would execute
    the entire Streamlit script.
    """
    from config import PROJECT_ROOT

    source = (PROJECT_ROOT / "streamlit_app" / "app.py").read_text(encoding="utf-8")
    assert "AI-annotated" in source
    assert "AI-generated ·" not in source
    assert '"AI-generated"' not in source


def test_visualization_does_not_mutate_input(sample_image):
    before = image_to_png_bytes(sample_image)
    visualize_detections(sample_image, [(1, 1, 20, 20)], ["apple"], [0.9])
    assert image_to_png_bytes(sample_image) == before


def test_visualize_accepts_label_strings_not_indices(sample_image):
    """The old signature forced a name->index->name round-trip through the very
    hardcoded list that caused F1."""
    import inspect

    params = inspect.signature(visualize_detections).parameters
    assert "coco_labels" not in params
    out = visualize_detections(sample_image, [(0, 0, 30, 30)], ["apple"], [0.99])
    assert isinstance(out, Image.Image)


# ---------------------------------------------------------------------------
# Persistence adapter
# ---------------------------------------------------------------------------
def test_persist_is_a_noop_when_disabled(sample_image, artifacts_settings):
    from utils import artifacts

    settings = artifacts_settings()  # persist_artifacts stays False (the default)
    assert not settings.persist_artifacts

    bundle = build_export_bundle(
        payload_json=b"{}", annotated=sample_image, crops=[("a.png", sample_image)]
    )
    assert artifacts.persist_bundle(bundle, run_id="abc123") is None
    assert not settings.runs_dir.exists(), "disabled path must create nothing"


@pytest.fixture
def artifacts_settings(tmp_path, monkeypatch):
    """Swap in a replaced Settings for the artifacts module.

    `Settings` is a frozen dataclass, so it cannot be mutated in place - that
    immutability is deliberate. Use dataclasses.replace and rebind the name that
    utils.artifacts imported. Pointing project_root at tmp_path relocates
    runs_dir with it, since runs_dir is derived.
    """
    import dataclasses

    import config
    from utils import artifacts

    def apply(**overrides):
        replaced = dataclasses.replace(
            config.settings, project_root=tmp_path, **overrides
        )
        monkeypatch.setattr(artifacts, "settings", replaced)
        return replaced

    return apply


def test_persist_writes_isolated_run_dir(sample_image, artifacts_settings):
    from utils import artifacts

    settings = artifacts_settings(persist_artifacts=True)

    bundle = build_export_bundle(
        payload_json=b'{"ok":true}',
        annotated=sample_image,
        crops=[("apple_0.png", sample_image)],
    )
    run_dir = artifacts.persist_bundle(bundle, run_id="run-aaa")

    assert run_dir is not None and run_dir.is_dir()
    assert (run_dir / "result.json").read_bytes() == b'{"ok":true}'
    assert (run_dir / "annotated.png").is_file()
    assert (run_dir / "crops" / "apple_0.png").is_file()


def test_retention_evicts_old_runs_but_never_the_current_one(artifacts_settings):
    import os

    from utils import artifacts

    settings = artifacts_settings(artifact_ttl_hours=1, max_stored_runs=100)
    runs = settings.runs_dir
    runs.mkdir(parents=True)

    stale_time = time.time() - 7200  # 2 hours old, past the 1h TTL
    for name in ("old-a", "old-b", "current"):
        directory = runs / name
        directory.mkdir()
        os.utime(directory, (stale_time, stale_time))

    removed = artifacts.enforce_retention(current_run_id="current")

    assert (runs / "current").is_dir(), "the in-flight run must never be evicted"
    assert {p.name for p in removed} == {"old-a", "old-b"}


def test_retention_respects_grace_window(artifacts_settings):
    """A run still being written must not be swept out from under itself."""
    from utils import artifacts

    settings = artifacts_settings(artifact_ttl_hours=1, max_stored_runs=1)
    runs = settings.runs_dir
    runs.mkdir(parents=True)

    (runs / "just-written").mkdir()  # mtime = now
    removed = artifacts.enforce_retention(current_run_id="other")

    assert removed == []
    assert (runs / "just-written").is_dir()


def test_persist_sanitizes_run_id_paths(sample_image, artifacts_settings):
    """run_id must never be able to escape the runs directory."""
    from utils import artifacts

    settings = artifacts_settings(persist_artifacts=True)

    bundle = build_export_bundle(payload_json=b"{}", annotated=sample_image, crops=[])
    run_dir = artifacts.persist_bundle(bundle, run_id="../../escape")

    assert run_dir is not None
    assert run_dir.parent == settings.runs_dir
    assert ".." not in run_dir.name
    assert settings.runs_dir.resolve() in run_dir.resolve().parents
