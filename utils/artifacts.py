"""Export artifacts - in-memory by default, optional bounded persistence.

The JSON, annotated PNG and crops exist to be **downloaded**, not archived, so
the default path builds them as bytes and hands them to ``st.download_button``.
That removes unbounded growth, concurrent-write collisions and stale artifacts
in one move.

**This is the only module permitted to create directories or open files for
writing.**  Keeping that in one place is what makes the retention policy
testable and the default path provably file-free.
"""

from __future__ import annotations

import io
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExportBundle:
    """Everything the UI can offer for download, as bytes."""

    payload_json: bytes
    annotated_png: bytes
    crop_pngs: tuple[tuple[str, bytes], ...]  # (filename, data)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_export_bundle(
    *,
    payload_json: bytes,
    annotated: Image.Image,
    crops: Sequence[tuple[str, Image.Image]],
) -> ExportBundle:
    """Assemble downloadable bytes. Touches no filesystem."""
    return ExportBundle(
        payload_json=payload_json,
        annotated_png=image_to_png_bytes(annotated),
        crop_pngs=tuple((name, image_to_png_bytes(img)) for name, img in crops),
    )


# ---------------------------------------------------------------------------
# Optional persistence (PERSIST_ARTIFACTS, default off)
# ---------------------------------------------------------------------------
def _safe_component(name: str) -> str:
    """Make a filename component safe: no separators, no traversal."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return cleaned.strip("._") or "object"


def persist_bundle(bundle: ExportBundle, *, run_id: str) -> Path | None:
    """Write a bundle to an isolated per-run directory. No-op unless enabled.

    Keyed by ``run_id`` (never ``content_hash``) so concurrent sessions cannot
    collide and so the directory name leaks no fingerprint of the image.
    """
    if not settings.persist_artifacts:
        return None

    run_dir = settings.runs_dir / _safe_component(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "result.json").write_bytes(bundle.payload_json)
    (run_dir / "annotated.png").write_bytes(bundle.annotated_png)

    if bundle.crop_pngs:
        crops_dir = run_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        for name, data in bundle.crop_pngs:
            (crops_dir / _safe_component(name)).write_bytes(data)

    logger.info("artifacts.persisted", extra={"run_id": run_id, "dir": str(run_dir)})
    enforce_retention(current_run_id=run_id)
    return run_dir


def enforce_retention(*, current_run_id: str | None = None) -> list[Path]:
    """Evict old run directories. Returns what was removed.

    Bounded by whichever of ``ARTIFACT_TTL_HOURS`` / ``MAX_STORED_RUNS`` binds
    first.  Never deletes the in-flight run, and uses a grace window so a
    concurrent run that is still writing is not swept out from under itself.
    """
    runs_dir = settings.runs_dir
    if not runs_dir.is_dir():
        return []

    grace_seconds = 300
    now = time.time()
    ttl_seconds = settings.artifact_ttl_hours * 3600
    protected = _safe_component(current_run_id) if current_run_id else None

    candidates = [
        d for d in runs_dir.iterdir() if d.is_dir() and d.name != protected
    ]
    # Newest first, so the tail is the eviction candidate set.
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    removed: list[Path] = []
    for index, directory in enumerate(candidates):
        age = now - directory.stat().st_mtime
        if age < grace_seconds:
            continue  # possibly still being written by a concurrent run
        too_old = age > ttl_seconds
        too_many = index >= max(0, settings.max_stored_runs - 1)
        if too_old or too_many:
            shutil.rmtree(directory, ignore_errors=True)
            removed.append(directory)

    if removed:
        logger.info("artifacts.evicted", extra={"count": len(removed)})
    return removed
