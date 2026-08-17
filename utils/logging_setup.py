"""Structured logging with two deliberately separate correlation scopes.

The pipeline's inference function is wrapped in ``st.cache_data``.  That creates
a hard constraint on correlation ids:

===============  ==========================================  ================  ==================
Scope            Stages                                      Correlation field  Runs on
===============  ==========================================  ================  ==================
**Request**      upload, render, export, errors, cache_hit   ``run_id``         every execution
**Compute**      detect, ocr, describe                       ``compute_id``     cache *misses* only
===============  ==========================================  ================  ==================

``run_id`` is minted per user-triggered execution in the orchestration layer and
must never be passed into, returned from, or stored inside the cached function -
otherwise a cache hit would replay a correlation id belonging to a *different*
user's request.  ``compute_id`` is minted *inside* the cached execution, used
only for its own log lines, and never escapes it.

``content_hash`` (the cache key) is never logged in either scope: it is a stable
fingerprint of the user's image, and emitting it would let anyone with log access
link the same picture across sessions.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from config import settings

_CONFIGURED = False

# Fields that must never appear in a log record, whatever the caller passes.
_FORBIDDEN_FIELDS = frozenset({"content_hash", "sha256", "image_hash"})


class _KeyValueFormatter(logging.Formatter):
    """``ts level logger msg key=value ...`` - greppable without a parser."""

    _RESERVED = frozenset(
        vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} "
            f"{record.levelname:<7} {record.name} {record.getMessage()}"
        )
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._RESERVED and not key.startswith("_")
        }
        if extras:
            base += " " + " ".join(f"{k}={_render(v)}" for k, v in sorted(extras.items()))
        return base


def _render(value: Any) -> str:
    text = "null" if value is None else str(value)
    return f'"{text}"' if " " in text else text


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_KeyValueFormatter())

    root = logging.getLogger("pipeline")
    root.setLevel(settings.log_level)
    root.handlers = [handler]
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``pipeline`` root."""
    _configure_once()
    short = name.split(".")[-1]
    return logging.getLogger(f"pipeline.{short}")


def new_run_id() -> str:
    """Mint a request-scope correlation id (one per user-triggered execution)."""
    return uuid.uuid4().hex[:12]


def new_compute_id() -> str:
    """Mint a compute-scope correlation id, *inside* the cached execution only.

    Never return this from the cached function or store it in its payload.
    """
    return uuid.uuid4().hex[:12]


def _scrub(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if k not in _FORBIDDEN_FIELDS}


@contextmanager
def stage(
    logger: logging.Logger,
    name: str,
    *,
    run_id: str | None = None,
    compute_id: str | None = None,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Time a pipeline stage and emit one record with ``duration_ms``.

    Yields a mutable dict; anything placed in it (counts, sizes) is merged into
    the final record.  Pass exactly one of ``run_id`` / ``compute_id`` according
    to which side of the cache boundary the stage sits on.

    These durations are the source for the p50/p95 figures in the acceptance
    plan, so every stage should be wrapped.
    """
    extra: dict[str, Any] = {}
    correlation = {}
    if run_id is not None:
        correlation["run_id"] = run_id
    if compute_id is not None:
        correlation["compute_id"] = compute_id

    started = time.perf_counter()
    try:
        yield extra
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.exception(
            f"stage.{name}.failed",
            extra=_scrub({**correlation, **fields, **extra, "duration_ms": duration_ms}),
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            f"stage.{name}",
            extra=_scrub({**correlation, **fields, **extra, "duration_ms": duration_ms}),
        )
