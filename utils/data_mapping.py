"""Result payload construction and serialization - PURE.

Creates no directories and writes no files.  The previous signature
(``map_data_to_objects(data, output_file)``) inverted this: it took a
destination path and wrote to it, which forced disk persistence into the default
path.  Filesystem access now lives solely in ``utils/artifacts.py``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1.0"


def _to_builtin(value: Any) -> Any:
    """Convert numpy/torch scalars and arrays to plain Python types.

    Without this, ``json.dump`` raises ``TypeError: Object of type Tensor is not
    JSON serializable`` on any value that escaped ``.item()``.
    """
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(v) for v in value]
    # numpy / torch scalars and arrays expose these without importing either.
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        try:
            return _to_builtin(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _to_builtin(value.tolist())
        except (ValueError, TypeError):
            pass
    return str(value)


def build_payload(
    *,
    objects: Sequence[Mapping[str, Any]],
    document_text: Sequence[str],
    image_meta: Mapping[str, Any],
    model_refs: Mapping[str, str],
    pipeline_version: str,
    detection_threshold: float,
) -> dict[str, Any]:
    """Assemble the canonical result payload.

    ``model_refs`` carries ``model_id@revision`` for each model so any exported
    artifact is traceable to the exact weights that produced it.

    Deliberately excludes ``run_id``, ``compute_id``, timestamps and any other
    request-scoped metadata: this payload is returned by the cached inference
    function, and request identity must never be cached (see P1.5).
    """
    return _to_builtin(
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": pipeline_version,
            "models": dict(model_refs),
            "detection_threshold": detection_threshold,
            "image": dict(image_meta),
            "document_text": list(document_text),
            "object_count": len(objects),
            "objects": list(objects),
        }
    )


def serialize_payload(payload: Mapping[str, Any], *, canonical: bool = False) -> bytes:
    """Serialize to UTF-8 JSON bytes.

    ``canonical=True`` sorts keys for stable snapshot comparison in regression
    tests, where the P0 payload is diffed against the P1 payload.
    """
    return json.dumps(
        _to_builtin(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=canonical,
    ).encode("utf-8")


def read_payload(data: bytes) -> dict[str, Any]:
    """Parse a payload previously produced by :func:`serialize_payload`."""
    return json.loads(data.decode("utf-8"))
