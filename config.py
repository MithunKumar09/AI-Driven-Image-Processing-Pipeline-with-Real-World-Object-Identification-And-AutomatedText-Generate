"""Typed, environment-driven configuration for the image-processing pipeline.

Validated at import so a bad setting fails fast and loudly rather than surfacing
as a confusing runtime error deep inside inference.

Deliberately a stdlib dataclass rather than pydantic: pydantic is not installed,
and explicit validation is the proportionate choice at this size.

Note on what is NOT here: there is no ``MAX_IMAGE_DIM`` for the detection path.
``DetrImageProcessor`` owns DETR resizing (see ``models/detection_model.py``), so
such a setting would be a knob that changes nothing.  ``OCR_MAX_DIM`` exists
because the OCR path genuinely does its own downscaling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
# config.py lives at the REPOSITORY ROOT, so the repo is `.parent`.
# `parents[1]` would resolve one level *above* the repo and scatter artifacts
# into the parent directory.  Only files nested one level down (e.g. utils/x.py)
# would use `parents[1]`.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

_ROOT_MARKER = "requirements.txt"


class ConfigError(RuntimeError):
    """Raised at import time when configuration is invalid."""


def _require_valid_root() -> None:
    """Fail loudly if this file is ever moved out of the repository root."""
    if not (PROJECT_ROOT / _ROOT_MARKER).is_file():
        raise ConfigError(
            f"PROJECT_ROOT={PROJECT_ROOT} does not contain {_ROOT_MARKER!r}. "
            "config.py must live at the repository root; if it moved, update "
            "PROJECT_ROOT (a nested module would need Path(__file__).parents[1])."
        )


# ---------------------------------------------------------------------------
# Env parsing helpers (each validates and reports the offending variable)
# ---------------------------------------------------------------------------
def _env_str(name: str, default: str, *, choices: tuple[str, ...] | None = None) -> str:
    raw = os.environ.get(name, default).strip()
    if choices is not None and raw.upper() not in choices:
        raise ConfigError(f"{name}={raw!r} must be one of {choices}")
    return raw


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name}={raw!r} is not a valid boolean")


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name}={value} out of range [{lo}, {hi}]")
    return value


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name}={value} out of range [{lo}, {hi}]")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration."""

    # --- identity ---------------------------------------------------------
    # Bump PIPELINE_VERSION by hand whenever composition/association logic
    # changes in a way the other cache-identity fields would not capture.
    pipeline_version: str

    # --- models (id + pinned revision; see models/registry.py) -------------
    detr_model_id: str
    detr_revision: str
    blip_model_id: str
    blip_revision: str

    # --- detection --------------------------------------------------------
    detection_threshold: float

    # --- OCR --------------------------------------------------------------
    ocr_max_dim: int
    ocr_min_confidence: float
    min_overlap_ratio: float

    # --- captioning caps --------------------------------------------------
    # MIN_CROP_PIXELS: BOTH crop width and height must be >= this value.
    # One knob, one meaning - there is deliberately no separate MIN_CROP_AREA.
    min_crop_pixels: int
    max_descriptions_per_image: int
    blip_max_new_tokens: int
    blip_num_beams: int

    # --- concurrency ------------------------------------------------------
    max_concurrent_inferences: int

    # --- uploads ----------------------------------------------------------
    max_upload_bytes: int
    max_pixels: int

    # --- demo guardrails --------------------------------------------------
    max_runs_per_hour: int

    # --- artifacts --------------------------------------------------------
    persist_artifacts: bool
    artifact_ttl_hours: int
    max_stored_runs: int

    # --- observability ----------------------------------------------------
    log_level: str

    # --- paths ------------------------------------------------------------
    project_root: Path = field(default=PROJECT_ROOT)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def runs_dir(self) -> Path:
        """Per-run artifact directories (only used when persist_artifacts)."""
        return self.data_dir / "runs"

    @property
    def detr_ref(self) -> str:
        return f"{self.detr_model_id}@{self.detr_revision}"

    @property
    def blip_ref(self) -> str:
        return f"{self.blip_model_id}@{self.blip_revision}"


def _load() -> Settings:
    _require_valid_root()

    settings = Settings(
        pipeline_version=_env_str("PIPELINE_VERSION", "1.0.0"),
        # Pinned to tested commit SHAs, not just model ids: a Hub repo can be
        # updated in place under the same name, which would change weights,
        # `id2label` or processor defaults with no local diff - and would make
        # cached results outlive the weights that produced them.
        detr_model_id=_env_str("DETR_MODEL_ID", "facebook/detr-resnet-50"),
        detr_revision=_env_str(
            "DETR_REVISION", "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
        ),
        blip_model_id=_env_str("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base"),
        blip_revision=_env_str(
            "BLIP_REVISION", "82a37760796d32b1411fe092ab5d4e227313294b"
        ),
        detection_threshold=_env_float("DETECTION_THRESHOLD", 0.5, lo=0.0, hi=1.0),
        ocr_max_dim=_env_int("OCR_MAX_DIM", 1600, lo=256, hi=8192),
        ocr_min_confidence=_env_float("OCR_MIN_CONFIDENCE", 0.30, lo=0.0, hi=1.0),
        min_overlap_ratio=_env_float("MIN_OVERLAP_RATIO", 0.5, lo=0.0, hi=1.0),
        min_crop_pixels=_env_int("MIN_CROP_PIXELS", 32, lo=1, hi=1024),
        max_descriptions_per_image=_env_int("MAX_DESCRIPTIONS_PER_IMAGE", 10, lo=1, hi=100),
        blip_max_new_tokens=_env_int("BLIP_MAX_NEW_TOKENS", 30, lo=5, hi=200),
        blip_num_beams=_env_int("BLIP_NUM_BEAMS", 3, lo=1, hi=10),
        max_concurrent_inferences=_env_int("MAX_CONCURRENT_INFERENCES", 1, lo=1, hi=16),
        max_upload_bytes=_env_int("MAX_UPLOAD_MB", 10, lo=1, hi=200) * 1024 * 1024,
        max_pixels=_env_int("MAX_PIXELS_MP", 50, lo=1, hi=500) * 1_000_000,
        max_runs_per_hour=_env_int("MAX_RUNS_PER_HOUR", 10, lo=1, hi=10_000),
        persist_artifacts=_env_bool("PERSIST_ARTIFACTS", False),
        artifact_ttl_hours=_env_int("ARTIFACT_TTL_HOURS", 24, lo=1, hi=24 * 365),
        max_stored_runs=_env_int("MAX_STORED_RUNS", 50, lo=1, hi=10_000),
        log_level=_env_str(
            "LOG_LEVEL",
            "INFO",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        ).upper(),
    )

    # Cross-field validation.
    if settings.blip_num_beams > 1 and settings.blip_max_new_tokens < 5:
        raise ConfigError("BLIP_MAX_NEW_TOKENS too small for beam search")

    return settings


settings: Settings = _load()
