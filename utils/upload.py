"""Upload validation and normalization.

Fixes F4 (empty read on rerun), F8 (unsafe path, no caps) and F17 (image mode /
EXIF orientation).

The uploaded image is **never written to disk to run inference**.  It is already
in memory as bytes; decode, validate and pass it straight down the pipeline.
The original code wrote it out only to read it back, which is where F8 came from.
Persisting the source happens only under ``PERSIST_ARTIFACTS`` (utils/artifacts).

Two identifiers, deliberately different values
----------------------------------------------
``content_hash``  SHA-256 of the image bytes.  **Internal cache key only** - never
                  logged, never rendered, never used in a path.
``run_id``        Fresh UUID per execution, minted by the orchestration layer.
                  The only public/observability identifier.

Reusing the content fingerprint as the correlation id would publish a stable
identifier for the image itself, letting anyone with log access link the same
picture across sessions and confirm whether a specific image was ever uploaded.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import settings

# Real image formats we accept, verified from decoded content - not from the
# filename extension, and not from the uploader's `type=` argument (a
# client-side hint only).
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


class UploadValidationError(ValueError):
    """Upload rejected for a reason worth showing the user."""


@dataclass(frozen=True)
class ValidatedImage:
    """A decoded, normalized, size-checked image plus its cache key."""

    image: Image.Image          # RGB, EXIF-normalized
    array: np.ndarray           # HxWx3 uint8
    content_hash: str           # cache key ONLY - never log this
    source_format: str
    width: int
    height: int
    size_bytes: int


def _reject(message: str) -> None:
    raise UploadValidationError(message)


def load_validated_image(uploaded_file) -> ValidatedImage:
    """Validate and normalize a Streamlit ``UploadedFile``.

    Uses ``.getvalue()``, never ``.read()``: ``UploadedFile`` is a ``BytesIO``
    subclass that **persists across reruns within a session**, and ``.read()``
    advances the stream position without resetting it.  On the second rerun with
    the same file ``.read()`` returns ``b""`` - the original code then wrote a
    0-byte file and failed with a generic error (F4).
    """
    data: bytes = uploaded_file.getvalue()
    return load_validated_bytes(data)


def load_validated_bytes(data: bytes) -> ValidatedImage:
    """Core validation path, decoupled from Streamlit for testability."""
    if not data:
        _reject("The uploaded file is empty.")

    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / (1024 * 1024)
        actual_mb = len(data) / (1024 * 1024)
        _reject(f"File is {actual_mb:.1f} MB; the limit is {limit_mb:.0f} MB.")

    # --- Step 1: verify the container is a real image of an allowed type -----
    try:
        with Image.open(io.BytesIO(data)) as probe:
            source_format = (probe.format or "").upper()
            declared_size = probe.size  # header only; pixels not decoded yet
            probe.verify()              # consumes the handle, hence the reopen
    except UnidentifiedImageError:
        _reject("That file is not a readable image.")
    except Image.DecompressionBombError:
        _reject("Image dimensions are implausibly large; refusing to decode it.")
    except Exception as exc:  # corrupt/truncated payloads
        _reject(f"The image could not be read: {exc}")

    if source_format not in ALLOWED_FORMATS:
        _reject(
            f"Unsupported image format {source_format or 'unknown'!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_FORMATS))}."
        )

    # --- Step 2: decompression-bomb guard, BEFORE materializing pixels -------
    # Deliberately does NOT assign Image.MAX_IMAGE_PIXELS: that is a
    # process-global shared by every session using the cached models, so
    # per-request mutation is a race that leaks across sessions.  Pillow is
    # lazy, so the header dimensions above are available without decoding.
    declared_pixels = declared_size[0] * declared_size[1]
    if declared_pixels > settings.max_pixels:
        _reject(
            f"Image is {declared_pixels / 1e6:.1f} megapixels; "
            f"the limit is {settings.max_pixels / 1e6:.0f} MP."
        )

    # --- Step 3: decode, normalizing orientation and mode -------------------
    try:
        with warnings.catch_warnings():
            # Pillow raises DecompressionBombWarning as a *warning*; promote it
            # so it lands in the same clean domain error as the hard failure.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                # EXIF-rotated phone photos: the browser shows them rotated
                # while PIL decodes them unrotated, so boxes would appear
                # misplaced relative to what the user sees.
                oriented = ImageOps.exif_transpose(opened)
                # Guarantees a 3-channel uint8 array downstream: grayscale gives
                # a 2-D array and RGBA/CMYK give 4 channels otherwise.
                image = oriented.convert("RGB")
    except Image.DecompressionBombWarning:
        _reject("Image dimensions are implausibly large; refusing to decode it.")
    except Image.DecompressionBombError:
        _reject("Image dimensions are implausibly large; refusing to decode it.")
    except Exception as exc:
        _reject(f"The image could not be decoded: {exc}")

    width, height = image.size
    if width < 2 or height < 2:
        _reject(f"Image is too small to analyze ({width}x{height} pixels).")

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        _reject("Image could not be normalized to 3-channel RGB.")

    return ValidatedImage(
        image=image,
        array=array,
        content_hash=hashlib.sha256(data).hexdigest(),
        source_format=source_format,
        width=width,
        height=height,
        size_bytes=len(data),
    )
