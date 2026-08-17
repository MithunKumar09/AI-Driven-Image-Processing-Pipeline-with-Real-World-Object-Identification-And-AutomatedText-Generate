"""Tier 1: upload validation, the F4 rerun bug, and image normalization."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from conftest import FakeUploadedFile, make_image_bytes
from config import settings
from utils.upload import UploadValidationError, load_validated_bytes, load_validated_image


# ---------------------------------------------------------------------------
# F4 - the rerun bug
# ---------------------------------------------------------------------------
def test_repeated_load_succeeds_across_reruns(rgb_png):
    """UploadedFile persists across reruns; .read() would return b"" the 2nd time.

    This is the direct regression test for F4: the original code used
    `.read()`, wrote a 0-byte file on the second rerun, and failed with a
    generic "An error occurred".
    """
    upload = FakeUploadedFile(rgb_png)

    first = load_validated_image(upload)
    second = load_validated_image(upload)
    third = load_validated_image(upload)

    assert first.content_hash == second.content_hash == third.content_hash
    assert first.width == second.width == third.width


def test_read_would_have_broken_it(rgb_png):
    """Demonstrates why .getvalue() is required, not a stylistic preference."""
    upload = FakeUploadedFile(rgb_png)
    assert upload.read() == rgb_png
    assert upload.read() == b""      # position never resets - the F4 mechanism
    assert upload.getvalue() == rgb_png


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_empty_upload_rejected():
    with pytest.raises(UploadValidationError, match="empty"):
        load_validated_bytes(b"")


def test_oversized_upload_rejected():
    payload = b"\x89PNG\r\n\x1a\n" + b"0" * (settings.max_upload_bytes + 1)
    with pytest.raises(UploadValidationError, match="limit"):
        load_validated_bytes(payload)


def test_renamed_text_file_rejected():
    """Extension is not evidence; content is."""
    with pytest.raises(UploadValidationError, match="not a readable image"):
        load_validated_bytes(b"this is definitely not an image, despite the .png name")


def test_unsupported_format_rejected():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="BMP")
    with pytest.raises(UploadValidationError, match="Unsupported image format"):
        load_validated_bytes(buffer.getvalue())


def test_tiny_image_rejected():
    with pytest.raises(UploadValidationError, match="too small"):
        load_validated_bytes(make_image_bytes(size=(1, 1)))


def test_decompression_bomb_rejected_from_header():
    """A small file declaring huge dimensions must be refused before decoding."""
    buffer = io.BytesIO()
    # 20000x20000 = 400 MP > the 50 MP default, but compresses to a tiny PNG.
    Image.new("L", (20000, 20000), color=0).save(buffer, format="PNG")
    payload = buffer.getvalue()

    with pytest.raises(UploadValidationError, match="megapixels|implausibly large"):
        load_validated_bytes(payload)


def test_bomb_guard_does_not_mutate_global_pil_state():
    """Image.MAX_IMAGE_PIXELS is process-global and shared across sessions.

    Mutating it per request would be a race that leaks between users.
    """
    before = Image.MAX_IMAGE_PIXELS
    buffer = io.BytesIO()
    Image.new("L", (20000, 20000), color=0).save(buffer, format="PNG")
    with pytest.raises(UploadValidationError):
        load_validated_bytes(buffer.getvalue())
    assert Image.MAX_IMAGE_PIXELS == before


# ---------------------------------------------------------------------------
# F17 - mode and orientation normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,fmt", [("L", "PNG"), ("RGBA", "PNG"), ("RGB", "JPEG")])
def test_all_modes_normalize_to_rgb(mode, fmt):
    validated = load_validated_bytes(make_image_bytes(mode=mode, fmt=fmt))
    assert validated.array.ndim == 3
    assert validated.array.shape[2] == 3
    assert validated.image.mode == "RGB"


def test_exif_orientation_applied():
    """An EXIF-rotated photo must be processed the way the browser shows it."""
    image = Image.new("RGB", (80, 40), color=(10, 20, 30))
    buffer = io.BytesIO()
    exif = image.getexif()
    exif[274] = 6  # Orientation: rotate 90 CW
    image.save(buffer, format="JPEG", exif=exif)

    validated = load_validated_bytes(buffer.getvalue())
    # exif_transpose swaps the axes for orientation 6.
    assert (validated.width, validated.height) == (40, 80)


def test_content_hash_is_stable_and_distinct():
    a = make_image_bytes(color=(1, 2, 3))
    b = make_image_bytes(color=(9, 9, 9))
    assert load_validated_bytes(a).content_hash == load_validated_bytes(a).content_hash
    assert load_validated_bytes(a).content_hash != load_validated_bytes(b).content_hash
