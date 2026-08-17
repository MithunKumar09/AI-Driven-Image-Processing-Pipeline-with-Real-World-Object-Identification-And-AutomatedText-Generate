"""Shared fixtures. Puts the repository root on sys.path for `models.` imports."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_IMAGE = PROJECT_ROOT / "data" / "input_images" / "Two_Apples.jpg"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def apples_bytes() -> bytes:
    if not FIXTURE_IMAGE.is_file():
        pytest.skip(f"regression fixture missing: {FIXTURE_IMAGE}")
    return FIXTURE_IMAGE.read_bytes()


def make_image_bytes(
    size: tuple[int, int] = (64, 48),
    mode: str = "RGB",
    fmt: str = "PNG",
    color=(120, 40, 40),
) -> bytes:
    """Build a small in-memory image for validation tests."""
    if mode == "L":
        image = Image.new("L", size, color=128)
    elif mode == "RGBA":
        image = Image.new("RGBA", size, color=(*color, 255))
    else:
        image = Image.new(mode, size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def rgb_png() -> bytes:
    return make_image_bytes()


class FakeUploadedFile(io.BytesIO):
    """Mimics Streamlit's UploadedFile: a BytesIO whose position persists.

    This is the whole point of the F4 regression test - `.read()` advances and
    never resets, so a second call returns b"".
    """

    def __init__(self, data: bytes, name: str = "upload.png") -> None:
        super().__init__(data)
        self.name = name


@pytest.fixture
def fake_upload(rgb_png):
    return FakeUploadedFile(rgb_png)


def box_of(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    return (x1, y1, x2, y2)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
