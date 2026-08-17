"""Centralized model lifecycle.

The original app constructed every model at **module scope**.  Streamlit
re-executes the script top-to-bottom on every widget interaction, so MaskRCNN +
CLIP + GPT-Neo-1.3B (loaded twice) + EasyOCR + DETR were rebuilt on each rerun.

``st.cache_resource`` makes each loader a singleton.  Two consequences follow
that must be designed for rather than discovered:

1. **These objects are process-global.**  ``st.cache_resource`` hands the *same*
   instance to every browser session in the process.  Concurrent sessions would
   otherwise contend for the same CPU and RAM, and neither EasyOCR's ``Reader``
   nor the HF processors document thread safety.  ``inference_slot()`` serializes
   the model-call section.
2. **Cached resources are read-only after load.**  Never mutate a cached model at
   request time (no per-request ``.to()``, no config edits).

Device discipline: moving the *model* to a device is not enough - every input
tensor batch must be moved too (see ``DetectionModel.detect``).  On the current
CPU-only torch build that is a no-op, which is exactly why it would silently rot
without an explicit test.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

import streamlit as st
import torch

from config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class InferenceBusyError(RuntimeError):
    """Raised when an inference slot could not be acquired in time."""


# One process-wide semaphore guarding the model-call section.  Deliberately NOT
# a Streamlit-cached object: it must exist once per process, independent of
# cache eviction.
_INFERENCE_SEMAPHORE = threading.Semaphore(settings.max_concurrent_inferences)

_ACQUIRE_TIMEOUT_SECONDS = 120


@contextmanager
def inference_slot(timeout: float = _ACQUIRE_TIMEOUT_SECONDS) -> Iterator[None]:
    """Serialize model calls across all sessions in this process.

    Hold this around **model calls only**.  Image decode, validation,
    composition and all Streamlit rendering must stay outside it, or one slow
    render stalls every other session.
    """
    acquired = _INFERENCE_SEMAPHORE.acquire(timeout=timeout)
    if not acquired:
        raise InferenceBusyError(
            "The server is busy running another analysis. Please try again in a moment."
        )
    try:
        yield
    finally:
        # try/finally so an exception inside inference cannot leak the permit.
        _INFERENCE_SEMAPHORE.release()


@st.cache_resource(show_spinner=False)
def get_device() -> torch.device:
    """Resolve the compute device once per process.

    The installed torch is a CPU-only build (``2.4.1+cpu``), so the CUDA branch
    will not activate until torch is reinstalled from the CUDA wheel index.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device.resolved", extra={"device": str(device), "torch": torch.__version__})
    return device


@st.cache_resource(show_spinner="Loading object detector…")
def get_detection_model():
    """Load DETR once per process, pinned to a tested revision."""
    from transformers import DetrForObjectDetection, DetrImageProcessor

    from models.detection_model import DetectionModel

    device = get_device()
    logger.info("model.loading", extra={"model": settings.detr_ref})

    processor = DetrImageProcessor.from_pretrained(
        settings.detr_model_id, revision=settings.detr_revision
    )
    model = DetrForObjectDetection.from_pretrained(
        settings.detr_model_id, revision=settings.detr_revision
    )
    model.to(device)
    model.eval()

    logger.info(
        "model.loaded",
        extra={"model": settings.detr_ref, "num_labels": len(model.config.id2label)},
    )
    return DetectionModel(model=model, processor=processor, device=device)


@st.cache_resource(show_spinner="Loading image captioner…")
def get_description_model():
    """Load BLIP once per process, pinned to a tested revision."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    from models.description_model import DescriptionModel

    device = get_device()
    logger.info("model.loading", extra={"model": settings.blip_ref})

    processor = BlipProcessor.from_pretrained(
        settings.blip_model_id, revision=settings.blip_revision
    )
    model = BlipForConditionalGeneration.from_pretrained(
        settings.blip_model_id, revision=settings.blip_revision
    )
    model.to(device)
    model.eval()

    logger.info("model.loaded", extra={"model": settings.blip_ref})
    return DescriptionModel(model=model, processor=processor, device=device)


@st.cache_resource(show_spinner="Loading text reader…")
def get_text_extraction_model():
    """Load EasyOCR once per process."""
    from models.text_extraction_model import TextExtractionModel

    device = get_device()
    logger.info("model.loading", extra={"model": "easyocr:en"})
    model = TextExtractionModel(use_gpu=device.type == "cuda")
    logger.info("model.loaded", extra={"model": "easyocr:en"})
    return model
