"""The inference pipeline: upload -> detect -> OCR -> caption -> payload.

Cache identity
--------------
``run_inference`` is wrapped in ``st.cache_data``.  A stale hit is worse than no
cache - it silently serves results from a different pipeline - so the key covers
everything that can change the output, assembled once in :func:`cache_identity`
so no call site can forget a component.

This is semantically safe **only because** generation is deterministic
(``do_sample=False``, beam search - see ``models/description_model``) and OCR
association ranks on intrinsic fields only (``utils/compose.associate_spans``).
Do not reintroduce sampling without removing the cache.

run_id vs compute_id
--------------------
``run_id`` is *execution* identity; ``cache_identity()`` is *inference* identity.
Conflating them poisons the cache: a cached payload carrying the ``run_id`` of
whichever execution populated it would hand every later cache hit a stale
correlation id belonging to a different user's request.

So ``run_inference`` never accepts, returns or stores ``run_id``.  Its own stage
logs use an ephemeral ``compute_id`` minted inside the cached call, which never
escapes.  Request-level logging happens in the orchestration layer (app.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st
from PIL import Image

from config import settings
from models import registry
from models.detection_model import Detection, crop
from models.text_extraction_model import OcrSpan, document_text
from utils import compose
from utils.data_mapping import build_payload
from utils.logging_setup import get_logger, new_compute_id, stage
from utils.upload import ValidatedImage
from utils.visualization import visualize_detections

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Reusable inference results. Deliberately free of request identity."""

    payload: dict[str, Any]
    detections: tuple[Detection, ...]
    spans: tuple[OcrSpan, ...]
    annotated: Image.Image
    crops: tuple[tuple[str, Image.Image], ...]


def cache_identity(detection_threshold: float) -> tuple:
    """Everything that can change inference output, in one place.

    Model *ids* alone are not identity - a Hub repo can be updated in place
    under the same name - so pinned revisions are included too.

    ``detection_threshold`` is a parameter rather than a read of global settings
    because it is user-adjustable per request: mutating the process-global
    ``settings`` would let concurrent sessions corrupt each other's threshold.
    """
    return (
        settings.pipeline_version,
        settings.detr_ref,
        settings.blip_ref,
        detection_threshold,
        settings.ocr_max_dim,
        settings.ocr_min_confidence,
        settings.min_overlap_ratio,
        settings.min_crop_pixels,
        settings.max_descriptions_per_image,
        settings.blip_max_new_tokens,
        settings.blip_num_beams,
        False,  # do_sample - pinned; flipping it must invalidate the cache
    )


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def _run_inference_cached(
    content_hash: str,
    detection_threshold: float,
    identity: tuple,
    _image_bytes: bytes,
):
    """Cached core. ``content_hash`` + ``detection_threshold`` + ``identity``
    are the cache key.

    ``_image_bytes`` is underscore-prefixed so Streamlit does not hash it: the
    content hash already identifies it, and hashing megabytes per call is waste.

    Note there is no ``run_id`` parameter and none in the return value.
    """
    from utils.upload import load_validated_bytes

    validated = load_validated_bytes(_image_bytes)
    return _execute(validated, detection_threshold=detection_threshold)


def _execute(validated: ValidatedImage, *, detection_threshold: float) -> PipelineResult:
    """Run the real work. Stage logs here use compute_id (cache misses only)."""
    compute_id = new_compute_id()

    detector = registry.get_detection_model()
    captioner = registry.get_description_model()
    reader = registry.get_text_extraction_model()

    # Hold the inference slot around MODEL CALLS ONLY. Composition, cropping and
    # all rendering stay outside it so one slow render cannot stall other
    # sessions.
    with registry.inference_slot():
        with stage(logger, "detect", compute_id=compute_id) as rec:
            detections = detector.detect(validated.image, threshold=detection_threshold)
            rec["detections"] = len(detections)

        with stage(logger, "ocr", compute_id=compute_id) as rec:
            spans = reader.extract(validated.image)
            rec["spans"] = len(spans)

        selected, skipped = compose.select_for_captioning(
            detections,
            min_crop_pixels=settings.min_crop_pixels,
            max_descriptions=settings.max_descriptions_per_image,
        )

        with stage(logger, "describe", compute_id=compute_id) as rec:
            captions: dict[int, str] = {}
            for index in sorted(selected):
                region = compose.pad_box(
                    detections[index].box, 4, validated.width, validated.height
                )
                captions[index] = captioner.caption(crop(validated.array, region))
            rec["captioned"] = len(captions)
            rec["skipped"] = len(skipped)

    # --- outside the inference slot from here -------------------------------
    attached, loose = compose.associate_spans(
        spans, detections, min_overlap_ratio=settings.min_overlap_ratio
    )

    objects = []
    crops: list[tuple[str, Image.Image]] = []
    for index, detection in enumerate(detections):
        region_text = [s.text for s in attached.get(index, [])]
        objects.append(
            compose.build_object_record(
                detection=detection,
                caption=captions.get(index),
                region_text=region_text,
                skipped_reason=skipped.get(index),
            )
        )
        crops.append(
            (
                f"{detection.label}_{index}.png",
                Image.fromarray(crop(validated.array, detection.box)),
            )
        )

    annotated = visualize_detections(
        validated.image,
        [d.box for d in detections],
        [d.label for d in detections],
        [d.score for d in detections],
    )

    payload = build_payload(
        objects=objects,
        document_text=document_text(list(loose)) if loose else [],
        image_meta={
            "width": validated.width,
            "height": validated.height,
            "format": validated.source_format,
            "size_bytes": validated.size_bytes,
        },
        model_refs={"detection": settings.detr_ref, "captioning": settings.blip_ref},
        pipeline_version=settings.pipeline_version,
        detection_threshold=detection_threshold,
    )

    return PipelineResult(
        payload=payload,
        detections=tuple(detections),
        spans=tuple(spans),
        annotated=annotated,
        crops=tuple(crops),
    )


def run_inference(
    validated: ValidatedImage,
    image_bytes: bytes,
    *,
    detection_threshold: float | None = None,
) -> PipelineResult:
    """Public entry point. Caching is transparent to the caller."""
    threshold = (
        settings.detection_threshold if detection_threshold is None else detection_threshold
    )
    return _run_inference_cached(
        validated.content_hash, threshold, cache_identity(threshold), image_bytes
    )


def run_inference_uncached(
    validated: ValidatedImage, *, detection_threshold: float | None = None
) -> PipelineResult:
    """Bypass the cache - used by tests that assert on determinism."""
    threshold = (
        settings.detection_threshold if detection_threshold is None else detection_threshold
    )
    return _execute(validated, detection_threshold=threshold)
