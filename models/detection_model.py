"""DETR object detection.

Fixes the repository's headline defect (F1) and the coordinate-ownership defect
(F19) that produced it.

F1 - label mapping
------------------
DETR predicts over the **91-entry** COCO *category-id* space, which contains
~11 ``N/A`` placeholder ids (12, 26, 29, 30, 45, 66, 68, 69, 71, 83, ...).  The
old code indexed those sparse ids into a hardcoded **81-entry contiguous** list,
so every class after the first gap was shifted.  Category id 53 is ``apple`` in
the real space but sits at ``hot dog`` in the contiguous list - which is exactly
what ``data/output/729c67a9-..._data_mapping.json`` recorded for a photo of two
apples, at 0.99 confidence.

The fix is to stop maintaining a label list at all and read
``model.config.id2label``, which is correct regardless of where the gaps fall.

F19 - coordinate ownership
--------------------------
The old code resized with ``cv2.resize`` to max-dim 800, handed the pre-resized
image to ``DetrImageProcessor`` (which then resized *again* internally), passed
the **scaled** size as ``target_sizes``, and finally divided every box by the
scale factor.  Boxes existed in three spaces with a manual round-trip between
them, so every downstream consumer had to guess which one it held.

Here the processor owns DETR's resizing and normalization end to end, and
``target_sizes`` is the **original** ``(height, width)``.  Post-processing then
returns boxes already in original-image pixels: one canonical space, no manual
round-trip, and no ``scale_factor`` anywhere in this path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from utils.compose import Box, clamp_box
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Detection:
    """One detected object, in original-image pixel coordinates."""

    label: str
    score: float
    box: Box


class DetectionModel:
    """Thin wrapper owning DETR inference and its coordinate contract."""

    def __init__(self, model, processor, device: torch.device) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    @property
    def id2label(self) -> dict[int, str]:
        """The single source of truth for class names."""
        return self._model.config.id2label

    def detect(self, image: Image.Image, *, threshold: float) -> list[Detection]:
        """Detect objects, returning boxes in original-image pixels.

        ``image`` must already be RGB and EXIF-normalized (see utils/upload.py).
        """
        width, height = image.size

        # The processor owns resize + normalization.  Do not pre-resize.
        inputs = self._processor(images=image, return_tensors="pt")
        # Move EVERY input tensor to the device, not just the model - otherwise
        # the CUDA path fails with a device mismatch the CPU build never shows.
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # target_sizes is the ORIGINAL (height, width): the processor maps its
        # internal resize back for us, so no manual scale factor is needed.
        target_sizes = torch.tensor([[height, width]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        id2label = self.id2label
        detections: list[Detection] = []
        skipped = 0

        scores = results["scores"].detach().cpu().tolist()
        labels = results["labels"].detach().cpu().tolist()
        boxes = results["boxes"].detach().cpu().tolist()

        for score, label_id, raw_box in zip(scores, labels, boxes):
            box = clamp_box(raw_box, width, height)
            if box is None:
                skipped += 1
                continue
            label = id2label.get(int(label_id), f"class_{int(label_id)}")
            detections.append(Detection(label=label, score=float(score), box=box))

        if skipped:
            logger.debug("detection.boxes_skipped", extra={"count": skipped})

        # Stable, intrinsic ordering so downstream consumers and the result
        # cache never depend on tensor iteration order.
        detections.sort(key=lambda d: (-d.score, d.box, d.label))
        return detections


def crop(image_np: np.ndarray, box: Box) -> np.ndarray:
    """Crop an already-clamped box from an HxWx3 array."""
    x1, y1, x2, y2 = box
    return image_np[y1:y2, x1:x2]
