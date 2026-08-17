"""Grounded per-object captioning with BLIP.

Replaces the GPT-Neo-1.3B path (F6), which had four compounding defects:

1. A **base** (non-instruction-tuned) LM was prompted ``"Define {label} in the
   real world."``  Base LMs continue text; they do not answer.
2. The prompt was never stripped, so saved output literally began
   ``"Define apple in the real world.\\n\\n"``.
3. ``max_length=100`` counts prompt **+** completion, so output truncated
   mid-sentence and a period was blindly appended -> ``"according to
   Merriam-Webster,."``.
4. The description **never saw the image**, which is how a photo of apples
   produced ``"This website is run by The Frugal Fawn."``

Here the caption is conditioned on the actual object crop.

Scope this honestly
-------------------
Grounding removes the *image-blind* generation path.  It does **not** make the
system hallucination-free: a VLM can still assert visually unsupported details -
invented colors, counts, materials or context - especially on small, blurry or
occluded crops.  The correct claim is *grounded captioning with a substantially
reduced ungrounded-generation surface*, never "verified" or "factual".
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class DescriptionModel:
    """BLIP image captioner over object crops."""

    def __init__(self, model, processor, device: torch.device) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    def caption(self, crop: Image.Image | np.ndarray) -> str:
        """Caption a single crop.

        Unconditional: no text prompt is supplied, so the caption is independent
        of the DETR label.  That independence is what makes the cross-model
        agreement check in ``utils.compose.captions_agree`` meaningful - do not
        condition on the label to "help" the model.
        """
        image = Image.fromarray(crop) if isinstance(crop, np.ndarray) else crop
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        # Move every input tensor to the device, not just the model.
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                # max_new_tokens, NEVER max_length: max_length counts the prompt
                # and was root cause #3 of F6's mid-sentence truncation.
                max_new_tokens=settings.blip_max_new_tokens,
                num_beams=settings.blip_num_beams,
                # Deterministic: this is the precondition that makes result
                # caching semantically safe (P1.5). Do not reintroduce sampling
                # without removing the cache.
                do_sample=False,
            )

        text = self._processor.decode(output[0], skip_special_tokens=True)
        return text.strip()

    def caption_batch(self, crops: list[Image.Image | np.ndarray]) -> list[str]:
        """Caption crops one at a time.

        Deliberately not batched: batching adds padding/attention-mask
        complexity and a real correctness risk for a gain that is unmeasured on
        a CPU-only build.  Revisit only if recorded p95 latency justifies it.
        """
        return [self.caption(crop) for crop in crops]
