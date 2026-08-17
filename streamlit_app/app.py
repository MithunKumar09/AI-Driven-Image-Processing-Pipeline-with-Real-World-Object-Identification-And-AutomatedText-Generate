"""Streamlit UI - thin orchestration only.

All inference logic lives in ``pipeline.py``; this file handles upload, layout,
request-level logging and downloads.

Streamlit version note: the pinned Streamlit is **1.38.0**, where ``st.image``
accepts ``use_column_width`` and does **not** accept ``use_container_width``
(verified via ``inspect.signature``).  ``st.dataframe`` does accept
``use_container_width`` on this version.  Do not "modernize" the ``st.image``
calls without also bumping Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repository root importable when run as `streamlit run streamlit_app/app.py`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from config import settings
from models.registry import InferenceBusyError
from pipeline import run_inference
from utils import rate_limit
from utils.artifacts import build_export_bundle, persist_bundle
from utils.data_mapping import serialize_payload
from utils.logging_setup import get_logger, new_run_id, stage
from utils.upload import UploadValidationError, load_validated_image
from utils.visualization import add_provenance_mark

logger = get_logger(__name__)

st.set_page_config(page_title="Image Analysis Pipeline", page_icon="🔍", layout="wide")

st.title("🔍 AI-Driven Image Analysis Pipeline")
st.caption(
    "Object detection (DETR) → grounded captioning (BLIP) → text recognition (EasyOCR). "
    "Descriptions are model-generated and may contain errors."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(settings.detection_threshold),
        step=0.05,
        help="Only objects scoring above this are reported.",
    )
    st.divider()
    st.caption(rate_limit.describe_limit(settings.max_runs_per_hour))
    st.caption(f"Detection: `{settings.detr_ref}`")
    st.caption(f"Captioning: `{settings.blip_ref}`")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "webp"], key="file_uploader_key"
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

# --- Request scope begins: fresh run_id for EVERY execution, cache hit or not.
run_id = new_run_id()

try:
    with stage(logger, "upload", run_id=run_id, name_len=len(uploaded_file.name)) as rec:
        validated = load_validated_image(uploaded_file)
        rec["width"] = validated.width
        rec["height"] = validated.height
        rec["format"] = validated.source_format
except UploadValidationError as exc:
    st.warning(f"⚠️ {exc}")
    logger.warning("upload.rejected", extra={"run_id": run_id, "reason": str(exc)})
    st.stop()

status = rate_limit.check(st.session_state, max_runs_per_hour=settings.max_runs_per_hour)
if not status.allowed:
    minutes = max(1, status.retry_after_seconds // 60)
    st.warning(f"⏳ Demo limit reached. Please try again in about {minutes} minute(s).")
    logger.info("ratelimit.blocked", extra={"run_id": run_id})
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Uploaded")
    st.image(validated.image, use_column_width=True)

try:
    with st.status("Analyzing image…", expanded=False) as status_box:
        st.write("Detecting objects, reading text, and generating descriptions…")
        result = run_inference(
            validated, uploaded_file.getvalue(), detection_threshold=threshold
        )
        status_box.update(label="Analysis complete", state="complete")
except InferenceBusyError as exc:
    st.warning(f"⏳ {exc}")
    logger.warning("inference.busy", extra={"run_id": run_id})
    st.stop()
except Exception as exc:  # unexpected: log the traceback, show an id, not a stack
    logger.exception("pipeline.failed", extra={"run_id": run_id})
    st.error(
        f"Something went wrong while analyzing this image. Reference: `{run_id}`\n\n"
        f"({type(exc).__name__})"
    )
    st.stop()

rate_limit.record_run(st.session_state)
logger.info(
    "request.completed",
    extra={
        "run_id": run_id,
        "objects": len(result.detections),
        "spans": len(result.spans),
    },
)

with right:
    st.subheader("Detected objects")
    st.image(result.annotated, use_column_width=True)

# --- Results -----------------------------------------------------------------
if not result.detections:
    st.info(
        "No COCO objects were detected above the confidence threshold. "
        "Try lowering the threshold in the sidebar."
    )
else:
    rows = [
        {
            "Object": obj["label"],
            "Confidence": f"{obj['score']:.0%}",
            "Description": obj["description"],
            "Box": str(obj["box"]),
        }
        for obj in result.payload["objects"]
    ]
    st.subheader(f"Summary ({len(rows)} object{'s' if len(rows) != 1 else ''})")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    flagged = [
        obj["label"]
        for obj in result.payload["objects"]
        if obj.get("caption_mentions_label") is False
    ]
    if flagged:
        st.caption(
            "⚠️ Diagnostic: the caption did not mention the detected label for "
            f"{', '.join(sorted(set(flagged)))}. This is a heuristic signal that the "
            "row may be less reliable — not a correctness check."
        )

    st.subheader("Extracted objects")
    columns = st.columns(3)
    for index, (name, image) in enumerate(result.crops):
        obj = result.payload["objects"][index]
        with columns[index % 3]:
            st.image(image, caption=f"{obj['label']} · {obj['score']:.0%}", use_column_width=True)

document_text = result.payload.get("document_text") or []
if document_text:
    st.subheader("Text found in the image")
    st.write(" ".join(document_text))

# --- Downloads (in-memory by default; no files written) ----------------------
with stage(logger, "export", run_id=run_id) as rec:
    payload_json = serialize_payload(result.payload)
    annotated_for_export = add_provenance_mark(
        result.annotated, f"AI-annotated · {settings.detr_model_id}"
    )
    bundle = build_export_bundle(
        payload_json=payload_json,
        annotated=annotated_for_export,
        crops=list(result.crops),
    )
    persisted = persist_bundle(bundle, run_id=run_id)
    rec["persisted"] = bool(persisted)

st.subheader("Download")
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "⬇️ Results (JSON)",
        data=bundle.payload_json,
        file_name="analysis.json",
        mime="application/json",
    )
with col_b:
    st.download_button(
        "⬇️ Annotated image (PNG)",
        data=bundle.annotated_png,
        file_name="annotated.png",
        mime="image/png",
    )

st.caption(
    "Object descriptions are generated by an image-captioning model and may include "
    "details not present in the photo. Labels, confidences and recognized text are "
    "measured outputs; descriptions are generated prose."
)
