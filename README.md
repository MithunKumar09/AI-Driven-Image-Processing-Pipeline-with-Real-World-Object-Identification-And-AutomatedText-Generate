# AI-Driven Image Analysis Pipeline

**Full name:** AI-Driven Image Processing Pipeline with Real-World Object Identification and Automated Text Generation
**Status of this document:** internal engineering reference (architecture, algorithms, complexity, contracts, operations)

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Problem statement and scope](#2-problem-statement-and-scope)
3. [Technology stack](#3-technology-stack)
4. [System architecture](#4-system-architecture)
5. [End-to-end execution walkthrough](#5-end-to-end-execution-walkthrough)
6. [Module reference](#6-module-reference)
7. [Algorithms used](#7-algorithms-used)
8. [Complexity analysis](#8-complexity-analysis)
9. [Data contracts and invariants](#9-data-contracts-and-invariants)
10. [Output schema](#10-output-schema)
11. [Caching model](#11-caching-model)
12. [Concurrency model](#12-concurrency-model)
13. [Configuration reference](#13-configuration-reference)
14. [Security and safety posture](#14-security-and-safety-posture)
15. [Observability](#15-observability)
16. [Testing strategy](#16-testing-strategy)
17. [Defect history — what was fixed and why](#17-defect-history--what-was-fixed-and-why)
18. [Known limitations and non-goals](#18-known-limitations-and-non-goals)
19. [Setup, run, and operations](#19-setup-run-and-operations)
20. [Extension points](#20-extension-points)

---

## 1. Executive summary

A Streamlit web application that ingests a single photograph and produces a structured,
downloadable analysis of it. Three independent models run locally over the image:

| Stage | Model | Produces |
|---|---|---|
| Object detection | `facebook/detr-resnet-50` | label, confidence, bounding box (original-image pixels) |
| Text recognition | EasyOCR (English) | text spans with geometry + confidence |
| Visual captioning | `Salesforce/blip-image-captioning-base` | one grounded caption per object crop |

A deterministic, model-free composition layer then joins these three outputs: each OCR
span is attached to the object it geometrically sits on, each object gets a prose
description assembled by string templating (not by an LLM), and the whole thing is
serialized to a versioned JSON payload plus an annotated PNG.

**Three properties define the design:**

1. **No API keys, no network calls at inference time.** All three models are downloaded
   once from the Hugging Face Hub and run on local CPU.
2. **Pinned model revisions.** Both HF models are pinned to commit SHAs, not branch names,
   so an upstream in-place repo update cannot silently change weights, `id2label`, or
   processor defaults.
3. **Measured vs. generated is kept visible.** Labels, scores, boxes and OCR text are
   *measured*; only the BLIP caption is *generated prose*. The UI, the payload, and the
   docstrings all preserve that distinction rather than blurring it into "AI insight".

---

## 2. Problem statement and scope

**Goal.** Given an arbitrary user photograph, answer: *what objects are in it, where are
they, what does each look like, and what text appears — and which object does that text
belong to?*

**In scope**
- Single-image, synchronous analysis triggered from a browser upload.
- The 80 real COCO object categories DETR was trained on (91-entry sparse id space).
- English printed/scene text.
- Downloadable JSON + annotated PNG artifacts.

**Explicitly out of scope**
- Instance segmentation masks (an earlier iteration had a Mask R-CNN path; it was removed).
- Video, batch, or multi-image processing.
- Open-vocabulary detection — anything outside COCO is invisible to the detector.
- Non-English OCR (the EasyOCR `Reader` is constructed with `["en"]`).
- Server-side authentication, quotas, or multi-tenant isolation.

---

## 3. Technology stack

### 3.1 Runtime

| Layer | Technology | Pinned version | Role |
|---|---|---|---|
| Language | Python | 3.12 (tested 3.12.6) | Entire codebase |
| Web UI | Streamlit | 1.38.0 | Upload, layout, caching primitives, downloads |
| DL framework | PyTorch | 2.4.1**+cpu** | Model execution |
| Vision ops | torchvision | 0.19.1+cpu | Torch companion pin |
| Model hub / API | Hugging Face `transformers` | 4.44.2 | DETR + BLIP loaders, processors, generation |
| Backbone provider | `timm` | 1.0.9 | Supplies DETR's ResNet-50 backbone |
| OCR | EasyOCR | 1.7.1 | CRAFT detector + CRNN recognizer |
| Image I/O | Pillow | 10.4.0 | Decode, EXIF, RGB normalization, drawing |
| Numerics | NumPy | 2.1.1 | `HxWx3 uint8` array representation |
| Tabular | pandas | 2.2.2 | Summary table rendering only |
| CV backend | `opencv-python-headless` | 4.10.0.84 | EasyOCR dependency; headless — no GUI window |
| Testing | pytest | 8.3.3 | Two-tier test suite |

**The `+cpu` local version is load-bearing.** `pip install torch==2.4.1` resolves a CUDA
build on many platforms, pulling gigabytes of GPU runtime and changing which kernels
execute. `requirements.txt` adds `--extra-index-url https://download.pytorch.org/whl/cpu`
to make the CPU build resolve deterministically. The code is nonetheless device-aware
(`get_device()`) and moves both models *and every input tensor* to the resolved device, so
swapping in a CUDA wheel works without code changes.

### 3.2 Standard-library choices worth noting

- **Configuration**: a frozen `@dataclass` with hand-written env parsers rather than
  pydantic. Pydantic is not a dependency; explicit validation is proportionate at this size.
- **Concurrency**: `threading.Semaphore`, not asyncio — Streamlit's execution model is
  thread-per-session and the workload is CPU-bound.
- **Logging**: `logging` with a custom key=value formatter, not `structlog`.
- **Serialization**: `json` with a recursive `_to_builtin` coercion for numpy/torch scalars.

### 3.3 Model weights (downloaded on first run, ~1.2 GB total)

| Model | Architecture | Approx. params | Pinned revision |
|---|---|---|---|
| DETR-ResNet-50 | CNN backbone + transformer encoder-decoder | ~41 M | `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b` |
| BLIP captioning base | ViT-B/16 encoder + BERT-style text decoder | ~250 M | `82a37760796d32b1411fe092ab5d4e227313294b` |
| EasyOCR `en` | CRAFT + CRNN (`english_g2`) | ~35 M combined | not pinned (EasyOCR fetches its own weights) |

---

## 4. System architecture

### 4.1 Layered view

```
┌──────────────────────────────────────────────────────────────────────┐
│  PRESENTATION            streamlit_app/app.py                        │
│  Upload widget, sidebar, layout, download buttons, request logging.   │
│  Thin — holds no inference logic.                                     │
├──────────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION           pipeline.py                                  │
│  Cache-identity boundary. Sequences detect → ocr → caption → compose. │
│  Owns the inference-slot scope. Free of request identity.             │
├──────────────────────────────────────────────────────────────────────┤
│  MODEL ADAPTERS          models/                                      │
│  registry.py (lifecycle, device, semaphore)                           │
│  detection_model.py · description_model.py · text_extraction_model.py │
│  Each owns its own coordinate/decoding contract.                      │
├──────────────────────────────────────────────────────────────────────┤
│  PURE LOGIC              utils/compose.py · utils/data_mapping.py     │
│  Geometry, budgeting, association, description assembly, payload.     │
│  No I/O, no models, no Streamlit state → highest-value test surface.  │
├──────────────────────────────────────────────────────────────────────┤
│  BOUNDARY SERVICES       utils/upload.py · artifacts.py ·             │
│                          visualization.py · rate_limit.py ·           │
│                          logging_setup.py                             │
│  artifacts.py is the ONLY module permitted to write to disk.          │
├──────────────────────────────────────────────────────────────────────┤
│  CONFIGURATION           config.py — validated at import, fails fast  │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Dataflow

```
 browser upload (bytes)
        │
        ▼
 ┌─────────────────────┐
 │ utils/upload.py     │  size cap → format allowlist (from decoded content)
 │ load_validated_bytes│  → megapixel bomb guard (header only) → decode
 └─────────┬───────────┘  → EXIF transpose → convert("RGB") → SHA-256
           │
           ▼  ValidatedImage {image, array, content_hash, w, h, format, size}
 ┌─────────────────────────────────────────────────────────────┐
 │ pipeline.run_inference   ── st.cache_data (key: hash +      │
 │                              threshold + cache_identity())  │
 │  ┌───────── inference_slot() (semaphore) ────────────────┐  │
 │  │  DETR.detect()      → list[Detection]  (boxes in      │  │
 │  │                        ORIGINAL image pixels)         │  │
 │  │  EasyOCR.extract()  → list[OcrSpan]    (rescaled back │  │
 │  │                        to ORIGINAL pixels)            │  │
 │  │  select_for_captioning() → which crops earn a caption │  │
 │  │  BLIP.caption(crop) × K → {index: caption}            │  │
 │  └───────────────────────────────────────────────────────┘  │
 │   associate_spans()      → {det_index: [spans]}, loose[]    │
 │   build_object_record()  → per-object dict + description    │
 │   visualize_detections() → annotated PIL image              │
 │   build_payload()        → canonical JSON-ready dict        │
 └─────────┬───────────────────────────────────────────────────┘
           │  PipelineResult {payload, detections, spans, annotated, crops}
           ▼
 ┌─────────────────────┐
 │ app.py rendering    │  summary dataframe · crop gallery · document text
 │ artifacts.py export │  serialize → provenance mark → in-memory PNG/JSON
 └─────────────────────┘  (disk write only if PERSIST_ARTIFACTS=true)
```

### 4.3 Design rules the code enforces

| Rule | Enforced by | Test |
|---|---|---|
| One canonical coordinate space (original-image pixels) | `DetrImageProcessor` owns DETR resize; OCR inverts its own scale at its boundary | `test_boxes_are_in_original_image_space`, `test_no_scale_factor_in_detection_path` |
| Class names come only from `model.config.id2label` | `DetectionModel.id2label` property | `test_no_hardcoded_coco_mapping_survives`, `test_labels_come_from_model_config` |
| Only `utils/artifacts.py` touches the filesystem | module docstring + code review | `test_end_to_end_returns_bytes_and_writes_nothing` |
| Request identity never enters the cache | `run_inference` accepts/returns no `run_id` | `test_run_id_never_enters_the_cache`, `test_payload_carries_no_request_identity` |
| Ordering is intrinsic, never positional | sort keys use score/box/label only | `test_selection_is_order_invariant`, `test_association_is_order_invariant` |
| `content_hash` is never logged | `_FORBIDDEN_FIELDS` scrub in `logging_setup.stage` | `test_content_hash_never_logged` |

---

## 5. End-to-end execution walkthrough

What happens, in order, when a user drops a JPEG into the browser:

**1 · Script rerun.** Streamlit re-executes `app.py` top to bottom. `st.set_page_config`,
title, sidebar slider (threshold), and file uploader are constructed. If no file is
present, `st.stop()`.

**2 · `run_id` minted.** `new_run_id()` produces a fresh 12-hex-char UUID slice. This is
the *request-scope* correlation id and is minted on **every** execution, cache hit or miss.

**3 · Upload validation** (`load_validated_image` → `load_validated_bytes`), inside a timed
`stage(logger, "upload", run_id=…)` block:
   - `.getvalue()` is used, never `.read()` — `UploadedFile` is a `BytesIO` whose position
     persists across reruns within a session, so `.read()` returns `b""` the second time.
   - Reject empty; reject `len(data) > MAX_UPLOAD_MB`.
   - `Image.open(...).verify()` on a probe handle establishes the *real* format from decoded
     content — the filename extension and the uploader's `type=` hint are both client-side
     and untrusted. Format must be in `{JPEG, PNG, WEBP}`.
   - **Decompression-bomb guard**: header dimensions (available lazily, before pixels are
     materialized) are multiplied and compared to `MAX_PIXELS_MP`. Deliberately does *not*
     assign `Image.MAX_IMAGE_PIXELS`, which is process-global and shared by every session.
   - Decode with `DecompressionBombWarning` promoted to an error, then `ImageOps.exif_transpose`
     (phone photos display rotated in the browser but decode unrotated — boxes would appear
     misplaced) and `.convert("RGB")` (guarantees `HxWx3 uint8`; grayscale gives 2-D, RGBA
     gives 4 channels).
   - Reject anything smaller than 2×2.
   - Compute `SHA-256(data)` as `content_hash` — **cache key only**, never logged, never
     rendered, never used in a path.

**4 · Rate-limit check.** `rate_limit.check(st.session_state, …)` prunes timestamps outside
the rolling 3600 s window and reports `allowed`. It does **not** consume budget.

**5 · Inference** (`run_inference`) inside `st.status`:
   - `cache_identity(threshold)` assembles the 12-tuple of everything that can change output.
   - `_run_inference_cached(content_hash, threshold, identity, _image_bytes)` — Streamlit
     hashes the first three args as the key; `_image_bytes` is underscore-prefixed so
     megabytes are not re-hashed per call.
   - **On a hit**, the stored `PipelineResult` is returned; no model runs, no `compute_id`
     is minted, no stage logs are emitted.
   - **On a miss**, `_execute` runs: mints an internal `compute_id`, resolves the three
     cached model singletons, then acquires the inference semaphore.

**6 · Inside the inference slot** (model calls only):
   - `detect()` — processor → forward → `post_process_object_detection(threshold, target_sizes=original)`
     → clamp → label lookup → stable sort by `(-score, box, label)`.
   - `extract()` — downscale to `OCR_MAX_DIM`, `readtext`, filter by `OCR_MIN_CONFIDENCE`,
     invert the scale back to original pixels, stable sort by reading order.
   - `select_for_captioning()` — drop crops smaller than `MIN_CROP_PIXELS` on either axis,
     rank the rest by `(-score, box, label)`, keep the top `MAX_DESCRIPTIONS_PER_IMAGE`.
   - For each selected index: `pad_box(box, 4, …)`, crop, `BLIP.caption(crop)` with
     `do_sample=False`, `num_beams=3`, `max_new_tokens=30`.

**7 · Outside the slot** (so a slow render cannot stall other sessions):
   - `associate_spans()` attaches each span to at most one detection.
   - `build_object_record()` per detection: measured fields + caption + region text +
     `caption_mentions_label` diagnostic + templated `description` string.
   - Every detection (not just captioned ones) gets a crop image for the gallery.
   - `visualize_detections()` draws boxes/labels onto a **copy**.
   - `build_payload()` produces the canonical dict; `_to_builtin` coerces any stray
     numpy/torch scalar so `json.dumps` cannot raise.

**8 · Post-inference.** `record_run()` consumes one unit of rate-limit budget;
`request.completed` is logged with object and span counts.

**9 · Rendering.** Two-column original/annotated view, summary dataframe, a heuristic
warning row if any caption failed to mention its label, a 3-column crop gallery, and
concatenated document-level text.

**10 · Export.** `serialize_payload` → `add_provenance_mark` (applied to the *exported*
image only, never to arrays fed to a model) → `build_export_bundle` (bytes) →
`persist_bundle` (no-op unless `PERSIST_ARTIFACTS=true`) → two `st.download_button`s.

**Failure paths.** `UploadValidationError` → user-facing warning + `upload.rejected` log.
`InferenceBusyError` → "server is busy" warning. Any other exception → full traceback to
the log via `logger.exception`, and the user sees only the `run_id` and the exception class
name — never a stack trace.

---

## 6. Module reference

### `config.py` (211 lines)
Frozen `Settings` dataclass, loaded and **validated at import time** so a bad env var fails
fast rather than surfacing deep inside inference. Typed parsers (`_env_str/_bool/_int/_float`)
each name the offending variable and its valid range. `_require_valid_root()` asserts that
`config.py` still sits beside `requirements.txt` — a guard against `PROJECT_ROOT` silently
resolving one directory too high and scattering artifacts. Derived properties: `data_dir`,
`runs_dir`, `detr_ref`, `blip_ref` (the `id@revision` strings that appear in every payload).
Cross-field check: beam search with `max_new_tokens < 5` is rejected.

**Deliberate omission:** there is no `MAX_IMAGE_DIM` for detection. `DetrImageProcessor` owns
DETR's resizing, so such a knob would change nothing — a dead setting is worse than no
setting. `OCR_MAX_DIM` exists precisely because the OCR path *does* do its own downscaling.
This is asserted by `test_no_dead_detr_resize_knob`.

### `pipeline.py` (218 lines)
The cache-identity boundary. `cache_identity()` assembles, in one place, everything that can
change inference output — so no call site can forget a component. `_run_inference_cached` is
the `st.cache_data`-wrapped core; `_execute` is the real work; `run_inference_uncached`
bypasses the cache for determinism tests.

The `run_id` / `compute_id` split lives here conceptually: a cached payload carrying the
`run_id` of whichever execution populated it would hand every later cache hit a stale
correlation id belonging to a different user's request. So `run_inference` never accepts,
returns, or stores a `run_id`.

### `models/registry.py` (141 lines)
`st.cache_resource` singletons for the three models and the device. Streamlit re-executes the
script on every widget interaction; the original app constructed every model at module scope,
so Mask R-CNN + CLIP + GPT-Neo-1.3B (loaded twice) + EasyOCR + DETR were rebuilt on each rerun.

Two consequences are designed for rather than discovered: cached resources are **process-global**
(the same instance serves every browser session), and they are **read-only after load** (never
mutate a cached model at request time — no per-request `.to()`, no config edits). The
process-wide `threading.Semaphore` is deliberately *not* a cached object; it must exist once
per process, independent of cache eviction.

### `models/detection_model.py` (120 lines)
`DetectionModel` wraps DETR and owns the coordinate contract. `id2label` is exposed as a
property and is the single source of truth for class names. `detect()` hands the raw PIL image
to the processor (no pre-resizing), moves **every** input tensor to the device, runs under
`torch.inference_mode()`, and passes the **original** `(height, width)` as `target_sizes` so
post-processing returns boxes already in original-image pixels. Degenerate boxes are dropped by
`clamp_box`; unknown ids fall back to `class_{id}`. Results are sorted by intrinsic fields.
Module-level `crop()` slices an already-clamped box from an `HxWx3` array.

### `models/description_model.py` (85 lines)
`DescriptionModel.caption()` runs BLIP **unconditionally** — no text prompt is supplied, so the
caption is statistically independent of the DETR label. That independence is exactly what makes
`captions_agree()` a meaningful cross-model signal; conditioning on the label to "help" the model
would destroy it. Generation uses `max_new_tokens` (never `max_length`, which counts the prompt)
and `do_sample=False` — the determinism precondition that makes result caching semantically safe.
`caption_batch` is explicitly *not* batched: batching adds padding/attention-mask complexity and
correctness risk for an unmeasured gain on a CPU build.

### `models/text_extraction_model.py` (103 lines)
`OcrSpan(text, confidence, box)` — box in **original-image pixels**. `extract()` downscales to
`OCR_MAX_DIM` for speed, remembers the ratio, and inverts it at this wrapper's boundary so no
consumer ever needs to know the OCR scale. `_to_original_coords` converts EasyOCR's 4-point
polygon to an axis-aligned box via min/max, clamps, and rejects degenerate results. Spans below
`OCR_MIN_CONFIDENCE` are dropped; the rest are sorted into reading order `(y, x, text)`.
`document_text()` flattens spans to strings.

### `utils/compose.py` (278 lines) — the pure core
Deterministic, I/O-free, model-free, Streamlit-free. Contains: `clamp_box`, `pad_box`,
`box_area`, `intersection_area`, `is_captionable`, `select_for_captioning`, `associate_spans`,
`captions_agree`, `compose_description`, `build_object_record`. See
[§7 Algorithms](#7-algorithms-used) for the logic.

### `utils/data_mapping.py` (91 lines) — pure
`build_payload` / `serialize_payload` / `read_payload`. Creates no directories, writes no files.
The previous signature was `map_data_to_objects(data, output_file)` — taking a destination path
and writing to it, which forced disk persistence into the default path. `_to_builtin` recursively
coerces mappings, sequences, and anything exposing `.item()`/`.tolist()` (numpy and torch scalars
without importing either), falling back to `str()`. `canonical=True` sorts keys for stable
snapshot diffs.

### `utils/upload.py` (153 lines)
Validation and normalization, described in [§5 step 3](#5-end-to-end-execution-walkthrough).
The uploaded image is **never written to disk to run inference** — it is already in memory as
bytes. Carries the two-identifier doctrine: `content_hash` (internal cache key, never emitted)
vs `run_id` (the only public/observability identifier). Reusing the content fingerprint as a
correlation id would publish a stable identifier for the image itself, letting anyone with log
access link the same picture across sessions.

### `utils/artifacts.py` (129 lines)
The only module permitted to create directories or open files for writing. `build_export_bundle`
produces `ExportBundle(payload_json, annotated_png, crop_pngs)` as pure bytes. `persist_bundle`
is a no-op unless `PERSIST_ARTIFACTS=true`; when enabled it writes to `data/runs/<safe(run_id)>/`
— keyed by `run_id`, never `content_hash`, so concurrent sessions cannot collide and the directory
name leaks no fingerprint of the image. `_safe_component` strips separators and traversal.
`enforce_retention` implements the eviction policy (see [§7.7](#77-artifact-retention--bounded-eviction)).

### `utils/visualization.py` (105 lines)
`visualize_detections` takes label **strings** directly — the old signature took label *indices*
plus a `coco_labels` list, forcing callers through the very hardcoded list that caused the
mislabeling defect. Draws onto a copy; the input is never mutated. Label placement prefers above
the box and drops inside when there is no room, clamped to canvas bounds.
`add_provenance_mark` composites a subtle corner label onto **exported** images only — never onto
arrays or crops fed to a model, which would corrupt inference inputs. The wording is
"AI-annotated", never "AI-generated": the source photograph is user-provided, not synthetic; only
the annotations and captions are model output. This is asserted by a test.

### `utils/rate_limit.py` (60 lines)
Rolling 1-hour window over timestamps in `st.session_state`. `check()` reports without consuming;
`record_run()` consumes. Keyed to actual pipeline executions, never to script reruns — Streamlit
reruns on every widget interaction, so counting reruns would exhaust the budget with no inference
happening. **Per-session, not server-side** — the UI says so plainly via `describe_limit`.

### `utils/logging_setup.py` (146 lines)
Custom `_KeyValueFormatter` emits `ts level logger msg key=value …` — greppable without a parser.
`stage()` is a context manager that times a block, yields a mutable dict for counts, and emits one
record with `duration_ms` on success or `logger.exception` on failure (then re-raises). `_scrub`
drops `_FORBIDDEN_FIELDS = {content_hash, sha256, image_hash}` whatever the caller passes.

### `streamlit_app/app.py` (206 lines)
Thin orchestration. Inserts the repo root into `sys.path` so `streamlit run streamlit_app/app.py`
resolves `models.` and `utils.` imports. Version note in the docstring: the pinned Streamlit 1.38.0
accepts `use_column_width` on `st.image` and does **not** accept `use_container_width` (verified via
`inspect.signature`), while `st.dataframe` does accept it — do not "modernize" without bumping.

---

## 7. Algorithms used

### 7.1 Object detection — DETR (DEtection TRansformer)

**Architecture.** ResNet-50 CNN backbone → 1×1 projection → transformer encoder (self-attention
over flattened feature-map positions with 2-D sinusoidal positional encoding) → transformer decoder
with **100 learned object queries** attending to encoder memory → two prediction heads per query
(a 92-way class softmax including a "no object" ∅ class, and a 4-vector box in normalized
`cxcywh`).

**Why it matters here:** DETR is *set prediction*. Each query emits at most one object, and the
Hungarian bipartite matching that makes this work is a **training-time** loss construction — it does
not run at inference. Consequently **there is no non-maximum suppression, no anchor generation, and
no region-proposal stage** in this codebase. Inference post-processing is just: sigmoid/softmax the
logits, threshold, convert `cxcywh → xyxy`, and rescale by `target_sizes`.

**The label-space subtlety (the project's headline defect).** DETR predicts over COCO's **sparse
91-entry category-id space**, which contains ~11 `N/A` placeholder ids (12, 26, 29, 30, 45, 66, 68,
69, 71, 83, …). Indexing those ids into a hardcoded **81-entry contiguous** list shifts every class
after the first gap. Category id 53 is `apple` in the real space and lands on `hot dog` at index 53
of the contiguous list. The fix is to maintain no list at all and read `model.config.id2label`,
which is correct regardless of where the gaps fall.

**Ordering.** `sort(key=(-score, box, label))` — intrinsic fields only, so downstream consumers and
the result cache never depend on tensor iteration order.

### 7.2 Text recognition — EasyOCR (CRAFT + CRNN/CTC)

Two-stage classical OCR pipeline:

1. **Detection — CRAFT** (Character Region Awareness for Text detection). A VGG/UNet-style fully
   convolutional network predicts two dense score maps per pixel: a *region* score (is this pixel
   inside a character?) and an *affinity* score (do these two adjacent characters belong to the same
   word?). Connected-component labeling with watershed-style grouping over the thresholded maps
   yields word-level quadrilaterals. This is why EasyOCR returns 4-point polygons rather than
   axis-aligned rectangles.
2. **Recognition — CRNN with CTC**. Each detected region is rectified to a fixed height, passed
   through a ResNet feature extractor, then a bidirectional LSTM sequence encoder, and decoded with
   **CTC** (Connectionist Temporal Classification) — an alignment-free decoder that collapses
   repeated frames and blanks, so no character-level segmentation is needed. The per-span confidence
   is derived from the CTC path probability.

**This wrapper's contribution.** The previous implementation kept only `result[1]` — the text
string — discarding geometry and confidence, so recovered text could never be tied to the object it
sat on. A grounding signal that was already being paid for was thrown away. Here the full triple is
kept, filtered by confidence, and rescaled to original-image coordinates.

### 7.3 Image captioning — BLIP

**Architecture.** ViT-B/16 image encoder produces patch embeddings; a BERT-style text decoder
attends to them via cross-attention and generates a caption autoregressively. The pretraining
objective that matters for this use is the **captioning (LM) head** of BLIP's three-objective
training (ITC / ITM / LM), plus BLIP's **CapFilt** bootstrapping (a captioner generates synthetic
captions for web images and a filter removes the noisy ones), which is why BLIP-base captions are
usable without fine-tuning.

**Decoding.** Beam search with `num_beams=3`, `max_new_tokens=30`, `do_sample=False`. Deterministic
by construction — this is the precondition that makes result caching semantically safe. Reintroducing
sampling would require removing the cache.

**Grounding, stated honestly.** Captioning the actual crop removes the *image-blind* generation path
that produced the original defect. It does **not** make the system hallucination-free: a VLM can still
assert visually unsupported details — invented colors, counts, materials, context — especially on
small, blurry, or occluded crops. The correct claim is *grounded captioning with a substantially
reduced ungrounded-generation surface*, never "verified" or "factual".

### 7.4 Box clamping — `clamp_box`

Guards a subtle Python defect: `int()` of a negative coordinate becomes a *negative index*, so
`image_np[-5:300]` silently produces an empty or wrong-region crop rather than erroring.

```
1. reject if len(box) != 4
2. swap inverted pairs (x2 < x1 → swap) — tolerate rather than emit an empty slice
3. round, then clamp each coordinate into [0, width] / [0, height]
4. return None if the result has zero area
```
Postcondition: `0 ≤ x1 < x2 ≤ width` and `0 ≤ y1 < y2 ≤ height`, or `None`.

### 7.5 Caption budgeting — `select_for_captioning`

BLIP is one generation per object, so cost is linear in detection count and **unbounded in
principle** — a crowd scene can yield 30+ boxes.

```
eligible = [i for i, d in enumerate(detections)
            if (x2-x1) >= MIN_CROP_PIXELS and (y2-y1) >= MIN_CROP_PIXELS]
ranked   = sort(eligible, key=(-score, box, label))
selected = set(ranked[:MAX_DESCRIPTIONS_PER_IMAGE])
skipped  = {small crops: "crop smaller than Npx",
            overflow:    "description skipped (cap reached)"}
```

`is_captionable` requires **both** dimensions to clear the threshold — one knob, one meaning. There
is deliberately no separate `MIN_CROP_AREA`, because a 12×9 crop yields a meaningless caption at full
model cost while satisfying any reasonable area bound. Selection ranks on intrinsic fields only, so
it is provably invariant under any permutation of the input.

### 7.6 OCR-to-object association — `associate_spans`

**Why centroid containment is insufficient.** Detections routinely overlap — a person holding a
bottle, nested fruit — so one span centroid can fall inside several boxes and be double-counted.

**The scoring rule.** For each span, compute `overlap = intersection_area(span, box) / span_area`
over all detections; keep candidates with `overlap ≥ MIN_OVERLAP_RATIO` (default 0.5), then select
the minimum under the lexicographic key:

```
overlap_ratio DESC → box_area ASC → detection_score DESC → (x1,y1,x2,y2) → label
```

- `overlap DESC` — the best geometric fit wins.
- `box_area ASC` — **the tighter of two nested boxes wins**. Text on a bottle held by a person
  belongs to the bottle, not to the person.
- The remaining keys are deterministic tie-breaks on intrinsic fields.

**List position is never a tie-break.** It is the input order, which would contradict the
shuffle-invariance the function guarantees. If two detections tie on every intrinsic field they are
duplicates of the same box, so either choice is by definition equivalent.

Each span attaches to **at most one** detection. Zero-area spans and spans below the threshold
against every box become *document-level* text (`loose`), surfaced separately in the UI.

### 7.7 Artifact retention — bounded eviction

Two independent bounds, whichever binds first, plus a concurrency guard:

```
candidates = subdirs of data/runs, EXCLUDING the in-flight run_id
sort candidates by mtime DESC              # newest first; tail = eviction set
for index, dir in candidates:
    age = now - mtime
    if age < 300s: continue                # grace window: a concurrent run
                                           # may still be writing into it
    evict if age > ARTIFACT_TTL_HOURS      # time bound
    evict if index >= MAX_STORED_RUNS - 1  # count bound (current run holds a slot)
```

### 7.8 Rate limiting — rolling window

A list of float timestamps in `st.session_state`. Both `check()` and `record_run()` prune entries
older than 3600 s before acting; `retry_after = 3600 - (now - oldest) + 1`. `check()` is
non-consuming so a blocked user does not extend their own penalty.

### 7.9 Description composition — deterministic templating

```
"{Label} — detected with {score:.0%} confidence."
  + ("Visual description not generated: {reason}."   if skipped
     else 'Visual description: "{cleaned caption}".' if caption)
  + ('Text found in this region: "{joined spans}".'  if region text)
```

**No language model is involved in the summary itself**, so the summary has no hallucination surface
of its own. The caption stays quoted and attributed so the UI can distinguish measured fields from
generated prose. `_clean_caption` collapses whitespace, capitalizes the first character, and strips
trailing spaces/periods (a terminal period is re-added by the template — this is what prevents the
old `"according to Merriam-Webster,."` artifact).

### 7.10 Cross-model agreement — `captions_agree` (a diagnostic, not a check)

Because BLIP captions the crop *unconditionally*, its output is independent of the DETR label, so
asking "does the caption mention the label?" is a free signal. Implementation: lowercase substring
test over label tokens longer than 2 characters.

**This is a heuristic for surfacing possibly-unreliable rows. It verifies nothing and must never be
presented as verification.** It is exposed as the nullable `caption_mentions_label` field (`null`
when there is no caption to compare) and rendered in the UI with an explicit disclaimer. A dedicated
test, `test_agreement_flag_is_a_heuristic`, exists to keep this honest.

---

## 8. Complexity analysis

### 8.1 Symbols

| Symbol | Meaning | Typical |
|---|---|---|
| `N` | uploaded file size in bytes | ≤ 10 MB |
| `W, H` | original image dimensions | ≤ 50 MP total |
| `W', H'` | OCR working dimensions, capped by `OCR_MAX_DIM` | ≤ 1600 on the long edge |
| `F` | DETR feature-map tokens ≈ `(W/32)·(H/32)` | 100–1000 |
| `Q` | DETR object queries — **fixed** | 100 |
| `C` | class-space size — **fixed** | 92 (91 + ∅) |
| `D` | detections above threshold | 0–30 |
| `K` | captioned crops, `K ≤ min(D, MAX_DESCRIPTIONS_PER_IMAGE)` | ≤ 10 |
| `S` | OCR spans surviving the confidence filter | 0–50 |
| `P` | BLIP ViT patches — **fixed** (384², 16² patches) | 577 |
| `B, L` | beam width, max new tokens — **fixed** | 3, 30 |
| `d` | model hidden width — **fixed** | 256 (DETR) / 768 (BLIP) |

### 8.2 Per-stage time complexity

| Stage | Time | Notes |
|---|---|---|
| SHA-256 hash | `O(N)` | one linear pass over the bytes |
| Header bomb guard | `O(1)` | Pillow is lazy; dimensions read without decoding pixels |
| Decode + EXIF + RGB | `O(W·H)` | dominated by JPEG entropy decode |
| DETR backbone (ResNet-50) | `O(W·H)` | ~4 GFLOPs at 800×1066; conv work is linear in pixels |
| DETR encoder self-attention | `O(F²·d)` | **quadratic in feature-map tokens** — the asymptotic ceiling of the detection path |
| DETR decoder cross-attention | `O(Q·F·d)` | linear in `F`; `Q` fixed at 100 |
| DETR post-processing | `O(Q·C)` = `O(1)` | 100×92 softmax, then threshold |
| Box clamping | `O(D)` | `O(1)` per box |
| Detection sort | `O(D log D)` | `D ≤ 100` by construction |
| OCR — CRAFT detection | `O(W'·H')` | fully convolutional; **capped by `OCR_MAX_DIM`, not by original size** |
| OCR — CRNN recognition | `O(Σ T_s)` ≈ `O(S · T̄)` | `T̄` = mean frames per span; BiLSTM is linear in sequence length |
| OCR coordinate inversion | `O(S)` | 4 points per span |
| OCR sort | `O(S log S)` | |
| Caption selection | `O(D log D)` | scan + sort |
| BLIP ViT encode (per crop) | `O(P²·d)` = `O(1)` | fixed 384×384 input regardless of crop size |
| BLIP beam decode (per crop) | `O(B·L·(L+P)·d)` = `O(1)` | fixed budget: 3 beams × 30 tokens |
| **BLIP total** | **`O(K)`** | `K` bounded by `MAX_DESCRIPTIONS_PER_IMAGE` (default 10) |
| OCR↔object association | **`O(S·D)`** | every span tested against every box; both small and bounded |
| Record building | `O(D + S)` | |
| Crop extraction | `O(Σ crop areas)` ⊆ `O(D·W·H)` | NumPy slicing is a view; `Image.fromarray` copies |
| Annotation rendering | `O(W·H + D)` | one full-canvas copy + `D` draw ops |
| Payload build + JSON | `O(D + S)` | `_to_builtin` recurses over a shallow structure |
| PNG encoding (export) | `O(W·H + Σ crop areas)` | zlib on the annotated image and each crop |
| Retention sweep | `O(R log R)` | `R` = stored run dirs; only when `PERSIST_ARTIFACTS=true` |
| Rate-limit check | `O(M)` | `M ≤ MAX_RUNS_PER_HOUR` timestamps |

### 8.3 Aggregate

**Asymptotic total (cache miss):**

```
O(N) + O(W·H) + O(F²·d) + O(W'·H') + O(K) + O(S·D) + O(W·H)
   ▲       ▲         ▲          ▲        ▲       ▲        ▲
 hash   decode    DETR        OCR      BLIP  associate  render
```

Since `F ∝ W·H/1024`, the DETR encoder term is formally `O((W·H)²)` in raw pixels — but the
processor's internal resize caps the long edge at 1333 px, so in practice `F` is bounded and the term
is effectively constant. Likewise the OCR term is capped by `OCR_MAX_DIM`. **Both model paths are
therefore bounded regardless of input size**; the only genuinely input-scaling terms are the linear
decode/hash/render passes.

**Cache hit:** `O(N)` for the SHA-256 plus `O(W·H)` for decode and render. No model executes.
Wall-clock drops from seconds to milliseconds.

**Empirical shape on CPU.** Wall-clock is overwhelmingly dominated by the `K` sequential BLIP beam
searches, since `caption_batch` deliberately does not batch. Detection is a single forward pass;
OCR is a single downscaled pass. This is why `MAX_DESCRIPTIONS_PER_IMAGE` is the single most
effective latency knob in the system.

### 8.4 Space complexity

| Resource | Cost | Notes |
|---|---|---|
| Model weights (resident) | ~1.2 GB | process-global via `st.cache_resource`; **paid once, not per session** |
| Image bytes | `O(N)` | |
| Decoded array + PIL image | `O(W·H)` ×2 | 3 bytes/px each |
| Annotated copy | `O(W·H)` | |
| Crops | `O(Σ crop areas)` ⊆ `O(D·W·H)` worst case | full-frame detections would duplicate the image `D` times |
| Result cache | ≤ 32 entries × `O(W·H + D + S)` | `max_entries=32`, `ttl=3600` |
| Activations (peak) | `O(F·d + Q·d)` / `O(P·d·B)` | transient, per forward pass |
| Persisted artifacts | `≤ MAX_STORED_RUNS` dirs | bounded by TTL and count |

**The one unbounded-in-principle term** is crop memory when many large overlapping detections exist.
It is bounded in practice by `MAX_UPLOAD_MB` and `MAX_PIXELS_MP`, and crops are held only for the
duration of a request plus its cache entry.

---

## 9. Data contracts and invariants

### 9.1 Coordinate contract

> **Every box crossing a module boundary is in original-image pixels.**

The old code resized with `cv2.resize` to max-dim 800, handed the pre-resized image to
`DetrImageProcessor` (which resized *again* internally), passed the **scaled** size as
`target_sizes`, and finally divided every box by the scale factor. Boxes existed in three spaces with
a manual round-trip between them, so every downstream consumer had to guess which one it held.

Now: the processor owns DETR's resize/normalization end to end; `target_sizes` is the **original**
`(height, width)`; and the OCR wrapper inverts its own downscale at its own boundary. There is no
`scale_factor` anywhere in the detection path — enforced by an **AST-based** test that inspects
executable identifiers rather than raw source text (the docstring legitimately mentions the term while
explaining its removal).

### 9.2 Determinism contract

Association and caption selection rank on **intrinsic fields only**, never on list position, because
`st.cache_data` caches results and an order-dependent result would make cached and fresh runs
disagree. BLIP runs with `do_sample=False`. Both are covered by shuffle-invariance tests.

**Scope of the determinism claim:** byte-identical output is guaranteed *within a process on one
machine*. Pinned revisions give model identity and version reproducibility, **not** cross-machine
numerical determinism — identical weights still run through different BLAS/oneDNN kernels and thread
counts. `test_determinism_same_process` documents exactly this boundary.

### 9.3 Identity contract

| Identifier | Scope | Value | Emitted? |
|---|---|---|---|
| `content_hash` | image | SHA-256 of upload bytes | **Never.** Cache key only — scrubbed from logs by name |
| `run_id` | request | fresh 12-hex UUID slice per execution | Yes — the only public/observability id |
| `compute_id` | cache miss | fresh UUID slice minted *inside* the cached call | Log lines only; never escapes the cached function |

**Why `content_hash` is never a correlation id:** it is a stable fingerprint of the user's image.
Publishing it would let anyone with log access link the same picture across sessions and confirm
whether a specific image was ever uploaded.

**Why `run_id` never enters the cache:** a cached payload carrying the `run_id` of whichever execution
populated it would hand every later cache hit a stale correlation id belonging to a *different* user's
request.

### 9.4 Filesystem contract

`utils/artifacts.py` is the **only** module permitted to create directories or open files for writing.
The default path (`PERSIST_ARTIFACTS=false`) is provably file-free — asserted end to end by
`test_end_to_end_returns_bytes_and_writes_nothing`.

---

## 10. Output schema

`schema_version: "1.0"`, produced by `build_payload`:

```jsonc
{
  "schema_version": "1.0",
  "pipeline_version": "1.0.0",
  "models": {
    "detection":  "facebook/detr-resnet-50@1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b",
    "captioning": "Salesforce/blip-image-captioning-base@82a37760796d32b1411fe092ab5d4e227313294b"
  },
  "detection_threshold": 0.5,
  "image":  { "width": 1024, "height": 768, "format": "JPEG", "size_bytes": 214_512 },
  "document_text": ["text spans not attached to any object"],
  "object_count": 2,
  "objects": [
    {
      "label": "apple",                       // MEASURED — from model.config.id2label
      "score": 0.9923,                        // MEASURED — rounded to 4dp
      "box": [120, 84, 388, 355],             // MEASURED — original-image pixels, xyxy
      "caption": "a red apple on a table",    // GENERATED — BLIP, unconditional
      "region_text": ["Organic"],             // MEASURED — OCR spans attached to this box
      "caption_mentions_label": true,         // DIAGNOSTIC heuristic; null if no caption
      "description_skipped": null,            // reason string when no caption was generated
      "description": "Apple — detected with 99% confidence. Visual description: \"a red apple on a table\". Text found in this region: \"Organic\"."
    }
  ]
}
```

**Deliberately absent:** `run_id`, `compute_id`, timestamps, filenames, `content_hash` — any
request-scoped metadata. This payload is returned by the cached inference function, and request
identity must never be cached.

**Provenance is embedded.** `models` carries `id@revision` for each model, so any exported artifact is
traceable to the exact weights that produced it. `test_payload_records_model_provenance` and the
integration test both assert the revision SHAs appear in the payload.

---

## 11. Caching model

Two distinct Streamlit caches with different semantics:

| Cache | Decorator | What it holds | Lifetime |
|---|---|---|---|
| Model singletons | `st.cache_resource` | DETR, BLIP, EasyOCR, `torch.device` | Process lifetime |
| Inference results | `st.cache_data` | `PipelineResult` | `max_entries=32`, `ttl=3600 s` |

**The cache key** is `(content_hash, detection_threshold, cache_identity(threshold))`, where
`cache_identity` is a 12-tuple assembled in one place so no call site can forget a component:

```python
(pipeline_version, detr_ref, blip_ref, detection_threshold,
 ocr_max_dim, ocr_min_confidence, min_overlap_ratio,
 min_crop_pixels, max_descriptions_per_image,
 blip_max_new_tokens, blip_num_beams,
 False)   # do_sample — pinned; flipping it must invalidate the cache
```

Notes:
- **Model *ids* alone are not identity** — a Hub repo can be updated in place under the same name — so
  pinned revisions are part of the key.
- `detection_threshold` is a **parameter**, not a read of global settings, because it is user-adjustable
  per request; mutating process-global `settings` would let concurrent sessions corrupt each other.
- `_image_bytes` is underscore-prefixed so Streamlit does not hash it — the content hash already
  identifies it, and hashing megabytes per call is pure waste.
- `pipeline_version` must be bumped by hand when composition/association logic changes in a way the
  other fields would not capture.
- **A stale hit is worse than no cache** — it silently serves results from a different pipeline. That
  premise is why the identity tuple is exhaustive and why three separate contract tests guard it.

**Behavioral note:** a cache hit still consumes one unit of rate-limit budget, because `record_run()`
is called after `run_inference` returns regardless of hit or miss. This is intentional — the limiter
counts user-triggered analyses, not GPU-seconds.

---

## 12. Concurrency model

Streamlit serves each browser session on its own thread within one process, while
`st.cache_resource` hands **the same model instance to every session**. Neither EasyOCR's `Reader`
nor the HF processors document thread safety, and concurrent sessions would contend for the same CPU
and RAM regardless.

**`inference_slot()`** — a process-wide `threading.Semaphore(MAX_CONCURRENT_INFERENCES)`, default 1 —
serializes the model-call section. Design details:

- Acquired with a **120 s timeout**; failure raises `InferenceBusyError`, which `app.py` renders as a
  polite "server is busy" message rather than a stack trace.
- Released in a `finally` block so an exception inside inference cannot leak the permit.
- **Not** an `st.cache_resource` object — it must exist once per process, independent of cache eviction.
- Held around **model calls only**. Image decode, validation, composition, cropping, annotation
  rendering, and all Streamlit rendering stay outside it, or one slow render would stall every other
  session.

**Cached resources are read-only after load.** Never mutate a cached model at request time — no
per-request `.to()`, no config edits. Similarly, the bomb guard deliberately does **not** assign
`Image.MAX_IMAGE_PIXELS`, since that is a process-global shared by every session, making per-request
mutation a race that leaks across sessions.

---

## 13. Configuration reference

All settings are optional environment variables, read from the process environment. **Nothing loads a
`.env` file** — `python-dotenv` is not a dependency; `.env.example` is documentation.

| Variable | Default | Range | Purpose |
|---|---|---|---|
| `PIPELINE_VERSION` | `1.0.0` | — | Manual cache-identity bump for logic changes |
| `DETR_MODEL_ID` | `facebook/detr-resnet-50` | — | Detection model |
| `DETR_REVISION` | `1d5f47bd…` | — | Pinned commit SHA |
| `BLIP_MODEL_ID` | `Salesforce/blip-image-captioning-base` | — | Captioning model |
| `BLIP_REVISION` | `82a37760…` | — | Pinned commit SHA |
| `DETECTION_THRESHOLD` | `0.5` | 0.0–1.0 | Min detection confidence (also a UI slider) |
| `OCR_MAX_DIM` | `1600` | 256–8192 | OCR downscale cap on the long edge |
| `OCR_MIN_CONFIDENCE` | `0.30` | 0.0–1.0 | Drop low-confidence spans |
| `MIN_OVERLAP_RATIO` | `0.5` | 0.0–1.0 | Span→object attachment threshold |
| `MIN_CROP_PIXELS` | `32` | 1–1024 | **Both** crop dimensions must meet this |
| `MAX_DESCRIPTIONS_PER_IMAGE` | `10` | 1–100 | Caption budget — the main latency knob |
| `BLIP_MAX_NEW_TOKENS` | `30` | 5–200 | Caption length (never `max_length`) |
| `BLIP_NUM_BEAMS` | `3` | 1–10 | Beam width |
| `MAX_CONCURRENT_INFERENCES` | `1` | 1–16 | Semaphore permits |
| `MAX_UPLOAD_MB` | `10` | 1–200 | Upload size cap |
| `MAX_PIXELS_MP` | `50` | 1–500 | Decompression-bomb guard |
| `MAX_RUNS_PER_HOUR` | `10` | 1–10000 | Per-session demo guard |
| `PERSIST_ARTIFACTS` | `false` | bool | Off: in-memory downloads only |
| `ARTIFACT_TTL_HOURS` | `24` | 1–8760 | Retention time bound |
| `MAX_STORED_RUNS` | `50` | 1–10000 | Retention count bound |
| `LOG_LEVEL` | `INFO` | DEBUG…CRITICAL | Log verbosity |

Invalid values raise `ConfigError` **at import**, naming the variable and its valid range —
parameterized in `test_invalid_setting_fails_fast`.

`.streamlit/config.toml` additionally sets `server.maxUploadSize = 10` (mirroring `MAX_UPLOAD_MB`, so
Streamlit rejects oversized files before they reach Python) and `server.headless = true`.

---

## 14. Security and safety posture

### 14.1 Upload handling

| Threat | Mitigation |
|---|---|
| Oversized upload / memory exhaustion | Byte cap before decode (`MAX_UPLOAD_MB`), plus Streamlit's own `maxUploadSize` |
| Decompression bomb | Header dimensions checked **before pixels are materialized**; `DecompressionBombWarning` promoted to an error; `Image.MAX_IMAGE_PIXELS` deliberately not mutated (process-global race) |
| Content-type spoofing (`.jpg` holding a script) | Format determined from **decoded content** via `Image.verify()`, not from filename or the uploader's `type=` hint (a client-side hint only) |
| Path traversal via filename | The uploaded file is **never written to disk to run inference**; when persistence is enabled, directories are keyed by `run_id` and passed through `_safe_component` |
| Malformed / truncated payload | Every decode path is wrapped; all failures collapse into a single user-facing `UploadValidationError` |

### 14.2 Privacy

- `content_hash` is scrubbed from logs by name (`_FORBIDDEN_FIELDS`) and asserted absent by
  `test_content_hash_never_logged`.
- Artifact directory names carry `run_id`, never a content fingerprint.
- Default operation writes **nothing** to disk.
- No network calls at inference time; no telemetry; no API keys anywhere in the project.

### 14.3 Error disclosure

Unexpected exceptions log a full traceback server-side via `logger.exception` but show the user only
the `run_id` and the exception **class name** — enough to correlate a support request, not enough to
leak internals.

### 14.4 Content provenance

Exported annotated images carry a composited `AI-annotated · {model_id}` mark. The wording is
deliberate and test-enforced: the source photograph is user-provided, not synthetic, so a bare
"AI-generated" claim about the image would be false. The mark is applied to exports only — never to
arrays or crops fed to a model, which would corrupt inference inputs.

### 14.5 What this is *not*

The rate limiter is **per browser session**, stored in `st.session_state`; a new session resets it. It
exists to stop a public demo from being casually used as free compute. It is not a server-side rate
limit and provides no protection against a determined caller. There is no authentication, no
authorization, and no per-user accounting. **Do not expose this to the open internet without a real
reverse-proxy rate limit in front of it.**

---

## 15. Observability

**Format.** `2026-08-17T12:18:17 INFO    pipeline.pipeline stage.detect compute_id=a1b2c3 detections=2 duration_ms=812.4`
— key=value, greppable without a parser. Values containing spaces are quoted.

**Two correlation scopes, deliberately separate:**

| Scope | Stages | Field | Runs on |
|---|---|---|---|
| **Request** | `upload`, `export`, errors, rate-limit blocks | `run_id` | every execution |
| **Compute** | `detect`, `ocr`, `describe` | `compute_id` | cache **misses** only |

Every stage is wrapped in `stage()`, which emits exactly one record carrying `duration_ms` — these
durations are the intended source for p50/p95 latency figures. Loggers are namespaced under a
`pipeline` root with `propagate = False`, so the app's records do not leak into third-party handlers.

**Event vocabulary:** `device.resolved`, `model.loading`, `model.loaded`, `stage.<name>`,
`stage.<name>.failed`, `upload.rejected`, `ratelimit.blocked`, `inference.busy`, `pipeline.failed`,
`request.completed`, `artifacts.persisted`, `artifacts.evicted`.

---

## 16. Testing strategy

Two tiers separated by a pytest marker, so the fast suite needs no model weights:

```bash
pytest -m "not slow"     # Tier 1 — unit/contract, no downloads (~8 s)
pytest -m slow           # Tier 2 — integration, downloads ~1.2 GB on first run
pytest                   # everything
```

### Tier 1 — 5 files, ~900 lines

| File | Focus |
|---|---|
| `test_geometry.py` | `clamp_box` (incl. the negative-index defect), `pad_box`, areas, `is_captionable`, caption selection, span association, **order-invariance of both ranking functions** |
| `test_contracts.py` | Project root guard, no hardcoded COCO list, no dead resize knob, no `MIN_CROP_AREA`, revisions are 40-hex SHAs, fail-fast config, **cache identity completeness**, `run_id` absent from cache and payload, JSON serialization of numpy/torch, canonical stability, description templating, agreement flag is a heuristic, `content_hash` never logged, rate-limit window |
| `test_upload.py` | Empty/oversized/renamed-text/unsupported rejection, tiny image, bomb guard from header, **no mutation of global PIL state**, all modes → RGB, EXIF orientation, hash stability |
| `test_artifacts.py` | Provenance mark non-destructive, crops untouched by export, provenance wording, visualization non-mutating, label **strings** not indices, persist no-op when disabled, isolated run dirs, retention eviction + grace window + `run_id` sanitization |
| `conftest.py` | `FakeUploadedFile` — a `BytesIO` whose position persists, reproducing Streamlit's rerun semantics; synthetic image builders; the `Two_Apples.jpg` regression fixture |

### Tier 2 — `test_integration.py` (217 lines)

The headline gate is `test_apples_are_labeled_apple`: `Two_Apples.jpg` must yield ≥1
high-confidence `apple` and **zero** `hot dog`. Against the pre-fix code this fails.

The assertion is deliberately `>= 1`, not `== 2` — the filename and the two saved crops describe the
*old broken run*, produced under both the mislabeling bug and the coordinate round-trip since removed,
and detection count is threshold-dependent anyway. *Never gate on a count without measurement behind
it.*

Also covered: `id2label` has exactly 91 entries and `id2label[53] == "apple"`; boxes lie inside the
original image; **AST inspection** proves no `scale_factor`/`resize` identifiers survive in the
detection path; captions contain no prompt echo and no `",."` truncation artifact; same-process
determinism; the default path writes nothing to disk; **input tensors land on the resolved device**
(a spy wraps `model.forward` — this is a no-op on the CPU build, which is exactly why it would
silently rot without an explicit test); and the zero-detection path produces a valid, serializable
empty result rather than a crash.

**Testing philosophy visible in the suite.** Tests assert *contracts and absences*, not just happy
paths: "no hardcoded list survives", "this knob does not exist", "this identifier is not in the AST",
"nothing was written to disk", "shuffling the input does not change the output". Several tests exist
purely to prevent a fixed defect from being reintroduced by a well-meaning refactor.

---

## 17. Defect history — what was fixed and why

This repository is a hardening of an earlier working version. The original saved output recorded two
independent defects in a single record — for a photograph of **two apples**:

```json
{ "label": "hot dog", "score": 0.9923,
  "description": "Define apple in the real world.\n\n…This website is run by
                  The Frugal Fawn…liking us on Facebook." }
```

| ID | Defect | Root cause | Fix |
|---|---|---|---|
| **F1** | `hot dog` at 99% for apples | Sparse 91-entry COCO ids indexed into a hardcoded 81-entry contiguous list; every class after the first `N/A` gap shifted | Delete the list; read `model.config.id2label` |
| **F6** | Descriptions were unrelated web text | (1) A **base**, non-instruction-tuned LM was prompted `"Define {label} in the real world."` — base LMs *continue* text, they do not answer; (2) the prompt was never stripped from the output; (3) `max_length=100` counts prompt **+** completion, truncating mid-sentence, then a period was blindly appended → `"according to Merriam-Webster,."`; (4) **the description never saw the image** | Caption the actual object crop with BLIP, unconditionally, using `max_new_tokens` |
| **F19** | Boxes in three coordinate spaces | Manual `cv2.resize` → processor resized *again* → scaled `target_sizes` → manual division by the scale factor | Processor owns resizing; `target_sizes` is the original size; AST test forbids the identifiers |
| **F4** | Empty read on rerun | `UploadedFile.read()` advances a position that persists across Streamlit reruns; the second call returned `b""` and a 0-byte file was written | Use `.getvalue()` |
| **F8** | Unsafe path, no caps | Upload written to disk just to read it back; no size or pixel limits | Never write to run inference; add byte + megapixel caps; sanitize any persisted component |
| **F17** | Wrong orientation / channel count | No EXIF transpose (browser shows rotated, PIL decodes unrotated → boxes visually misplaced); grayscale/RGBA gave non-3-channel arrays | `ImageOps.exif_transpose` + `.convert("RGB")` |
| **F18** | OCR geometry discarded | Only `result[1]` kept; ran at full resolution | Keep the full triple, downscale, invert coordinates at the boundary, associate spans to objects |
| **—** | Models rebuilt on every widget interaction | Mask R-CNN + CLIP + GPT-Neo-1.3B (loaded twice) + EasyOCR + DETR constructed at module scope | `st.cache_resource` singletons in `models/registry.py` |
| **—** | Negative-index crops | `int()` of a negative coordinate becomes a Python negative index; `image_np[-5:300]` silently returns the wrong region | `clamp_box` with an explicit `None` for degenerate boxes |
| **P1.5** | — | — | Result caching, made semantically safe by pinning `do_sample=False` and ranking on intrinsic fields only |

F1 and F6 are both covered by regression tests that fail against the old code.

---

## 18. Known limitations and non-goals

**Model capability**
1. **Closed vocabulary.** Only the 80 real COCO categories are detectable. A laptop charger, a road
   sign, or a specific brand of anything is invisible or mislabeled into the nearest COCO class.
2. **Captions can hallucinate.** BLIP is grounded in the crop but can still assert invented colors,
   counts, materials, or context — especially on small, blurry, or occluded crops. Descriptions are
   *suggestive, not authoritative*.
3. **`caption_mentions_label` verifies nothing.** It is a lowercase substring test. It will flag a
   correct caption that uses a synonym, and it will pass a fluent but wrong caption that happens to
   echo the label.
4. **English OCR only**, and EasyOCR struggles with handwriting, heavy skew, and very low contrast.
5. **No segmentation masks** — bounding boxes only. Crops of irregular objects include background.

**System**

6. **Rate limiting is per browser session**, resettable by opening a new one. Not a security control.
7. **Single-process, in-memory cache.** No horizontal scaling; a restart empties both caches and
   incurs a cold model load.
8. **Serialized inference by default** (`MAX_CONCURRENT_INFERENCES=1`). Concurrent users queue and can
   time out after 120 s with `InferenceBusyError`.
9. **CPU-bound latency**, dominated by `K` sequential BLIP beam searches. `caption_batch` is
   deliberately unbatched.
10. **Determinism is in-process only.** Different machines can produce different floating-point
    results from identical weights.
11. **No `.env` loading.** Configuration must come from the real process environment.
12. **Streamlit 1.38.0 API coupling** — `st.image` takes `use_column_width`, not
    `use_container_width`, on this pinned version.
13. **EasyOCR weights are not revision-pinned** (unlike the two HF models); EasyOCR fetches its own.
14. **Association is a heuristic.** Text overlapping ≥50% of its area with a box is attributed to that
    object, whether or not it is physically printed on it.

---

## 19. Setup, run, and operations

### Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

**Verify the CPU build actually installed** — this is the single most common setup failure:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# expected: 2.4.1+cpu 0.19.1+cpu False
```

For a GPU machine, drop the `--extra-index-url` line from `requirements.txt` and install the CUDA
build from <https://pytorch.org/get-started/locally/>. No code changes are needed — `get_device()`
resolves CUDA automatically and every input tensor is already moved to the resolved device.

### Run

```bash
streamlit run streamlit_app/app.py     # → http://localhost:8501
```

First run downloads ~1.2 GB of weights (several minutes). Subsequent runs load from the local Hugging
Face cache (`~/.cache/huggingface`, `~/.EasyOCR`).

### Test

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"     # ~8 s, no downloads
pytest -m slow           # integration
```

### Operating notes

| Symptom | Likely cause | Action |
|---|---|---|
| `ConfigError` at startup | Bad env var — the message names it and its range | Fix or unset the variable |
| "server is busy" | Semaphore timeout (120 s) under concurrent load | Raise `MAX_CONCURRENT_INFERENCES` (watch RAM: models are shared, activations are not) |
| Slow analyses | `K` sequential BLIP generations | Lower `MAX_DESCRIPTIONS_PER_IMAGE`, raise `MIN_CROP_PIXELS`, or lower `BLIP_NUM_BEAMS` |
| Nothing detected | Object outside COCO, or threshold too high | Lower the sidebar threshold; check the label vocabulary |
| Missed text | Below `OCR_MIN_CONFIDENCE`, or lost to `OCR_MAX_DIM` downscaling | Lower the confidence floor or raise `OCR_MAX_DIM` (costs latency) |
| Text attached to the wrong object | Overlapping boxes near the ratio boundary | Tune `MIN_OVERLAP_RATIO` |
| `data/runs/` growing | `PERSIST_ARTIFACTS=true` | Lower `ARTIFACT_TTL_HOURS` / `MAX_STORED_RUNS`; eviction runs after each persist |
| Results look stale after a logic change | Cache identity did not capture it | Bump `PIPELINE_VERSION` |

**Deployment checklist for anything public:** put a real reverse-proxy rate limit in front of the app
(the built-in limiter is per-session only); pre-warm the model cache into the image so first-request
latency is not several minutes; size RAM for ~1.2 GB of resident weights plus per-request activations;
decide `PERSIST_ARTIFACTS` deliberately, since enabling it means storing user images.

---

## 20. Extension points

Ordered roughly by value-to-effort, with the specific contract each change must respect:

| Change | Where | Contract to respect |
|---|---|---|
| **Swap the detector** (e.g. DETR-ResNet-101, YOLOS) | `models/registry.get_detection_model` + `DETR_MODEL_ID`/`_REVISION` | Keep reading `id2label`; keep `target_sizes` as the original size; add the new ref to `cache_identity` |
| **Swap the captioner** (e.g. BLIP-2) | `models/description_model` | Keep `do_sample=False`, or **remove the result cache** |
| **Add languages** | `TextExtractionModel.__init__` `["en"]` | Adjust `captions_agree`, whose substring test is English-shaped |
| **GPU deployment** | `requirements.txt` only | `get_device()` and the per-tensor `.to()` calls already handle it — `test_input_tensors_land_on_the_resolved_device` guards the path |
| **Batch captioning** | `DescriptionModel.caption_batch` | Padding + attention masks must be correct; the docstring asks for recorded p95 evidence first |
| **Server-side rate limiting** | New module, replacing `utils/rate_limit` at the call site | Needs shared state (Redis) — `st.session_state` cannot express it |
| **Segmentation masks** | New `models/segmentation_model.py` | Would extend, not replace, the box contract; crops become masked |
| **A REST API** | New FastAPI entry point calling `pipeline._execute` | `pipeline` is already Streamlit-coupled only through `st.cache_data`; `_execute` itself is framework-free |
| **Persistent result store** | `utils/artifacts.py` | The filesystem contract says all writes go through this module |
| **Better agreement checking** | `utils/compose.captions_agree` | Whatever replaces it must stay honest about being a heuristic unless it genuinely verifies |

---

## Appendix A — File inventory

| Path | Lines | Role |
|---|---|---|
| `config.py` | 211 | Typed, validated, env-driven settings |
| `pipeline.py` | 218 | Cached inference pipeline; the cache-identity boundary |
| `models/registry.py` | 141 | Cached model loaders, device resolution, inference semaphore |
| `models/detection_model.py` | 120 | DETR wrapper; owns the coordinate contract |
| `models/description_model.py` | 85 | BLIP captioner |
| `models/text_extraction_model.py` | 103 | EasyOCR wrapper; spans in original coordinates |
| `utils/compose.py` | 278 | Geometry, budgeting, association, composition — pure |
| `utils/upload.py` | 153 | Validation, normalization, bomb guard |
| `utils/logging_setup.py` | 146 | Structured logging, dual correlation scopes |
| `utils/artifacts.py` | 129 | Export bytes; the only filesystem writer |
| `utils/visualization.py` | 105 | Annotated rendering + provenance mark |
| `utils/data_mapping.py` | 91 | Payload construction and serialization — pure |
| `utils/rate_limit.py` | 60 | Per-session demo guard |
| `streamlit_app/app.py` | 206 | Thin UI orchestration |
| `tests/test_contracts.py` | 304 | Contract and invariant tests |
| `tests/test_integration.py` | 217 | Tier 2, real models |
| `tests/test_geometry.py` | 196 | Pure-logic tests |
| `tests/test_artifacts.py` | 195 | Export, provenance, retention |
| `tests/test_upload.py` | 127 | Upload validation |
| `tests/conftest.py` | 78 | Fixtures, `FakeUploadedFile` |
| | **3,163** | total |

Supporting files: `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.env.example`,
`.streamlit/config.toml`, `.gitignore`, `README.md`, `data/input_images/Two_Apples.jpg` (regression
fixture).

## Appendix B — Glossary

| Term | Meaning |
|---|---|
| **CTC** | Connectionist Temporal Classification — alignment-free sequence decoding used by the OCR recognizer |
| **CRAFT** | Character Region Awareness for Text detection — EasyOCR's detection stage |
| **Hungarian matching** | Bipartite assignment used in DETR's *training* loss; absent at inference |
| **Object query** | One of DETR's 100 learned decoder embeddings, each emitting at most one object |
| **`id2label`** | The model's own class-id → name mapping; the single source of truth for labels |
| **Cache identity** | The 12-tuple of every value that can change inference output |
| **`run_id` / `compute_id`** | Request-scope vs. cache-miss-scope correlation ids, deliberately never conflated |
| **Loose span** | An OCR span attached to no object; reported as document-level text |
| **Sparse COCO id space** | The 91-entry category space containing `N/A` gaps — the root of the `hot dog` defect |

test demo link: https://image-processing-pipeline-with-real-world-object-identification.streamlit.app
