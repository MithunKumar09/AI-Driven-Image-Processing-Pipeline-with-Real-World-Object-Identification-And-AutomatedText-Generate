# AI-Driven Image Analysis Pipeline

A Streamlit application that analyzes an uploaded photograph: it detects objects,
generates a grounded description of each one from its actual pixels, reads any
text in the image, and ties that text back to the object it sits on.

Everything runs **locally**. No API keys are required.

---

## What it actually does

```
upload ──▶ validate & normalize (EXIF, RGB, size/bomb caps)
       ──▶ DETR object detection            → label, confidence, box
       ──▶ EasyOCR text recognition         → text spans + geometry
       ──▶ BLIP captioning of each crop     → grounded visual description
       ──▶ associate spans to objects       → per-object text evidence
       ──▶ deterministic composition        → summary table + JSON + annotated PNG
```

| Stage | Model | Notes |
|---|---|---|
| Detection | [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50) | Labels read from `model.config.id2label` |
| Captioning | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) | Captions the object crop, unconditionally |
| Text | EasyOCR (English) | Spans carry geometry and confidence |

Both models are pinned to specific Hugging Face **commit revisions**, so results
do not change when an upstream repository is updated in place.

### A note on generated text

Labels, confidence scores, bounding boxes and recognized text are **measured**
outputs. Object descriptions are **generated prose** from an image-captioning
model. Captioning is grounded in the actual crop, which is a large improvement
over an image-blind language model — but a vision-language model can still
assert details that are not in the photo, especially on small or blurry crops.
Treat descriptions as suggestive, not authoritative.

The app reports a diagnostic flag when a caption does not mention the detected
label. That is a **heuristic** for surfacing possibly-unreliable rows, not a
verification step.

---

## Requirements

* **Python 3.12** (tested on 3.12.6)
* ~1.2 GB of disk for model weights, downloaded on first run
* No GPU required — the pinned PyTorch is a CPU build

## Installation

```bash
git clone <this-repo>
cd AI-Driven-Image-Processing-Pipeline-with-Real-World-Object-Identification-And-AutomatedText-Generation

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verify the **CPU** build actually installed — a plain `pip install torch` pulls a
CUDA build on many platforms:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# expected: 2.4.1+cpu 0.19.1+cpu False
```

For a GPU machine, drop the `--extra-index-url` line from `requirements.txt` and
install the CUDA build from <https://pytorch.org/get-started/locally/>. The code
is device-aware and will use CUDA automatically when it is available.

## Running

```bash
streamlit run streamlit_app/app.py
```

Then open <http://localhost:8501>.

The first run downloads model weights and takes a few minutes. Subsequent runs
load from the local Hugging Face cache.

---

## Configuration

All settings are optional environment variables with sensible defaults; see
[`.env.example`](.env.example) for the full list and
[`config.py`](config.py) for validation rules. Nothing loads a `.env` file
automatically — these are read from the process environment.

| Variable | Default | Purpose |
|---|---|---|
| `DETECTION_THRESHOLD` | `0.5` | Minimum detection confidence (also a UI slider) |
| `MAX_UPLOAD_MB` | `10` | Upload size cap |
| `MAX_PIXELS_MP` | `50` | Decompression-bomb guard |
| `MAX_DESCRIPTIONS_PER_IMAGE` | `10` | Caps captioning cost per image |
| `MIN_CROP_PIXELS` | `32` | Both crop dimensions must meet this to be captioned |
| `MAX_CONCURRENT_INFERENCES` | `1` | Models are process-global and shared across sessions |
| `MAX_RUNS_PER_HOUR` | `10` | Per-session demo guard (see caveat below) |
| `PERSIST_ARTIFACTS` | `false` | Off: results are in-memory downloads, nothing is written |
| `LOG_LEVEL` | `INFO` | Structured log verbosity |

**Rate-limit caveat:** the limiter is **per browser session**, stored in
`st.session_state`. A new session resets it. It exists to stop a public demo
being casually used as free compute; it is not a server-side rate limit.

**Artifacts:** by default the app writes **nothing to disk** — the JSON and
annotated PNG are served as in-memory downloads. Setting `PERSIST_ARTIFACTS=true`
writes per-run directories under `data/runs/` with a bounded retention policy.

---

## Project structure

```
config.py                    Typed, validated, env-driven settings
pipeline.py                  Cached inference pipeline (the cache-identity boundary)
models/
  registry.py                Cached model loaders, device resolution, inference semaphore
  detection_model.py         DETR wrapper; owns the coordinate contract
  description_model.py       BLIP captioner
  text_extraction_model.py   EasyOCR wrapper; returns spans in original coordinates
utils/
  upload.py                  Validation, normalization, bomb guard
  compose.py                 Geometry, caption budgeting, OCR association, composition
  data_mapping.py            Pure payload construction and serialization
  artifacts.py               Export bytes; the ONLY module that touches the filesystem
  visualization.py           Annotated image rendering and export provenance mark
  rate_limit.py              Per-session demo guard
  logging_setup.py           Structured logging (run_id / compute_id scopes)
streamlit_app/app.py         Thin UI orchestration
tests/                       Tier 1 unit tests + Tier 2 `slow` integration tests
```

---

## Tests

```bash
pip install -r requirements-dev.txt

pytest -m "not slow"     # fast unit tests, no model downloads (~8s)
pytest -m slow           # integration tests; downloads weights on first run
pytest                   # everything
```

---

## Notable fix: correct object labels

An earlier version of this project reported **`hot dog`** at 99% confidence for a
photograph of two apples. Its own saved output recorded the failure:

```json
{ "label": "hot dog", "score": 0.9923,
  "description": "Define apple in the real world.\n\n…This website is run by
                  The Frugal Fawn…liking us on Facebook." }
```

Two independent defects, visible in one record:

1. **The label was wrong while the model was right.** DETR predicts over the
   sparse **91**-entry COCO category-id space, which contains `N/A` placeholders.
   The old code indexed those ids into a hardcoded **81**-entry *contiguous* list,
   so everything after the first gap shifted. Category id 53 is `apple` in the
   real space and lands on `hot dog` at index 53 of the contiguous list.
   The fix: read `model.config.id2label` and delete the hardcoded list.

2. **The description never saw the image.** A base (non-instruction-tuned)
   language model was prompted `"Define {label} in the real world."`, the prompt
   was never stripped from the output, and the token budget counted the prompt so
   text truncated mid-sentence. The fix: caption the actual object crop with BLIP.

Both are covered by regression tests in `tests/test_integration.py`.

---

## License

MIT
