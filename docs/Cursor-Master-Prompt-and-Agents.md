# Cursor Pro Cloud — Master Prompt & Agent Prompts
## Companion to `ASL-Bangla-System-Architecture.md` — read that file first for full rationale.

Paste **Part 17** into Cursor Pro Cloud as the top-level task. If Cursor supports spawning the sub-agents from Part 16 directly, paste each of the Part 18 prompts into its respective agent; otherwise, Part 17 alone is sufficient — it contains the full sequential/parallel plan.

---

## PART 17 — Cursor Master Prompt (copy-paste this whole block)

```
You are building a complete, working, production-quality-but-not-deployed system:
"Real-Time American Sign Language Recognition and English-to-Bangla Neural
Translation." You have 5 days. Total budget is $0 — no paid APIs, no paid
datasets, no paid cloud services. Do not ask unnecessary questions; make
reasonable engineering decisions yourself using the specification below.

============================================================
NON-NEGOTIABLE RULES
============================================================
1. Do NOT merely write code. EXECUTE the project: run every command, inspect
   real output/logs, run real tests, and only declare a step "done" after it
   has actually been run and verified. Writing code that "should work" is not
   done. A checklist item is only checked off after you produced and looked at
   the artifact that proves it (a log file, a test report, a metrics.json).
2. NEVER invent a dataset, model, checkpoint, URL, or license that isn't in
   the sources list below. If you discover during execution that something
   below is wrong or inaccessible, fall back per the decision tree in section
   "FALLBACK LOGIC" — do not silently substitute something you made up.
3. NEVER fabricate results. Never hardcode a fake accuracy number, never
   return random/mocked predictions from what looks like a real inference
   endpoint, never claim a test passed without having run it.
4. NEVER use a paid API, paid dataset, or paid cloud service. The translation
   model runs locally via `transformers`, not via a hosted API.
5. NEVER leave placeholder code (TODO, `pass  # implement later`, stub
   functions) in any of these critical paths: model inference, landmark
   extraction, sentence-state logic, translation. Placeholders are acceptable
   only in clearly-marked, non-critical, optional stretch features, and must
   be listed explicitly in your final report if any remain.
6. NEVER hardcode personal file paths or secrets. Everything configurable
   lives in config/config.yaml or environment variables. The only credential
   in this entire project is a free Kaggle API token (KAGGLE_USERNAME /
   KAGGLE_KEY) — nothing else requires auth.
7. Operate as a loop: BUILD → RUN → TEST → DETECT FAILURE → DIAGNOSE → FIX →
   RE-RUN → RE-TEST. Continue until every item in ACCEPTANCE CRITERIA below is
   actually satisfied and verified.
8. Work in parallel where genuinely independent (see AGENT PLAN). Do not
   create parallelism where a real dependency exists — respect the order
   given.

============================================================
PROJECT OBJECTIVE
============================================================
Build a web application that:
- Accepts live webcam ASL input and uploaded video as an alternative input.
- Extracts hand/pose/face landmarks via MediaPipe and classifies isolated ASL
  signs from the landmark sequence (NOT raw video classification — see
  ARCHITECTURE below for why).
- Recognizes signs from a vocabulary of at least 250 words, extended to
  450-520 words if time and data allow (see DATASETS and 5-DAY PLAN).
- Accumulates recognized signs into an English sentence with duplicate
  suppression, confidence thresholding, and debounce.
- Translates the recognized English into Bangla, both instantly per-word
  (precomputed dictionary) and as a fluent sentence (on-demand model call).
- Runs with real-time latency in webcam mode (browser-side landmark
  extraction, WebSocket streaming of landmarks — not video frames — to the
  backend for classification).
- Is production-quality in code (config-driven, tested, logged, versioned
  checkpoints, no hardcoded paths/secrets) but is NOT deployed anywhere.
- Is explicitly and honestly scoped as isolated-sign recognition with
  sequence accumulation — NOT full continuous ASL linguistic translation.
  State this plainly in the README and in the UI.

============================================================
ARCHITECTURE (already decided — do not re-litigate, implement this)
============================================================
Camera/Video → MediaPipe landmark extraction (browser WASM for live webcam;
server-side Python HolisticLandmarker for uploaded video) → normalize to a
~130-point subset (both hands full 21pt sets, ~12 upper-body pose points,
~76-point reduced face mesh), fixed to T=64 frames per sequence via
interpolation/padding with an attention mask → Conv1D(x2, 392→256) →
positional encoding → TransformerEncoder (4 layers, 4 heads, d_model=256,
ffn=512, dropout=0.1) → attention pooling → Linear classifier head → softmax
→ English gloss + confidence → temporal EMA smoothing + confidence threshold
(0.60) + cooldown debounce (800ms) + duplicate suppression → English sentence
buffer → (a) instant Bangla word from a precomputed dictionary cache, (b)
on-demand full-sentence Bangla via csebuetnlp/banglat5_nmt_en_bn.

IMPORTANT — MediaPipe API version: use the current MediaPipe Tasks API
(`mediapipe.tasks.python.vision.HolisticLandmarker` for server-side Python;
`@mediapipe/tasks-vision` for browser JS). The legacy `mediapipe.solutions.
holistic` API is deprecated by Google and some of its convenience functions
no longer exist in current mediapipe releases. If you find yourself about to
copy a tutorial that imports `mediapipe.solutions.holistic`, STOP and use the
Tasks API instead. If the installed mediapipe version genuinely lacks the
Tasks Holistic landmarker, fall back to separate HandLandmarker + PoseLandmarker
+ FaceLandmarker tasks (all three are stable in the Tasks API) rather than the
deprecated solutions module.

Model export: train in PyTorch, export to ONNX, serve via ONNX Runtime in the
FastAPI backend (not a full PyTorch runtime in the serving path).

============================================================
DATASETS (in priority order — verified accessible at time of writing)
============================================================
TIER 1a — PRIMARY, start here:
  Kaggle competition "asl-signs" (Google Isolated Sign Language Recognition).
  Download: `kaggle competitions download -c asl-signs` (requires free
  KAGGLE_USERNAME/KAGGLE_KEY env vars — this is the ONLY credential needed in
  the whole project). Data is ALREADY pre-extracted MediaPipe Holistic
  landmark parquet files (250-word vocabulary, ~210k+ examples) — do NOT
  re-extract landmarks from video for this source, just load the parquets.

TIER 1b — VOCABULARY EXTENSION, start downloading in parallel with 1a:
  ASL Citizen (Microsoft Research). Direct download, no auth:
  `wget https://download.microsoft.com/download/b/8/8/...` — locate the exact
  current URL from https://www.microsoft.com/en-us/research/project/asl-citizen/
  (the download page lists the current direct link; do not guess the URL,
  fetch it from the official page). ~42.8GB zip, raw video, 2,731 signs. You
  WILL need to run MediaPipe landmark extraction on this one yourself
  (parallelize across all available CPU cores — this is the single most
  compute-heavy step in the project, start it as early as possible on Day 1).
  License: Microsoft Research License (non-commercial research use — fine
  for this project since it is not being deployed/sold).

TIER 2 — FALLBACK if Tier 1a/1b become inaccessible:
  WLASL2000 video mirror with actual downloadable .mp4 files (not YouTube
  links): Hugging Face dataset `Voxel51/WLASL`. Download via `huggingface_hub`
  (free HF account, no payment). License inherited from WLASL's original
  C-UDA terms — non-commercial research use.

TIER 3 — INSTANT FALLBACK, no training required at all:
  Pretrained WLASL I3D checkpoints hosted on a public Hugging Face Space
  (search for "I3D_Sign_Language_Classification" on huggingface.co/spaces —
  verify the exact current space name before use, since community spaces can
  be renamed/removed; if the specific one referenced in the architecture doc
  is gone, search Hugging Face Spaces for an equivalent WLASL I3D checkpoint
  before falling further back). Includes a Kinetics-pretrained I3D backbone
  plus WLASL100 (100-class) and WLASL2000 (2000-class) fine-tuned heads. Use
  this ONLY as a fallback if Tier 1/2 fail — it is a heavier RGB-video model,
  not the landmark architecture, and should be wrapped behind the SAME
  inference API contract so the rest of the app doesn't need to change.

DO NOT put WLASL's raw YouTube-link-based download (the official
dxli94/WLASL repo's own video_downloader.py) on the critical path — its own
README documents significant link rot. Only use WLASL via the Tier 2/3 routes
above.

============================================================
TRANSLATION
============================================================
Model: `csebuetnlp/banglat5_nmt_en_bn` via Hugging Face `transformers`
(AutoModelForSeq2SeqLM + AutoTokenizer, use_fast=False). Install and use the
`normalizer` package (`pip install git+https://github.com/csebuetnlp/
normalizer`) to normalize text before tokenization, per the model card.
Runs on CPU, no API key, license cc-by-nc-sa-4.0 (non-commercial — fine here).

Build TWO translation paths:
1. A precomputed dictionary: after the final vocabulary is fixed, run every
   vocabulary word through the model ONCE, cache word→Bangla in
   `bangla_dictionary.json`. This is what powers instant per-word display.
2. An on-demand full-sentence call: when the user taps "Translate sentence,"
   send the full accumulated English string through the same model for a
   fluent Bangla sentence. Label this output differently from the
   word-by-word dictionary output in the UI — they are not the same thing and
   must not be presented as if they were.

============================================================
FALLBACK LOGIC (encode this literally as control flow, not just as a plan)
============================================================
IF Kaggle asl-signs download+load succeeds:
    PRIMARY_MODEL = train on 250-word vocab (guaranteed baseline, ship this
    no matter what happens next)
    IF ASL Citizen download + MediaPipe extraction complete with time to
    spare (check against the Day 3 checkpoint in the execution plan):
        EXTEND vocabulary to 450-520 words (union of Kaggle's 250 + top
        ASL-Citizen words by example count, deduplicated), retrain/fine-tune
        the SAME architecture, evaluate against held-out data
        IF extended model's held-out top-1 accuracy is not catastrophically
        worse than the 250-word model's (do not ship a much worse model just
        because it has more classes — log both, prefer the extended model
        only if it's reasonably competitive):
            SERVE the extended model
        ELSE:
            SERVE the 250-word model, log the extended attempt and its
            metrics for the final report, do not silently discard the work
    ELSE:
        SERVE the 250-word model as final; this still satisfies "several
        hundred signs" even though it misses the 400+ stretch target — do
        not treat this as failure, it is an explicitly acceptable outcome
ELSE (Kaggle access fails — auth issue, competition access changed, etc.):
    FALL BACK to the WLASL pretrained I3D checkpoint (Tier 3) as the served
    model immediately so SOMETHING real works; wrap it behind the identical
    FastAPI inference contract used everywhere else in the app
IF banglat5_nmt_en_bn fails to load or is too slow to be usable even for the
on-demand sentence call on available hardware:
    DISABLE the "Translate sentence" button, keep the precomputed dictionary
    path working, and label sentence output clearly as "word-by-word Bangla"
    — do not hide the limitation, state it in the UI
IF no GPU is available at all:
    This is EXPECTED and FINE for the primary landmark architecture (it's
    designed to be CPU-trainable). If you end up on the Tier 3 I3D fallback,
    restrict it to upload-video mode only on CPU and clearly disable/label
    live-webcam mode as unavailable in that fallback state — do not silently
    serve a broken/laggy live mode.

============================================================
EXECUTION ORDER / PARALLEL AGENT PLAN
============================================================
Run as follows (see the companion architecture doc, Part 16, for full detail
on responsibilities and produced artifacts per agent):

SEQUENTIAL FIRST STEP (fast, blocks everything):
  Agent 1 — Environment & Scaffolding: probe hardware (CPU/GPU/RAM/disk),
  create the full repo structure, set up Python 3.11 venv + Node environment,
  write the config/config.yaml skeleton with every path/hyperparameter/
  threshold mentioned in this prompt as an explicit, named field.

THEN, IN PARALLEL:
  Agent 2 — Data Acquisition: kick off both dataset downloads (Kaggle first
  since it's small and unblocks training fastest; ASL Citizen in the
  background since it's large), verify integrity of each as it completes.
  Agent 7 — Frontend scaffolding: can start building the React app shell,
  camera capture component, and UI panels against a MOCKED WebSocket response
  before the real backend/model exist — do not wait idle for other agents.
  Agent 8 — Testing scaffolding: set up pytest/Vitest structure and write
  tests for logic that doesn't depend on data yet (config loading, sentence-
  state debounce logic with synthetic inputs, landmark normalization math).

THEN, AS DATA BECOMES AVAILABLE:
  Agent 3 — Data Preprocessing: build vocabulary + splits from Kaggle data as
  soon as it's verified (don't wait for ASL Citizen); run MediaPipe
  extraction on ASL Citizen across parallel workers as soon as ITS download
  is verified.
  Agent 5 — Translation: fully independent of the above — implement the
  BanglaT5 wrapper and can be tested with placeholder English words
  immediately; build the real word-dictionary cache once Agent 3's final
  vocabulary is fixed.

THEN:
  Agent 4 — Model Training & Evaluation: implement model/train/evaluate code
  as soon as Agent 1 is done (code doesn't need data to exist to be written),
  actually run training as soon as Agent 3's processed 250-word tensors
  exist; re-run for the extended vocabulary later if that path is taken.
  Agent 6 — Backend/Inference: implement the FastAPI app and WebSocket
  contract as soon as Agent 1 is done (can be built/tested against a stub
  model), swap in the real ONNX model as soon as Agent 4 produces one.

FINALLY:
  Agent 9 — Integration/QA: run the ENTIRE pipeline from a clean checkout
  (`scripts/run_full_pipeline.sh`, then `run_backend.sh` and `run_frontend.sh`
  simultaneously), execute the full test suite, fix any integration bugs
  found, and only then evaluate the ACCEPTANCE CRITERIA below. This agent is
  not allowed to declare the project complete based on code review alone — it
  must have actually run the system.

============================================================
5-DAY SCHEDULE (see companion doc Part 15 for full detail/fallbacks)
============================================================
Day 1: environment + both downloads started + Kaggle-data baseline trained
       (even a rough/short-epoch model, to prove the pipeline works)
Day 2: full 250-word training + evaluation + ONNX export; ASL Citizen
       extraction running in parallel
Day 3: real-time inference pipeline (backend WebSocket + browser landmark
       extraction) wired end-to-end; vocabulary extension IF Day-2 extraction
       finished in time
Day 4: frontend/backend full integration, sentence accumulation, both
       translation paths, uploaded-video mode
Day 5: full test suite, bug fixing, latency measurement, README, final
       acceptance-checklist verification

============================================================
TESTING YOU MUST ACTUALLY RUN (not just write)
============================================================
- Unit tests: landmark normalization, sequence padding, vocabulary/label-map
  construction, sentence-state debounce/cooldown/duplicate-suppression logic,
  Bangla dictionary lookup, ONNX-vs-PyTorch output parity.
- Integration tests: WebSocket round-trip with a synthetic landmark window,
  REST upload-video endpoint, translation endpoint (both paths).
- End-to-end tests: at least one real recorded test clip through the FULL
  pipeline (camera/video → landmarks → model → sentence → Bangla).
- Model tests: actual top-1/top-5 accuracy, per-class precision/recall/F1,
  confusion matrix, "no-sign" class true-negative rate — REPORT the real
  numbers, do not assert an arbitrary pass/fail threshold you made up.
- Real-time tests: measure actual browser-side FPS, actual WebSocket+
  inference latency (p50/p95), actual end-to-end sign-to-word latency.
- Edge cases: no hands visible, very fast/slow signing, low light, repeated
  identical sign performed twice deliberately (must yield two words once a
  rest period is detected), empty/invalid uploaded file, WebSocket
  disconnect/reconnect mid-session.

Save all test/metrics output as artifacts under a `reports/` directory (JSON
or plain text), not just printed to a terminal that disappears.

============================================================
ACCEPTANCE CRITERIA (all must be true, and verified by an actual run/artifact)
============================================================
- Clean-checkout setup script succeeds with no manual steps.
- Both datasets (or the actually-reached fallback tier) are downloaded and
  integrity-verified — show the verification output.
- Preprocessing produces the documented tensor shapes and vocabulary size.
- At least one model checkpoint loads in both PyTorch and ONNX Runtime.
- Live webcam mode and uploaded-video mode both work end-to-end with REAL
  (never mocked/random) predictions.
- Bangla translation works via both the dictionary path and the on-demand
  sentence-model path (or the documented fallback if the model path was
  disabled per FALLBACK LOGIC).
- Sentence accumulation, duplicate suppression, remove-last, and clear all
  demonstrably work.
- Full test suite passes; actual accuracy and latency numbers are recorded in
  `reports/`.
- No hardcoded paths, no secrets in source, no paid dependency anywhere
  (grep for API keys / billing SDK imports as a final check), no leftover
  placeholder/TODO in a critical path.
- README documents exact setup/train/inference/evaluation commands, the
  final vocabulary size and which fallback tier was actually used, and the
  explicit honest-scope statement (isolated-sign recognition + accumulation,
  not full continuous ASL linguistic translation).

Do not report the project as complete until every item above is checked
against an actual artifact you produced by running the system, not by
inspection of code alone.
```

---

## PART 18 — Individual Agent Prompts

Use these if your Cursor setup lets you dispatch separate sub-agents. Each assumes the shared context (config.yaml, repo structure, architecture) from Part 17 is already established by Agent 1.

### Agent 1 — Environment & Scaffolding
```
Probe this machine: CPU core count, total RAM, GPU presence/model/VRAM (via
torch.cuda if available, else report "none"), free disk space. Write the
results to reports/environment.json. Based on this, set sane defaults in
config/config.yaml for batch size (128 default, drop to 32 if RAM is
constrained or no GPU), device selection, and number of parallel workers for
data preprocessing (default to CPU core count - 1). Create the full repo
structure exactly as specified in the architecture doc's Part 13. Set up a
Python 3.11 virtual environment and a Node environment for the frontend. Do
not proceed to write any ML/backend/frontend logic yet — this agent's only
job is a working, empty-but-structured project plus a working environment.
Verify by actually activating the venv and running `python --version` and
`node --version` inside it, and paste the real output into your report.
```

### Agent 2 — Data Acquisition
```
Download two datasets. (1) Run `kaggle competitions download -c asl-signs`
into data/raw/kaggle_asl_signs/ — this requires KAGGLE_USERNAME and
KAGGLE_KEY environment variables; if they are not set, stop and clearly
report that a free Kaggle account + API token is needed, do not fabricate
credentials or skip silently. (2) Fetch the current official ASL Citizen
direct-download URL from https://www.microsoft.com/en-us/research/project/
asl-citizen/ (do not guess the URL — read it off the live page) and `wget`
it into data/raw/asl_citizen/ in the background, since it is ~42.8GB and will
take a while; do not block other agents on this download. After each
download completes, run an integrity check: for Kaggle, confirm the parquet
file count matches train.csv row count; for ASL Citizen, confirm the
extracted file count is in the expected range for an 84k-video dataset. Log
results to reports/data_acquisition.json. If either download fails after 3
retries, report this clearly and hand off to the fallback tier described in
the master prompt's FALLBACK LOGIC section — do not silently continue as if
it succeeded.
```

### Agent 3 — Data Preprocessing
```
Once Kaggle data is verified: load the landmark parquet files, select the
~130-point landmark subset (both full hand sets, ~12 pose points, ~76-point
reduced face mesh — exact indices are your engineering decision, document
whichever MediaPipe landmark indices you select in config/config.yaml),
normalize (center on shoulder midpoint, scale by shoulder width), resize/pad
every sequence to T=64 frames with an attention mask for padded positions,
and cache as tensors. Build label_map.json from the 250-word vocabulary.
Build stratified 80/10/10 train/val/test splits, splitting by signer/
participant id where available to avoid person-specific memorization.
Once ASL Citizen is verified: run the current MediaPipe Tasks API
HolisticLandmarker (NOT the deprecated mediapipe.solutions.holistic module)
across all videos, parallelized across every available CPU core, extracting
the identical landmark subset/normalization/T=64 packaging so both sources
produce identical tensor shapes. This step is the heaviest compute in the
project — checkpoint progress periodically so a crash doesn't lose completed
work, and log estimated time remaining. Build the extended vocabulary
(union of Kaggle's 250 words + top ASL-Citizen words by example count,
deduplicated, capped at 450-520 total) once extraction is far enough along.
Apply class-balancing: oversample (via mirrored + temporally-cropped copies,
not naive duplication) any class below 15 examples up to that floor; drop
any class below 5 examples entirely and log exactly which classes were
dropped and why.
```

### Agent 4 — Model Training & Evaluation
```
Implement ml/model.py: Conv1D(x2, input_dim→256) → positional encoding →
TransformerEncoder(4 layers, 4 heads, d_model=256, ffn=512, dropout=0.1) →
attention pooling → Linear(256→num_classes). Implement ml/train.py:
AdamW, weight decay 0.01, cosine LR schedule with 5% warmup, base LR 3e-4 at
batch size 128 (scale linearly if batch size differs), label smoothing 0.1,
early stopping on val top-1 accuracy with patience 8, checkpoint best+last
every epoch, must support --resume from a checkpoint. Train first on the
250-word Kaggle-derived tensors as soon as Agent 3 produces them — do not
wait for ASL Citizen. Implement ml/evaluate.py: top-1, top-5, macro-F1,
per-class precision/recall, confusion matrix, all saved to reports/ as JSON/
image artifacts, not just printed. Implement ml/export_onnx.py and verify
numerically that ONNX Runtime output matches PyTorch output on a validation
batch within a small tolerance — this check must actually run and its result
recorded, not assumed. If/when the extended vocabulary tensors from Agent 3
become available, repeat training/evaluation for that vocabulary and compare
its held-out accuracy against the 250-word model before deciding which one
to serve, per the FALLBACK LOGIC in the master prompt.
```

### Agent 5 — Translation
```
Implement ml/translation/translate.py wrapping csebuetnlp/banglat5_nmt_en_bn
via Hugging Face transformers (AutoModelForSeq2SeqLM, AutoTokenizer with
use_fast=False), using the csebuetnlp normalizer package on input text before
tokenization per the model card. Verify it actually loads and produces
Bengali-script output for a handful of test English sentences — print and
save these test outputs to reports/translation_smoke_test.json. Implement
ml/translation/build_word_dictionary.py, which takes the final vocabulary
(from Agent 3's label_map.json once available) and runs each word through
the model exactly once, caching results to bangla_dictionary.json — this
must be run once as a build step, not per-request at serving time. If model
load time or per-call latency is high enough that it would be unusable for
an on-demand "translate sentence" button (measure this, don't guess), report
that clearly so the FALLBACK LOGIC (dictionary-only mode) can be applied.
```

### Agent 6 — Backend/Inference Service
```
Implement backend/main.py (FastAPI), backend/inference.py (ONNX Runtime
session wrapper — load whichever model.onnx is currently the "served" model
per config.yaml, do not hardcode a specific run name), backend/
sentence_state.py (debounce 800ms, confidence threshold 0.60, EMA smoothing
over the last 3 prediction windows, duplicate-adjacent suppression, remove-
last, clear), backend/landmark_server.py (server-side MediaPipe Tasks
HolisticLandmarker for uploaded-video mode — again, current Tasks API, not
the deprecated solutions module), and backend/schemas.py (Pydantic models
for every WebSocket message and REST payload). Expose a WebSocket endpoint
that accepts a landmark window and returns {word, confidence, top_k}, and a
REST endpoint that accepts an uploaded video file and returns a full
recognition result. You may build and test this against a small stub model
before Agent 4's real ONNX export exists, but must re-test against the real
model as soon as it's available — do not ship the stub. Never return a
random/hardcoded prediction from these endpoints under any circumstance.
```

### Agent 7 — Frontend
```
Build the React + Vite + Tailwind app: camera panel (getUserMedia, canvas
overlay), in-browser landmark extraction using @mediapipe/tasks-vision
(current Tasks API — HolisticLandmarker/HandLandmarker+PoseLandmarker+
FaceLandmarker running via WASM in the browser, not sent to the server as
video), a WebSocket hook streaming landmark windows to the backend and
receiving predictions, a recognition panel (current sign, confidence,
English word), a Bangla panel (translation, clearly distinguishing the
instant word-by-word result from the on-demand full-sentence result), a
sentence bar (accumulated English + accumulated Bangla, remove-last, clear,
manual punctuation controls), video-upload controls as an alternative input,
and a togglable debug overlay (FPS, inference latency, current model
version/vocab size, raw confidence, landmark count detected, connection
state). You may build and test this against a mocked WebSocket response
before the real backend exists, but must re-test against the real backend
once Agent 6 has it running. Keep the UI clean and functional — no
unnecessary complexity.
```

### Agent 8 — Testing
```
Write and continuously run: unit tests for landmark normalization, sequence
padding, vocabulary construction, sentence-state logic (debounce/cooldown/
duplicate suppression/remove-last/clear), Bangla dictionary lookup, and
ONNX-vs-PyTorch parity; integration tests for the WebSocket and REST
endpoints and the translation endpoints; end-to-end tests running at least
one real test clip through the full pipeline; explicit edge-case tests (no
hands visible, very fast/slow signing, low light, repeated identical sign
with a rest period in between, empty/invalid upload, WebSocket disconnect/
reconnect). Measure and record real latency numbers (browser FPS, WebSocket+
inference p50/p95, end-to-end sign-to-word latency) — do not estimate these,
actually measure them and save to reports/. Run the suite after every other
agent's major change, not just once at the end, and report failures back
with enough detail (actual error/traceback, not just "test failed") for
fixes to be made quickly.
```

### Agent 9 — Integration/QA (final, not parallel)
```
Starting from a clean checkout, run scripts/setup_env.sh, then the full data
pipeline, then run_backend.sh and run_frontend.sh simultaneously, then the
complete test suite (pytest + frontend tests). Manually exercise webcam mode
and upload mode at least once each and confirm real predictions and real
Bangla output appear. Walk through every line of the ACCEPTANCE CRITERIA in
the master prompt and check each one only against an actual artifact you
produced in this run (a log, a reports/ file, an observed UI behavior) — not
against code you merely read. Fix any integration-level bugs you find
(cross-agent contract mismatches, config drift, path issues) directly. Write
the final README covering setup/train/inference/evaluation commands, the
actual final vocabulary size and which fallback tier of DATASETS/FALLBACK
LOGIC was actually reached, the actual measured accuracy and latency numbers,
and the explicit honest-scope statement about isolated-sign recognition vs.
full continuous ASL translation. Only after all of this should you report
the project as complete.
```
