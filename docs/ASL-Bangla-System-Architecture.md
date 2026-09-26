# Real-Time ASL Recognition → English → Bangla Translation
## Complete Architecture & Execution Blueprint (5-Day, $0 Budget, Cursor Pro Cloud)

Research cutoff for dataset/model availability claims below: **September 27, 2026** (web-verified). Every load-bearing claim has a source URL. Nothing below is invented — where something could not be verified, it is explicitly marked as such.

---

## PART 1 — Executive Summary

**Chosen approach in one paragraph:** Skip raw-video 3D-CNN training entirely. Use **MediaPipe landmark extraction → a small Conv1D+Transformer sequence classifier**, trained first on a dataset that already ships pre-extracted landmarks (zero preprocessing risk), then extended with a second, larger, directly-downloadable video dataset to push vocabulary past 400 words. Translation to Bangla uses a small, free, locally-run Hugging Face model, with a pre-computed word-level cache for instant lookups and live model inference for full-sentence fluency. Everything runs as scripts Cursor can execute end-to-end, with pretrained checkpoints as an instant-working fallback at every stage so the system is never "broken" for more than a few hours.

**Why this and not raw video / 3D-CNN:** The single largest risk in this project is *not* modeling — it's **data acquisition**. The most famous ASL dataset (WLASL) depends on downloading individual clips from YouTube, and a large fraction of those links are now dead (confirmed by the dataset's own GitHub issues and README, which as of this dataset's last update still ships a `find_missing.py` script and a manual video-request form specifically because of this). Any plan that puts WLASL video-scraping on the critical path for Day 1 risks losing most of 5 days to link-chasing. Landmark-based recognition also happens to be the *correct* engineering choice for a real-time browser app anyway: landmarks are ~100–1000x smaller than video, MediaPipe runs the extraction in the browser itself (no server-side GPU needed for inference), and this is exactly the architecture Google used for its own production-oriented Kaggle competitions on this identical problem.

**Vocabulary:** 250 words guaranteed on Day 2 (from pre-extracted landmark data, no scraping), extended to a **450–520 word target** by Day 3 using a second dataset that is directly `wget`-able with no YouTube dependency at all.

**Translation:** `csebuetnlp/banglat5_nmt_en_bn` — a 247M-parameter T5 model fine-tuned specifically for English→Bangla, free, runs on CPU, no API key.

**Bottom line:** this plan trades "biggest famous dataset" for "dataset that will actually be sitting on disk on Day 1," and trades "novel research architecture" for "the exact architecture that already won $100,000 in prize money on this exact task in a Google-run competition." That is a deliberate, documented choice to maximize P(working system in 5 days).

---

## PART 2 — Current Research Findings (with sources)

### 2.1 WLASL (Word-Level ASL) — the "obvious" choice has a real accessibility problem
- WLASL provides only a JSON file of YouTube (and some non-YouTube) video URLs + timestamps, not the videos themselves. The official repo's own README states videos "can disappear over time due to expired urls," ships a `find_missing.py` helper, and offers a manual request form with up to 7-day turnaround for missing clips. (github.com/dxli94/WLASL, dxli94.github.io/WLASL)
- A user issue on the repo reports downloading only 3,000/20,000 clips in 5 hours with most URLs returning "missing" (github.com/dxli94/WLASL/issues/36) — this is exactly the failure mode the project brief warns against.
- **Mitigation found:** a community mirror, `Voxel51/WLASL` on Hugging Face Hub, hosts the **actual `.mp4` files** (not links) for the WLASL video-classification dataset (huggingface.co/datasets/Voxel51/WLASL) — directly downloadable with `huggingface_hub`, no YouTube dependency, no auth beyond a free HF account. License tag on the HF card is `other` (i.e., inherits WLASL's original C-UDA research-only terms) — non-commercial research use only.
- **Pretrained checkpoints exist and are directly downloadable:** the official WLASL authors released pretrained **I3D** and **Pose-TGCN** checkpoints. A public, still-live Hugging Face Space mirrors the exact I3D weights, including a Kinetics-pretrained RGB backbone (`rgb_imagenet.pt`) plus two fine-tuned heads: **WLASL100** (top-1 65.89%, top-5 84.11%, top-10 89.92%) and **WLASL2000** (top-1 32.48%, top-5 57.31%, top-10 66.31%) (huggingface.co/spaces/cedssama/I3D_Sign_Language_Classification, mirrored at Gyufyjk/I3D_Sign_Language_Classification). These can be `git clone`d or fetched via LFS directly — **zero training required** for an instant, if modest-accuracy, 2000-class demo.
- Known data-density problem even if fully downloaded: WLASL2000's classes have a **median of only ~6 clips per gloss**, which is very sparse for training a robust classifier from scratch — independently reported by two unrelated projects analyzed during this research (github.com/KISHORE-KUMAR-S/sign-language-translator; github.com/streamsl/islr-mlops).

### 2.2 ASL Citizen (Microsoft Research) — best video-based option found
- 84,000 videos across **2,731 distinct ASL signs**, crowdsourced from everyday Deaf signers under IRB approval, framed by Microsoft as a dictionary-retrieval task with published baselines improving ISLR accuracy from ~32% to ~62% (microsoft.com/en-us/research/project/asl-citizen/).
- **Directly downloadable, no scraping:** a single `ASL_Citizen.zip` (42.8 GB) via Microsoft's official download center, `wget`-able from the command line, no YouTube dependency (microsoft.com/download/details.aspx?id=105253).
- License: **Microsoft Research License Terms** — non-commercial, non-revenue-generating research use only; commercial use requires contacting `ASL_Citizen@microsoft.com`. Fine for this project (explicitly not deployed), not fine if the person later wants to sell it.
- **This dataset directly solves requirement #13** (vocabulary well above 400) on its own, and solves it with zero link-rot risk.

### 2.3 MS-ASL (Microsoft) — looks similar to WLASL, has the same weakness
- 1,000-word vocabulary (16,054 train / 5,287 dev / 4,172 test clips) (service.tib.eu/ldmservice/dataset/msasl).
- The file Microsoft hosts directly is only **1.9 MB** (microsoft.com/en-us/download/details.aspx?id=100121) — this is metadata/URLs only, not video. Actual clips still require downloading from YouTube via a separate script (confirmed pattern in github.com/gulvarol/bsl1k, which ships `download_msasl.py` alongside `download_wlasl.py`). **Same link-rot risk as WLASL — not preferred as a primary source.**

### 2.4 PopSign ASL v1.0 / Google "Isolated Sign Language Recognition" Kaggle competition — best *fast-start* option found
- 250 isolated ASL signs, **210,000+ examples**, collected via smartphone selfie camera from 47 Deaf adult signers, licensed **CC-BY 4.0** at its canonical host (signdata.cc.gatech.edu), described in the NeurIPS 2023 Datasets & Benchmarks track (neurips.cc/virtual/2023/poster/73414).
- Google ran this exact dataset as a **$100,000-prize Kaggle competition** ("Isolated Sign Language Recognition," `kaggle.com/competitions/asl-signs`) and — critically — **released the data pre-processed as MediaPipe Holistic landmark parquet files**, not raw video. This means the single most time-consuming pipeline stage (running MediaPipe over hundreds of thousands of clips) is **already done** for this vocabulary.
- Multiple public, documented open-source solutions exist (bronze-medal and above), including exact input feature specs: **~130 selected landmarks** — 21 keypoints × 2 hands, ~6 pose points per arm, and ~76 reduced face-mesh points (lips/eyes/nose) — fed into a **Conv1D + Transformer** classifier (~1.7M parameters), reported reaching ~82–86% validation accuracy with a baseline LSTM per the paper's own benchmark (element14 community writeup; github.com/abhinand5/isolated-sign-language-recognition; github.com/nlevites/signchat; neurips.cc/virtual/2023/poster/73414).
- **Licensing caveat (must be disclosed, not hidden):** standard Kaggle competition rules typically restrict *competition-hosted* data to "the purpose and duration of the Competition" unless the organizer separately opens it — this is a generic Kaggle policy pattern, not something specific to this dataset that could be independently confirmed for post-competition use of the *Kaggle-hosted copy* specifically. The **underlying PopSign ASL v1.0 dataset is independently CC-BY 4.0 licensed and hosted by Georgia Tech/NTID** outside Kaggle (per the NeurIPS supplement), which is the clean path if strict compliance matters. Practically: thousands of public GitHub repos already use the Kaggle-hosted competition data for study/research well after competition close with no enforcement action reported; this is treated here as low legal risk for a non-commercial student project but is called out explicitly in the Risk Register (Part 21).

### 2.5 How2Sign, AUTSL, ASLLVD — investigated, deliberately excluded from the critical path
- **How2Sign**: continuous (not isolated-word) ASL with English translation, multi-view + depth modalities, tens of hours, license requires accepting research terms (how2sign.github.io). Excellent for *continuous* sign translation research, but far too heavy (multi-modal, multi-GB per clip, continuous segmentation is a much harder problem) to be safely on a 5-day critical path. Recommended only as **future work**, not part of the MVP.
- **AUTSL**: Turkish Sign Language, not ASL — relevant only as a transfer-learning/pretraining source for general sign-recognition backbones, never as a source of ASL labels. Excluded.
- **ASLLVD** (Boston University): a long-standing, well-known ASL lexicon video dataset. Based on its long-stable structure, bulk video access has historically required a direct request/agreement with BU rather than instant bulk download. Not independently re-verified as instantly downloadable within a 5-day window during this research pass — **excluded from the critical path** on the "accessible over famous" principle. If independently confirmed accessible, it is a reasonable *additional* vocabulary source later.

### 2.6 Hugging Face model search — what already exists that we don't have to build
- Pretrained WLASL **I3D** checkpoints (100-class and 2000-class heads + Kinetics backbone) — see 2.1. Usable as an instant RGB-based fallback demo.
- No turnkey "ASL landmark → English gloss" *general-purpose* pretrained classifier for an open vocabulary was found that isn't tied to one of the specific datasets above — this is expected; isolated sign recognition models are essentially always trained per-vocabulary, which is why this plan trains a small model rather than searching further for a magic universal checkpoint.
- **Translation:** `csebuetnlp/banglat5_nmt_en_bn` — BanglaT5 checkpoint fine-tuned specifically on the BanglaNMT English→Bengali parallel corpus (2.75M sentence pairs), 247M params, license `cc-by-nc-sa-4.0` (huggingface.co/csebuetnlp/banglat5_nmt_en_bn). The companion paper's own benchmark table shows BanglaT5 outperforming mT5-base, mBART-50, IndicBART, and XLM-ProphetNet on the BanglaNMT test set (SacreBLEU 38.8 vs. 36.6/23.6/22.7/23.3 respectively) (github.com/csebuetnlp/BanglaNLG). This is both the most accurate *and* the smallest of the compared options — an unusually clean win.
- **MediaPipe API note (important, current):** the legacy `mediapipe.solutions.holistic` Python API is being phased out in favor of the new **Tasks API** (`HolisticLandmarker`, or separately `HandLandmarker`+`PoseLandmarker`+`FaceLandmarker`). Google's own maintainers have stated in GitHub issues that "support for legacy Holistic solution is completely ended" in favor of the new Task API, and some legacy convenience functions (e.g., `solutions.drawing_utils`) are no longer available in current `mediapipe` releases (github.com/google/mediapipe/issues/5353; github.com/google-ai-edge/mediapipe/issues/6206). **Many older public tutorials still use the deprecated API** — Cursor must be explicitly instructed to use the Tasks API (with a version-checked fallback), or it will likely copy a broken pattern from an old tutorial.

---

## PART 3 — Dataset Comparison Table

| Dataset | Vocab size | Video or landmarks | Labels | Access method | Auth needed | License | Size | Training difficulty | Real-time fit | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **PopSign ASL v1.0 (Kaggle `asl-signs`)** | 250 | **Pre-extracted MediaPipe landmark parquet** | English gloss | Kaggle API download | Free Kaggle account + API token | CC-BY 4.0 (canonical GT host); Kaggle competition terms caveat on the Kaggle-hosted copy | ~2–4 GB (landmarks only) | **Lowest** — no video processing needed | Excellent (landmark model = the target architecture) | **Tier 1 — primary, Day 1–2** |
| **ASL Citizen (Microsoft)** | 2,731 | Raw RGB video | English gloss | Direct `wget` .zip from Microsoft | None | Microsoft Research License (non-commercial) | 42.8 GB | Medium — needs MediaPipe extraction at scale, but no scraping | Good after extraction | **Tier 1 — vocabulary extension, Day 2–3** |
| **WLASL2000 (HF mirror `Voxel51/WLASL`)** | 2,000 | Raw RGB video (actual `.mp4`s, not links) | English gloss | `huggingface_hub` download | Free HF account | `other` (inherits C-UDA, non-commercial) | ~10 GB (12k clips) | Medium, but median ~6 clips/class — sparse | Fair | **Tier 2 — supplementary / pretrained-checkpoint fallback** |
| **WLASL pretrained I3D (HF Space)** | 100 or 2000 | N/A — checkpoint only | English gloss | `git clone`/LFS from HF Space | None | Inherits WLASL C-UDA | ~50–110 MB | **None — already trained** | Fair (RGB model, heavier than landmark model) | **Tier 3 — instant fallback demo** |
| **MS-ASL** | 1,000 | Metadata only; video via YouTube | English gloss | JSON download direct; video via scraping | None for metadata | Research use | 1.9 MB metadata + scraped video | High — same link-rot risk as WLASL | Fair | **Excluded from critical path** |
| **How2Sign** | Continuous (not word-level) | Multi-view RGB + depth | English sentences | Signed license agreement | Yes (agreement) | Research | Very large (100s GB across views) | Very high — continuous segmentation + huge multimodal size | Poor fit for 5-day MVP | **Excluded — future work** |
| **AUTSL** | N/A (Turkish SL) | RGB + depth | Turkish gloss | Kaggle/ChaLearn request | Yes | Research | Large | N/A — wrong language | **Excluded — not ASL** |
| **ASLLVD** | ~3,000 (lexicon) | RGB video | English gloss | Historically request-based with BU | Likely yes | Research | Unknown, not re-verified this pass | Not independently confirmed | Not evaluated | **Excluded — accessibility not re-confirmed in time budget** |

---

## PART 4 — Model/Architecture Comparison

| Option | Accuracy potential | Latency | Training time (5-day budget) | Dataset fit | Engineering complexity | GPU need | Real-time suitability | Automation ease | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **A. Raw RGB → 3D CNN/Video Transformer** | High (with enough data) | High compute per frame | Days per serious run; infeasible from scratch in 5 days | Needs video, not landmarks | High | Strong GPU required | Poor (heavy per-frame cost) | Hard to fully automate reliably | Rejected as primary |
| **B. MediaPipe landmarks → temporal model (GRU/LSTM/Transformer)** | Proven 80%+ on 250-class benchmark; competitive on larger vocab with enough examples/class | Very low (landmarks are ~100s of floats, not pixels) | Hours, not days, even on CPU | **Matches Tier-1 dataset exactly (pre-extracted)** | Low–medium | None required (CPU-trainable at this scale) | Excellent | Easy to fully automate | **Selected** |
| **C. Hybrid RGB + landmarks** | Potentially highest | Medium-high | Multi-day | Needs both modalities aligned | High | GPU strongly preferred | Medium | Hard | Rejected — complexity not justified by 5-day gain |
| **D. Pretrained video model → fine-tune** | Medium (I3D 2000-class caps at ~32% top-1 out of the box) | Medium (3D CNN inference cost) | Low if used as-is; medium if fine-tuned | Best with WLASL | Low if used as a fallback only | GPU preferred for fine-tuning | Fair | Easy as a fallback | **Selected as Tier-3 fallback only**, not primary |
| **E. Existing pretrained landmark model → adapt** | N/A — no general-purpose open checkpoint found for this exact task/vocab | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Not applicable — no such checkpoint exists; closest thing is training B from the Kaggle-style recipe, which is what we do |

**Decision: Option B**, using the exact feature representation and model family (Conv1D+Transformer over ~130 MediaPipe landmarks) that public solutions to Google's own $100K competition on this precise task already validated. Option D's checkpoints are retained purely as an always-available fallback (Part 6), not the primary path.

---

## PART 5 — Final Architecture Decision

**Primary pipeline:**
`Camera / uploaded video → MediaPipe landmark extraction (browser-side via WASM for live, server-side Python for uploaded video) → fixed-length landmark sequence → Conv1D+Transformer classifier → English gloss + confidence → temporal debounce/segmentation → sentence buffer → (a) instant dictionary lookup for per-word Bangla, (b) BanglaT5 model call for full-sentence Bangla → UI.`

**Why this is the safest route for a 5-day deadline, explicitly:**
1. **No YouTube dependency on the critical path.** Both Tier-1 datasets (Kaggle landmarks, ASL Citizen video) are directly downloadable with a single command each. WLASL is demoted to a fallback specifically because of its documented link-rot problem.
2. **The hardest pipeline stage (video → landmarks) is pre-solved for the first 250 words.** Day 1 can go straight to training a real model instead of fighting MediaPipe batch extraction.
3. **The model itself is small** (~1–5M params) — trains in minutes-to-low-hours on CPU, faster on any GPU Cursor Cloud happens to expose. No risk of "still training on day 4."
4. **Landmarks are the natural fit for a browser real-time app.** Running MediaPipe client-side and streaming ~100s of floats per frame over WebSocket is dramatically lower-latency and lower-bandwidth than streaming video frames to a server for server-side vision inference.
5. **A working fallback exists at every layer** (Part 6), so "the ambitious path is behind schedule" never means "nothing works" — it means "we're serving the previous tier's checkpoint."

---

## PART 6 — Fallback Architecture (explicit decision logic for Cursor)

```
IF Kaggle "asl-signs" landmark data downloads and trains successfully (Day 1-2):
    → PRIMARY model = Conv1D+Transformer on 250-word vocab. Ship this as the guaranteed baseline.
    IF ASL Citizen also downloads and MediaPipe extraction completes in time (Day 2-3):
        → EXTEND vocabulary: fine-tune / retrain the SAME architecture on the union
          of the 250-word set + top-frequency ASL Citizen words not already covered,
          targeting 450-520 total classes (Tier 1 stretch, satisfies "well above 400").
    ELSE:
        → SHIP the 250-word model as final. This still satisfies "several hundred
          useful signs" (requirement 11) even though it misses the stretch target.
ELSE IF Kaggle landmark data is inaccessible (auth/API failure):
    → FALL BACK to WLASL100 pretrained I3D checkpoint from the HF Space mirror
      (no training required, top-1 65.89%). Vocabulary = 100 words. Wrap it in the
      same FastAPI inference contract so the rest of the app is unaffected.
    IF time remains after this fallback is live:
        → ATTEMPT WLASL2000 pretrained checkpoint (top-1 32.48%, weaker but 2000
          words) as an additional selectable "large vocabulary / lower accuracy" mode.
IF the translation model (banglat5_nmt_en_bn) fails to load or is too slow on
available hardware:
    → FALL BACK to the precomputed word-level EN→BN dictionary only (no live
      sentence-level model call); disable the "fluent sentence" button and label
      sentence output as "word-by-word Bangla" instead of hiding the feature.
IF Cursor Cloud exposes no GPU at all:
    → Landmark model: unaffected (CPU-trainable by design).
    → If the WLASL I3D fallback is needed, run it in inference-only mode on CPU
      (slower per-clip, acceptable for uploaded-video mode; disable it for live
      webcam mode and show a clear "GPU not available, live I3D mode disabled"
      message rather than silently degrading).
```

This logic must be encoded literally in the Cursor master prompt (Part 17) as an if/else decision tree the agents execute, not just documentation.

---

## PART 7 — Complete System Architecture

**Components:**
1. **Frontend (React + Vite)** — camera capture, MediaPipe Tasks-Vision (WASM) running *in the browser* for live landmark extraction, video upload widget, recognition/Bangla/sentence panels, debug overlay.
2. **Input/camera layer** — `getUserMedia` webcam stream; file input for uploaded video (decoded client-side or sent to backend for server-side MediaPipe processing).
3. **Video/frame processing** — live: browser WASM MediaPipe per frame. Uploaded video: server-side Python MediaPipe Tasks (`HolisticLandmarker`), frame-sampled.
4. **Landmark extraction** — ~130-point subset (2×21 hand, ~12 pose, ~76 face) per frame, (x, y, z) — see Part 9 for exact spec.
5. **Feature extraction / normalization** — center on torso/shoulder-midpoint, scale by shoulder width, handle missing-hand frames with learned mask token or zero-fill + visibility flag.
6. **Temporal modeling** — Conv1D stem (local temporal smoothing) → Transformer encoder (global temporal attention) → attention pooling.
7. **Sign classifier** — linear head over pooled representation → softmax over vocabulary.
8. **Temporal segmentation/debouncing** — rest-position detector for sign boundaries in live mode; confidence-margin + cooldown debounce before committing a word.
9. **English sequence builder** — accumulates committed words into an ordered buffer.
10. **English normalization** — gloss→natural-word mapping table (e.g., merge inflectional variants recorded in the dataset label files), duplicate-adjacent suppression.
11. **Bangla translation** — dictionary cache (instant) + BanglaT5 model call (sentence-level, on demand).
12. **API/service layer** — FastAPI: REST for uploaded-video jobs + static config; WebSocket for live landmark streaming and low-latency prediction push.
13. **Model storage** — versioned checkpoint directory (`models/<name>/<version>/model.onnx` + `labels.json` + `metrics.json`).
14. **Configuration** — single `config.yaml` (+ environment overrides) covering dataset paths, model hyperparameters, thresholds — no hardcoded paths anywhere else.
15. **Logging** — structured JSON logs (request id, latency, confidence, model version) via Python `logging` + Uvicorn access logs.
16. **Testing** — `pytest` suite (unit + integration) — see Part 19.
17. **Training pipeline** — scripted, resumable, checkpointed (Part 9).
18. **Evaluation pipeline** — held-out split, top-1/top-5, per-class confusion matrix, latency benchmark script.

**Data flow:** `Camera/Video → Frames → Landmarks → Normalized feature sequence → Conv1D+Transformer → Sign (English gloss + confidence) → Debounce/segment → Sentence buffer → English normalization → Bangla dictionary/model → UI render.`

---

## PART 8 — Data Pipeline

**Stage 1 — Acquire (parallelizable across two sources):**
- `kaggle competitions download -c asl-signs` (requires `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars — the only "credential" needed anywhere in this project, and it's a free account, not a paid one). Produces per-clip landmark parquet files + `train.csv` (sequence id → sign label) + `sign_to_prediction_index_map.json`.
- `wget` the ASL Citizen `.zip` from the Microsoft download URL directly; unzip; read the accompanying label CSV/splits.

**Stage 2 — Verify integrity:** checksum/row-count sanity checks (parquet row counts vs. `train.csv` entries; zip file count vs. published 84k for ASL Citizen); log and hard-fail with a clear message rather than silently continuing on a truncated download.

**Stage 3 — Preprocess:**
- Kaggle data: already landmarks — just load, select the ~130-point subset, normalize, pad/resize each sequence to a fixed length `T` (default 64 frames; configurable), cache as `.npy`/`.pt` tensors sharded by split.
- ASL Citizen: run MediaPipe `HolisticLandmarker` over each video (parallelized across CPU cores — this step is the main compute cost of the whole project and should be sharded across Cursor's parallel agents), extract the same ~130-point subset, same normalization, same fixed-length packaging, so both sources produce **identical tensor shapes** and can share one training pipeline.

**Stage 4 — Vocabulary construction:** union the 250 Kaggle words with the highest-example-count ASL Citizen words not already present, capped at the compute/time-budget-adjusted target (450–520 default), producing a single `label_map.json` (gloss → normalized English word → class index).

**Stage 5 — Splits:** stratified 80/10/10 train/val/test **per signer** where signer identity is available (ASL Citizen), to avoid a model that memorizes a specific person's motion style rather than the sign itself; fall back to stratified-by-example split where signer id isn't available (Kaggle set does include participant id and should also be split by signer).

**Stage 6 — Class balancing:** oversample classes below a minimum example count (configurable floor, default 15) up to the floor via temporal-crop + mirroring augmentation (see Part 9); drop classes below an absolute minimum (default 5 examples) rather than training on unlearnable classes — log every dropped class explicitly.

**Stage 7 — Caching:** processed tensors cached to disk keyed by a hash of (source, landmark-subset config, `T`) so re-running preprocessing after a config tweak doesn't redo untouched work.

---

## PART 9 — Model Pipeline

- **Backbone:** Kinetics/ImageNet pretraining is irrelevant here (input is landmarks, not pixels) — the model is trained from scratch, which is fine because it is small and the data volume (hundreds of thousands of sequences) is adequate for a ~1–5M parameter model.
- **Input representation:** per frame, 130 landmarks × 3 coordinates (x, y, z) = 390 features, plus a binary visibility/presence flag per hand (2 extra features) = **392-dim per frame**.
- **Landmark subset (default, overridable in config):** both hands full 21-point sets (42 points), 12 upper-body pose points (shoulders, elbows, wrists, hips ×2 sides ×... configurable to exactly reproduce the ~76-face/~130-total split reported by the reference solutions), and a reduced ~76-point face-mesh subset (lips, eyes, eyebrows, nose) — face is included because mouth/eyebrow shape is linguistically meaningful in ASL (non-manual markers), but is the first thing to drop if latency becomes a problem.
- **Sequence length:** fixed `T=64` frames via linear-interpolation resize for sequences longer than 64, and edge-padding + attention mask for sequences shorter than 64 (do not zero-pad without masking — the model must ignore pad positions).
- **Architecture:** `Conv1D(k=3) ×2 (temporal smoothing, channel 392→256) → positional encoding → TransformerEncoder (4 layers, 4 heads, d_model=256, dim_feedforward=512, dropout=0.1) → attention pooling over time → LayerNorm → Linear(256→num_classes)`.
- **Normalization:** per-frame, translate so the midpoint between the two shoulder landmarks is the origin; scale so shoulder-to-shoulder distance = 1.0. This makes the model invariant to camera distance and roughly invariant to body size.
- **Augmentation:** horizontal mirroring (re-run landmark left/right index swap, *not* naive x-negation, since left/right hand landmarks are semantically distinct — mirroring must swap which landmark array is "left hand" vs "right hand"); temporal random-crop/speed jitter (±15%); small per-frame Gaussian jitter on coordinates (σ≈0.01 in normalized units) for robustness to landmark-detector noise.
- **Class balancing:** weighted random sampler inversely proportional to (post-oversampling) class frequency, in addition to the oversampling in Part 8.
- **Loss:** cross-entropy with label smoothing (0.1).
- **Optimizer:** AdamW, weight decay 0.01.
- **LR schedule:** cosine decay with 5% linear warmup; base LR 3e-4 for batch size 128 (linearly scale LR with batch size if hardware forces a smaller batch).
- **Batch size:** 128 default; drop to 32–64 automatically if CPU-only or low-memory GPU is detected (Cursor must probe available RAM/VRAM first and write the chosen value into the run's config snapshot for reproducibility).
- **Epochs:** up to 60, with **early stopping** on validation top-1 accuracy (patience 8 epochs).
- **Checkpointing:** save best-by-val-accuracy and last-epoch checkpoints every epoch; training must be resumable from the last checkpoint if interrupted.
- **Evaluation metrics:** top-1, top-5, macro-F1, per-class precision/recall, confusion matrix (saved as an artifact, not just printed).
- **Inference/confidence threshold:** default top-1 softmax probability ≥ 0.60 to accept a prediction as "recognized"; below that, UI shows "uncertain" rather than a wrong word.
- **Temporal smoothing:** exponential moving average of the last 3 prediction-window softmax vectors before thresholding, to reduce single-frame-window flicker.
- **Debouncing/repetition suppression:** once a word is committed to the sentence buffer, suppress committing the *same* word again for a cooldown window (default 800 ms) even if the model keeps predicting it, so holding a sign doesn't spam the sentence.
- **Unknown/no-sign handling:** an explicit "no-sign / rest" class trained on segments of the signer's hands at rest (synthesize from the gaps between clips / low-motion segments) so the model has a way to say "nothing is happening" instead of forcing a guess.
- **Low-confidence behavior:** show the top-3 candidates with scores in debug mode; in normal mode, show "…" and do not add anything to the sentence buffer.
- **Sign segmentation strategy (live mode):** hybrid — (a) a lightweight motion-energy detector on wrist landmarks to find likely start/end of a signing motion (silence → motion → silence), combined with (b) a fixed sliding window (default 1.0–1.5s, 50% overlap) as a fallback when motion segmentation is ambiguous (continuous signing with little pause).
- **Sequence accumulation logic:** see Part 11.

---

## PART 10 — Real-Time Inference Pipeline

**Live webcam path:** browser captures frames → MediaPipe Tasks-Vision (WASM, runs in-browser) extracts the ~130-point landmark subset per frame at the browser's native frame rate → landmark vectors are batched into the current sliding window client-side → the window is sent over an open **WebSocket** to FastAPI as compact JSON/binary (not video) → backend runs the ONNX-exported classifier (CPU-friendly, sub-10ms per window on typical hardware for a model this size) → backend pushes back `{word, confidence, top_k}` over the same socket → frontend updates the recognition panel, runs debounce/segmentation logic, and updates the sentence buffer.

**Why landmarks are computed client-side, not server-side, for live mode:** it removes video-frame transport from the latency budget entirely (only small landmark arrays cross the network), it parallelizes naturally (every client does its own extraction, no server GPU/CPU contention), and it matches exactly how the reference Kaggle-competition solutions were designed to run on-device.

**Uploaded video path:** file is sent to a REST endpoint → server-side Python `HolisticLandmarker` processes the whole file → the same classifier runs over the extracted sequence(s) → results (word-by-word with timestamps) are returned as a single response (or streamed via Server-Sent Events for longer videos so the UI can show progress).

**Latency budget (targets, to be measured and reported honestly, not assumed):** browser landmark extraction < 30 ms/frame on typical laptop hardware (MediaPipe's own published performance envelope for Holistic-class models); WebSocket round-trip + ONNX inference < 50 ms per window; end-to-end "sign performed" → "word shown" target **under ~500 ms** for the debounce-committed case. These are **targets to validate empirically** in Part 19, not guarantees — the acceptance checklist requires the actual measured numbers to be reported, not assumed to pass.

---

## PART 11 — Sentence Construction

`sign → (confidence ≥ threshold?) → (different from last committed word, or cooldown elapsed?) → append to English buffer → normalize (gloss→word mapping, e.g. collapse plural/inflection variants the dataset may label separately) → on "Translate" action or automatically every N committed words: run Bangla translation.`

- **Duplicate suppression:** adjacent identical words are collapsed unless the cooldown window has fully elapsed and the sign was clearly re-performed (motion-energy detector saw a rest period in between).
- **Temporal smoothing:** see Part 9 (EMA over recent prediction windows before a commit is even considered).
- **Minimum confidence threshold:** configurable, default 0.60 (Part 9).
- **Word cooldown/debounce:** default 800 ms (Part 9).
- **Sentence buffer:** ordered list of `{word, confidence, timestamp}`; exposed to the frontend as both the raw list and a joined string.
- **Correction/removal of last word:** explicit "⌫ remove last" control, backed by a simple pop-from-buffer call — no ML involved, must be instant.
- **Clear/reset sentence:** explicit control, clears buffer and resets debounce state.
- **Punctuation/end-of-sentence handling:** a manual "." / "?" / "!" control the user taps; **not** inferred from ASL (ASL punctuation-marking is a non-manual, prosodic phenomenon out of scope for an isolated-word classifier — this limitation is stated in the UI, not hidden).
- **Translation trigger logic:** (a) instant per-word Bangla gloss from the precomputed dictionary as each word is committed (near-zero latency), and (b) an explicit "Translate sentence" button that sends the full accumulated English string to BanglaT5 for a fluent, grammatically-adjusted Bangla sentence (a word-by-word dictionary concatenation is *not* a grammatical sentence, and the UI must not claim it is — label the two outputs separately: "Bangla (word-by-word)" vs. "Bangla (translated sentence)").

**Explicit scope statement (per project requirement — do not overclaim):** this system performs **isolated-sign recognition with sequence accumulation**, not continuous ASL-to-English linguistic translation. ASL grammar (non-manual markers, classifiers, spatial referencing, topic-comment structure) is not modeled. The output is a sequence of recognized English words in signing order, which is a genuinely useful and honestly-scoped deliverable — it is not full linguistic machine translation of ASL discourse, and the UI/README must say so plainly.

---

## PART 12 — Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 (backend/training), TypeScript (frontend) | Wide library support, MediaPipe Python bindings, Cursor's strongest languages |
| ML framework (training) | PyTorch | Flexible for the custom Conv1D+Transformer; easy export to ONNX |
| Inference runtime | ONNX Runtime (CPU-optimized) | Framework-agnostic, fast CPU inference, no TF/PyTorch runtime needed in the serving container |
| Landmark extraction | MediaPipe Tasks-Vision — **JS/WASM in-browser for live**, Python `HolisticLandmarker` for server-side uploaded-video processing | Current, non-deprecated API (Part 2.6); in-browser extraction removes video transport from the latency path |
| Backend framework | FastAPI + Uvicorn | Async, native WebSocket support, good typing, fast to scaffold |
| Realtime transport | WebSocket (native, via FastAPI) | Lower overhead than polling REST for a continuous landmark stream |
| Frontend framework | React + Vite | Fast dev loop, huge ecosystem, easy webcam/canvas integration |
| Styling | Tailwind CSS | Fast to build a clean, functional UI without hand-rolled CSS |
| Translation model | Hugging Face `transformers` running `csebuetnlp/banglat5_nmt_en_bn` | Free, local, best measured accuracy among compared options (Part 2.6) |
| Dataset download | `kaggle` CLI, `wget`, `huggingface_hub` | Matches each source's actual access method (Part 3) — no invented download paths |
| Storage (metadata/config) | SQLite (if any server-side persistence is needed for run history) + JSON/YAML config files | Lightweight, no external DB service, zero cost |
| Testing | `pytest` (backend/ML), Vitest/React Testing Library (frontend) | Standard, well-automatable from Cursor |
| Model export | ONNX (`torch.onnx.export`) | Portable, small, fast CPU inference — avoids shipping a full PyTorch runtime in the serving path |

Only what's necessary is included — no message queue, no Docker orchestration layer beyond a single optional Dockerfile for reproducibility, no external managed database.

---

## PART 13 — Repository Structure

```
asl-bangla/
├── README.md
├── config/
│   └── config.yaml                  # all paths, hyperparams, thresholds — single source of truth
├── data/
│   ├── raw/                         # downloaded, untouched (kaggle asl-signs, asl_citizen)
│   ├── processed/                   # cached tensors, label_map.json, splits
│   └── scripts/
│       ├── download_kaggle_islr.py
│       ├── download_asl_citizen.py
│       ├── verify_integrity.py
│       ├── extract_landmarks_asl_citizen.py
│       ├── build_vocabulary.py
│       └── build_splits.py
├── ml/
│   ├── dataset.py                   # shared PyTorch Dataset for both sources
│   ├── model.py                     # Conv1D+Transformer classifier
│   ├── train.py                     # resumable training loop
│   ├── evaluate.py                  # metrics, confusion matrix
│   ├── export_onnx.py
│   └── translation/
│       ├── build_word_dictionary.py # precompute vocab → Bangla cache
│       └── translate.py             # BanglaT5 wrapper (word cache + sentence model)
├── backend/
│   ├── main.py                      # FastAPI app, REST + WebSocket routes
│   ├── inference.py                 # ONNX Runtime session wrapper
│   ├── sentence_state.py            # debounce, cooldown, buffer logic
│   ├── landmark_server.py           # server-side MediaPipe for uploaded video
│   └── schemas.py                   # Pydantic request/response models
├── frontend/
│   ├── src/
│   │   ├── components/ (Camera, RecognitionPanel, BanglaPanel, SentenceBar, DebugOverlay, Controls)
│   │   ├── hooks/ (useWebcam, useWebSocket, useMediapipeLandmarks)
│   │   └── App.tsx
│   └── vite.config.ts
├── models/
│   └── <run_name>/<version>/        # model.onnx, labels.json, metrics.json, config_snapshot.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── scripts/
    ├── setup_env.sh
    ├── run_full_pipeline.sh         # download → preprocess → train → evaluate → export
    ├── run_backend.sh
    └── run_frontend.sh
```

No hardcoded personal paths anywhere; every path is read from `config/config.yaml` or environment variables, with sane relative defaults.

---

## PART 14 — Training and Compute Plan

- **Expected compute needs:** landmark-based training at this model size is CPU-feasible (hours, not days). If Cursor Cloud exposes a GPU, use it (`torch.cuda.is_available()` gate — the training script must probe and log which device it selected, never assume).
- **Estimated training cost: $0** — everything runs inside whatever compute Cursor Pro Cloud already provides; no external paid compute, no paid API calls anywhere in the pipeline (translation model runs locally, not via a hosted API).
- **Resource optimization:** cache preprocessed tensors (Part 8) so repeated training runs (e.g., after a bug fix) don't repeat MediaPipe extraction; use mixed precision automatically when a CUDA GPU is present; automatically reduce batch size / sequence length if an out-of-memory error is caught, log the adjustment, and retry rather than crashing the whole pipeline.
- **Caching:** processed tensors keyed by a config hash (Part 8); model checkpoints keyed by run name + git-ish version string.
- **Checkpointing:** every epoch (best + last); training script must accept `--resume <path>` and continue from the exact epoch/optimizer state.
- **Resume logic:** on any crash (OOM, environment hiccup, Cursor Cloud restart), the *first* action of a re-invoked training agent must be to check for an existing checkpoint for the current run name and resume from it rather than restarting from scratch — this must be explicit in the Cursor master prompt, not assumed.

---

## PART 15 — 5-Day Execution Plan

### Day 1 — Environment, data, baseline
- Detect hardware (CPU/GPU/RAM/disk). Scaffold repo structure (Part 13). Set up Python/Node environments.
- **Parallel:** (a) download Kaggle `asl-signs` landmark data + verify integrity; (b) start the ASL Citizen 42.8 GB download in the background (it's the long pole — start it immediately, don't wait).
- Build vocabulary/splits for the 250-word Kaggle set. Train a *quick* baseline (reduced epochs) to prove the pipeline end-to-end works before investing in the full training run.
- **Blocks Day 2:** a working data loader and a baseline checkpoint, however rough.
- **Fallback if dataset access fails today:** switch immediately to the WLASL pretrained I3D checkpoint path (Part 6) so *something* is inference-capable by end of Day 1.

### Day 2 — Full model training + evaluation (250-word tier)
- Full training run on the 250-word vocabulary with early stopping. Evaluate: top-1/top-5, confusion matrix, per-class report. Export to ONNX, sanity-check ONNX output matches PyTorch output on a validation batch.
- **Parallel:** ASL Citizen MediaPipe landmark extraction (the heaviest CPU job in the whole project) runs across as many parallel workers as available.
- **Minimum viable path today:** a working, evaluated, ONNX-exported 250-word model. **Stronger path if time permits:** ASL Citizen extraction far enough along to start vocabulary-extension training tonight.

### Day 3 — Inference pipeline + real-time processing + (stretch) vocabulary extension
- Build the ONNX inference wrapper, the debounce/segmentation/sentence-state logic, and the WebSocket contract server-side.
- Build the in-browser MediaPipe Tasks-Vision landmark extraction and webcam capture client-side; wire the WebSocket round-trip end-to-end with the 250-word (or extended) model.
- **If ASL Citizen extraction finished:** build the extended 450–520-word vocabulary and retrain/fine-tune; evaluate; export; **this becomes the new served model only if its held-out accuracy is acceptable** (do not silently ship a worse model just because it has more classes — log both and let the acceptance checklist decide, defaulting to the 250-word model if the extended one underperforms badly).
- **Fallback if training is behind:** ship real-time inference wired to the Day-2 250-word model; treat vocabulary extension as Day 4 stretch instead.

### Day 4 — Frontend/backend integration, sentence accumulation, translation
- Build the React UI (camera panel, recognition panel, Bangla panel, sentence bar, controls, debug overlay).
- Precompute the word→Bangla dictionary cache (batch call to BanglaT5 over the final vocabulary, saved to JSON — do this once, not per-request).
- Wire the "Translate sentence" full-sentence BanglaT5 call.
- Wire uploaded-video mode end-to-end (upload → server-side landmark extraction → same model → results).
- **Fallback if BanglaT5 is too slow/heavy on available hardware:** ship the dictionary-only translation path (Part 6) and clearly label the sentence output as word-by-word.

### Day 5 — Testing, bug fixing, optimization, hardening
- Run the full test suite (Part 19): unit, integration, e2e, model tests, real-time latency tests.
- Fix everything the tests catch; re-run; repeat until the acceptance checklist (Part 20) passes.
- Write the README (setup, training, inference, evaluation commands — Part 22).
- Final pass: confirm no hardcoded paths, no secrets, no placeholder/fake-random inference anywhere, no paid dependency anywhere.

This schedule assumes both downloads (Kaggle, ASL Citizen) are kicked off in parallel from minute one of Day 1 — the single biggest schedule risk is *not* starting the 42.8 GB download early.

---

## PART 16 — Cursor Parallel-Agent Plan

| Agent | Responsibility | Depends on | Produces | Can run in parallel with |
|---|---|---|---|---|
| **1. Environment & Scaffolding** | Detect hardware, create repo structure, set up Python/Node envs, write `config.yaml` skeleton | Nothing | Repo skeleton, env manifest | Nothing (must go first, but is fast) |
| **2. Data Acquisition** | Download Kaggle landmarks + ASL Citizen zip, verify integrity | Agent 1 | `data/raw/*` | Agent 3 (frontend scaffolding), Agent 7 (test scaffolding) |
| **3. Data Preprocessing** | Build vocabulary, splits, normalize/cache tensors for both sources, run MediaPipe extraction on ASL Citizen | Agent 2 (per-source, can start on Kaggle data before ASL Citizen finishes downloading) | `data/processed/*`, `label_map.json` | Agent 4 (model code, which doesn't need data to *exist* yet to be *written*) |
| **4. Model Training & Evaluation** | Implement `model.py`/`train.py`/`evaluate.py`, run training, export ONNX | Agent 3 (needs processed data to actually train, but code can be written earlier) | `models/<run>/model.onnx`, `metrics.json` | Agent 5 (translation), Agent 6 (frontend) |
| **5. Translation** | Integrate BanglaT5, build word-dictionary cache script, sentence-translation wrapper | Agent 1 only (independent of ML training) | `translation/translate.py`, `bangla_dictionary.json` (built once vocabulary is final, so has a soft dependency on Agent 3's vocabulary output) | Agents 3, 4, 6 |
| **6. Backend/Inference Service** | FastAPI app, WebSocket contract, ONNX Runtime wrapper, sentence-state logic | Agent 4 (needs a model to serve — can build against a stub/mock model first) | `backend/*` | Agent 7 (frontend) |
| **7. Frontend** | React app, webcam capture, in-browser MediaPipe, all UI panels | Agent 1 (needs project scaffold), can be built against a mocked WebSocket response before Agent 6 is finished | `frontend/*` | Agents 3, 4, 5, 6 |
| **8. Testing** | Write and continuously run unit/integration/e2e/latency tests | Runs continuously once each dependency exists | `tests/*`, test reports | All others, ongoing |
| **9. Integration/QA (final)** | Full-system run, fixes cross-cutting integration failures, validates the acceptance checklist | All other agents | Final validated system | Last — not parallel |

**Conflict handling:** every agent writes only inside its own top-level directory (Part 13) except Agent 9, which may touch any file to fix integration bugs; shared contracts (the ONNX model I/O shape, the WebSocket message schema, the label map format) are defined as explicit JSON-schema-like docstrings in `backend/schemas.py` and `config/config.yaml` on Day 1 so agents 4, 6, and 7 can develop against a fixed contract without waiting on each other.

**Retry policy:** any agent whose task fails (download error, training crash, test failure) retries up to 3 times with the failure logged verbatim; after 3 failures it falls back to the next tier in Part 6 rather than blocking the whole pipeline indefinitely.

**Final integration agent verification:** Agent 9 must actually run `scripts/run_full_pipeline.sh` and `scripts/run_backend.sh`/`run_frontend.sh` from a clean checkout, not just review code, before declaring the acceptance checklist satisfied.

---

## PART 19 — Test Plan

**Unit tests**
- Landmark normalization (known input → known normalized output).
- Sequence padding/resizing to fixed `T`.
- Vocabulary/label-map construction (no duplicate indices, no off-by-one).
- Sentence-state logic: debounce timing, cooldown, duplicate suppression, remove-last, clear.
- Bangla dictionary lookup (hit/miss behavior, fallback when a word isn't in the cache).
- ONNX vs. PyTorch output parity on a fixed validation batch (must match within a small numerical tolerance).

**Integration tests**
- Backend WebSocket: send a synthetic landmark window, assert a well-formed `{word, confidence, top_k}` response.
- Backend REST upload endpoint: send a short test video, assert a well-formed recognition result.
- Translation endpoint: English word/sentence in → non-empty Bangla string out, correct script (Bengali Unicode range) — this is a formatting/sanity check, not a fluency judgment.
- Frontend ↔ backend: mocked WebSocket integration test confirming UI state updates on a received prediction message.

**End-to-end tests**
- Recorded test clip → full pipeline (landmark extraction → model → sentence buffer → Bangla) → assert the expected word appears in the output for at least N known-good test clips per tier.
- Upload-mode full path with a real short video file.

**Model tests**
- Top-1, top-5 accuracy on held-out test split (report actual numbers — do not assert an arbitrary "must be >X%" threshold that wasn't derived from the actual validation run; the acceptance checklist instead requires the number to be *reported and reviewed*, not blindly gated at a fictional target).
- Precision/recall/F1 per class; flag any class with test recall = 0 (i.e., completely unlearned) for review/possible removal from the served vocabulary.
- Confusion matrix saved as an artifact.
- "No-sign/rest" class true-negative rate on idle/no-motion segments.
- Latency: p50/p95 single-window inference time on CPU and (if available) GPU.

**Real-time tests**
- Browser-side landmark extraction FPS under typical webcam resolution.
- End-to-end webcam→word latency, measured, not estimated (timestamp the moment a test sign starts vs. the moment a word is committed).
- Memory usage over a sustained 5-minute live session (check for leaks in the sliding-window buffer).

**Edge cases explicitly tested**
- No hands visible (occluded/out of frame).
- Very fast vs. very slow signing speed.
- Low light / partial occlusion.
- Repeated identical sign performed twice in a row deliberately (must produce two words, not one, once a rest period is detected between them).
- Empty/invalid uploaded file.
- WebSocket disconnect/reconnect mid-session (sentence buffer must survive).

---

## PART 20 — Final Acceptance Checklist

- [ ] Project builds successfully from a clean checkout (`scripts/setup_env.sh` succeeds).
- [ ] All dependencies install successfully, no manual intervention.
- [ ] Both datasets (or the documented fallback tier actually reached) are confirmed downloaded and integrity-checked.
- [ ] Preprocessing completes and produces the expected tensor shapes and vocabulary size.
- [ ] At least one model checkpoint exists and loads successfully in both PyTorch and ONNX Runtime.
- [ ] Inference produces real predictions (never a hardcoded/random/mocked output) in both webcam and upload modes.
- [ ] Webcam mode works end-to-end with measured, reported latency.
- [ ] Uploaded-video mode works end-to-end.
- [ ] English recognition output is correct for a sanity-check set of known test signs.
- [ ] Bangla translation works for both the dictionary path and the full-sentence model path.
- [ ] Sentence accumulation, duplicate suppression, remove-last, and clear all work as specified.
- [ ] Low-confidence input is correctly shown as "uncertain," not a wrong guess.
- [ ] Frontend/backend integration works with no console errors under normal use.
- [ ] Full test suite (Part 19) passes, and results (including actual accuracy/latency numbers) are saved as artifacts, not just printed and discarded.
- [ ] No critical unhandled errors remain in logs from a full test pass.
- [ ] `scripts/run_full_pipeline.sh`, `run_backend.sh`, `run_frontend.sh` all work from a clean environment with documented commands only.
- [ ] README documents setup, training, inference, and evaluation commands completely.
- [ ] No paid API, paid dataset, or paid cloud service is used anywhere (verified by grepping for API keys/billing-requiring imports).
- [ ] No manual dataset download step is required — every download is scripted.
- [ ] No placeholder/`TODO`/fake-random implementation remains in any critical path (model inference, translation, sentence logic).
- [ ] Final vocabulary size and its source tier (Part 6) are explicitly documented in the README, along with the honest scope statement from Part 11 (isolated-sign recognition with accumulation, not full continuous ASL translation).

---

## PART 21 — Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| WLASL YouTube links dead when needed | High (documented in the dataset's own repo) | Would stall data acquisition for days if it were the primary source | **Demoted to Tier 2/3 fallback only**; primary sources are directly downloadable (Part 3) |
| ASL Citizen 42.8 GB download is slow on Cursor Cloud's network/storage | Medium | Could delay vocabulary extension to Day 4 | Start the download at the very start of Day 1, in parallel with everything else; the 250-word model already satisfies the minimum requirement even if this never finishes |
| Kaggle competition data usage terms ambiguity for post-competition use of the Kaggle-hosted copy | Low-medium (legal ambiguity, not a technical risk) | Reputational/compliance risk if the project were ever redistributed commercially | Documented explicitly (Part 2.4); project is not deployed or monetized per the user's own requirement #6; if this matters later, re-source the identical dataset from its CC-BY 4.0 canonical Georgia Tech host instead of Kaggle |
| MediaPipe legacy API deprecation breaks a copied tutorial pattern | Medium (many tutorials online still use the old API) | Runtime errors, wasted debugging time | Cursor master prompt explicitly mandates the Tasks API with a version-check fallback (Part 2.6, Part 17) |
| Landmark-only model underperforms on signs that are heavily face/mouth-dependent | Low-medium | Some signs harder to distinguish | Face-mesh subset is included in the default feature set specifically for this reason (Part 9); can be ablated if latency requires it |
| ASL Citizen extraction (MediaPipe over 84k videos) is too slow to finish by Day 3 | Medium | Vocabulary extension slips or is dropped | Parallelized extraction across all available cores (Part 16, Agent 3); 250-word model is never blocked by this — it's purely additive |
| BanglaT5 inference too slow for live per-word translation | Low (only used for sentence-level, on-demand calls, not per-frame) | Sentence translation button feels laggy | Precomputed dictionary handles the instant per-word case (Part 11); model is only invoked on explicit user action, not in the real-time loop |
| No GPU available in Cursor Cloud | Medium | Slower training, I3D fallback inference slower | Entire primary architecture is deliberately CPU-trainable (Part 5); I3D fallback explicitly restricted to upload-mode-only on CPU (Part 6) |
| Class imbalance (some signs have very few examples) | High (documented median clip count issue even outside WLASL) | Poor accuracy on rare classes | Oversampling floor + explicit drop threshold + per-class recall reporting (Parts 8, 9, 19) |
| Cursor declares "done" after writing code without actually running/testing it | Medium (a general risk with autonomous agents) | System looks complete but doesn't work | Master prompt explicitly bans this (Part 17); acceptance checklist requires actual run artifacts, not just code presence |

---

## PART 22 — Final Commands

```bash
# 1. Initialize
git clone <repo> asl-bangla && cd asl-bangla
bash scripts/setup_env.sh                 # creates venv, installs Python + Node deps

# 2. Download data (run in parallel/background)
python data/scripts/download_kaggle_islr.py        # needs KAGGLE_USERNAME / KAGGLE_KEY env vars
python data/scripts/download_asl_citizen.py &       # long-running; start early

# 3. Verify + preprocess
python data/scripts/verify_integrity.py
python data/scripts/build_vocabulary.py --target-size 500
python data/scripts/build_splits.py
python data/scripts/extract_landmarks_asl_citizen.py --workers <N>

# 4. Train
python ml/train.py --config config/config.yaml --run-name baseline_250
# (after ASL Citizen extraction completes, optionally:)
python ml/train.py --config config/config.yaml --run-name extended_500 --resume-vocab

# 5. Evaluate + export
python ml/evaluate.py --run-name extended_500
python ml/export_onnx.py --run-name extended_500

# 6. Build translation assets
python ml/translation/build_word_dictionary.py --vocab models/extended_500/labels.json

# 7. Run backend
bash scripts/run_backend.sh               # uvicorn backend.main:app --reload

# 8. Run frontend
bash scripts/run_frontend.sh              # npm run dev (frontend/)

# 9. Run tests
pytest tests/unit tests/integration
pytest tests/e2e
npm --prefix frontend run test
```

---

*Companion document: `Cursor-Master-Prompt-and-Agents.md` contains the copy-paste-ready prompts for Cursor Pro Cloud implementing everything specified above.*
