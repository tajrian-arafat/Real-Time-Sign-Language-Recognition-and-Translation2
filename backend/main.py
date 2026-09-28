"""FastAPI app: WebSocket landmark inference and REST video recognition."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.bangla_service import (
    DictionaryNotFoundError,
    load_bangla_dictionary,
    lookup_instant_bangla,
    translate_sentence,
)
from backend.config_loader import get_config, get_inference_settings
from backend.inference import ModelNotLoadedError, get_classifier
from backend.landmark_server import VideoLandmarkExtractor, sliding_windows
from backend.schemas import (
    InstantBatchTranslateRequest,
    InstantBatchTranslateResponse,
    InstantTranslateRequest,
    InstantTranslateResponse,
    LandmarkWindowMessage,
    PredictionMessage,
    SentenceTranslateRequest,
    SentenceTranslateResponse,
    StatusMessage,
    TopKCandidate,
    VideoRecognitionResponse,
    VideoRecognitionSegment,
)
from backend.sentence_state import PredictionSample, SentenceState, SentenceStateConfig
from ml.translation.translate import load_translation_config


def _sentence_state_from_config(config: dict[str, Any]) -> SentenceState:
    settings = get_inference_settings(config)
    return SentenceState(
        config=SentenceStateConfig(
            confidence_threshold=float(settings["confidence_threshold"]),
            debounce_cooldown_ms=int(settings["debounce_cooldown_ms"]),
            temporal_ema_windows=int(settings["temporal_ema_windows"]),
            duplicate_suppression=bool(settings["duplicate_suppression"]),
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    app.state.config = config
    try:
        app.state.classifier = get_classifier(config)
    except ModelNotLoadedError as exc:
        app.state.classifier = None
        app.state.model_error = str(exc)
    else:
        app.state.model_error = None
    yield


app = FastAPI(title="ASL → Bangla Recognition API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_classifier() -> Any:
    classifier = app.state.classifier
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=app.state.model_error or "Model not loaded",
        )
    return classifier


def _to_prediction_message(result: Any, committed_word: str | None = None) -> PredictionMessage:
    return PredictionMessage(
        word=result.word,
        confidence=result.confidence,
        top_k=[
            TopKCandidate(word=word, confidence=conf) for word, conf in result.top_k
        ],
        model_version=result.model_version,
        vocab_size=result.vocab_size,
        latency_ms=result.latency_ms,
        committed_word=committed_word,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    classifier = app.state.classifier
    dictionary_ok = True
    dictionary_error: str | None = None
    try:
        load_bangla_dictionary()
    except (DictionaryNotFoundError, ValueError) as exc:
        dictionary_ok = False
        dictionary_error = str(exc)
    return {
        "status": "ok"
        if classifier is not None and dictionary_ok
        else "degraded",
        "model_loaded": classifier is not None,
        "model_path": str(classifier.onnx_path) if classifier else None,
        "is_stub": classifier.is_stub if classifier else None,
        "model_error": app.state.model_error,
        "bangla_dictionary_loaded": dictionary_ok,
        "bangla_dictionary_error": dictionary_error,
    }


@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket) -> None:
    await websocket.accept()
    config: dict[str, Any] = app.state.config
    ws_path = config.get("backend", {}).get("websocket_path", "/ws/recognize")
    if ws_path != "/ws/recognize":
        # Config-doc parity; endpoint remains /ws/recognize for vite proxy.
        pass

    classifier = app.state.classifier
    if classifier is None:
        await websocket.send_json(
            StatusMessage(
                connection="error",
                message=app.state.model_error or "Model not loaded",
            ).model_dump()
        )
        await websocket.close()
        return

    await websocket.send_json(
        StatusMessage(connection="connected", message="ready").model_dump()
    )

    session_state = _sentence_state_from_config(config)

    try:
        while True:
            payload = await websocket.receive_json()
            try:
                window = LandmarkWindowMessage.model_validate(payload)
            except Exception as exc:
                await websocket.send_json(
                    StatusMessage(
                        connection="error",
                        message=f"Invalid landmark_window: {exc}",
                    ).model_dump()
                )
                continue

            if not window.frames:
                await websocket.send_json(
                    StatusMessage(
                        connection="error",
                        message="landmark_window.frames must not be empty",
                    ).model_dump()
                )
                continue

            frame_vectors = [frame.landmarks for frame in window.frames]
            result = classifier.predict_window(frame_vectors)
            timestamp_ms = window.frames[-1].timestamp_ms
            committed = session_state.process_prediction(
                PredictionSample(
                    word=result.word,
                    confidence=result.confidence,
                    timestamp_ms=timestamp_ms,
                )
            )
            await websocket.send_json(
                _to_prediction_message(result, committed_word=committed).model_dump()
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json(
            StatusMessage(connection="error", message=str(exc)).model_dump()
        )
        raise


@app.post("/api/recognize/video", response_model=VideoRecognitionResponse)
@app.post("/api/video", response_model=VideoRecognitionResponse)
async def recognize_video(file: UploadFile = File(...)) -> VideoRecognitionResponse:
    classifier = _require_classifier()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ".mp4"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    started = time.perf_counter()
    extractor = VideoLandmarkExtractor()
    try:
        frames = extractor.extract_from_bytes(data, suffix=suffix)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Video landmark extraction failed: {exc}",
        ) from exc

    if not frames:
        raise HTTPException(status_code=422, detail="No frames extracted from video")

    sentence_state = _sentence_state_from_config(app.state.config)
    segments: list[VideoRecognitionSegment] = []
    window_size = min(32, max(8, len(frames)))
    stride = max(1, window_size // 2)
    min_segment_confidence = 0.05

    for chunk in sliding_windows(frames, window_size=window_size, stride=stride):
        vectors = [f.landmarks for f in chunk]
        result = classifier.predict_window(vectors)
        if result.confidence >= min_segment_confidence:
            segments.append(
                VideoRecognitionSegment(
                    timestamp_ms=chunk[-1].timestamp_ms,
                    word=result.word,
                    confidence=result.confidence,
                    top_k=[
                        TopKCandidate(word=w, confidence=c) for w, c in result.top_k
                    ],
                )
            )
        sentence_state.process_prediction(
            PredictionSample(
                word=result.word,
                confidence=result.confidence,
                timestamp_ms=chunk[-1].timestamp_ms,
            )
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    committed_words = [seg.word for seg in segments]
    return VideoRecognitionResponse(
        model_version=classifier.model_version,
        vocab_size=classifier.vocab_size,
        latency_ms=latency_ms,
        segments=segments,
        sentence_english=sentence_state.sentence,
        committed_words=committed_words,
    )


@app.post("/api/translate/word", response_model=InstantTranslateResponse)
def translate_word_instant(body: InstantTranslateRequest) -> InstantTranslateResponse:
    """Instant gloss lookup from models/bangla_dictionary.json (Agent 5 cache)."""
    try:
        dictionary = load_bangla_dictionary()
    except DictionaryNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    bangla = lookup_instant_bangla(body.english, dictionary)
    return InstantTranslateResponse(
        english=body.english.strip(),
        bangla=bangla,
        source="dictionary" if bangla is not None else "missing",
    )


@app.post("/api/translate/words", response_model=InstantBatchTranslateResponse)
def translate_words_instant(
    body: InstantBatchTranslateRequest,
) -> InstantBatchTranslateResponse:
    """Word-by-word Bangla glosses for the sentence bar (non-grammatical)."""
    try:
        dictionary = load_bangla_dictionary()
    except DictionaryNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    glosses = [lookup_instant_bangla(word, dictionary) for word in body.words]
    joined = " ".join(g for g in glosses if g)
    return InstantBatchTranslateResponse(
        english_words=body.words,
        bangla_glosses=glosses,
        joined_bangla=joined,
    )


@app.post("/api/translate/sentence", response_model=SentenceTranslateResponse)
def translate_full_sentence(body: SentenceTranslateRequest) -> SentenceTranslateResponse:
    """On-demand fluent Bangla via BanglaT5 (matches Bangla panel button)."""
    text = body.english.strip()
    if not text:
        raise HTTPException(status_code=400, detail="english must not be empty")
    try:
        result = translate_sentence(text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Sentence translation failed: {exc}",
        ) from exc
    model_id = load_translation_config().get("model_id", "")
    return SentenceTranslateResponse(
        english=result.english,
        normalized_english=result.normalized_english,
        bangla=result.bangla,
        latency_sec=result.latency_sec,
        model_id=str(model_id),
    )
