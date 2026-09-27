"""Pydantic models for WebSocket and REST payloads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TopKCandidate(BaseModel):
    word: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionMessage(BaseModel):
    type: Literal["prediction"] = "prediction"
    word: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[TopKCandidate]
    model_version: str | None = None
    vocab_size: int | None = None
    latency_ms: float | None = None
    committed_word: str | None = None


class StatusMessage(BaseModel):
    type: Literal["status"] = "status"
    connection: Literal["connected", "disconnected", "error"]
    message: str | None = None


class LandmarkFrame(BaseModel):
    timestamp_ms: int
    landmarks: list[float]


class LandmarkWindowMessage(BaseModel):
    type: Literal["landmark_window"] = "landmark_window"
    frames: list[LandmarkFrame]
    session_id: str | None = None


class VideoRecognitionSegment(BaseModel):
    timestamp_ms: int
    word: str
    confidence: float
    top_k: list[TopKCandidate]


class VideoRecognitionResponse(BaseModel):
    model_version: str
    vocab_size: int
    latency_ms: float
    segments: list[VideoRecognitionSegment]
    sentence_english: str
    committed_words: list[str]


class InstantTranslateRequest(BaseModel):
    english: str


class InstantTranslateResponse(BaseModel):
    english: str
    bangla: str | None
    source: Literal["dictionary", "missing"]


class InstantBatchTranslateRequest(BaseModel):
    words: list[str]


class InstantBatchTranslateResponse(BaseModel):
    english_words: list[str]
    bangla_glosses: list[str | None]
    joined_bangla: str


class SentenceTranslateRequest(BaseModel):
    english: str


class SentenceTranslateResponse(BaseModel):
    english: str
    normalized_english: str
    bangla: str
    latency_sec: float
    model_id: str
