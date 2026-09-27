"""Unit tests for sentence accumulation gating."""

from __future__ import annotations

import pytest

from backend.sentence_state import (
    PredictionSample,
    SentenceState,
    SentenceStateConfig,
)


def test_low_confidence_rejected() -> None:
    state = SentenceState()
    assert (
        state.process_prediction(
            PredictionSample(word="hello", confidence=0.59, timestamp_ms=0)
        )
        is None
    )
    assert state.words == []


def test_ema_allows_commit_after_smoothing() -> None:
    state = SentenceState(SentenceStateConfig(temporal_ema_windows=3))
    assert state.process_prediction(PredictionSample("a", 0.55, 0)) is None
    assert state.process_prediction(PredictionSample("a", 0.55, 50)) is None
    committed = state.process_prediction(PredictionSample("a", 0.70, 100))
    assert committed == "a"


def test_cooldown_blocks_rapid_duplicate_words() -> None:
    state = SentenceState()
    assert state.process_prediction(PredictionSample("hello", 0.9, 0)) == "hello"
    assert (
        state.process_prediction(PredictionSample("world", 0.9, 100)) is None
    )
    assert state.process_prediction(PredictionSample("world", 0.9, 801)) == "world"
    assert state.sentence == "hello world"


def test_duplicate_suppression_adjacent_same_word() -> None:
    state = SentenceState()
    state.process_prediction(PredictionSample("again", 0.95, 0))
    assert state.process_prediction(PredictionSample("again", 0.95, 900)) is None
    assert state.words == ["again"]


def test_duplicate_suppression_disabled() -> None:
    state = SentenceState(SentenceStateConfig(duplicate_suppression=False))
    state.process_prediction(PredictionSample("again", 0.95, 0))
    assert state.process_prediction(PredictionSample("again", 0.95, 900)) == "again"
    assert state.words == ["again", "again"]


def test_remove_last_and_clear() -> None:
    state = SentenceState()
    state.process_prediction(PredictionSample("one", 0.9, 0))
    state.process_prediction(PredictionSample("two", 0.9, 900))
    assert state.remove_last() == "two"
    assert state.sentence == "one"
    state.reset()
    assert state.sentence == ""


def test_process_batch_order() -> None:
    state = SentenceState()
    samples = [
        PredictionSample("I", 0.9, 0),
        PredictionSample("love", 0.9, 850),
        PredictionSample("you", 0.9, 1700),
    ]
    assert state.process_batch(samples) == ["I", "love", "you"]
