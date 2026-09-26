"""Debounce, cooldown, EMA smoothing, and sentence buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SentenceStateConfig:
    confidence_threshold: float = 0.60
    debounce_cooldown_ms: int = 800
    temporal_ema_windows: int = 3
    duplicate_suppression: bool = True


@dataclass
class PredictionSample:
    word: str
    confidence: float
    timestamp_ms: int


@dataclass
class SentenceState:
    """Accumulates committed English words with temporal gating."""

    config: SentenceStateConfig = field(default_factory=SentenceStateConfig)
    words: list[str] = field(default_factory=list)
    _confidence_history: list[float] = field(default_factory=list)
    _last_committed_word: str | None = None
    _last_commit_timestamp_ms: int | None = None

    def reset(self) -> None:
        self.words.clear()
        self._confidence_history.clear()
        self._last_committed_word = None
        self._last_commit_timestamp_ms = None

    def remove_last(self) -> str | None:
        if not self.words:
            return None
        removed = self.words.pop()
        if self.words:
            self._last_committed_word = self.words[-1]
        else:
            self._last_committed_word = None
            self._last_commit_timestamp_ms = None
        return removed

    def _ema_confidence(self, raw: float) -> float:
        history = self._confidence_history
        history.append(raw)
        window = max(1, self.config.temporal_ema_windows)
        tail = history[-window:]
        return sum(tail) / len(tail)

    def _cooldown_elapsed(self, timestamp_ms: int) -> bool:
        if self._last_commit_timestamp_ms is None:
            return True
        elapsed = timestamp_ms - self._last_commit_timestamp_ms
        return elapsed >= self.config.debounce_cooldown_ms

    def _should_suppress_duplicate(self, word: str) -> bool:
        if not self.config.duplicate_suppression:
            return False
        return word == self._last_committed_word

    def process_prediction(self, sample: PredictionSample) -> str | None:
        """Return newly committed word, or None if gated."""
        smoothed = self._ema_confidence(sample.confidence)
        if smoothed < self.config.confidence_threshold:
            return None
        if self._should_suppress_duplicate(sample.word):
            return None
        if not self._cooldown_elapsed(sample.timestamp_ms):
            return None

        self.words.append(sample.word)
        self._last_committed_word = sample.word
        self._last_commit_timestamp_ms = sample.timestamp_ms
        return sample.word

    def process_batch(self, samples: Sequence[PredictionSample]) -> list[str]:
        committed: list[str] = []
        for sample in samples:
            word = self.process_prediction(sample)
            if word is not None:
                committed.append(word)
        return committed

    @property
    def sentence(self) -> str:
        return " ".join(self.words)
