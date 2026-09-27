"""BanglaT5 English→Bangla translation wrapper (Agent 5)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "config.yaml"

# Rough UI threshold: on-demand sentence button should feel responsive (<3s on CPU).
SENTENCE_LATENCY_UI_THRESHOLD_SEC = 3.0

SMOKE_TEST_SENTENCES = [
    "Hello, how are you?",
    "Thank you very much.",
    "I love learning sign language.",
    "Good morning.",
    "Where is the bathroom?",
]


def repo_root() -> Path:
    return _REPO_ROOT


def load_translation_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _DEFAULT_CONFIG
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    translation = raw.get("translation", {})
    paths = raw.get("paths", {})
    return {
        "model_id": translation.get("model_id", "csebuetnlp/banglat5_nmt_en_bn"),
        "tokenizer_use_fast": translation.get("tokenizer_use_fast", False),
        "reports_dir": paths.get("reports_dir", "reports"),
        "max_new_tokens": translation.get("max_new_tokens", 128),
    }


def _contains_bengali_script(text: str) -> bool:
    return bool(re.search(r"[\u0980-\u09FF]", text))


@dataclass
class TranslationResult:
    english: str
    normalized_english: str
    bangla: str
    latency_sec: float


@dataclass
class LoadMetrics:
    load_sec: float
    model_id: str
    device: str


class BanglaTranslator:
    """Lazy-loaded BanglaT5 NMT on CPU (no API keys)."""

    def __init__(
        self,
        model_id: str | None = None,
        use_fast_tokenizer: bool = False,
        max_new_tokens: int = 128,
    ) -> None:
        cfg = load_translation_config()
        self.model_id = model_id or cfg["model_id"]
        self.use_fast_tokenizer = use_fast_tokenizer
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._normalizer: Any | None = None
        self._load_metrics: LoadMetrics | None = None

    def _ensure_normalizer(self) -> Any:
        if self._normalizer is None:
            from normalizer import normalize as normalize_fn

            self._normalizer = normalize_fn
        return self._normalizer

    def load(self) -> LoadMetrics:
        if self._load_metrics is not None:
            return self._load_metrics

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=self.use_fast_tokenizer,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self._model.eval()
        load_sec = time.perf_counter() - t0

        import torch

        device = "cpu"
        self._model.to(device)

        self._load_metrics = LoadMetrics(
            load_sec=load_sec,
            model_id=self.model_id,
            device=device,
        )
        return self._load_metrics

    def normalize(self, text: str) -> str:
        normalize_fn = self._ensure_normalizer()
        return normalize_fn(text.strip())

    def translate(self, english: str) -> TranslationResult:
        self.load()
        assert self._tokenizer is not None and self._model is not None

        normalized = self.normalize(english)
        inputs = self._tokenizer(normalized, return_tensors="pt")
        t0 = time.perf_counter()
        with __import__("torch").no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        latency = time.perf_counter() - t0
        bangla = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return TranslationResult(
            english=english,
            normalized_english=normalized,
            bangla=bangla,
            latency_sec=latency,
        )

    def lookup_dictionary(self, english_word: str, dictionary: dict[str, str]) -> str | None:
        key = english_word.strip().lower()
        if key in dictionary:
            return dictionary[key]
        return dictionary.get(english_word.strip())


def run_smoke_test(
    config_path: Path | None = None,
    sentences: list[str] | None = None,
) -> dict[str, Any]:
    cfg = load_translation_config(config_path)
    translator = BanglaTranslator(
        model_id=cfg["model_id"],
        use_fast_tokenizer=cfg["tokenizer_use_fast"],
        max_new_tokens=int(cfg["max_new_tokens"]),
    )
    load_metrics = translator.load()

    samples: list[dict[str, Any]] = []
    latencies: list[float] = []
    test_sentences = sentences or SMOKE_TEST_SENTENCES

    for sentence in test_sentences:
        result = translator.translate(sentence)
        latencies.append(result.latency_sec)
        samples.append(
            {
                "english": result.english,
                "normalized_english": result.normalized_english,
                "bangla": result.bangla,
                "latency_sec": round(result.latency_sec, 4),
                "bengali_script_detected": _contains_bengali_script(result.bangla),
            }
        )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    sentence_path_ok = max_latency <= SENTENCE_LATENCY_UI_THRESHOLD_SEC

    return {
        "model_id": load_metrics.model_id,
        "device": load_metrics.device,
        "model_load_sec": round(load_metrics.load_sec, 4),
        "sentence_latency_ui_threshold_sec": SENTENCE_LATENCY_UI_THRESHOLD_SEC,
        "per_sentence_latency_sec": {
            "mean": round(avg_latency, 4),
            "max": round(max_latency, 4),
            "samples": [round(x, 4) for x in latencies],
        },
        "sentence_path_usable_for_ui": sentence_path_ok,
        "latency_notes": (
            "One-time model load is paid at process startup (~several seconds on CPU). "
            "Per-sentence inference measured here is suitable for an on-demand "
            "'Translate sentence' button if max latency stays below "
            f"{SENTENCE_LATENCY_UI_THRESHOLD_SEC}s."
            if sentence_path_ok
            else (
                "Per-sentence latency exceeds the UI threshold; prefer dictionary-only "
                "word-by-word display and disable or warn on full-sentence translation."
            )
        ),
        "samples": samples,
    }


def write_smoke_test_report(
    output_path: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    cfg = load_translation_config(config_path)
    reports_dir = repo_root() / cfg["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = output_path or (reports_dir / "translation_smoke_test.json")
    payload = run_smoke_test(config_path=config_path)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    out = write_smoke_test_report()
    payload = json.loads(out.read_text(encoding="utf-8"))
    print(f"Wrote smoke test report to {out}")
    for row in payload["samples"]:
        print(f"  EN: {row['english']}")
        print(f"  BN: {row['bangla']} ({row['latency_sec']}s)")
    print(
        f"Load: {payload['model_load_sec']}s | "
        f"mean sentence: {payload['per_sentence_latency_sec']['mean']}s | "
        f"UI OK: {payload['sentence_path_usable_for_ui']}"
    )


if __name__ == "__main__":
    main()
