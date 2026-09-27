#!/usr/bin/env python3
"""Precompute English vocabulary → Bangla cache via BanglaT5 (Agent 5).

Run once after Agent 3 produces ``label_map.json``:

    python ml/translation/build_word_dictionary.py \\
        --label-map data/processed/label_map.json

Until that file exists, the script uses a tiny placeholder vocabulary so the
pipeline can be exercised early. Re-run with the real label map when Agent 3
finishes vocabulary construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.translation.translate import BanglaTranslator, repo_root

# Minimal gloss-like words for smoke builds before Agent 3's label_map.json exists.
PLACEHOLDER_VOCAB = [
    "hello",
    "thank you",
    "yes",
    "no",
    "please",
    "water",
    "help",
    "good",
    "morning",
    "friend",
]


def load_paths_config(config_path: Path) -> dict[str, str]:
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    paths = raw.get("paths", {})
    return {
        "models_dir": paths.get("models_dir", "models"),
        "bangla_dictionary_filename": paths.get(
            "bangla_dictionary_filename", "bangla_dictionary.json"
        ),
        "label_map_filename": paths.get("label_map_filename", "label_map.json"),
        "repo_relative_data_root": paths.get("repo_relative_data_root", "data"),
    }


def default_label_map_path(root: Path, paths_cfg: dict[str, str]) -> Path:
    return root / paths_cfg["repo_relative_data_root"] / "processed" / paths_cfg[
        "label_map_filename"
    ]


def default_dictionary_path(root: Path, paths_cfg: dict[str, str]) -> Path:
    return root / paths_cfg["models_dir"] / paths_cfg["bangla_dictionary_filename"]


def extract_english_words(label_map: Any) -> list[str]:
    """Parse Agent 3 label_map.json shapes into unique English words."""
    words: list[str] = []

    if isinstance(label_map, list):
        for item in label_map:
            if isinstance(item, str):
                words.append(item)
            elif isinstance(item, dict):
                w = item.get("word") or item.get("english") or item.get("label")
                if isinstance(w, str):
                    words.append(w)
    elif isinstance(label_map, dict):
        label_to_index = label_map.get("label_to_index")
        if isinstance(label_to_index, dict):
            words.extend(str(gloss) for gloss in label_to_index.keys())
        elif "words" in label_map and isinstance(label_map["words"], list):
            words.extend(str(w) for w in label_map["words"])
        elif "labels" in label_map and isinstance(label_map["labels"], list):
            words.extend(str(w) for w in label_map["labels"])
        elif "entries" in label_map and isinstance(label_map["entries"], list):
            for entry in label_map["entries"]:
                if isinstance(entry, dict):
                    w = entry.get("word") or entry.get("english")
                    if isinstance(w, str):
                        words.append(w)
        else:
            for gloss, value in label_map.items():
                if gloss in ("metadata", "version", "source"):
                    continue
                if isinstance(value, str):
                    words.append(value)
                elif isinstance(value, dict):
                    w = value.get("word") or value.get("english")
                    if isinstance(w, str):
                        words.append(w)
                elif isinstance(value, int):
                    words.append(str(gloss))

    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        key = w.strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        unique.append(key)
    return unique


def load_vocabulary(label_map_path: Path | None, use_placeholder: bool) -> tuple[list[str], str]:
    if label_map_path is not None and label_map_path.is_file():
        data = json.loads(label_map_path.read_text(encoding="utf-8"))
        words = extract_english_words(data)
        if words:
            return words, f"label_map:{label_map_path}"
    if use_placeholder:
        return list(PLACEHOLDER_VOCAB), "placeholder_vocabulary"
    raise FileNotFoundError(
        f"No vocabulary at {label_map_path}; pass --placeholder or wait for Agent 3 label_map.json"
    )


def build_dictionary(
    words: list[str],
    translator: BanglaTranslator,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    dictionary: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    for word in words:
        result = translator.translate(word)
        key = word.strip().lower()
        dictionary[key] = result.bangla
        details.append(
            {
                "english": word,
                "bangla": result.bangla,
                "latency_sec": round(result.latency_sec, 4),
            }
        )
    return dictionary, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bangla_dictionary.json from vocabulary.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "config" / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Path to label_map.json (default: data/processed/label_map.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: models/bangla_dictionary.json)",
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Use tiny placeholder vocab if label_map.json is missing",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    paths_cfg = load_paths_config(args.config)
    label_map_path = args.label_map or default_label_map_path(root, paths_cfg)
    output_path = args.output or default_dictionary_path(root, paths_cfg)
    use_placeholder = args.placeholder or not label_map_path.is_file()

    words, source = load_vocabulary(label_map_path if label_map_path.is_file() else None, use_placeholder)

    translator = BanglaTranslator()
    t0 = time.perf_counter()
    load_metrics = translator.load()
    dictionary, word_details = build_dictionary(words, translator)
    total_sec = time.perf_counter() - t0

    payload = {
        "source": source,
        "model_id": load_metrics.model_id,
        "word_count": len(dictionary),
        "model_load_sec": round(load_metrics.load_sec, 4),
        "total_build_sec": round(total_sec, 4),
        "note": (
            "Placeholder vocabulary — re-run after Agent 3 writes data/processed/label_map.json"
            if source == "placeholder_vocabulary"
            else "Built from final label_map.json"
        ),
        "dictionary": dictionary,
        "details": word_details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(dictionary)} entries to {output_path} (source: {source})")
    if source == "placeholder_vocabulary":
        print(
            "Using placeholder vocab; run again with Agent 3's label_map.json for full cache."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
