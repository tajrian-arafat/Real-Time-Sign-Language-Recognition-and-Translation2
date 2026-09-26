"""Download and cache MediaPipe task model bundles for server-side video."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from backend.config_loader import resolve_models_dir, get_config

_TASK_URLS: dict[str, str] = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def task_model_path(filename: str) -> Path:
    config = get_config()
    cache_dir = resolve_models_dir(config) / "mediapipe_tasks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    url = _TASK_URLS.get(filename)
    if url is None:
        raise ValueError(f"No download URL configured for {filename}")
    urllib.request.urlretrieve(url, target)
    return target
