"""Server-side MediaPipe Tasks landmark extraction for uploaded video."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from backend.config_loader import get_config, get_landmark_settings


@dataclass(frozen=True)
class ExtractedFrame:
    timestamp_ms: int
    landmarks: list[float]


class VideoLandmarkExtractor:
    """Sample video frames and extract flattened landmark vectors."""

    def __init__(self, target_fps: float = 15.0) -> None:
        self._config = get_config()
        self._feature_dim = get_landmark_settings(self._config)["input_dim_per_frame"]
        self._target_fps = target_fps

    def _flatten_detection(
        self,
        hand_landmarks: list | None,
        pose_landmarks: list | None,
        face_landmarks: list | None,
    ) -> list[float]:
        coords: list[float] = []

        def append_points(points: list | None, max_points: int) -> None:
            if not points:
                return
            for pt in points[:max_points]:
                coords.extend([float(pt.x), float(pt.y), float(getattr(pt, "z", 0.0))])

        if hand_landmarks:
            for hand in hand_landmarks[:2]:
                append_points(hand, 21)
        if pose_landmarks:
            append_points(pose_landmarks, 33)
        if face_landmarks:
            append_points(face_landmarks, 40)

        if len(coords) < self._feature_dim:
            coords.extend([0.0] * (self._feature_dim - len(coords)))
        return coords[: self._feature_dim]

    def extract_from_path(self, video_path: Path) -> list[ExtractedFrame]:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from backend.mediapipe_assets import task_model_path

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(task_model_path("hand_landmarker.task"))
            ),
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(task_model_path("pose_landmarker_lite.task"))
            ),
            running_mode=vision.RunningMode.VIDEO,
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(task_model_path("face_landmarker.task"))
            ),
            running_mode=vision.RunningMode.VIDEO,
        )

        frames: list[ExtractedFrame] = []
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        native_fps = capture.get(cv2.CAP_PROP_FPS) or self._target_fps
        step = max(1, int(round(native_fps / self._target_fps)))

        with (
            vision.HandLandmarker.create_from_options(hand_options) as hand_lm,
            vision.PoseLandmarker.create_from_options(pose_options) as pose_lm,
            vision.FaceLandmarker.create_from_options(face_options) as face_lm,
        ):
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                if index % step != 0:
                    index += 1
                    continue
                timestamp_ms = int((index / native_fps) * 1000)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)
                pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
                face_result = face_lm.detect_for_video(mp_image, timestamp_ms)

                hand_pts = hand_result.hand_landmarks
                pose_pts = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                face_pts = (
                    face_result.face_landmarks[0] if face_result.face_landmarks else None
                )
                flat = self._flatten_detection(hand_pts, pose_pts, face_pts)
                frames.append(ExtractedFrame(timestamp_ms=timestamp_ms, landmarks=flat))
                index += 1

        capture.release()
        return frames

    def extract_from_bytes(self, data: bytes, suffix: str = ".mp4") -> list[ExtractedFrame]:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            return self.extract_from_path(Path(tmp.name))


def sliding_windows(
    frames: list[ExtractedFrame], window_size: int, stride: int
) -> Iterator[list[ExtractedFrame]]:
    if not frames:
        return
    if len(frames) <= window_size:
        yield frames
        return
    for start in range(0, len(frames) - window_size + 1, stride):
        yield frames[start : start + window_size]
