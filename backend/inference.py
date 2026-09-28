"""ONNX Runtime inference session and landmark-window preprocessing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from backend.config_loader import (
    get_config,
    get_landmark_settings,
    repo_root,
    resolve_label_map_path,
    resolve_models_dir,
    resolve_onnx_model_path,
)
from ml.sequence_padding import fit_sequence_length
from ml.preprocess.live_window import preprocess_landmark_frames


@dataclass(frozen=True)
class InferenceResult:
    word: str
    confidence: float
    top_k: list[tuple[str, float]]
    model_version: str
    vocab_size: int
    latency_ms: float


class ModelNotLoadedError(RuntimeError):
    """Raised when no ONNX model is available and stubs are disabled."""


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def ensure_stub_onnx(stub_path: Path, label_map_path: Path) -> None:
    """Create a tiny deterministic ONNX classifier when Agent 4 export is absent."""
    if stub_path.is_file():
        try:
            ort.InferenceSession(str(stub_path), providers=["CPUExecutionProvider"])
            return
        except Exception:
            stub_path.unlink(missing_ok=True)
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    if not label_map_path.is_file():
        raise FileNotFoundError(f"Missing stub label map at {label_map_path}")

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    with label_map_path.open(encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    num_classes = len(labels)
    feature_dim = 392
    sequence_t = 64

    rng = np.random.default_rng(42)
    weights = (rng.standard_normal((feature_dim, num_classes)) * 0.01).astype(
        np.float32
    )
    bias = (rng.standard_normal(num_classes) * 0.01).astype(np.float32)

    landmarks_info = helper.make_tensor_value_info(
        "landmarks",
        TensorProto.FLOAT,
        ["batch", sequence_t, feature_dim],
    )
    logits_info = helper.make_tensor_value_info(
        "logits",
        TensorProto.FLOAT,
        ["batch", num_classes],
    )
    w_tensor = numpy_helper.from_array(weights, name="W")
    b_tensor = numpy_helper.from_array(bias, name="B")
    mean_node = helper.make_node(
        "ReduceMean",
        inputs=["landmarks"],
        outputs=["pooled"],
        axes=[1],
        keepdims=0,
    )
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["pooled", "W", "B"],
        outputs=["logits"],
    )
    graph = helper.make_graph(
        [mean_node, gemm_node],
        "stub_sign_classifier",
        [landmarks_info],
        [logits_info],
        initializer=[w_tensor, b_tensor],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    if stub_path.is_file():
        stub_path.unlink()
    onnx.save(model, stub_path)


class OnnxSignClassifier:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or get_config()
        self._landmark_cfg = get_landmark_settings(self._config)
        self._sequence_T = self._landmark_cfg["sequence_length_T"]
        self._feature_dim = self._landmark_cfg["input_dim_per_frame"]
        inference = self._config.get("inference", {})
        self._allow_stub = bool(inference.get("allow_stub_model", True))

        self._onnx_path = resolve_onnx_model_path(self._config)
        self._label_map_path = resolve_label_map_path(self._config, self._onnx_path)
        if self._allow_stub and "stub" in self._onnx_path.parts:
            ensure_stub_onnx(self._onnx_path, self._label_map_path)

        if not self._onnx_path.is_file():
            raise ModelNotLoadedError(
                f"ONNX model not found at {self._onnx_path}. "
                "Export with ml/export_onnx.py or enable allow_stub_model."
            )

        with self._label_map_path.open(encoding="utf-8") as f:
            label_data = json.load(f)
        labels = label_data.get("labels") or label_data.get("classes")
        if not isinstance(labels, list) or not labels:
            index_to_label = label_data.get("index_to_label")
            if isinstance(index_to_label, dict):
                labels = [
                    str(index_to_label[str(i)])
                    for i in range(len(index_to_label))
                    if str(i) in index_to_label
                ]
            else:
                label_to_index = label_data.get("label_to_index")
                if isinstance(label_to_index, dict):
                    labels = [
                        gloss
                        for gloss, _idx in sorted(
                            label_to_index.items(), key=lambda item: item[1]
                        )
                    ]
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Invalid label map at {self._label_map_path}")
        self._labels: list[str] = [str(x) for x in labels]

        providers = inference.get("onnx_runtime_providers") or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(self._onnx_path),
            providers=[str(p) for p in providers],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self.model_version = (
            "stub-deterministic-v1"
            if "stub" in self._onnx_path.parts
            else self._onnx_path.stem
        )

    @property
    def onnx_path(self) -> Path:
        return self._onnx_path

    @property
    def is_stub(self) -> bool:
        return "stub" in self._onnx_path.parts

    @property
    def vocab_size(self) -> int:
        return len(self._labels)

    def _frames_to_tensor(self, frames: list[list[float]]) -> np.ndarray:
        if not frames:
            raise ValueError("landmark window must include at least one frame")
        packed = preprocess_landmark_frames(frames)  # (T_in, 392)
        seq3 = packed[:, np.newaxis, :]
        fitted, _mask = fit_sequence_length(seq3, self._sequence_T)
        flat = fitted.reshape(self._sequence_T, self._feature_dim)
        return flat[np.newaxis, ...].astype(np.float32)

    def predict_window(self, frames: list[list[float]], top_k: int = 5) -> InferenceResult:
        started = time.perf_counter()
        tensor = self._frames_to_tensor(frames)
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: tensor},
        )
        logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if logits.size != len(self._labels):
            raise RuntimeError(
                f"Model output size {logits.size} != vocab {len(self._labels)}"
            )
        probs = _softmax(logits)
        order = np.argsort(probs)[::-1]
        k = min(top_k, len(self._labels))
        top_indices = order[:k]
        pairs = [(self._labels[int(i)], float(probs[int(i)])) for i in top_indices]
        best_word, best_conf = pairs[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return InferenceResult(
            word=best_word,
            confidence=best_conf,
            top_k=pairs,
            model_version=self.model_version,
            vocab_size=self.vocab_size,
            latency_ms=latency_ms,
        )


_classifier: OnnxSignClassifier | None = None


def get_classifier(config: dict[str, Any] | None = None) -> OnnxSignClassifier:
    global _classifier
    if _classifier is None:
        _classifier = OnnxSignClassifier(config)
    return _classifier


def reset_classifier_for_tests() -> None:
    global _classifier
    _classifier = None
