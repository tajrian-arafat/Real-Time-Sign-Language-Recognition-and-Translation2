"""~130-point MediaPipe Holistic subset aligned with architecture (392-dim frames)."""
from __future__ import annotations

from typing import Any

import yaml

from ml.preprocess.paths import repo_root

HOLISTIC_LANDMARK_COUNT = 543
NUM_HAND_LANDMARKS = 21
POSE_LANDMARK_COUNT = 33
FACE_MESH_LANDMARK_COUNT = 468

# Upper-body pose (MediaPipe pose indices 0–32 in holistic layout).
POSE_LANDMARK_INDICES: list[int] = [
    0,  # nose
    11,
    12,  # shoulders (normalization anchors)
    13,
    14,  # elbows
    15,
    16,  # wrists
    17,
    18,  # pinkies
    23,
    24,  # hips
    25,
]

# Face mesh indices (local 0–467); offset +33 in holistic flat index.
# Lips, eyes, brows, nose — trimmed to 76 points (matches 130 total with both hands).
FACE_MESH_LOCAL_INDICES: list[int] = [
    0,
    1,
    2,
    4,
    5,
    6,
    7,
    13,
    14,
    17,
    19,
    33,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    94,
    95,
    146,
    178,
    181,
    185,
    191,
    195,
    197,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
    144,
    145,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    163,
    173,
    246,
    249,
    263,
    362,
    373,
    374,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    390,
    398,
]

assert len(FACE_MESH_LOCAL_INDICES) == 76

LEFT_HAND_HOLISTIC_START = POSE_LANDMARK_COUNT + FACE_MESH_LANDMARK_COUNT  # 501
RIGHT_HAND_HOLISTIC_START = LEFT_HAND_HOLISTIC_START + NUM_HAND_LANDMARKS  # 522

HAND_LEFT_INDICES: list[int] = list(
    range(LEFT_HAND_HOLISTIC_START, LEFT_HAND_HOLISTIC_START + NUM_HAND_LANDMARKS)
)
HAND_RIGHT_INDICES: list[int] = list(
    range(RIGHT_HAND_HOLISTIC_START, RIGHT_HAND_HOLISTIC_START + NUM_HAND_LANDMARKS)
)

FACE_HOLISTIC_INDICES: list[int] = [POSE_LANDMARK_COUNT + i for i in FACE_MESH_LOCAL_INDICES]

SELECTED_HOLISTIC_INDICES: list[int] = (
    POSE_LANDMARK_INDICES + FACE_HOLISTIC_INDICES + HAND_LEFT_INDICES + HAND_RIGHT_INDICES
)

assert len(SELECTED_HOLISTIC_INDICES) == 130

# Shoulder pair for normalization (pose indices in holistic space).
SHOULDER_LEFT_HOLISTIC = 11
SHOULDER_RIGHT_HOLISTIC = 12

SELECTED_LANDMARK_COUNT = len(SELECTED_HOLISTIC_INDICES)
COORDS_PER_LANDMARK = 3
HAND_PRESENCE_FEATURES = 2
FEATURES_PER_FRAME = SELECTED_LANDMARK_COUNT * COORDS_PER_LANDMARK + HAND_PRESENCE_FEATURES  # 392


def load_landmark_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return landmark subset spec; prefers config.yaml when lists are populated."""
    lm = config.get("landmarks", {})
    hand = lm.get("hand_landmark_indices") or []
    pose = lm.get("pose_landmark_indices") or []
    face = lm.get("face_landmark_indices") or []
    if hand and pose and face:
        indices = pose + face + hand
        return {
            "selected_holistic_indices": indices,
            "shoulder_left": SHOULDER_LEFT_HOLISTIC,
            "shoulder_right": SHOULDER_RIGHT_HOLISTIC,
            "features_per_frame": len(indices) * 3 + 2,
        }
    return {
        "selected_holistic_indices": SELECTED_HOLISTIC_INDICES,
        "shoulder_left": SHOULDER_LEFT_HOLISTIC,
        "shoulder_right": SHOULDER_RIGHT_HOLISTIC,
        "features_per_frame": FEATURES_PER_FRAME,
    }


def sync_landmark_indices_to_config_yaml() -> None:
    """Write default index lists into config/config.yaml for documentation."""
    path = repo_root() / "config" / "config.yaml"
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["landmarks"]["hand_landmark_indices"] = HAND_LEFT_INDICES + HAND_RIGHT_INDICES
    cfg["landmarks"]["pose_landmark_indices"] = POSE_LANDMARK_INDICES
    cfg["landmarks"]["face_landmark_indices"] = FACE_HOLISTIC_INDICES
    cfg["landmarks"]["input_dim_per_frame"] = FEATURES_PER_FRAME
    cfg["landmarks"]["selected_landmark_count"] = SELECTED_LANDMARK_COUNT
    cfg["landmarks"]["holistic_index_documentation"] = (
        "Pose uses MediaPipe pose indices 0-32 at start of 543-vector; "
        "face local indices offset +33; left hand +501; right hand +522."
    )
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
