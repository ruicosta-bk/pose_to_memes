"""Scale- and translation-invariant pose & hand feature extraction."""

from __future__ import annotations

from typing import Optional

import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
POSE_VECTOR_DIM = (POSE_LANDMARK_COUNT + 2 * HAND_LANDMARK_COUNT) * 2

# Weight upper-body & arm joints (13..22) while zeroing redundant head (1..10)
# and noisy/off-screen lower-body (25..32) landmarks so distinct poses separate cleanly.
_POSE_WEIGHTS = np.zeros((POSE_LANDMARK_COUNT, 1), dtype=np.float32)
_POSE_WEIGHTS[0, 0] = 0.5
_POSE_WEIGHTS[11:13, 0] = 0.5
_POSE_WEIGHTS[13:23, 0] = 1.0
_POSE_WEIGHTS[23:25, 0] = 0.15


def extract_pose_vector(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
) -> Optional[np.ndarray]:
    """Extract a translation/scale-invariant, L2-normalized 1D float32 feature vector.

    Coordinates are centered at mid-shoulder ``(L_SHOULDER + R_SHOULDER) / 2``
    and scaled by shoulder width ``||L_SHOULDER - R_SHOULDER||``. Missing hand
    slots (Left/Right) are zero-padded so the output length is always
    ``POSE_VECTOR_DIM`` (150).
    """
    if pose is None:
        return None

    pose_arr = np.asarray(pose, dtype=np.float32)
    if pose_arr.ndim != 2 or pose_arr.shape[0] < POSE_LANDMARK_COUNT or pose_arr.shape[1] < 2:
        return None

    pose_xy = pose_arr[:POSE_LANDMARK_COUNT, :2]
    l_shoulder = pose_xy[LEFT_SHOULDER]
    r_shoulder = pose_xy[RIGHT_SHOULDER]

    mid_shoulder = (l_shoulder + r_shoulder) * 0.5
    shoulder_width = float(np.linalg.norm(l_shoulder - r_shoulder))
    if not np.isfinite(shoulder_width) or shoulder_width < 1e-6:
        return None

    pose_norm = ((pose_xy - mid_shoulder) / shoulder_width) * _POSE_WEIGHTS

    left_hand = np.zeros((HAND_LANDMARK_COUNT, 2), dtype=np.float32)
    right_hand = np.zeros((HAND_LANDMARK_COUNT, 2), dtype=np.float32)
    left_filled = False
    right_filled = False

    if hands:
        for label, arr in hands:
            if arr is None:
                continue
            hand_arr = np.asarray(arr, dtype=np.float32)
            if hand_arr.ndim != 2 or hand_arr.shape[0] < HAND_LANDMARK_COUNT or hand_arr.shape[1] < 2:
                continue

            hand_norm = (hand_arr[:HAND_LANDMARK_COUNT, :2] - mid_shoulder) / shoulder_width
            lbl = str(label).strip().lower()
            if "left" in lbl and not left_filled:
                left_hand = hand_norm
                left_filled = True
            elif "right" in lbl and not right_filled:
                right_hand = hand_norm
                right_filled = True
            elif not left_filled:
                left_hand = hand_norm
                left_filled = True
            elif not right_filled:
                right_hand = hand_norm
                right_filled = True

    vec = np.concatenate(
        [pose_norm.ravel(), left_hand.ravel(), right_hand.ravel()]
    ).astype(np.float32, copy=False)

    if not np.all(np.isfinite(vec)):
        return None

    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None

    return (vec / norm).astype(np.float32, copy=False)


def cosine_similarity(
    vec_a: Optional[np.ndarray],
    vec_b: Optional[np.ndarray],
) -> float:
    """Compute cosine similarity in ``[-1.0, 1.0]`` between two 1D vectors."""
    if vec_a is None or vec_b is None:
        return 0.0

    a = np.asarray(vec_a, dtype=np.float32).ravel()
    b = np.asarray(vec_b, dtype=np.float32).ravel()
    if a.size == 0 or a.shape != b.shape:
        return 0.0
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(sim, -1.0, 1.0))
