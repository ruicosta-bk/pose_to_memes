"""Tier 1 rule-based geometric and blendshape detectors for built-in meme poses."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

BUILTIN_POSES: tuple[str, ...] = (
    "ABSOLUTE_CINEMA",
    "ERM_ACTUALLY",
    "THINKING",
    "SHOCKED",
    "DOMAIN_EXPANSION",
    "PEACE_SIGN",
    "SIGMA_RIZZ",
)

MIN_RULE_CONFIDENCE = 0.35


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ramp_up(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return float((value - low) / (high - low))


def _ramp_down(value: float, low: float, high: float) -> float:
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    return float((high - value) / (high - low))


def _band_score(
    value: float,
    ideal_low: float,
    ideal_high: float,
    outer_low: float,
    outer_high: float,
) -> float:
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return _ramp_up(value, outer_low, ideal_low)
    return _ramp_down(value, ideal_high, outer_high)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    cos_val = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return math.degrees(math.acos(cos_val))


def _valid_pose_metrics(pose: Optional[np.ndarray]) -> Optional[dict[str, object]]:
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 33 or arr.shape[1] < 2:
        return None
    if not np.all(np.isfinite(arr[:25, :2])):
        return None

    l_sh = arr[11, :2]
    r_sh = arr[12, :2]
    sh_width = _dist(l_sh, r_sh)
    if sh_width < 1e-6:
        return None

    return {
        "arr": arr,
        "nose": arr[0, :2],
        "l_ear": arr[7, :2],
        "r_ear": arr[8, :2],
        "mouth_l": arr[9, :2],
        "mouth_r": arr[10, :2],
        "l_sh": l_sh,
        "r_sh": r_sh,
        "l_el": arr[13, :2],
        "r_el": arr[14, :2],
        "l_wr": arr[15, :2],
        "r_wr": arr[16, :2],
        "mid_sh": (l_sh + r_sh) * 0.5,
        "sh_width": sh_width,
    }


def _valid_hands(hands: Optional[list[tuple[str, np.ndarray]]]) -> list[np.ndarray]:
    if not hands:
        return []
    out: list[np.ndarray] = []
    for _, arr in hands:
        if arr is None:
            continue
        h = np.asarray(arr, dtype=np.float32)
        if h.ndim == 2 and h.shape[0] >= 21 and h.shape[1] >= 2 and np.all(np.isfinite(h[:21, :2])):
            out.append(h[:21, :2])
    return out


def _valid_face(face: Optional[np.ndarray], min_points: int = 292) -> Optional[np.ndarray]:
    if face is None:
        return None
    arr = np.asarray(face, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < min_points or arr.shape[1] < 2:
        return None
    if not np.all(np.isfinite(arr[:min_points, :2])):
        return None
    return arr


def _finger_extended_score(hand: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> float:
    wrist = hand[0]
    d_tip = _dist(wrist, hand[tip])
    d_pip = _dist(wrist, hand[pip])
    tip_pip_ratio = d_tip / max(d_pip, 1e-6)

    seg_sum = (
        _dist(hand[mcp], hand[pip])
        + _dist(hand[pip], hand[dip])
        + _dist(hand[dip], hand[tip])
    )
    straight = _dist(hand[mcp], hand[tip]) / max(seg_sum, 1e-6)

    score = 0.55 * _ramp_up(tip_pip_ratio, 1.05, 1.32) + 0.45 * _ramp_up(straight, 0.72, 0.90)
    return float(np.clip(score, 0.0, 1.0))


def _finger_curled_score(hand: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> float:
    wrist = hand[0]
    d_tip = _dist(wrist, hand[tip])
    d_pip = _dist(wrist, hand[pip])
    tip_pip_ratio = d_tip / max(d_pip, 1e-6)

    seg_sum = (
        _dist(hand[mcp], hand[pip])
        + _dist(hand[pip], hand[dip])
        + _dist(hand[dip], hand[tip])
    )
    straight = _dist(hand[mcp], hand[tip]) / max(seg_sum, 1e-6)

    curl_dist = _ramp_down(tip_pip_ratio, 0.98, 1.22)
    curl_bend = _ramp_down(straight, 0.62, 0.85)
    return float(max(curl_dist, curl_bend))


def detect_absolute_cinema(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
    face: Optional[np.ndarray] = None,
) -> float:
    """Both hands raised near/above head level, elbows bent (~60-135 deg), arms wide."""
    m = _valid_pose_metrics(pose)
    if m is None:
        return 0.0

    sh_w = float(m["sh_width"])
    mid_sh = m["mid_sh"]
    l_sh, r_sh = m["l_sh"], m["r_sh"]
    l_el, r_el = m["l_el"], m["r_el"]
    l_wr, r_wr = m["l_wr"], m["r_wr"]

    lift_l = float(mid_sh[1] - l_wr[1]) / sh_w
    lift_r = float(mid_sh[1] - r_wr[1]) / sh_w
    height_l = _ramp_up(lift_l, 0.15, 0.45)
    height_r = _ramp_up(lift_r, 0.15, 0.45)
    height_score = min(height_l, height_r)

    forearm_l = _ramp_up(float(l_el[1] - l_wr[1]) / sh_w, 0.05, 0.25)
    forearm_r = _ramp_up(float(r_el[1] - r_wr[1]) / sh_w, 0.05, 0.25)
    forearm_score = math.sqrt(forearm_l * forearm_r)

    ang_l = _angle_deg(l_sh, l_el, l_wr)
    ang_r = _angle_deg(r_sh, r_el, r_wr)
    elbow_l = _band_score(ang_l, 60.0, 135.0, 35.0, 160.0)
    elbow_r = _band_score(ang_r, 60.0, 135.0, 35.0, 160.0)
    elbow_score = math.sqrt(elbow_l * elbow_r)

    wrist_sep = abs(float(l_wr[0] - r_wr[0])) / sh_w
    elbow_sep = abs(float(l_el[0] - r_el[0])) / sh_w
    wide_score = _ramp_up(wrist_sep, 0.85, 1.30) * _ramp_up(elbow_sep, 0.85, 1.25)

    combined = height_score * forearm_score * elbow_score * wide_score
    return float(np.clip(combined ** 0.25 if combined > 0.0 else 0.0, 0.0, 1.0))


def detect_erm_actually(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
    face: Optional[np.ndarray] = None,
) -> float:
    """One index finger extended straight up, middle/ring/pinky curled, near shoulder/face."""
    hand_list = _valid_hands(hands)
    if not hand_list:
        return 0.0

    m = _valid_pose_metrics(pose)
    best = 0.0

    for h in hand_list:
        idx_ext = _finger_extended_score(h, 5, 6, 7, 8)
        mid_curl = _finger_curled_score(h, 9, 10, 11, 12)
        ring_curl = _finger_curled_score(h, 13, 14, 15, 16)
        pinky_curl = _finger_curled_score(h, 17, 18, 19, 20)

        vec = h[8] - h[5]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            continue
        up_cos = float(-vec[1]) / norm
        up_score = _ramp_up(up_cos, 0.50, 0.85)

        if m is not None:
            rel_y = float(m["mid_sh"][1] - h[0, 1]) / float(m["sh_width"])
            height_score = _band_score(rel_y, -0.25, 0.90, -0.60, 1.30)
        else:
            height_score = 1.0

        curls = mid_curl * (0.5 * min(ring_curl, pinky_curl) + 0.25 * (ring_curl + pinky_curl))
        raw = idx_ext * curls * up_score * height_score
        score = float(raw ** 0.35) if raw > 0.0 else 0.0
        best = max(best, score)

    return float(np.clip(best, 0.0, 1.0))


def detect_thinking(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
    face: Optional[np.ndarray] = None,
) -> float:
    """At least one fingertip (thumb/index/middle) touching or close to chin/mouth."""
    m = _valid_pose_metrics(pose)
    targets: list[np.ndarray] = []
    scale = 0.0

    face_arr = _valid_face(face, 292)

    if face_arr is not None:
        mouth_c = (face_arr[13, :2] + face_arr[14, :2]) * 0.5
        mouth_w = _dist(face_arr[61, :2], face_arr[291, :2])
        chin = face_arr[152, :2]
        chin_dist = _dist(chin, mouth_c)
        if chin[1] <= mouth_c[1] or chin_dist < max(mouth_w * 0.2, 1e-5):
            chin = mouth_c + np.array([0.0, max(mouth_w * 0.65, 0.03)], dtype=np.float32)

        targets = [mouth_c, chin, (mouth_c + chin) * 0.5]
        if m is not None:
            scale = float(m["sh_width"])
        else:
            scale = max(mouth_w * 2.5, _dist(mouth_c, chin) * 3.0, 0.05)
    elif m is not None:
        mouth_l, mouth_r = m["mouth_l"], m["mouth_r"]
        nose = m["nose"]
        sh_w = float(m["sh_width"])
        if _dist(mouth_l, mouth_r) > 1e-5:
            mouth_c = (mouth_l + mouth_r) * 0.5
        else:
            mouth_c = nose + np.array([0.0, 0.12 * sh_w], dtype=np.float32)

        if mouth_c[1] > nose[1] + 1e-4:
            chin_est = mouth_c + (mouth_c - nose) * 0.85
        else:
            chin_est = mouth_c + np.array([0.0, 0.14 * sh_w], dtype=np.float32)

        targets = [mouth_c, chin_est, (mouth_c + chin_est) * 0.5]
        scale = sh_w
    else:
        return 0.0

    hand_list = _valid_hands(hands)
    candidate_pts: list[np.ndarray] = []
    used_pose_fallback = False

    if hand_list:
        for h in hand_list:
            for tip_idx in (4, 8, 12):
                candidate_pts.append(h[tip_idx])
    elif m is not None:
        used_pose_fallback = True
        pose_arr = m["arr"]
        for idx in (19, 20, 21, 22):
            candidate_pts.append(pose_arr[idx, :2])
    else:
        return 0.0

    min_d = min(_dist(pt, tgt) for pt in candidate_pts for tgt in targets)
    rel_d = min_d / max(scale, 1e-6)
    score = _ramp_down(rel_d, low=0.07, high=0.22)
    if used_pose_fallback:
        score *= 0.85
    return float(np.clip(score, 0.0, 1.0))


def detect_shocked(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
    face: Optional[np.ndarray] = None,
    face_blendshapes: Optional[dict[str, float]] = None,
) -> float:
    """Large Mouth Aspect Ratio (MAR) optionally combined with jawOpen/eyeWide/browInnerUp."""
    mar_score = 0.0
    arr = _valid_face(face, 292)
    if arr is not None:
        vertical = _dist(arr[13, :2], arr[14, :2])
        w_inner = (
            _dist(arr[78, :2], arr[308, :2])
            if arr.shape[0] >= 309 and np.all(np.isfinite(arr[[78, 308], :2]))
            else 0.0
        )
        w_outer = _dist(arr[61, :2], arr[291, :2])
        width = w_inner if w_inner > 1e-4 else w_outer
        if width >= 1e-5:
            mar = vertical / width
            mar_score = _ramp_up(mar, 0.22, 0.55)

    if not face_blendshapes:
        return mar_score

    jaw_open = float(face_blendshapes.get("jawOpen", 0.0))
    eye_wide = 0.5 * (
        float(face_blendshapes.get("eyeWideLeft", 0.0))
        + float(face_blendshapes.get("eyeWideRight", 0.0))
    )
    brow_up = float(face_blendshapes.get("browInnerUp", 0.0))

    jaw_score = _ramp_up(jaw_open, 0.25, 0.65)
    surprise_expr = 0.55 * _ramp_up(eye_wide, 0.15, 0.55) + 0.45 * _ramp_up(brow_up, 0.15, 0.55)
    bs_score = float(np.clip(0.70 * jaw_score + 0.30 * surprise_expr, 0.0, 1.0))

    if mar_score > 0.1 and bs_score > 0.1:
        return float(np.clip(0.4 * mar_score + 0.6 * bs_score, 0.0, 1.0))
    return float(np.clip(max(mar_score, bs_score), 0.0, 1.0))


def detect_sigma_rizz(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
    face: Optional[np.ndarray] = None,
    face_blendshapes: Optional[dict[str, float]] = None,
) -> float:
    """Detect Sigma/Rizz expression via ARKit blendshapes with FaceMesh landmark fallback."""
    if face_blendshapes:
        jaw_open = float(face_blendshapes.get("jawOpen", 0.0))
        if jaw_open >= 0.35:
            return 0.0

        pucker = max(
            float(face_blendshapes.get("mouthPucker", 0.0)),
            float(face_blendshapes.get("mouthFunnel", 0.0)) * 0.85,
            0.45
            * (
                float(face_blendshapes.get("mouthPressLeft", 0.0))
                + float(face_blendshapes.get("mouthPressRight", 0.0))
            ),
        )
        squint = 0.5 * (
            float(face_blendshapes.get("eyeSquintLeft", 0.0))
            + float(face_blendshapes.get("eyeSquintRight", 0.0))
        )
        brow_l = float(face_blendshapes.get("browOuterUpLeft", 0.0))
        brow_r = float(face_blendshapes.get("browOuterUpRight", 0.0))
        brow_signal = min(1.0, 0.75 * max(brow_l, brow_r) + 0.35 * abs(brow_l - brow_r))

        mouth_score = _ramp_up(pucker, 0.18, 0.48)
        squint_score = _ramp_up(squint, 0.15, 0.45)
        brow_score = _ramp_up(brow_signal, 0.15, 0.45)

        if mouth_score <= 0.15 or (squint_score <= 0.15 and brow_score <= 0.15):
            return 0.0

        jaw_gate = _ramp_down(jaw_open, 0.25, 0.35)
        raw = (0.42 * mouth_score + 0.33 * squint_score + 0.25 * brow_score) * jaw_gate
        return float(np.clip(raw, 0.0, 1.0))

    # Landmark fallback when face_blendshapes is None.
    arr = _valid_face(face, 387)
    if arr is None:
        return 0.0

    inter_ocular = _dist(arr[33, :2], arr[263, :2])
    w_l = _dist(arr[33, :2], arr[133, :2])
    w_r = _dist(arr[362, :2], arr[263, :2])
    mouth_w = _dist(arr[61, :2], arr[291, :2])
    if inter_ocular < 1e-5 or w_l < 1e-5 or w_r < 1e-5 or mouth_w < 1e-5:
        return 0.0

    mar = _dist(arr[13, :2], arr[14, :2]) / mouth_w
    if mar >= 0.35:
        return 0.0
    mar_gate = _ramp_down(mar, 0.18, 0.35)

    ear_l = _dist(arr[159, :2], arr[145, :2]) / w_l
    ear_r = _dist(arr[386, :2], arr[374, :2]) / w_r
    squint_score = _ramp_down(0.5 * (ear_l + ear_r), 0.18, 0.28)

    brow_l = float(arr[159, 1] - arr[105, 1]) / inter_ocular
    brow_r = float(arr[386, 1] - arr[334, 1]) / inter_ocular
    brow_asym = abs(brow_l - brow_r)
    brow_raise = max(brow_l, brow_r)
    brow_score = max(_ramp_up(brow_asym, 0.04, 0.10), _ramp_up(brow_raise, 0.16, 0.24))

    mouth_ratio = mouth_w / inter_ocular
    pucker_score = _ramp_down(mouth_ratio, 0.40, 0.52)

    if pucker_score <= 0.15 or (squint_score <= 0.15 and brow_score <= 0.15):
        return 0.0

    raw = (0.42 * pucker_score + 0.33 * squint_score + 0.25 * brow_score) * mar_gate
    return float(np.clip(raw, 0.0, 1.0))


def detect_domain_expansion(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
    face: Optional[np.ndarray] = None,
) -> float:
    """Both hands clasped or close together in front of chest/chin."""
    hand_list = _valid_hands(hands)
    if len(hand_list) < 2:
        return 0.0

    m = _valid_pose_metrics(pose)
    best = 0.0

    for i in range(len(hand_list)):
        for j in range(i + 1, len(hand_list)):
            h1, h2 = hand_list[i], hand_list[j]
            palm1 = np.mean(h1[[0, 5, 9, 13, 17]], axis=0)
            palm2 = np.mean(h2[[0, 5, 9, 13, 17]], axis=0)
            tips1 = np.mean(h1[[4, 8, 12, 16, 20]], axis=0)
            tips2 = np.mean(h2[[4, 8, 12, 16, 20]], axis=0)

            wrist_dist = _dist(h1[0], h2[0])
            palm_dist = _dist(palm1, palm2)
            finger_dist = _dist(tips1, tips2)

            if m is not None:
                scale = float(m["sh_width"])
            else:
                palm_size = max(_dist(h1[0], h1[9]), _dist(h2[0], h2[9]), 1e-4)
                scale = palm_size * 4.0

            s_palm = _ramp_down(palm_dist / scale, 0.20, 0.55)
            s_wrist = _ramp_down(wrist_dist / scale, 0.25, 0.70)
            s_finger = _ramp_down(finger_dist / scale, 0.20, 0.55)
            prox = 0.45 * s_palm + 0.30 * s_wrist + 0.25 * s_finger

            if m is not None:
                center = 0.5 * (palm1 + palm2)
                dx = abs(float(center[0] - m["mid_sh"][0])) / scale
                dy = float(center[1] - m["mid_sh"][1]) / scale
                horiz = _ramp_down(dx, 0.30, 0.65)
                vert = _band_score(dy, -0.35, 0.65, -0.75, 1.05)
                pos_score = horiz * vert
            else:
                pos_score = 1.0

            best = max(best, prox * pos_score)

    return float(np.clip(best, 0.0, 1.0))


def detect_peace_sign(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
    face: Optional[np.ndarray] = None,
) -> float:
    """V-sign: index & middle extended and separated, ring & pinky curled."""
    hand_list = _valid_hands(hands)
    if not hand_list:
        return 0.0

    best = 0.0
    for h in hand_list:
        idx_ext = _finger_extended_score(h, 5, 6, 7, 8)
        mid_ext = _finger_extended_score(h, 9, 10, 11, 12)
        ring_curl = _finger_curled_score(h, 13, 14, 15, 16)
        pinky_curl = _finger_curled_score(h, 17, 18, 19, 20)

        v1 = h[8] - h[5]
        v2 = h[12] - h[9]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cos_v = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang_v = math.degrees(math.acos(cos_v))
        tip_gap = _dist(h[8], h[12])
        mcp_gap = max(_dist(h[5], h[9]), 1e-6)

        ang_score = _band_score(ang_v, 15.0, 65.0, 6.0, 95.0)
        gap_score = _ramp_up(tip_gap / mcp_gap, 1.15, 1.75)
        sep_score = max(ang_score, 0.7 * gap_score)

        ext_part = math.sqrt(idx_ext * mid_ext)
        curl_part = math.sqrt(ring_curl * pinky_curl)
        score = ext_part * curl_part * sep_score
        best = max(best, score)

    return float(np.clip(best, 0.0, 1.0))


def classify_builtin(
    pose: Optional[np.ndarray],
    hands: Optional[list[tuple[str, np.ndarray]]] = None,
    face: Optional[np.ndarray] = None,
    face_blendshapes: Optional[dict[str, float]] = None,
) -> tuple[Optional[str], float, dict[str, float]]:
    """Evaluate all built-in geometric and blendshape meme poses.

    Returns ``(best_pose, best_confidence, scores_by_pose)``. ``best_pose`` is
    ``None`` when no pose reaches ``MIN_RULE_CONFIDENCE``.
    """
    domain_score = detect_domain_expansion(pose, hands, face)
    thinking_score = detect_thinking(pose, hands, face)
    # Suppress single-hand chin-touch ambiguity when both hands are clasped together.
    if domain_score > 0.5:
        thinking_score *= max(0.0, 1.0 - 0.75 * domain_score)

    scores: dict[str, float] = {
        "ABSOLUTE_CINEMA": detect_absolute_cinema(pose, hands, face),
        "ERM_ACTUALLY": detect_erm_actually(pose, hands, face),
        "THINKING": thinking_score,
        "SHOCKED": detect_shocked(pose, hands, face, face_blendshapes=face_blendshapes),
        "DOMAIN_EXPANSION": domain_score,
        "PEACE_SIGN": detect_peace_sign(pose, hands, face),
        "SIGMA_RIZZ": detect_sigma_rizz(pose, hands, face, face_blendshapes=face_blendshapes),
    }

    best_pose = max(scores, key=scores.__getitem__)
    best_conf = scores[best_pose]
    if best_conf < MIN_RULE_CONFIDENCE:
        return None, best_conf, scores
    return best_pose, best_conf, scores
