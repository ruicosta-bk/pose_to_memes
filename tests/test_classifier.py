"""Unit and behavioral tests for Pose-to-Meme Studio classifier package."""

from __future__ import annotations

import numpy as np
import pytest

from classifier import (
    BUILTIN_POSES,
    FACE_VECTOR_DIM,
    POSE_VECTOR_DIM,
    CustomPoseEntry,
    CustomPoseStore,
    TemporalStabilizer,
    classify_builtin,
    cosine_similarity,
    extract_face_blendshape_vector,
    extract_pose_vector,
)


def make_neutral_pose(
    cx: float = 0.5,
    mid_y: float = 0.45,
    shoulder_w: float = 0.30,
) -> np.ndarray:
    """Create a synthetic (33, 4) neutral MediaPipe Pose array in image coords."""
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 0.99

    half = shoulder_w * 0.5
    # MediaPipe convention: +x right in image, +y down in image.
    pose[0, :2] = [cx, mid_y - 0.55 * shoulder_w]  # nose
    # Eyes (1-6) near nose
    pose[1, :2] = [cx + 0.05 * shoulder_w, mid_y - 0.62 * shoulder_w]
    pose[2, :2] = [cx + 0.08 * shoulder_w, mid_y - 0.62 * shoulder_w]
    pose[3, :2] = [cx + 0.11 * shoulder_w, mid_y - 0.62 * shoulder_w]
    pose[4, :2] = [cx - 0.05 * shoulder_w, mid_y - 0.62 * shoulder_w]
    pose[5, :2] = [cx - 0.08 * shoulder_w, mid_y - 0.62 * shoulder_w]
    pose[6, :2] = [cx - 0.11 * shoulder_w, mid_y - 0.62 * shoulder_w]
    pose[7, :2] = [cx + 0.22 * shoulder_w, mid_y - 0.55 * shoulder_w]  # left ear
    pose[8, :2] = [cx - 0.22 * shoulder_w, mid_y - 0.55 * shoulder_w]  # right ear
    pose[9, :2] = [cx + 0.08 * shoulder_w, mid_y - 0.42 * shoulder_w]  # mouth left
    pose[10, :2] = [cx - 0.08 * shoulder_w, mid_y - 0.42 * shoulder_w]  # mouth right

    # Shoulders: L=11 (+x), R=12 (-x)
    pose[11, :2] = [cx + half, mid_y]
    pose[12, :2] = [cx - half, mid_y]

    # Relaxed arms hanging down
    pose[13, :2] = [cx + half * 1.15, mid_y + 0.65 * shoulder_w]  # L elbow
    pose[14, :2] = [cx - half * 1.15, mid_y + 0.65 * shoulder_w]  # R elbow
    pose[15, :2] = [cx + half * 1.15, mid_y + 1.25 * shoulder_w]  # L wrist
    pose[16, :2] = [cx - half * 1.15, mid_y + 1.25 * shoulder_w]  # R wrist
    pose[17, :2] = [cx + half * 1.20, mid_y + 1.36 * shoulder_w]  # L pinky
    pose[18, :2] = [cx - half * 1.20, mid_y + 1.36 * shoulder_w]  # R pinky
    pose[19, :2] = [cx + half * 1.15, mid_y + 1.35 * shoulder_w]
    pose[20, :2] = [cx - half * 1.15, mid_y + 1.35 * shoulder_w]
    pose[21, :2] = [cx + half * 1.10, mid_y + 1.32 * shoulder_w]
    pose[22, :2] = [cx - half * 1.10, mid_y + 1.32 * shoulder_w]

    # Hips & legs (23-32)
    pose[23, :2] = [cx + 0.35 * shoulder_w, mid_y + 1.40 * shoulder_w]
    pose[24, :2] = [cx - 0.35 * shoulder_w, mid_y + 1.40 * shoulder_w]
    pose[25, :2] = [cx + 0.35 * shoulder_w, mid_y + 2.20 * shoulder_w]
    pose[26, :2] = [cx - 0.35 * shoulder_w, mid_y + 2.20 * shoulder_w]
    pose[27, :2] = [cx + 0.35 * shoulder_w, mid_y + 2.95 * shoulder_w]
    pose[28, :2] = [cx - 0.35 * shoulder_w, mid_y + 2.95 * shoulder_w]
    pose[29, :2] = [cx + 0.37 * shoulder_w, mid_y + 3.05 * shoulder_w]
    pose[30, :2] = [cx - 0.37 * shoulder_w, mid_y + 3.05 * shoulder_w]
    pose[31, :2] = [cx + 0.33 * shoulder_w, mid_y + 3.08 * shoulder_w]
    pose[32, :2] = [cx - 0.33 * shoulder_w, mid_y + 3.08 * shoulder_w]
    return pose


def make_neutral_face(
    cx: float = 0.5,
    mouth_y: float = 0.324,
    mouth_w: float = 0.08,
    mar: float = 0.05,
    eye_ear: float = 0.34,
    left_brow_lift: float = 0.012,
    right_brow_lift: float = 0.012,
) -> np.ndarray:
    """Create a synthetic (478, 3) MediaPipe FaceMesh array."""
    face = np.zeros((478, 3), dtype=np.float32)
    face[:, 0] = cx
    face[:, 1] = mouth_y

    half_w = mouth_w * 0.5
    half_h = (mar * mouth_w) * 0.5

    face[61, :2] = [cx - half_w, mouth_y]
    face[291, :2] = [cx + half_w, mouth_y]
    face[78, :2] = [cx - half_w * 0.85, mouth_y]
    face[308, :2] = [cx + half_w * 0.85, mouth_y]
    face[13, :2] = [cx, mouth_y - half_h]
    face[14, :2] = [cx, mouth_y + half_h]
    face[152, :2] = [cx, mouth_y + mouth_w * 0.70]  # chin

    # Eyes & brows (inter-ocular outer corners 33..263 = 0.14)
    eye_y = mouth_y - 0.065
    eye_w = 0.036
    face[33, :2] = [cx - 0.07, eye_y]
    face[133, :2] = [cx - 0.07 + eye_w, eye_y]
    face[159, :2] = [cx - 0.07 + eye_w * 0.5, eye_y - eye_ear * eye_w * 0.5]
    face[145, :2] = [cx - 0.07 + eye_w * 0.5, eye_y + eye_ear * eye_w * 0.5]
    face[105, :2] = [cx - 0.07 + eye_w * 0.5, face[159, 1] - left_brow_lift]

    face[263, :2] = [cx + 0.07, eye_y]
    face[362, :2] = [cx + 0.07 - eye_w, eye_y]
    face[386, :2] = [cx + 0.07 - eye_w * 0.5, eye_y - eye_ear * eye_w * 0.5]
    face[374, :2] = [cx + 0.07 - eye_w * 0.5, eye_y + eye_ear * eye_w * 0.5]
    face[334, :2] = [cx + 0.07 - eye_w * 0.5, face[386, 1] - right_brow_lift]
    return face


def _set_finger(
    hand: np.ndarray,
    mcp: int,
    pip: int,
    dip: int,
    tip: int,
    base_xy: np.ndarray,
    direction_xy: np.ndarray,
    length: float,
    curled: bool,
) -> None:
    d = direction_xy / max(float(np.linalg.norm(direction_xy)), 1e-6)
    perp = np.array([-d[1], d[0]], dtype=np.float32)
    hand[mcp, :2] = base_xy
    if not curled:
        hand[pip, :2] = base_xy + d * (0.40 * length)
        hand[dip, :2] = base_xy + d * (0.72 * length)
        hand[tip, :2] = base_xy + d * length
    else:
        # Folded back toward palm/wrist so tip is closer to wrist than PIP.
        hand[pip, :2] = base_xy + d * (0.35 * length)
        hand[dip, :2] = base_xy + d * (0.18 * length) + perp * (0.04 * length)
        hand[tip, :2] = base_xy - d * (0.10 * length)


def make_hand(
    wrist_xy: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.14,
    extended: tuple[bool, bool, bool, bool, bool] = (False, False, False, False, False),
    v_spread_deg: float = 30.0,
) -> np.ndarray:
    """Create a synthetic (21, 3) hand with configurable finger extension states."""
    hand = np.zeros((21, 3), dtype=np.float32)
    wrist = np.array(wrist_xy, dtype=np.float32)
    hand[0, :2] = wrist

    palm_h = 0.45 * scale
    # Thumb
    thumb_dir = np.array([-0.7, -0.7], dtype=np.float32)
    hand[1, :2] = wrist + np.array([-0.12 * scale, -0.10 * scale], dtype=np.float32)
    hand[2, :2] = wrist + np.array([-0.22 * scale, -0.22 * scale], dtype=np.float32)
    if extended[0]:
        hand[3, :2] = hand[2, :2] + thumb_dir * (0.20 * scale)
        hand[4, :2] = hand[2, :2] + thumb_dir * (0.42 * scale)
    else:
        hand[3, :2] = wrist + np.array([-0.10 * scale, -0.30 * scale], dtype=np.float32)
        hand[4, :2] = wrist + np.array([0.02 * scale, -0.25 * scale], dtype=np.float32)

    half_rad = np.deg2rad(v_spread_deg * 0.5)
    up_idx = np.array([-np.sin(half_rad), -np.cos(half_rad)], dtype=np.float32)
    up_mid = (
        np.array([np.sin(half_rad), -np.cos(half_rad)], dtype=np.float32)
        if (extended[1] and extended[2] and not extended[3] and not extended[4])
        else np.array([0.0, -1.0], dtype=np.float32)
    )
    up = np.array([0.0, -1.0], dtype=np.float32)

    _set_finger(
        hand, 5, 6, 7, 8,
        wrist + np.array([-0.14 * scale, -palm_h], dtype=np.float32),
        up_idx if (extended[1] and extended[2]) else up,
        0.55 * scale,
        curled=not extended[1],
    )
    _set_finger(
        hand, 9, 10, 11, 12,
        wrist + np.array([-0.04 * scale, -palm_h], dtype=np.float32),
        up_mid,
        0.60 * scale,
        curled=not extended[2],
    )
    _set_finger(
        hand, 13, 14, 15, 16,
        wrist + np.array([0.06 * scale, -palm_h], dtype=np.float32),
        up,
        0.55 * scale,
        curled=not extended[3],
    )
    _set_finger(
        hand, 17, 18, 19, 20,
        wrist + np.array([0.16 * scale, -palm_h], dtype=np.float32),
        up,
        0.46 * scale,
        curled=not extended[4],
    )
    return hand


def build_pose_sample(pose_name: str):
    """Return synthetic (pose, hands, face) for each named pose or NEUTRAL."""
    pose = make_neutral_pose()
    face = make_neutral_face()
    hands: list[tuple[str, np.ndarray]] = []

    if pose_name == "ABSOLUTE_CINEMA":
        # Elbows out wide, wrists raised above head/shoulder level with ~90 deg bend.
        pose[13, :2] = [0.76, 0.45]  # L elbow
        pose[14, :2] = [0.24, 0.45]  # R elbow
        pose[15, :2] = [0.76, 0.24]  # L wrist high
        pose[16, :2] = [0.24, 0.24]  # R wrist high
    elif pose_name == "ERM_ACTUALLY":
        hands = [("Right", make_hand(wrist_xy=(0.35, 0.35), extended=(False, True, False, False, False)))]
    elif pose_name == "THINKING":
        chin_xy = face[152, :2]
        # Resting curled/sideways thinking hand with thumb/index under chin (not vertical Erm Actually).
        h = make_hand(wrist_xy=(float(chin_xy[0]), float(chin_xy[1]) + 0.12), extended=(True, False, False, False, False))
        offset = chin_xy - h[4, :2]
        h[:, :2] += offset
        hands = [("Right", h)]
    elif pose_name == "SHOCKED":
        face = make_neutral_face(mar=0.65)
    elif pose_name == "DOMAIN_EXPANSION":
        h_l = make_hand(wrist_xy=(0.51, 0.46), extended=(True, True, True, True, True))
        h_r = make_hand(wrist_xy=(0.49, 0.46), extended=(True, True, True, True, True))
        hands = [("Left", h_l), ("Right", h_r)]
    elif pose_name == "PEACE_SIGN":
        hands = [("Right", make_hand(wrist_xy=(0.35, 0.40), extended=(False, True, True, False, False), v_spread_deg=35.0))]
    elif pose_name == "SIGMA_RIZZ":
        # Narrow pursed mouth, squinted eyes, asymmetric raised brow (landmark fallback)
        face = make_neutral_face(
            mouth_w=0.050,
            mar=0.05,
            eye_ear=0.14,
            left_brow_lift=0.030,
            right_brow_lift=0.010,
        )
    elif pose_name == "NEUTRAL":
        pass
    else:
        raise ValueError(f"Unknown pose sample: {pose_name}")

    return pose, hands, face


@pytest.mark.parametrize("pose_name", BUILTIN_POSES)
def test_builtin_poses_detected_with_high_confidence(pose_name: str) -> None:
    pose, hands, face = build_pose_sample(pose_name)
    best_pose, best_conf, scores = classify_builtin(pose, hands, face)
    assert set(scores.keys()) == set(BUILTIN_POSES)
    assert best_pose == pose_name, f"Expected {pose_name}, got {best_pose} ({scores})"
    assert best_conf >= 0.72, f"Expected >= 0.72 for {pose_name}, got {best_conf:.3f}"


def test_neutral_and_negative_cases() -> None:
    pose, hands, face = build_pose_sample("NEUTRAL")
    best_pose, best_conf, scores = classify_builtin(pose, hands, face)
    assert best_pose is None
    assert best_conf < 0.35
    assert all(0.0 <= v <= 1.0 for v in scores.values())

    # None inputs
    best_none, conf_none, scores_none = classify_builtin(None, None, None)
    assert best_none is None
    assert conf_none == 0.0
    assert all(v == 0.0 for v in scores_none.values())

    # NaN / Inf corrupted landmarks must not crash or trigger
    bad_pose = pose.copy()
    bad_pose[11, 0] = np.nan
    bad_face = face.copy()
    bad_face[13, 1] = np.inf
    bad_hand = make_hand()
    bad_hand[8, 0] = np.nan
    best_bad, conf_bad, _ = classify_builtin(bad_pose, [("Left", bad_hand)], bad_face)
    assert best_bad is None
    assert conf_bad == 0.0


def test_extract_pose_vector_invariance_and_edge_cases() -> None:
    pose, hands, _ = build_pose_sample("PEACE_SIGN")
    vec_base = extract_pose_vector(pose, hands)
    assert vec_base is not None
    assert vec_base.shape == (POSE_VECTOR_DIM,)
    assert vec_base.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec_base), 1.0, atol=1e-5)

    # Translation invariance
    pose_shift = pose.copy()
    pose_shift[:, 0] += 0.27
    pose_shift[:, 1] -= 0.19
    hands_shift = [(lbl, arr + np.array([0.27, -0.19, 0.0], dtype=np.float32)) for lbl, arr in hands]
    vec_shift = extract_pose_vector(pose_shift, hands_shift)
    assert vec_shift is not None
    assert np.allclose(vec_base, vec_shift, atol=1e-5)
    assert cosine_similarity(vec_base, vec_shift) > 0.9999

    # Scale invariance (uniform scaling around origin)
    pose_scale = pose.copy()
    pose_scale[:, :2] *= 2.35
    hands_scale = [
        (lbl, np.column_stack([arr[:, :2] * 2.35, arr[:, 2]]))
        for lbl, arr in hands
    ]
    vec_scale = extract_pose_vector(pose_scale, hands_scale)
    assert vec_scale is not None
    assert np.allclose(vec_base, vec_scale, atol=1e-5)
    assert cosine_similarity(vec_base, vec_scale) > 0.9999

    # Missing hands still yields fixed-length L2-normalized vector
    vec_no_hands = extract_pose_vector(pose, None)
    assert vec_no_hands is not None
    assert vec_no_hands.shape == (POSE_VECTOR_DIM,)
    assert np.isclose(np.linalg.norm(vec_no_hands), 1.0, atol=1e-5)

    # Degenerate shoulders (width == 0) -> None
    zero_sh = pose.copy()
    zero_sh[12, :2] = zero_sh[11, :2]
    assert extract_pose_vector(zero_sh, hands) is None
    assert extract_pose_vector(None, hands) is None


def test_custom_pose_store_matching_and_persistence(tmp_path) -> None:
    store_path = tmp_path / "custom_poses.json"
    store = CustomPoseStore(store_path)

    pose_a, hands_a, _ = build_pose_sample("PEACE_SIGN")
    vec_a = extract_pose_vector(pose_a, hands_a)
    assert vec_a is not None

    pose_b, hands_b, _ = build_pose_sample("ABSOLUTE_CINEMA")
    vec_b = extract_pose_vector(pose_b, hands_b)
    assert vec_b is not None

    entry = store.add_pose(
        "My Custom Dab",
        vector=vec_a,
        meme_path="assets/memes/dab.png",
        audio_path="assets/memes/dab.wav",
        threshold=0.88,
        pose_id="CUSTOM_DAB",
    )
    assert isinstance(entry, CustomPoseEntry)
    assert store_path.exists()

    # Exact match exceeds threshold
    matched, sim = store.match(vec_a)
    assert matched is not None
    assert matched.id == "CUSTOM_DAB"
    assert sim > 0.999

    # Unrelated poses must not match above 0.88
    miss_entry, miss_sim = store.match(vec_b)
    assert miss_entry is None
    assert miss_sim < 0.88

    pose_n, hands_n, _ = build_pose_sample("NEUTRAL")
    vec_n = extract_pose_vector(pose_n, hands_n)
    assert vec_n is not None
    assert cosine_similarity(vec_n, vec_b) < 0.88

    # Reload from disk
    reloaded = CustomPoseStore(store_path)
    listed = reloaded.list_poses()
    assert len(listed) == 1
    assert listed[0].id == "CUSTOM_DAB"
    assert listed[0].name == "My Custom Dab"
    reloaded.update_pose("CUSTOM_DAB", name="Renamed Dab")
    assert reloaded.get_pose("CUSTOM_DAB").name == "Renamed Dab"

    assert reloaded.remove_pose("CUSTOM_DAB") is True
    assert reloaded.remove_pose("CUSTOM_DAB") is False
    assert reloaded.list_poses() == []


def test_temporal_stabilizer_hold_hysteresis_and_cooldown() -> None:
    stabilizer = TemporalStabilizer(
        hold_ms=250,
        trigger_threshold=0.72,
        release_threshold=0.50,
        cooldown_ms=900,
        ema_alpha=0.50,
    )

    cinema_pose, cinema_hands, cinema_face = build_pose_sample("ABSOLUTE_CINEMA")
    neutral_pose, neutral_hands, neutral_face = build_pose_sample("NEUTRAL")

    # t=0: first detection starts hold timer, not active yet
    r0 = stabilizer.update(cinema_pose, cinema_hands, cinema_face, timestamp_ms=0)
    assert r0.active_pose is None
    assert r0.triggered_pose is None
    assert r0.candidate_pose == "ABSOLUTE_CINEMA"
    assert 0.0 < r0.confidence < r0.raw_confidence

    # t=200ms (<250ms): still holding, not triggered yet
    r1 = stabilizer.update(cinema_pose, cinema_hands, cinema_face, timestamp_ms=200)
    assert r1.active_pose is None
    assert r1.triggered_pose is None

    # t=250ms (>=250ms): triggers!
    r2 = stabilizer.update(cinema_pose, cinema_hands, cinema_face, timestamp_ms=250)
    assert r2.active_pose == "ABSOLUTE_CINEMA"
    assert r2.triggered_pose == "ABSOLUTE_CINEMA"

    # Hysteresis: confidence dips to ~0.595 (between release=0.50 and trigger=0.72)
    mid_pose = cinema_pose.copy()
    mid_pose[15, 1] = 0.39
    mid_pose[16, 1] = 0.39
    _, mid_conf, _ = classify_builtin(mid_pose, cinema_hands, cinema_face)
    assert 0.50 <= mid_conf < 0.72
    r_hyst = stabilizer.update(mid_pose, cinema_hands, cinema_face, timestamp_ms=400)
    assert r_hyst.active_pose == "ABSOLUTE_CINEMA"
    assert r_hyst.triggered_pose is None

    # Release when neutral (<0.50)
    r_rel = stabilizer.update(neutral_pose, neutral_hands, neutral_face, timestamp_ms=500)
    assert r_rel.active_pose is None

    # Cooldown: last trigger was at t=250ms, cooldown=900ms -> blocked until t=1150ms
    stabilizer.update(cinema_pose, cinema_hands, cinema_face, timestamp_ms=600)
    r_cd_block = stabilizer.update(cinema_pose, cinema_hands, cinema_face, timestamp_ms=900)
    assert r_cd_block.active_pose is None
    assert r_cd_block.triggered_pose is None

    # At t=1150ms (250 + 900) and held >=250ms since t=600 -> triggers again
    r_cd_ok = stabilizer.update(cinema_pose, cinema_hands, cinema_face, timestamp_ms=1150)
    assert r_cd_ok.active_pose == "ABSOLUTE_CINEMA"
    assert r_cd_ok.triggered_pose == "ABSOLUTE_CINEMA"


def test_temporal_stabilizer_with_custom_pose(tmp_path) -> None:
    store = CustomPoseStore(tmp_path / "custom.json")
    # Build a distinct custom pose (left arm straight out sideways, right arm straight up)
    custom_pose = make_neutral_pose()
    custom_pose[13, :2] = [0.80, 0.45]
    custom_pose[15, :2] = [0.95, 0.45]
    custom_pose[14, :2] = [0.32, 0.25]
    custom_pose[16, :2] = [0.32, 0.08]
    vec = extract_pose_vector(custom_pose, None)
    assert vec is not None
    entry = store.add_pose("Salute", vector=vec, threshold=0.90, pose_id="CUSTOM_SALUTE")

    stabilizer = TemporalStabilizer(custom_store=store, hold_ms=250, cooldown_ms=800)
    r0 = stabilizer.update(custom_pose, None, None, timestamp_ms=1000)
    assert r0.active_pose is None
    assert r0.candidate_pose == entry.id

    r1 = stabilizer.update(custom_pose, None, None, timestamp_ms=1260)
    assert r1.active_pose == entry.id
    assert r1.triggered_pose == entry.id
    assert r1.is_custom is True
    assert r1.custom_entry is not None and r1.custom_entry.id == entry.id


def test_face_blendshapes_sigma_rizz_and_shocked() -> None:
    neutral_pose = make_neutral_pose()
    neutral_face = make_neutral_face()

    # Neutral resting blendshapes -> None vector and no built-in trigger
    resting_bs = {"mouthPucker": 0.03, "eyeSquintLeft": 0.02, "jawOpen": 0.02}
    assert extract_face_blendshape_vector(None) is None
    assert extract_face_blendshape_vector(resting_bs) is None
    best_rest, conf_rest, _ = classify_builtin(neutral_pose, None, neutral_face, face_blendshapes=resting_bs)
    assert best_rest is None
    assert conf_rest < 0.35

    # SIGMA_RIZZ via blendshapes
    rizz_bs = {
        "mouthPucker": 0.68,
        "eyeSquintLeft": 0.55,
        "eyeSquintRight": 0.50,
        "browOuterUpLeft": 0.62,
        "browOuterUpRight": 0.12,
        "jawOpen": 0.05,
    }
    vec = extract_face_blendshape_vector(rizz_bs)
    assert vec is not None
    assert vec.shape == (FACE_VECTOR_DIM,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    best_rizz, conf_rizz, scores_rizz = classify_builtin(
        neutral_pose, None, neutral_face, face_blendshapes=rizz_bs
    )
    assert best_rizz == "SIGMA_RIZZ"
    assert conf_rizz >= 0.72
    assert scores_rizz["SHOCKED"] < 0.35

    # SHOCKED via blendshapes must not collide with SIGMA_RIZZ (jawOpen >= 0.35 gates SIGMA_RIZZ to 0)
    shocked_bs = {
        "jawOpen": 0.78,
        "eyeWideLeft": 0.65,
        "eyeWideRight": 0.62,
        "browInnerUp": 0.58,
        "mouthPucker": 0.40,
    }
    best_shock, conf_shock, scores_shock = classify_builtin(
        neutral_pose, None, make_neutral_face(mar=0.60), face_blendshapes=shocked_bs
    )
    assert best_shock == "SHOCKED"
    assert conf_shock >= 0.72
    assert scores_shock["SIGMA_RIZZ"] == 0.0


def test_custom_face_expression_store_and_stabilizer(tmp_path) -> None:
    store_path = tmp_path / "custom_face_poses.json"
    store = CustomPoseStore(store_path)

    face_only_bs = {"mouthSmileLeft": 0.85, "mouthSmileRight": 0.82, "cheekSquintLeft": 0.45}
    face_vec = extract_face_blendshape_vector(face_only_bs)
    assert face_vec is not None

    entry = store.add_pose(
        "Smug Face",
        vector=[],
        meme_path="assets/memes/smug.png",
        threshold=0.88,
        pose_id="CUSTOM_SMUG",
        include_face=True,
        face_vector=face_vec,
    )
    assert entry.include_face is True
    assert entry.face_vector is not None and len(entry.face_vector) == FACE_VECTOR_DIM

    # Reload from disk and verify backward-compatible JSON persistence
    reloaded = CustomPoseStore(store_path)
    loaded_entry = reloaded.get_pose("CUSTOM_SMUG")
    assert loaded_entry is not None
    assert loaded_entry.include_face is True
    assert loaded_entry.vector == []

    # Matching requires face_vector when include_face=True
    assert reloaded.match(None, face_vector=None)[0] is None
    matched, sim = reloaded.match(None, face_vector=face_vec)
    assert matched is not None and matched.id == "CUSTOM_SMUG" and sim > 0.99

    # TemporalStabilizer triggers face-only custom expression deterministically
    stabilizer = TemporalStabilizer(custom_store=reloaded, hold_ms=250, cooldown_ms=800)
    r0 = stabilizer.update(None, None, None, timestamp_ms=100, face_blendshapes=face_only_bs)
    assert r0.active_pose is None
    assert r0.candidate_pose == "CUSTOM_SMUG"
    r1 = stabilizer.update(None, None, None, timestamp_ms=360, face_blendshapes=face_only_bs)
    assert r1.active_pose == "CUSTOM_SMUG"
    assert r1.triggered_pose == "CUSTOM_SMUG"
    assert r1.is_custom is True

    # Combined body + face custom pose entry + process_detection()
    pose_combo, hands_combo, _ = build_pose_sample("PEACE_SIGN")
    body_vec = extract_pose_vector(pose_combo, hands_combo)
    assert body_vec is not None
    combo_bs = {"mouthSmileLeft": 0.65, "eyeSquintRight": 0.60, "jawOpen": 0.05}
    combo_face_vec = extract_face_blendshape_vector(combo_bs)
    assert combo_face_vec is not None
    reloaded.add_pose(
        "Wink Peace",
        vector=body_vec,
        threshold=0.88,
        pose_id="CUSTOM_WINK_PEACE",
        include_face=True,
        face_vector=combo_face_vec,
    )
    matched_combo, sim_combo = reloaded.match(body_vec, face_vector=combo_face_vec)
    assert matched_combo is not None and matched_combo.id == "CUSTOM_WINK_PEACE" and sim_combo > 0.99

    class _Det:
        def __init__(self, pose, hands, face, face_blendshapes, timestamp_ms):
            self.pose = pose
            self.hands = hands
            self.face = face
            self.face_blendshapes = face_blendshapes
            self.timestamp_ms = timestamp_ms

    stab2 = TemporalStabilizer(custom_store=reloaded, hold_ms=250, cooldown_ms=800)
    stab2.process_detection(_Det(pose_combo, hands_combo, None, combo_bs, 1000))
    res_combo = stab2.process_detection(_Det(pose_combo, hands_combo, None, combo_bs, 1260))
    assert res_combo.active_pose == "CUSTOM_WINK_PEACE"
    assert res_combo.triggered_pose == "CUSTOM_WINK_PEACE"
