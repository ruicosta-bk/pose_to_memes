"""Neon cyberpunk skeleton HUD and Picture-in-Picture compositor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from .tracker import FrameDetection

# BGR neon palette
NEON_CYAN = (255, 230, 30)
NEON_MAGENTA = (220, 60, 255)
NEON_LIME = (120, 255, 100)
NEON_VIOLET = (255, 110, 170)
NEON_WHITE = (255, 255, 255)
HUD_DARK_BG = (20, 18, 30)

# MediaPipe Pose body connections (excluding facial landmarks 0..10 when face mesh is present)
POSE_BODY_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # Left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    # Right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
)

POSE_HEAD_FALLBACK_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 2),
    (2, 7),
    (0, 5),
    (5, 8),
    (9, 10),
)

# MediaPipe Hand 21-landmark topology
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

HAND_TIPS: frozenset[int] = frozenset((4, 8, 12, 16, 20))

# MediaPipe FaceMesh contour indices (compatible with both 468 and 478 landmark models)
FACE_OUTER_LIPS: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
)
FACE_INNER_LIPS: tuple[int, ...] = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
)
FACE_LEFT_EYE: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33,
)
FACE_RIGHT_EYE: tuple[int, ...] = (
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263,
)
FACE_LEFT_BROW: tuple[int, ...] = (70, 63, 105, 66, 107)
FACE_RIGHT_BROW: tuple[int, ...] = (300, 293, 334, 296, 336)
FACE_JAWLINE: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
)


def _to_pixel_pts(
    landmarks: np.ndarray,
    width: int,
    height: int,
    vis_threshold: float = 0.0,
) -> list[Optional[tuple[int, int]]]:
    """Convert normalized (N, >=2) landmarks to integer pixel coordinates or None if invalid/hidden."""
    has_vis = landmarks.shape[1] >= 4 and vis_threshold > 0.0
    pts: list[Optional[tuple[int, int]]] = []
    for row in landmarks:
        x_val, y_val = float(row[0]), float(row[1])
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            pts.append(None)
            continue
        if has_vis and float(row[3]) < vis_threshold:
            pts.append(None)
            continue
        px = int(round(x_val * width))
        py = int(round(y_val * height))
        pts.append((px, py))
    return pts


def _draw_connections(
    glow_img: np.ndarray,
    core_img: np.ndarray,
    pts: Sequence[Optional[tuple[int, int]]],
    connections: Sequence[tuple[int, int]],
    color: tuple[int, int, int],
    glow_thickness: int = 7,
    core_thickness: int = 2,
) -> None:
    """Draw neon skeleton edges onto separate glow and core layers."""
    n = len(pts)
    for i, j in connections:
        if i >= n or j >= n:
            continue
        p1, p2 = pts[i], pts[j]
        if p1 is None or p2 is None:
            continue
        cv2.line(glow_img, p1, p2, color, glow_thickness, cv2.LINE_AA)
        cv2.line(core_img, p1, p2, color, core_thickness, cv2.LINE_AA)


def _draw_polyline(
    img: np.ndarray,
    pts: Sequence[Optional[tuple[int, int]]],
    indices: Sequence[int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a subtle polyline across valid landmark indices."""
    n = len(pts)
    poly: list[tuple[int, int]] = []
    for idx in indices:
        if idx < n and pts[idx] is not None:
            poly.append(pts[idx])  # type: ignore[arg-type]
    if len(poly) >= 2:
        arr = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_joints(
    glow_img: np.ndarray,
    core_img: np.ndarray,
    pts: Sequence[Optional[tuple[int, int]]],
    indices: Sequence[int],
    color: tuple[int, int, int],
    radius: int = 4,
    highlight_indices: frozenset[int] = frozenset(),
) -> None:
    """Draw glowing joint nodes with white hot inner core."""
    n = len(pts)
    for idx in indices:
        if idx >= n:
            continue
        pt = pts[idx]
        if pt is None:
            continue
        r = radius + 1 if idx in highlight_indices else radius
        cv2.circle(glow_img, pt, r + 3, color, -1, cv2.LINE_AA)
        cv2.circle(core_img, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(core_img, pt, max(1, r - 2), NEON_WHITE, -1, cv2.LINE_AA)


def draw_neon_hud(
    frame_bgr: np.ndarray,
    detection: Optional[FrameDetection],
    active_pose: Optional[str] = None,
    confidence: float = 0.0,
) -> np.ndarray:
    """Render a neon cyberpunk skeleton HUD onto a copy of ``frame_bgr``."""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    if detection is not None:
        glow = np.zeros_like(canvas)
        has_glow = False

        # Subtle face contours first (underneath hands/pose)
        face = getattr(detection, "face", None)
        if face is not None and len(face) >= 468:
            face_pts = _to_pixel_pts(face, w, h)
            _draw_polyline(canvas, face_pts, FACE_JAWLINE, (110, 95, 50), thickness=1)
            _draw_polyline(canvas, face_pts, FACE_LEFT_BROW, NEON_VIOLET, thickness=1)
            _draw_polyline(canvas, face_pts, FACE_RIGHT_BROW, NEON_VIOLET, thickness=1)
            _draw_polyline(canvas, face_pts, FACE_LEFT_EYE, NEON_CYAN, thickness=1)
            _draw_polyline(canvas, face_pts, FACE_RIGHT_EYE, NEON_CYAN, thickness=1)
            _draw_polyline(glow, face_pts, FACE_OUTER_LIPS, NEON_MAGENTA, thickness=4)
            _draw_polyline(canvas, face_pts, FACE_OUTER_LIPS, NEON_MAGENTA, thickness=1)
            _draw_polyline(canvas, face_pts, FACE_INNER_LIPS, NEON_CYAN, thickness=1)
            has_glow = True

        # Pose skeleton
        pose = getattr(detection, "pose", None)
        if pose is not None and len(pose) >= 33:
            pose_pts = _to_pixel_pts(pose, w, h, vis_threshold=0.35)
            _draw_connections(glow, canvas, pose_pts, POSE_BODY_CONNECTIONS, NEON_CYAN, glow_thickness=7, core_thickness=2)
            if face is None:
                _draw_connections(glow, canvas, pose_pts, POSE_HEAD_FALLBACK_CONNECTIONS, NEON_CYAN, glow_thickness=5, core_thickness=1)
                pose_joint_ids = range(len(pose_pts))
            else:
                pose_joint_ids = range(11, len(pose_pts))
            _draw_joints(glow, canvas, pose_pts, pose_joint_ids, NEON_CYAN, radius=4)
            has_glow = True

        # Hands
        hands = getattr(detection, "hands", None) or []
        for label, hand_lms in hands:
            if hand_lms is None or len(hand_lms) < 21:
                continue
            hand_pts = _to_pixel_pts(hand_lms, w, h)
            hand_color = NEON_MAGENTA if label == "Left" else NEON_LIME
            _draw_connections(glow, canvas, hand_pts, HAND_CONNECTIONS, hand_color, glow_thickness=6, core_thickness=2)
            _draw_joints(glow, canvas, hand_pts, range(21), hand_color, radius=3, highlight_indices=HAND_TIPS)
            has_glow = True

        if has_glow:
            cv2.addWeighted(canvas, 1.0, glow, 0.45, 0.0, dst=canvas)

    # Optional active pose badge in top-left corner
    if active_pose and active_pose.upper() != "NEUTRAL":
        badge_text = f"{active_pose}  {max(0.0, min(1.0, float(confidence))) * 100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.7, w / 1100.0))
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(badge_text, font, scale, thickness)
        x1, y1 = 14, 14
        x2, y2 = min(w - 10, x1 + tw + 20), min(h - 10, y1 + th + baseline + 14)
        if x2 > x1 and y2 > y1:
            roi = canvas[y1:y2, x1:x2]
            bg = np.full_like(roi, HUD_DARK_BG)
            cv2.addWeighted(bg, 0.72, roi, 0.28, 0.0, dst=roi)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), NEON_CYAN, 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                badge_text,
                (x1 + 10, y1 + th + 6),
                font,
                scale,
                NEON_CYAN,
                thickness,
                cv2.LINE_AA,
            )

    return canvas


def composite_pip(
    frame_bgr: np.ndarray,
    meme_bgr_or_bgra: Optional[np.ndarray],
    label: str = "",
) -> np.ndarray:
    """Composite a meme image/frame (BGR or BGRA) onto the bottom-right corner of ``frame_bgr``."""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    canvas = frame_bgr.copy()
    if meme_bgr_or_bgra is None or meme_bgr_or_bgra.size == 0:
        return canvas

    fh, fw = canvas.shape[:2]
    if fh < 40 or fw < 40:
        return canvas

    meme = meme_bgr_or_bgra
    if meme.ndim == 2:
        meme = cv2.cvtColor(meme, cv2.COLOR_GRAY2BGR)
    elif meme.shape[2] not in (3, 4):
        return canvas

    mh, mw = meme.shape[:2]
    if mh <= 0 or mw <= 0:
        return canvas

    max_w = max(48, int(fw * 0.28))
    max_h = max(48, int(fh * 0.38))
    scale = min(max_w / mw, max_h / mh)
    pip_w = max(16, int(round(mw * scale)))
    pip_h = max(16, int(round(mh * scale)))

    resized = cv2.resize(meme, (pip_w, pip_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    pad = max(10, int(min(fw, fh) * 0.025))
    x2 = fw - pad
    y2 = fh - pad
    x1 = max(0, x2 - pip_w)
    y1 = max(0, y2 - pip_h)
    actual_w = x2 - x1
    actual_h = y2 - y1
    if actual_w <= 0 or actual_h <= 0:
        return canvas
    resized = resized[:actual_h, :actual_w]

    # Drop shadow
    shadow_off = 5
    sx1 = min(fw, x1 + shadow_off)
    sy1 = min(fh, y1 + shadow_off)
    sx2 = min(fw, x2 + shadow_off)
    sy2 = min(fh, y2 + shadow_off)
    if sx2 > sx1 and sy2 > sy1:
        shadow_roi = canvas[sy1:sy2, sx1:sx2]
        shadow_roi[:] = (shadow_roi.astype(np.float32) * 0.45).astype(np.uint8)

    # Composite BGR or BGRA
    roi = canvas[y1:y2, x1:x2]
    if resized.shape[2] == 4:
        alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
        fg = resized[:, :, :3].astype(np.float32)
        blended = fg * alpha + roi.astype(np.float32) * (1.0 - alpha)
        roi[:] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        roi[:] = resized

    # Neon border
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), NEON_CYAN, 2, cv2.LINE_AA)

    # Optional label strip at the bottom of the PiP card
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.38, min(0.58, actual_w / 320.0))
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        bar_h = min(actual_h, th + baseline + 10)
        by1 = y2 - bar_h
        bar_roi = canvas[by1:y2, x1:x2]
        bg = np.full_like(bar_roi, HUD_DARK_BG)
        cv2.addWeighted(bg, 0.78, bar_roi, 0.22, 0.0, dst=bar_roi)
        tx = x1 + max(6, (actual_w - tw) // 2)
        ty = y2 - baseline - 4
        cv2.putText(canvas, label, (tx, ty), font, font_scale, NEON_WHITE, thickness, cv2.LINE_AA)

    return canvas
