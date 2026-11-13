"""Procedural default meme card generator and pose media config resolver."""

from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw, ImageFont

from assets.audio import ensure_default_audio

CARD_WIDTH = 800
CARD_HEIGHT = 600
_SUPERSAMPLE = 2
_CANVAS_W = CARD_WIDTH * _SUPERSAMPLE
_CANVAS_H = CARD_HEIGHT * _SUPERSAMPLE

_IMAGE_EXTS: tuple[str, ...] = (".gif", ".webp", ".png", ".jpg", ".jpeg")
_AUDIO_EXTS: tuple[str, ...] = (".wav", ".mp3", ".ogg")

POSE_ORDER: tuple[str, ...] = (
    "ABSOLUTE_CINEMA",
    "ERM_ACTUALLY",
    "THINKING",
    "SHOCKED",
    "DOMAIN_EXPANSION",
    "PEACE_SIGN",
    "NEUTRAL",
)

DEFAULT_MEMES_CONFIG: dict[str, dict[str, Optional[str]]] = {
    "ABSOLUTE_CINEMA": {
        "title": "ABSOLUTE CINEMA",
        "meme_path": "assets/default/absolute_cinema.png",
        "audio_path": "assets/default/dramatic_hit.wav",
        "accent_color": "#F59E0B",
    },
    "ERM_ACTUALLY": {
        "title": "ERM, ACTUALLY...",
        "meme_path": "assets/default/erm_actually.png",
        "audio_path": "assets/default/erm_actually.wav",
        "accent_color": "#84CC16",
    },
    "THINKING": {
        "title": "HMMM... THINKING",
        "meme_path": "assets/default/thinking.png",
        "audio_path": "assets/default/anime_chime.wav",
        "accent_color": "#A855F7",
    },
    "SHOCKED": {
        "title": "WHAT?! SHOCKED",
        "meme_path": "assets/default/shocked.png",
        "audio_path": "assets/default/vine_boom.wav",
        "accent_color": "#EF4444",
    },
    "DOMAIN_EXPANSION": {
        "title": "DOMAIN EXPANSION",
        "meme_path": "assets/default/domain_expansion.png",
        "audio_path": "assets/default/domain_pulse.wav",
        "accent_color": "#C026D3",
    },
    "PEACE_SIGN": {
        "title": "PEACE OUT!",
        "meme_path": "assets/default/peace_sign.png",
        "audio_path": "assets/default/peace_chime.wav",
        "accent_color": "#06B6D4",
    },
    "NEUTRAL": {
        "title": "WAITING FOR POSE...",
        "meme_path": "assets/default/neutral.png",
        "audio_path": None,
        "accent_color": "#64748B",
    },
}

_CARD_SPECS: dict[str, dict[str, Any]] = {
    "ABSOLUTE_CINEMA": {
        "filename": "absolute_cinema.png",
        "title": "ABSOLUTE CINEMA",
        "badge": "POSE // CINEMA",
        "bg_top": (14, 12, 18),
        "bg_bot": (28, 22, 14),
        "accent": (245, 158, 11),
        "glow": (251, 191, 36),
    },
    "ERM_ACTUALLY": {
        "filename": "erm_actually.png",
        "title": "ERM, ACTUALLY...",
        "badge": "POSE // NERD ALERT",
        "bg_top": (10, 18, 14),
        "bg_bot": (17, 29, 22),
        "accent": (132, 204, 22),
        "glow": (163, 230, 53),
    },
    "THINKING": {
        "filename": "thinking.png",
        "title": "HMMM... THINKING",
        "badge": "POSE // DEEP THOUGHT",
        "bg_top": (15, 12, 26),
        "bg_bot": (26, 20, 42),
        "accent": (168, 85, 247),
        "glow": (192, 132, 252),
    },
    "SHOCKED": {
        "filename": "shocked.png",
        "title": "WHAT?! SHOCKED",
        "badge": "POSE // JAW DROP",
        "bg_top": (24, 10, 14),
        "bg_bot": (36, 14, 18),
        "accent": (239, 68, 68),
        "glow": (251, 113, 133),
    },
    "DOMAIN_EXPANSION": {
        "filename": "domain_expansion.png",
        "title": "DOMAIN EXPANSION",
        "badge": "POSE // CURSED SEAL",
        "bg_top": (16, 8, 26),
        "bg_bot": (30, 10, 44),
        "accent": (192, 38, 211),
        "glow": (232, 121, 249),
    },
    "PEACE_SIGN": {
        "filename": "peace_sign.png",
        "title": "PEACE OUT!",
        "badge": "POSE // V-SIGN",
        "bg_top": (8, 16, 26),
        "bg_bot": (12, 28, 42),
        "accent": (6, 182, 212),
        "glow": (34, 211, 238),
    },
    "NEUTRAL": {
        "filename": "neutral.png",
        "title": "WAITING FOR POSE...",
        "badge": "STUDIO // STANDBY",
        "bg_top": (11, 15, 25),
        "bg_bot": (17, 24, 39),
        "accent": (100, 116, 139),
        "glow": (56, 189, 248),
    },
}


def _default_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


class MemesConfig(dict):
    """Dictionary mapping pose IDs to media settings with its origin root_dir."""

    def __init__(self, data: Optional[dict[str, Any]] = None, root_dir: Optional[Path] = None) -> None:
        super().__init__(data or {})
        self.root_dir: Path = root_dir if root_dir is not None else _default_root_dir()


def _load_font(size: int, bold: bool = True, impact: bool = False) -> ImageFont.ImageFont:
    win_fonts = Path("C:/Windows/Fonts")
    if impact:
        names = ("impact.ttf", "arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf")
    elif bold:
        names = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
    else:
        names = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")

    candidates: list[str] = []
    for name in names:
        win_candidate = win_fonts / name
        if win_candidate.is_file():
            candidates.append(str(win_candidate))
        candidates.append(name)

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _lerp_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _create_base_canvas(spec: dict[str, Any]) -> Image.Image:
    """Build 2x dark editorial card canvas with vertical gradient, grid, glow, and frame."""
    bg_top = spec["bg_top"]
    bg_bot = spec["bg_bot"]
    accent = spec["accent"]
    glow = spec["glow"]

    # Fast 1px-wide gradient scaled up
    grad = Image.new("RGB", (1, _CANVAS_H))
    px = grad.load()
    for y in range(_CANVAS_H):
        px[0, y] = _lerp_rgb(bg_top, bg_bot, y / max(1, _CANVAS_H - 1))
    canvas = grad.resize((_CANVAS_W, _CANVAS_H), resample=Image.Resampling.BILINEAR).convert("RGBA")

    overlay = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    # Subtle studio grid
    for gx in range(0, _CANVAS_W, 80):
        od.line([(gx, 0), (gx, _CANVAS_H)], fill=(255, 255, 255, 8), width=2)
    for gy in range(0, _CANVAS_H, 80):
        od.line([(0, gy), (_CANVAS_W, gy)], fill=(255, 255, 255, 8), width=2)

    # Soft central radial aura
    cx, cy = _CANVAS_W // 2, int(_CANVAS_H * 0.44)
    for r, alpha in ((440, 14), (330, 22), (220, 30), (130, 36)):
        od.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*glow, alpha))

    # Outer border + corner viewfinder brackets
    margin = 36
    od.rounded_rectangle(
        (margin, margin, _CANVAS_W - margin, _CANVAS_H - margin),
        radius=36,
        outline=(*accent, 110),
        width=4,
    )
    b_len = 56
    b_pad = 54
    for x0, y0, dx, dy in (
        (b_pad, b_pad, 1, 1),
        (_CANVAS_W - b_pad, b_pad, -1, 1),
        (b_pad, _CANVAS_H - b_pad, 1, -1),
        (_CANVAS_W - b_pad, _CANVAS_H - b_pad, -1, -1),
    ):
        od.line([(x0, y0), (x0 + dx * b_len, y0)], fill=(*glow, 220), width=6)
        od.line([(x0, y0), (x0, y0 + dy * b_len)], fill=(*glow, 220), width=6)

    return Image.alpha_composite(canvas, overlay)


def _draw_badge_and_title(canvas: Image.Image, spec: dict[str, Any]) -> Image.Image:
    accent = spec["accent"]
    glow = spec["glow"]
    badge_text = spec["badge"]
    title_text = spec["title"]

    layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Top-left pill badge
    badge_font = _load_font(28, bold=True)
    bbox = d.textbbox((0, 0), badge_text, font=badge_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bx, by = 82, 76
    pill_w, pill_h = tw + 48, th + 28
    d.rounded_rectangle(
        (bx, by, bx + pill_w, by + pill_h),
        radius=pill_h // 2,
        fill=(15, 23, 42, 220),
        outline=(*accent, 210),
        width=3,
    )
    d.ellipse((bx + 18, by + pill_h // 2 - 7, bx + 32, by + pill_h // 2 + 7), fill=(*glow, 255))
    d.text((bx + 42, by + (pill_h - th) // 2 - bbox[1]), badge_text, font=badge_font, fill=(226, 232, 240, 255))

    # Bottom title plate
    plate_y0 = _CANVAS_H - 250
    d.rounded_rectangle(
        (96, plate_y0, _CANVAS_W - 96, _CANVAS_H - 74),
        radius=28,
        fill=(8, 12, 22, 210),
        outline=(*accent, 140),
        width=3,
    )
    # Accent bar
    d.rounded_rectangle(
        (_CANVAS_W // 2 - 120, plate_y0 + 18, _CANVAS_W // 2 + 120, plate_y0 + 26),
        radius=4,
        fill=(*glow, 245),
    )

    # Fit title cleanly inside plate
    font_size = 92
    title_font = _load_font(font_size, bold=True, impact=True)
    max_w = _CANVAS_W - 260
    while font_size > 44:
        title_font = _load_font(font_size, bold=True, impact=True)
        tb = d.textbbox((0, 0), title_text, font=title_font)
        if (tb[2] - tb[0]) <= max_w:
            break
        font_size -= 6

    tb = d.textbbox((0, 0), title_text, font=title_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = (_CANVAS_W - tw) // 2 - tb[0]
    ty = plate_y0 + 44 + (110 - th) // 2 - tb[1]

    d.text((tx + 4, ty + 6), title_text, font=title_font, fill=(0, 0, 0, 200))
    d.text((tx, ty), title_text, font=title_font, fill=(248, 250, 252, 255))
    return Image.alpha_composite(canvas, layer)


def _draw_absolute_cinema(canvas: Image.Image, spec: dict[str, Any]) -> Image.Image:
    layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = _CANVAS_W // 2, 500
    accent = spec["accent"]
    glow = spec["glow"]

    # Cinema letterbox bars
    d.rectangle((40, 40, _CANVAS_W - 40, 145), fill=(4, 4, 8, 220))
    d.rectangle((40, _CANVAS_H - 145, _CANVAS_W - 40, _CANVAS_H - 40), fill=(4, 4, 8, 220))

    # Dramatic spotlight beams
    d.polygon([(210, 145), (320, 145), (cx + 220, 880), (cx - 260, 880)], fill=(*glow, 28))
    d.polygon([(_CANVAS_W - 210, 145), (_CANVAS_W - 320, 145), (cx + 260, 880), (cx - 220, 880)], fill=(*accent, 24))

    # Stylized raised-arms cinema director silhouette
    d.ellipse((cx - 108, cy - 220, cx + 108, cy - 4), fill=(241, 245, 249, 255), outline=(*glow, 255), width=6)
    # Expressive brow & eyes
    d.line([(cx - 64, cy - 135), (cx - 18, cy - 122)], fill=(15, 23, 42, 255), width=12)
    d.line([(cx + 18, cy - 122), (cx + 64, cy - 135)], fill=(15, 23, 42, 255), width=12)
    d.ellipse((cx - 52, cy - 110, cx - 22, cy - 82), fill=(15, 23, 42, 255))
    d.ellipse((cx + 22, cy - 110, cx + 52, cy - 82), fill=(15, 23, 42, 255))

    # Shoulders & raised cinema arms
    d.rounded_rectangle((cx - 165, cy + 10, cx + 165, cy + 255), radius=80, fill=(30, 41, 59, 255), outline=(*accent, 230), width=6)
    # Left & Right raised forearms + open palms
    for sign in (-1, 1):
        pts = [
            (cx + sign * 145, cy + 95),
            (cx + sign * 290, cy + 45),
            (cx + sign * 305, cy - 170),
            (cx + sign * 245, cy - 185),
            (cx + sign * 230, cy - 10),
            (cx + sign * 125, cy + 35),
        ]
        d.polygon(pts, fill=(241, 245, 249, 255), outline=(*glow, 255))
        px = cx + sign * 276
        py = cy - 215
        d.ellipse((px - 44, py - 52, px + 44, py + 44), fill=(248, 250, 252, 255), outline=(*glow, 255), width=5)

    return Image.alpha_composite(canvas, layer)


def _draw_erm_actually(canvas: Image.Image, spec: dict[str, Any]) -> Image.Image:
    layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = _CANVAS_W // 2, 485
    accent = spec["accent"]
    glow = spec["glow"]

    # Oversized nerd glasses
    gx_off = 148
    r_glass = 115
    for sign in (-1, 1):
        gx = cx - 60 + sign * gx_off
        gy = cy - 55
        d.ellipse((gx - r_glass, gy - r_glass, gx + r_glass, gy + r_glass), fill=(15, 23, 42, 230), outline=(*glow, 255), width=14)
        # Diagonal lens glint
        d.line([(gx - 55, gy - 45), (gx + 10, gy - 80)], fill=(255, 255, 255, 190), width=12)
        d.line([(gx - 35, gy + 5), (gx + 55, gy - 45)], fill=(255, 255, 255, 130), width=8)
    # Bridge
    d.line([(cx - 60 - gx_off + r_glass - 6, cy - 55), (cx - 60 + gx_off - r_glass + 6, cy - 55)], fill=(*glow, 255), width=14)
    # Smug mouth
    d.arc((cx - 125, cy + 75, cx - 5, cy + 145), start=15, end=155, fill=(226, 232, 240, 255), width=10)

    # Raised pointing index finger on right
    fx, fy = cx + 275, cy + 40
    d.rounded_rectangle((fx - 68, fy + 40, fx + 68, fy + 225), radius=48, fill=(241, 245, 249, 255), outline=(*accent, 255), width=6)
    d.rounded_rectangle((fx - 26, fy - 245, fx + 26, fy + 85), radius=26, fill=(248, 250, 252, 255), outline=(*glow, 255), width=6)
    # Curled finger knuckles
    for k in range(3):
        kx = fx + 12 + k * 20
        d.ellipse((kx - 20, fy + 48, kx + 18, fy + 105), fill=(203, 213, 225, 255), outline=(*accent, 220), width=4)

    # Sparkle lines around fingertip
    for angle_deg in (-55, -20, 20, 55):
        rad = math.radians(angle_deg - 90)
        x1 = fx + int(math.cos(rad) * 55)
        y1 = fy - 255 + int(math.sin(rad) * 55)
        x2 = fx + int(math.cos(rad) * 95)
        y2 = fy - 255 + int(math.sin(rad) * 95)
        d.line([(x1, y1), (x2, y2)], fill=(*glow, 240), width=8)

    return Image.alpha_composite(canvas, layer)


def _draw_thinking(canvas: Image.Image, spec: dict[str, Any]) -> Image.Image:
    layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = _CANVAS_W // 2, 480
    accent = spec["accent"]
    glow = spec["glow"]

    # Thoughtful face circle
    d.ellipse((cx - 175, cy - 195, cx + 175, cy + 155), fill=(30, 27, 52, 245), outline=(*glow, 255), width=8)
    # Asymmetric raised/furrowed eyebrows
    d.arc((cx - 115, cy - 125, cx - 25, cy - 65), start=190, end=340, fill=(248, 250, 252, 255), width=12)
    d.line([(cx + 25, cy - 82), (cx + 115, cy - 95)], fill=(248, 250, 252, 255), width=12)
    # Side-glancing eyes
    for ex in (cx - 65, cx + 65):
        d.ellipse((ex - 30, cy - 55, ex + 30, cy - 15), fill=(248, 250, 252, 255))
        d.ellipse((ex - 4, cy - 48, ex + 20, cy - 22), fill=(15, 23, 42, 255))
    # Contemplative mouth
    d.line([(cx - 48, cy + 48), (cx + 42, cy + 38)], fill=(226, 232, 240, 255), width=10)

    # Hand resting on chin
    d.rounded_rectangle((cx - 95, cy + 100, cx + 45, cy + 255), radius=42, fill=(241, 245, 249, 255), outline=(*accent, 255), width=6)
    d.rounded_rectangle((cx - 110, cy + 75, cx - 35, cy + 155), radius=24, fill=(248, 250, 252, 255), outline=(*glow, 255), width=5)

    # Floating "?" emblem & rising thought bubbles
    q_font = _load_font(170, bold=True, impact=True)
    d.text((cx + 225, cy - 245), "?", font=q_font, fill=(*glow, 245))
    d.text((cx - 335, cy - 175), "?", font=_load_font(115, bold=True, impact=True), fill=(*accent, 210))
    for r, bx, by in ((16, cx + 195, cy + 45), (24, cx + 235, cy - 15), (34, cx + 275, cy - 85)):
        d.ellipse((bx - r, by - r, bx + r, by + r), fill=(*glow, 180), outline=(255, 255, 255, 220), width=3)

    return Image.alpha_composite(canvas, layer)


def _draw_shocked(canvas: Image.Image, spec: dict[str, Any]) -> Image.Image:
    layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = _CANVAS_W // 2, 485
    accent = spec["accent"]
    glow = spec["glow"]

    # Radial manga/anime shock burst lines
    for i in range(24):
        ang = i * (2 * math.pi / 24)
        r_in = 250 if i % 2 == 0 else 295
        r_out = 680
        x1 = cx + int(math.cos(ang) * r_in)
        y1 = cy + int(math.sin(ang) * r_in)
        x2 = cx + int(math.cos(ang) * r_out)
        y2 = cy + int(math.sin(ang) * r_out)
        d.line([(x1, y1), (x2, y2)], fill=(*accent, 110 if i % 2 == 0 else 65), width=8)

    # Shocked head
    d.ellipse((cx - 175, cy - 195, cx + 175, cy + 155), fill=(32, 14, 20, 245), outline=(*glow, 255), width=8)
    # Wide eyes with tiny shocked pupils
    for ex in (cx - 68, cx + 68):
        d.ellipse((ex - 48, cy - 105, ex + 48, cy - 5), fill=(255, 255, 255, 255), outline=(*accent, 255), width=6)
        d.ellipse((ex - 12, cy - 67, ex + 12, cy - 43), fill=(15, 23, 42, 255))
    # Huge jaw-drop open mouth
    d.ellipse((cx - 64, cy + 12, cx + 64, cy + 132), fill=(12, 8, 12, 255), outline=(*glow, 255), width=8)

    # Jagged lightning bolts + "?!"
    for sign in (-1, 1):
        bx = cx + sign * 265
        bolt = [
            (bx, cy - 180),
            (bx - sign * 42, cy - 65),
            (bx + sign * 12, cy - 55),
            (bx - sign * 35, cy + 65),
            (bx + sign * 45, cy - 75),
            (bx - sign * 8, cy - 85),
        ]
        d.polygon(bolt, fill=(*glow, 245))
    d.text((cx + 210, cy - 265), "?!", font=_load_font(125, bold=True, impact=True), fill=(254, 226, 226, 255))

    return Image.alpha_composite(canvas, layer)


def _draw_domain_expansion(canvas: Image.Image, spec: dict[str, Any]) -> Image.Image:
    layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = _CANVAS_W // 2, 485
    accent = spec["accent"]
    glow = spec["glow"]

    # Concentric arcane cursed seal rings
    for r, w, alpha in ((315, 6, 170), (265, 3, 210), (195, 5, 190)):
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*glow, alpha), width=w)
    # 12 radial sigil ticks
    for i in range(12):
        ang = i * (2 * math.pi / 12)
        x1 = cx + int(math.cos(ang) * 265)
        y1 = cy + int(math.sin(ang) * 265)
        x2 = cx + int(math.cos(ang) * 315)
        y2 = cy + int(math.sin(ang) * 315)
        d.line([(x1, y1), (x2, y2)], fill=(*glow, 230), width=5)

    # Minimalist torii-gate silhouette behind hand seal
    d.rectangle((cx - 210, cy - 175, cx + 210, cy - 148), fill=(*accent, 140))
    d.rectangle((cx - 175, cy - 125, cx + 175, cy - 106), fill=(*accent, 120))
    d.rectangle((cx - 132, cy - 150, cx - 104, cy + 165), fill=(*accent, 120))
    d.rectangle((cx + 104, cy - 150, cx + 132, cy + 165), fill=(*accent, 120))

    # Clasped hand-seal core emblem
    d.polygon(
        [(cx, cy - 145), (cx + 125, cy + 35), (cx + 75, cy + 165), (cx - 75, cy + 165), (cx - 125, cy + 35)],
        fill=(18, 10, 30, 245),
        outline=(*glow, 255),
    )
    # Interlocked vertical seal fingers
    for dx, h in ((-34, 105), (-11, 130), (11, 130), (34, 105)):
        d.rounded_rectangle(
            (cx + dx - 10, cy - h, cx + dx + 10, cy + 95),
            radius=10,
            fill=(248, 250, 252, 245),
            outline=(*accent, 255),
            width=3,
        )

    return Image.alpha_composite(canvas, layer)


def _draw_peace_sign(canvas: Image.Image, spec: dict[str, Any]) -> Image.Image:
    layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = _CANVAS_W // 2, 495
    accent = spec["accent"]
    glow = spec["glow"]

    # Neon halo circle
    d.ellipse((cx - 230, cy - 230, cx + 230, cy + 230), outline=(*glow, 180), width=8)

    # V-sign fingers (index + middle)
    d.line([(cx - 22, cy + 15), (cx - 125, cy - 235)], fill=(248, 250, 252, 255), width=48)
    d.line([(cx + 22, cy + 15), (cx + 125, cy - 235)], fill=(248, 250, 252, 255), width=48)
    # Rounded finger tips + neon outline
    for tx, ty in ((cx - 125, cy - 235), (cx + 125, cy - 235)):
        d.ellipse((tx - 24, ty - 24, tx + 24, ty + 24), fill=(*glow, 255))

    # Palm base + curled ring/pinky knuckles
    d.rounded_rectangle((cx - 95, cy - 10, cx + 95, cy + 190), radius=64, fill=(241, 245, 249, 255), outline=(*accent, 255), width=8)
    for kx in (cx - 48, cx, cx + 48):
        d.ellipse((kx - 24, cy + 10, kx + 24, cy + 72), fill=(203, 213, 225, 255), outline=(*accent, 230), width=4)

    # Sparkle diamonds
    for sx, sy, sr in ((cx - 245, cy - 140, 34), (cx + 255, cy - 115, 42), (cx + 215, cy + 125, 26)):
        d.polygon([(sx, sy - sr), (sx + sr // 2, sy), (sx, sy + sr), (sx - sr // 2, sy)], fill=(*glow, 240))

    return Image.alpha_composite(canvas, layer)


def _draw_neutral(canvas: Image.Image, spec: dict[str, Any]) -> Image.Image:
    layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = _CANVAS_W // 2, 485
    glow = spec["glow"]

    # Studio radar reticle
    for r, alpha in ((250, 120), (175, 160), (95, 200)):
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*glow, alpha), width=4)
    d.line([(cx - 285, cy), (cx - 115, cy)], fill=(*glow, 160), width=4)
    d.line([(cx + 115, cy), (cx + 285, cy)], fill=(*glow, 160), width=4)
    d.line([(cx, cy - 285), (cx, cy - 115)], fill=(*glow, 160), width=4)
    d.line([(cx, cy + 115), (cx, cy + 285)], fill=(*glow, 160), width=4)

    # Minimalist standby human silhouette
    d.ellipse((cx - 46, cy - 120, cx + 46, cy - 28), fill=(203, 213, 225, 230), outline=(*glow, 240), width=5)
    d.chord((cx - 95, cy - 10, cx + 95, cy + 165), start=180, end=360, fill=(148, 163, 184, 215), outline=(*glow, 240), width=5)

    return Image.alpha_composite(canvas, layer)


_ARTWORK_DRAWERS: dict[str, Callable[[Image.Image, dict[str, Any]], Image.Image]] = {
    "ABSOLUTE_CINEMA": _draw_absolute_cinema,
    "ERM_ACTUALLY": _draw_erm_actually,
    "THINKING": _draw_thinking,
    "SHOCKED": _draw_shocked,
    "DOMAIN_EXPANSION": _draw_domain_expansion,
    "PEACE_SIGN": _draw_peace_sign,
    "NEUTRAL": _draw_neutral,
}


def _render_meme_card(pose_id: str, out_path: Path) -> None:
    spec = _CARD_SPECS[pose_id]
    canvas = _create_base_canvas(spec)
    canvas = _ARTWORK_DRAWERS[pose_id](canvas, spec)
    canvas = _draw_badge_and_title(canvas, spec)

    final_img = canvas.convert("RGB").resize((CARD_WIDTH, CARD_HEIGHT), resample=Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".png", dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        final_img.save(str(tmp_path), format="PNG", optimize=True)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def ensure_default_assets(root_dir: Optional[str | Path] = None) -> dict[str, Any]:
    """Idempotently generate default meme PNGs, SFX WAVs, memes/.gitkeep, and memes_config.json."""
    base = Path(root_dir) if root_dir is not None else _default_root_dir()
    default_dir = base / "assets" / "default"
    memes_dir = base / "assets" / "memes"
    data_dir = base / "data"

    default_dir.mkdir(parents=True, exist_ok=True)
    memes_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    gitkeep = memes_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    image_paths: list[Path] = []
    for pose_id in POSE_ORDER:
        spec = _CARD_SPECS[pose_id]
        img_path = default_dir / spec["filename"]
        if not img_path.is_file() or img_path.stat().st_size == 0:
            _render_meme_card(pose_id, img_path)
        image_paths.append(img_path)

    audio_paths = ensure_default_audio(base)

    config_path = data_dir / "memes_config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        config_path.write_text(json.dumps(DEFAULT_MEMES_CONFIG, indent=2) + "\n", encoding="utf-8")

    return {
        "images": image_paths,
        "audio": audio_paths,
        "config": config_path,
    }


def load_memes_config(root_dir: Optional[str | Path] = None) -> MemesConfig:
    """Load data/memes_config.json merged with defaults for all 6 poses + NEUTRAL."""
    base = Path(root_dir) if root_dir is not None else _default_root_dir()
    config_path = base / "data" / "memes_config.json"
    merged = deepcopy(DEFAULT_MEMES_CONFIG)

    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for pose_id, entry in loaded.items():
                    if isinstance(entry, dict):
                        base_entry = merged.get(pose_id, {}).copy()
                        base_entry.update(entry)
                        merged[pose_id] = base_entry
        except Exception:
            pass

    return MemesConfig(merged, root_dir=base)


def _resolve_existing_path(raw_path: Optional[str | Path], base_dir: Path) -> Optional[str]:
    if not raw_path:
        return None
    p = Path(raw_path)
    if not p.is_absolute():
        p = base_dir / p
    return str(p.resolve()) if p.is_file() else None


def _find_override_file(memes_dir: Path, stems: list[str], exts: tuple[str, ...]) -> Optional[str]:
    if not memes_dir.is_dir():
        return None
    normalized_stems = {s.lower(): idx for idx, s in enumerate(stems) if s}
    matches: list[tuple[int, int, Path]] = []
    for item in memes_dir.iterdir():
        if not item.is_file():
            continue
        ext_lower = item.suffix.lower()
        if ext_lower not in exts:
            continue
        stem_idx = normalized_stems.get(item.stem.lower())
        if stem_idx is not None:
            matches.append((stem_idx, exts.index(ext_lower), item))
    if not matches:
        return None
    matches.sort(key=lambda m: (m[0], m[1]))
    return str(matches[0][2].resolve())


def _find_custom_entry(custom_store: Any, pose_id: str) -> Any:
    if custom_store is None:
        return None
    entries: list[Any] = []
    if hasattr(custom_store, "list_poses") and callable(custom_store.list_poses):
        try:
            entries = list(custom_store.list_poses())
        except Exception:
            entries = []
    elif isinstance(custom_store, (list, tuple)):
        entries = list(custom_store)

    pose_lower = pose_id.lower()
    for entry in entries:
        eid = str(getattr(entry, "id", "") or "")
        ename = str(getattr(entry, "name", "") or "")
        if eid == pose_id or ename == pose_id:
            return entry
    for entry in entries:
        eid = str(getattr(entry, "id", "") or "").lower()
        ename = str(getattr(entry, "name", "") or "").lower()
        if eid == pose_lower or ename == pose_lower:
            return entry
    return None


def resolve_pose_media(
    pose_id: Optional[str],
    config: Optional[dict[str, Any]] = None,
    custom_store: Any = None,
    root_dir: Optional[str | Path] = None,
) -> dict[str, Optional[str]]:
    """Resolve title, meme_path, audio_path, and accent_color for a built-in or custom pose.

    Resolution order for media files:
    1. User drop-in overrides in ``assets/memes/`` (e.g. ``ABSOLUTE_CINEMA.gif`` / ``.png`` / ``.wav``)
    2. Custom pose entry paths (if ``custom_store`` matches) or configured path in ``config``
    3. Default bundled asset in ``assets/default/``
    """
    raw_id = (pose_id or "NEUTRAL").strip() or "NEUTRAL"
    base_dir = (
        Path(root_dir)
        if root_dir is not None
        else (config.root_dir if isinstance(config, MemesConfig) else _default_root_dir())
    )
    cfg = config if config is not None else load_memes_config(base_dir)
    memes_dir = base_dir / "assets" / "memes"

    custom_entry = _find_custom_entry(custom_store, raw_id)
    lookup_id = raw_id if raw_id in cfg else raw_id.upper()
    cfg_entry: dict[str, Any] = cfg.get(lookup_id, {}) if isinstance(cfg, dict) else {}
    default_entry: dict[str, Any] = DEFAULT_MEMES_CONFIG.get(lookup_id, {})

    stems = [raw_id]
    if custom_entry is not None:
        cname = str(getattr(custom_entry, "name", "") or "").strip()
        if cname and cname not in stems:
            stems.append(cname)
        title = cname or str(cfg_entry.get("title") or raw_id)
        accent_color = str(cfg_entry.get("accent_color") or "#A855F7")
    else:
        title = str(cfg_entry.get("title") or default_entry.get("title") or raw_id)
        accent_color = str(cfg_entry.get("accent_color") or default_entry.get("accent_color") or "#38BDF8")

    # 1. User overrides in assets/memes/
    meme_path = _find_override_file(memes_dir, stems, _IMAGE_EXTS)
    audio_path = _find_override_file(memes_dir, stems, _AUDIO_EXTS)

    # 2. CustomPoseEntry paths or configured paths
    if meme_path is None and custom_entry is not None:
        meme_path = _resolve_existing_path(getattr(custom_entry, "meme_path", None), base_dir)
    if audio_path is None and custom_entry is not None:
        audio_path = _resolve_existing_path(getattr(custom_entry, "audio_path", None), base_dir)

    if meme_path is None:
        meme_path = _resolve_existing_path(cfg_entry.get("meme_path"), base_dir)
    if audio_path is None:
        audio_path = _resolve_existing_path(cfg_entry.get("audio_path"), base_dir)

    # 3. Default asset fallback
    if meme_path is None:
        meme_path = _resolve_existing_path(default_entry.get("meme_path"), base_dir)
    if audio_path is None:
        audio_path = _resolve_existing_path(default_entry.get("audio_path"), base_dir)

    return {
        "pose_id": lookup_id if lookup_id in DEFAULT_MEMES_CONFIG else raw_id,
        "title": title,
        "meme_path": meme_path,
        "audio_path": audio_path,
        "accent_color": accent_color,
    }
