"""Integration and headless UI smoke tests for Pose-to-Meme Studio."""

from __future__ import annotations

import os
import wave
from pathlib import Path

# Ensure headless Qt platform is configured before importing PySide6 widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mediapipe as mp
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from assets import (
    DEFAULT_AUDIO_FILES,
    POSE_ORDER,
    AudioPlayer,
    ensure_default_assets,
    load_memes_config,
    resolve_pose_media,
)
from classifier import CustomPoseStore, PosePipeline
from main import build_studio, create_app
from tests.test_classifier import build_pose_sample
from ui import MainWindow, MemeStageWidget, RecorderDialog
from vision import (
    CameraTracker,
    FrameDetection,
    MediaPipeLandmarker,
    composite_pip,
    draw_neon_hud,
    get_mediapipe_backend,
)

EXPECTED_PNG_NAMES = {
    "absolute_cinema.png",
    "erm_actually.png",
    "thinking.png",
    "shocked.png",
    "domain_expansion.png",
    "peace_sign.png",
    "neutral.png",
}


def _get_qapp() -> QApplication:
    return create_app(["pytest-offscreen"])


def test_ensure_default_assets_idempotent_and_valid(tmp_path: Path) -> None:
    res1 = ensure_default_assets(tmp_path)
    default_dir = tmp_path / "assets" / "default"
    assert (tmp_path / "assets" / "memes" / ".gitkeep").is_file()
    assert (tmp_path / "data" / "memes_config.json").is_file()

    # All 7 default PNGs exist and open cleanly as 800x600 images
    png_names = {p.name for p in default_dir.glob("*.png")}
    assert EXPECTED_PNG_NAMES.issubset(png_names)
    for name in EXPECTED_PNG_NAMES:
        with Image.open(default_dir / name) as img:
            assert img.size == (800, 600)

    # All default WAVs exist and have valid RIFF/PCM frames
    wav_names = {p.name for p in default_dir.glob("*.wav")}
    assert set(DEFAULT_AUDIO_FILES).issubset(wav_names)
    for name in DEFAULT_AUDIO_FILES:
        with wave.open(str(default_dir / name), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            assert wf.getnframes() > 1000

    # Idempotency: second call preserves existing files without regenerating/failing
    sentinel = default_dir / "absolute_cinema.png"
    mtime_before = sentinel.stat().st_mtime_ns
    res2 = ensure_default_assets(tmp_path)
    assert sentinel.stat().st_mtime_ns == mtime_before
    assert len(res1["images"]) == len(POSE_ORDER)
    assert len(res2["images"]) == len(POSE_ORDER)

    # Config + user override resolution
    cfg = load_memes_config(tmp_path)
    for pose_id in POSE_ORDER:
        media = resolve_pose_media(pose_id, cfg, root_dir=tmp_path)
        assert media["meme_path"] and Path(media["meme_path"]).is_file()

    override_gif = tmp_path / "assets" / "memes" / "ABSOLUTE_CINEMA.gif"
    Image.new("RGB", (32, 32), color=(255, 0, 255)).save(override_gif, format="GIF")
    overridden = resolve_pose_media("ABSOLUTE_CINEMA", cfg, root_dir=tmp_path)
    assert overridden["meme_path"] == str(override_gif)


def test_vision_overlay_hud_and_pip() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (25, 30, 40)

    # Without detection (None)
    out_empty = draw_neon_hud(frame, None)
    assert out_empty.shape == frame.shape
    assert out_empty is not frame

    # With synthetic full detection
    pose, hands, face = build_pose_sample("DOMAIN_EXPANSION")
    det = FrameDetection(pose=pose, hands=hands, face=face, timestamp_ms=100)
    out_hud = draw_neon_hud(frame, det, active_pose="DOMAIN_EXPANSION", confidence=0.95)
    assert out_hud.shape == frame.shape
    assert not np.array_equal(out_hud, frame)

    # PiP with BGR meme
    meme_bgr = np.full((120, 160, 3), (0, 200, 255), dtype=np.uint8)
    pip_bgr = composite_pip(out_hud, meme_bgr, label="DOMAIN EXPANSION")
    assert pip_bgr.shape == frame.shape
    assert not np.array_equal(pip_bgr, out_hud)

    # PiP with BGRA meme (semi-transparent alpha) and None meme
    meme_bgra = np.full((120, 160, 4), (255, 100, 50, 180), dtype=np.uint8)
    pip_bgra = composite_pip(out_hud, meme_bgra, label="ALPHA")
    assert pip_bgra.shape == frame.shape
    assert np.array_equal(composite_pip(out_hud, None), out_hud)


def test_vision_tracker_mediapipe_compatibility(tmp_path: Path) -> None:
    backend = get_mediapipe_backend()
    if hasattr(mp, "solutions"):
        assert backend in {"tasks", "solutions"}
    else:
        assert backend == "tasks"

    landmarker = MediaPipeLandmarker(models_dir=tmp_path / "models")
    assert landmarker.backend == backend

    _get_qapp()
    tracker = CameraTracker(camera_index=0, mirror=True, models_dir=tmp_path / "models")
    assert tracker.camera_index() == 0
    assert tracker.is_mirror() is True
    tracker.set_mirror(False)
    assert tracker.is_mirror() is False
    tracker.set_camera(1)
    assert tracker.camera_index() == 1
    tracker.stop(timeout_ms=200)


def test_headless_ui_smoke(tmp_path: Path) -> None:
    app = _get_qapp()
    ensure_default_assets(tmp_path)

    # 1. MemeStageWidget with static PNG and animated GIF
    stage = MemeStageWidget()
    png_path = tmp_path / "assets" / "default" / "absolute_cinema.png"
    stage.set_meme("ABSOLUTE_CINEMA", "ABSOLUTE CINEMA", png_path, "#F59E0B")
    bgr_png = stage.get_current_frame_bgr()
    assert bgr_png is not None and bgr_png.ndim == 3 and bgr_png.shape[2] == 3

    gif_path = tmp_path / "anim.gif"
    f1 = Image.new("RGB", (64, 64), (255, 0, 0))
    f2 = Image.new("RGB", (64, 64), (0, 255, 0))
    f1.save(gif_path, save_all=True, append_images=[f2], duration=50, loop=0)
    stage.set_meme("CUSTOM_GIF", "GIF TEST", gif_path, "#22d3ee")
    bgr_gif = stage.get_current_frame_bgr()
    assert bgr_gif is not None and bgr_gif.shape[:2] == (64, 64)
    stage.cleanup()
    stage.close()

    # 2. RecorderDialog capture flow with synthetic FrameDetection
    recorder = RecorderDialog()
    recorder._name_edit.setText("Smoke Pose")
    pose, hands, face = build_pose_sample("PEACE_SIGN")
    det = FrameDetection(pose=pose, hands=hands, face=face, timestamp_ms=500)
    recorder._phase = "capturing"
    recorder.on_frame(np.zeros((240, 320, 3), dtype=np.uint8), det, 30.0)
    recorder._finish_capture()
    entry = recorder.get_recorded_entry()
    assert entry is not None
    assert entry.name == "Smoke Pose"
    assert len(entry.vector) == 150
    recorder.close()

    # 3. MainWindow full pipeline smoke test via build_studio + synthetic frames
    window: MainWindow = build_studio(tmp_path, probe_cameras=False)
    window.audio_player.set_muted(True)
    window.show()
    app.processEvents()

    cinema_pose, cinema_hands, cinema_face = build_pose_sample("ABSOLUTE_CINEMA")
    frame = np.full((360, 480, 3), (18, 22, 32), dtype=np.uint8)

    res1 = window.handle_frame(
        frame,
        FrameDetection(pose=cinema_pose, hands=cinema_hands, face=cinema_face, timestamp_ms=0),
        fps=60.0,
    )
    assert res1 is not None
    assert res1.active_pose is None

    res2 = window.handle_frame(
        frame,
        FrameDetection(pose=cinema_pose, hands=cinema_hands, face=cinema_face, timestamp_ms=300),
        fps=60.0,
    )
    assert res2 is not None
    assert res2.active_pose == "ABSOLUTE_CINEMA"
    assert res2.triggered_pose == "ABSOLUTE_CINEMA"
    assert "ABSOLUTE CINEMA" in window._status_pill.text()

    app.processEvents()
    window.close()
    app.processEvents()
