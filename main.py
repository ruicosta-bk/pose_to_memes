"""Pose-to-Meme Studio application entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication

from assets import AudioPlayer, ensure_default_assets, load_memes_config
from classifier import CustomPoseStore, PosePipeline
from ui import MainWindow

BASE_DIR = Path(__file__).resolve().parent


def create_app(argv: Optional[list[str]] = None) -> QApplication:
    """Return existing QApplication or create a new one cleanly."""
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication(argv if argv is not None else sys.argv)


def build_studio(
    base_dir: Optional[Path] = None,
    *,
    probe_cameras: bool = True,
) -> MainWindow:
    """Initialize assets, config, classifier pipeline, audio, and MainWindow."""
    root = Path(base_dir) if base_dir is not None else BASE_DIR
    ensure_default_assets(root)

    config = load_memes_config(root)
    custom_store = CustomPoseStore(root / "data" / "custom_poses.json")
    audio_player = AudioPlayer()
    pipeline = PosePipeline(custom_store=custom_store)

    return MainWindow(
        root_dir=root,
        config=config,
        custom_store=custom_store,
        audio_player=audio_player,
        pipeline=pipeline,
        probe_cameras=probe_cameras,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Launch Pose-to-Meme Studio GUI event loop."""
    app = create_app(argv)
    app.setApplicationName("Pose-to-Meme Studio")
    window = build_studio(BASE_DIR)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
