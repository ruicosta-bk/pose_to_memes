"""Vision package: threaded webcam tracking, MediaPipe compatibility, and neon HUD overlays."""

from .overlay import composite_pip, draw_neon_hud
from .tracker import (
    DEFAULT_MODELS_DIR,
    MODEL_SPECS,
    CameraTracker,
    FrameDetection,
    MediaPipeLandmarker,
    ensure_task_models,
    get_mediapipe_backend,
    list_available_cameras,
)

__all__ = [
    "CameraTracker",
    "DEFAULT_MODELS_DIR",
    "FrameDetection",
    "MODEL_SPECS",
    "MediaPipeLandmarker",
    "composite_pip",
    "draw_neon_hud",
    "ensure_task_models",
    "get_mediapipe_backend",
    "list_available_cameras",
]
