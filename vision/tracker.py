"""Threaded webcam capture and MediaPipe landmark inference (Tasks API + legacy fallback)."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_SPECS: dict[str, tuple[str, str]] = {
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
    "face": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
}


@dataclass
class FrameDetection:
    """Per-frame normalized landmark detections from MediaPipe."""

    pose: Optional[np.ndarray] = None  # shape (33, 4) float32: x, y, z, visibility
    hands: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (label in {"Left", "Right"}, (21, 3) float32)
    face: Optional[np.ndarray] = None  # shape (N, 3) float32, N >= 468
    timestamp_ms: int = 0


def _parse_version_tuple(ver: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in ver.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def get_mediapipe_backend() -> str:
    """Return active MediaPipe backend: ``'tasks'``, ``'solutions'``, or ``'unavailable'``."""
    has_tasks = False
    try:
        from mediapipe.tasks.python import vision as _mp_vision  # noqa: F401
        has_tasks = True
    except Exception:
        has_tasks = False

    has_solutions = hasattr(mp, "solutions")
    ver = _parse_version_tuple(getattr(mp, "__version__", "0.0.0"))
    if has_tasks and (not has_solutions or ver >= (0, 10, 30)):
        return "tasks"
    if has_solutions:
        return "solutions"
    if has_tasks:
        return "tasks"
    return "unavailable"


def ensure_task_models(
    models_dir: Union[str, Path] = DEFAULT_MODELS_DIR,
    status_cb: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 60.0,
) -> dict[str, Path]:
    """Ensure MediaPipe ``.task`` model files exist locally, downloading missing ones atomically."""
    target_dir = Path(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for key, (filename, url) in MODEL_SPECS.items():
        dest = target_dir / filename
        if dest.is_file() and dest.stat().st_size > 0:
            paths[key] = dest
            continue

        if status_cb is not None:
            status_cb(f"Downloading {filename}...")

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(target_dir),
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                req = urllib.request.Request(url, headers={"User-Agent": "PoseToMemeStudio/1.0"})
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())

            if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty model file for {filename}")

            os.replace(str(tmp_path), str(dest))
            tmp_path = None
        except Exception as exc:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Failed to download {filename} from {url}: {exc}") from exc

        paths[key] = dest

    if status_cb is not None:
        status_cb("MediaPipe models ready")
    return paths


def _normalize_handedness(raw_label: Optional[str]) -> str:
    if not raw_label:
        return "Right"
    low = raw_label.strip().lower()
    if low.startswith("l"):
        return "Left"
    return "Right"


def _landmarks_to_pose_array(landmarks: object) -> np.ndarray:
    arr = np.zeros((33, 4), dtype=np.float32)
    for idx, lm in enumerate(landmarks):  # type: ignore[arg-type]
        if idx >= 33:
            break
        vis = getattr(lm, "visibility", None)
        if vis is None:
            vis = getattr(lm, "presence", None)
        arr[idx, 0] = float(lm.x)
        arr[idx, 1] = float(lm.y)
        arr[idx, 2] = float(lm.z)
        arr[idx, 3] = float(1.0 if vis is None else vis)
    return arr


def _landmarks_to_xyz_array(landmarks: object) -> np.ndarray:
    pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in landmarks]  # type: ignore[attr-defined]
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


class MediaPipeLandmarker:
    """Unified wrapper over MediaPipe Tasks API (>=0.10.30) and legacy mp.solutions."""

    def __init__(
        self,
        models_dir: Union[str, Path] = DEFAULT_MODELS_DIR,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.status_cb = status_cb
        self.backend = get_mediapipe_backend()
        self._last_ts = -1

        # Tasks API handles
        self._pose_task = None
        self._hand_task = None
        self._face_task = None

        # Legacy solutions handles
        self._pose_sol = None
        self._hands_sol = None
        self._face_sol = None

    def initialize(self) -> None:
        """Load models and create MediaPipe landmarker instances."""
        if self.backend == "tasks":
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision

            model_paths = ensure_task_models(self.models_dir, status_cb=self.status_cb)
            if self.status_cb is not None:
                self.status_cb("Initializing MediaPipe Tasks landmarkers...")

            self._pose_task = mp_vision.PoseLandmarker.create_from_options(
                mp_vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_paths["pose"])),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            self._hand_task = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_paths["hand"])),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            self._face_task = mp_vision.FaceLandmarker.create_from_options(
                mp_vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_paths["face"])),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
        elif self.backend == "solutions":
            if self.status_cb is not None:
                self.status_cb("Initializing MediaPipe legacy solutions...")
            mp_sol = mp.solutions  # type: ignore[attr-defined]
            self._pose_sol = mp_sol.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands_sol = mp_sol.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_sol = mp_sol.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            raise RuntimeError("Neither mediapipe.tasks.python.vision nor mediapipe.solutions is available.")

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> FrameDetection:
        """Run pose, hand, and face landmark inference on an RGB uint8 image."""
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb_contig = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

        pose_arr: Optional[np.ndarray] = None
        hands_list: list[tuple[str, np.ndarray]] = []
        face_arr: Optional[np.ndarray] = None

        if self.backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contig)

            if self._pose_task is not None:
                pose_res = self._pose_task.detect_for_video(mp_image, ts)
                if pose_res.pose_landmarks:
                    pose_arr = _landmarks_to_pose_array(pose_res.pose_landmarks[0])

            if self._hand_task is not None:
                hand_res = self._hand_task.detect_for_video(mp_image, ts)
                for idx, h_lms in enumerate(hand_res.hand_landmarks):
                    label = "Right"
                    if idx < len(hand_res.handedness) and hand_res.handedness[idx]:
                        label = _normalize_handedness(hand_res.handedness[idx][0].category_name)
                    arr = _landmarks_to_xyz_array(h_lms)
                    if arr.shape == (21, 3):
                        hands_list.append((label, arr))

            if self._face_task is not None:
                face_res = self._face_task.detect_for_video(mp_image, ts)
                if face_res.face_landmarks:
                    arr = _landmarks_to_xyz_array(face_res.face_landmarks[0])
                    if arr.shape[0] >= 468:
                        face_arr = arr

        elif self.backend == "solutions":
            if self._pose_sol is not None:
                pose_res = self._pose_sol.process(rgb_contig)
                if pose_res.pose_landmarks:
                    pose_arr = _landmarks_to_pose_array(pose_res.pose_landmarks.landmark)

            if self._hands_sol is not None:
                hand_res = self._hands_sol.process(rgb_contig)
                if hand_res.multi_hand_landmarks:
                    handed = hand_res.multi_handedness or []
                    for idx, h_lms in enumerate(hand_res.multi_hand_landmarks):
                        label = "Right"
                        if idx < len(handed) and handed[idx].classification:
                            label = _normalize_handedness(handed[idx].classification[0].label)
                        arr = _landmarks_to_xyz_array(h_lms.landmark)
                        if arr.shape == (21, 3):
                            hands_list.append((label, arr))

            if self._face_sol is not None:
                face_res = self._face_sol.process(rgb_contig)
                if face_res.multi_face_landmarks:
                    arr = _landmarks_to_xyz_array(face_res.multi_face_landmarks[0].landmark)
                    if arr.shape[0] >= 468:
                        face_arr = arr

        return FrameDetection(
            pose=pose_arr,
            hands=hands_list,
            face=face_arr,
            timestamp_ms=ts,
        )

    def close(self) -> None:
        """Release all MediaPipe landmarker resources."""
        for obj in (
            self._pose_task,
            self._hand_task,
            self._face_task,
            self._pose_sol,
            self._hands_sol,
            self._face_sol,
        ):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._pose_task = None
        self._hand_task = None
        self._face_task = None
        self._pose_sol = None
        self._hands_sol = None
        self._face_sol = None


def _open_capture(index: int) -> cv2.VideoCapture:
    """Open an OpenCV VideoCapture cleanly (preferring CAP_DSHOW on Windows)."""
    if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def list_available_cameras(max_idx: int = 5) -> list[int]:
    """Probe camera indices ``0..max_idx-1`` and return available indices."""
    available: list[int] = []
    for idx in range(max(0, int(max_idx))):
        cap = None
        try:
            cap = _open_capture(idx)
            if cap is not None and cap.isOpened():
                available.append(idx)
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return available


class CameraTracker(QThread):
    """Background QThread capturing webcam frames and running MediaPipe inference."""

    frame_ready = Signal(np.ndarray, object, float)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        camera_index: int = 0,
        mirror: bool = True,
        models_dir: Optional[Union[str, Path]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._camera_index = int(camera_index)
        self._pending_camera: Optional[int] = None
        self._mirror = bool(mirror)
        self._models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
        self._running = False

    def camera_index(self) -> int:
        with self._lock:
            return self._camera_index

    def is_mirror(self) -> bool:
        with self._lock:
            return self._mirror

    def set_camera(self, index: int) -> None:
        """Switch active camera index (applied cleanly on the next capture loop iteration)."""
        idx = int(index)
        with self._lock:
            if idx == self._camera_index and self._pending_camera is None:
                return
            if self._running:
                self._pending_camera = idx
            else:
                self._camera_index = idx

    def set_mirror(self, enabled: bool) -> None:
        """Toggle horizontal mirror flip for capture and inference."""
        with self._lock:
            self._mirror = bool(enabled)

    def stop(self, timeout_ms: int = 3000) -> None:
        """Signal the capture loop to exit and wait cleanly for thread termination."""
        self._running = False
        if self.isRunning() and QThread.currentThread() is not self:
            if not self.wait(max(100, int(timeout_ms))):
                self.terminate()
                self.wait(1000)

    def _configure_capture(self, cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def run(self) -> None:
        self._running = True
        landmarker: Optional[MediaPipeLandmarker] = None
        cap: Optional[cv2.VideoCapture] = None

        try:
            landmarker = MediaPipeLandmarker(
                models_dir=self._models_dir,
                status_cb=lambda msg: self.status_changed.emit(msg),
            )
            landmarker.initialize()

            with self._lock:
                cam_idx = self._camera_index
                self._pending_camera = None

            self.status_changed.emit(f"Opening camera {cam_idx}...")
            cap = _open_capture(cam_idx)
            if not cap.isOpened():
                self.error_occurred.emit(f"Unable to open camera index {cam_idx}.")
                return

            self._configure_capture(cap)
            self.status_changed.emit(f"Camera {cam_idx} active")

            ema_fps = 0.0
            last_frame_t = time.perf_counter()
            consecutive_failures = 0

            while self._running:
                with self._lock:
                    switch_to = self._pending_camera
                    self._pending_camera = None
                    mirror = self._mirror

                if switch_to is not None:
                    self.status_changed.emit(f"Switching to camera {switch_to}...")
                    if cap is not None:
                        cap.release()
                    cap = _open_capture(switch_to)
                    if not cap.isOpened():
                        self.error_occurred.emit(f"Unable to open camera index {switch_to}.")
                        break
                    self._configure_capture(cap)
                    with self._lock:
                        self._camera_index = switch_to
                    consecutive_failures = 0
                    self.status_changed.emit(f"Camera {switch_to} active")

                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None or frame_bgr.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures > 60:
                        self.error_occurred.emit("Camera stream interrupted (no frames received).")
                        break
                    self.msleep(10)
                    continue

                consecutive_failures = 0
                if mirror:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ts_ms = int(time.monotonic() * 1000)
                detection = landmarker.detect(rgb_frame, ts_ms)

                now = time.perf_counter()
                dt = max(1e-4, now - last_frame_t)
                last_frame_t = now
                inst_fps = 1.0 / dt
                ema_fps = inst_fps if ema_fps == 0.0 else (0.85 * ema_fps + 0.15 * inst_fps)

                self.frame_ready.emit(frame_bgr, detection, float(ema_fps))

        except Exception as exc:
            self.error_occurred.emit(f"Vision tracker error: {exc}")
        finally:
            self._running = False
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if landmarker is not None:
                landmarker.close()
            self.status_changed.emit("Camera stopped")
