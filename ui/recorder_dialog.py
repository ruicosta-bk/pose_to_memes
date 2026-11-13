"""Dialog for recording a custom pose with a 3-second countdown and 0.5s vector capture."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from classifier import (
    CustomPoseEntry,
    extract_face_blendshape_vector,
    extract_pose_vector,
)
from vision import draw_neon_hud


def bgr_to_qpixmap(frame_bgr: np.ndarray, target_size: Optional[Any] = None) -> QPixmap:
    """Convert a BGR uint8 numpy frame into a scaled QPixmap safely."""
    if frame_bgr is None or frame_bgr.size == 0:
        return QPixmap()
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if target_size is not None and target_size.width() > 10 and target_size.height() > 10:
        return pix.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pix


class RecorderDialog(QDialog):
    """Modal/non-blocking studio dialog to record a Tier 2 custom pose."""

    pose_recorded = Signal(object)  # Emits CustomPoseEntry

    COUNTDOWN_MS = 3000
    CAPTURE_MS = 550
    TICK_MS = 50

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Record New Custom Pose")
        self.setMinimumSize(640, 600)
        self.setModal(True)

        self._phase: str = "idle"  # "idle" | "countdown" | "capturing"
        self._elapsed_ms: int = 0
        self._captured_vectors: list[np.ndarray] = []
        self._captured_face_vectors: list[np.ndarray] = []
        self._last_detection: Any = None
        self.recorded_entry: Optional[CustomPoseEntry] = None

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        header = QLabel("CUSTOM POSE RECORDER", self)
        header.setObjectName("DialogHeading")
        root.addWidget(header)

        subtitle = QLabel(
            "Strike your pose, enter a name, optionally attach a meme image/GIF and audio SFX, "
            "then press Start 3s Countdown.",
            self,
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("DialogSubtitle")
        root.addWidget(subtitle)

        # Camera preview card
        preview_card = QFrame(self)
        preview_card.setObjectName("PreviewCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(8)

        self._preview_label = QLabel("Waiting for camera feed...", preview_card)
        self._preview_label.setObjectName("PreviewCanvas")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(560, 280)
        preview_layout.addWidget(self._preview_label, stretch=1)

        self._status_label = QLabel("Ready to record.", preview_card)
        self._status_label.setObjectName("RecorderStatus")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self._status_label)

        root.addWidget(preview_card, stretch=1)

        # Form inputs
        form_card = QFrame(self)
        form_card.setObjectName("FormCard")
        form = QFormLayout(form_card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)

        self._name_edit = QLineEdit(form_card)
        self._name_edit.setPlaceholderText("e.g. GIGA_CHAD or SALUTE")
        form.addRow("Pose Name:", self._name_edit)

        # Meme media picker
        meme_row = QHBoxLayout()
        self._meme_path_edit = QLineEdit(form_card)
        self._meme_path_edit.setPlaceholderText("Optional: PNG, JPG, WEBP, or GIF")
        meme_browse = QPushButton("Browse...", form_card)
        meme_browse.clicked.connect(self._browse_meme)
        meme_row.addWidget(self._meme_path_edit, stretch=1)
        meme_row.addWidget(meme_browse)
        form.addRow("Meme Image/GIF:", meme_row)

        # Audio SFX picker
        audio_row = QHBoxLayout()
        self._audio_path_edit = QLineEdit(form_card)
        self._audio_path_edit.setPlaceholderText("Optional: WAV, MP3, or OGG")
        audio_browse = QPushButton("Browse...", form_card)
        audio_browse.clicked.connect(self._browse_audio)
        audio_row.addWidget(self._audio_path_edit, stretch=1)
        audio_row.addWidget(audio_browse)
        form.addRow("Audio SFX:", audio_row)

        self._threshold_spin = QDoubleSpinBox(form_card)
        self._threshold_spin.setRange(0.65, 0.99)
        self._threshold_spin.setSingleStep(0.01)
        self._threshold_spin.setValue(0.88)
        form.addRow("Match Threshold:", self._threshold_spin)

        self._include_face_chk = QCheckBox("Include Face Expression (Blendshapes)", form_card)
        self._include_face_chk.setChecked(False)
        form.addRow("", self._include_face_chk)

        root.addWidget(form_card)

        # Action buttons
        actions = QHBoxLayout()
        actions.setSpacing(10)

        self._start_btn = QPushButton("Start 3s Countdown", self)
        self._start_btn.setObjectName("PrimaryCta")
        self._start_btn.clicked.connect(self.start_countdown)

        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.clicked.connect(self.reject)

        actions.addStretch(1)
        actions.addWidget(self._cancel_btn)
        actions.addWidget(self._start_btn)
        root.addLayout(actions)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #090d16;
                color: #e2e8f0;
                font-family: "Segoe UI", sans-serif;
            }
            QLabel#DialogHeading {
                font-size: 16px;
                font-weight: 700;
                color: #22d3ee;
                letter-spacing: 1px;
            }
            QLabel#DialogSubtitle {
                font-size: 12px;
                color: #94a3b8;
            }
            QFrame#PreviewCard, QFrame#FormCard {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }
            QLabel#PreviewCanvas {
                background-color: #050811;
                border-radius: 8px;
                color: #64748b;
            }
            QLabel#RecorderStatus {
                font-size: 13px;
                font-weight: 700;
                color: #38bdf8;
                padding: 4px;
            }
            QLineEdit, QDoubleSpinBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 8px;
                color: #f8fafc;
            }
            QLineEdit:focus, QDoubleSpinBox:focus {
                border-color: #22d3ee;
            }
            QCheckBox {
                spacing: 7px;
                color: #cbd5e1;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #475569;
                background-color: #1e293b;
            }
            QCheckBox::indicator:checked {
                background-color: #22d3ee;
                border-color: #22d3ee;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 14px;
                color: #e2e8f0;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #334155;
            }
            QPushButton#PrimaryCta {
                background-color: #7c3aed;
                border: 1px solid #a855f7;
                color: #ffffff;
            }
            QPushButton#PrimaryCta:hover {
                background-color: #6d28d9;
            }
            QPushButton#PrimaryCta:disabled {
                background-color: #334155;
                border-color: #475569;
                color: #94a3b8;
            }
            """
        )

    def _browse_meme(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Meme Image or GIF",
            "",
            "Media Files (*.png *.jpg *.jpeg *.webp *.gif *.bmp);;All Files (*)",
        )
        if file_path:
            self._meme_path_edit.setText(file_path)

    def _browse_audio(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio SFX",
            "",
            "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)",
        )
        if file_path:
            self._audio_path_edit.setText(file_path)

    def _set_status(self, text: str, color: str = "#38bdf8") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {color}; padding: 4px;"
        )

    def start_countdown(self) -> None:
        """Validate pose name and begin the 3-second countdown before capturing."""
        name = self._name_edit.text().strip()
        if not name:
            self._set_status("Please enter a Pose Name before recording.", "#f87171")
            self._name_edit.setFocus()
            return

        self.recorded_entry = None
        self._captured_vectors.clear()
        self._captured_face_vectors.clear()
        self._elapsed_ms = 0
        self._phase = "countdown"
        self._start_btn.setEnabled(False)
        self._set_status("3... Hold steady!", "#fbbf24")
        self._timer.start()

    def _on_tick(self) -> None:
        self._elapsed_ms += self.TICK_MS
        if self._phase == "countdown":
            if self._elapsed_ms < 1000:
                self._set_status("3... Get ready!", "#fbbf24")
            elif self._elapsed_ms < 2000:
                self._set_status("2... Strike your pose!", "#fb923c")
            elif self._elapsed_ms < self.COUNTDOWN_MS:
                self._set_status("1... Hold it!", "#f472b6")
            else:
                self._phase = "capturing"
                self._elapsed_ms = 0
                self._captured_vectors.clear()
                self._captured_face_vectors.clear()
                if self._last_detection is not None:
                    self.feed_detection(self._last_detection)
                self._set_status("CAPTURING! Hold still...", "#22d3ee")
        elif self._phase == "capturing":
            if self._elapsed_ms >= self.CAPTURE_MS:
                self._finish_capture()

    def feed_detection(self, detection: Any) -> None:
        """Accept a FrameDetection (useful for live capture and deterministic tests)."""
        self._last_detection = detection
        if self._phase != "capturing" or detection is None:
            return
        pose = getattr(detection, "pose", None)
        hands = getattr(detection, "hands", None)
        vec = extract_pose_vector(pose, hands)
        if vec is not None and np.linalg.norm(vec) > 1e-6:
            self._captured_vectors.append(vec)

        if self._include_face_chk.isChecked():
            blendshapes = getattr(detection, "face_blendshapes", None)
            face_vec = extract_face_blendshape_vector(blendshapes)
            if face_vec is not None and np.linalg.norm(face_vec) > 1e-6:
                self._captured_face_vectors.append(face_vec)

    def on_frame(self, frame_bgr: np.ndarray, detection: Any, _fps: float = 0.0) -> None:
        """Receive live webcam frame + detection from MainWindow/CameraTracker."""
        self.feed_detection(detection)
        if frame_bgr is None or frame_bgr.size == 0:
            return

        display = draw_neon_hud(frame_bgr, detection)
        overlay_text = ""
        overlay_color = (34, 211, 238)
        if self._phase == "countdown":
            rem = max(1, 3 - (self._elapsed_ms // 1000))
            overlay_text = f"{rem}..."
            overlay_color = (60, 190, 255)
        elif self._phase == "capturing":
            if self._include_face_chk.isChecked():
                overlay_text = (
                    f"CAPTURING (B:{len(self._captured_vectors)} F:{len(self._captured_face_vectors)})"
                )
            else:
                overlay_text = f"CAPTURING ({len(self._captured_vectors)})"
            overlay_color = (238, 211, 34)

        if overlay_text:
            cv2.putText(
                display,
                overlay_text,
                (28, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 0),
                6,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                overlay_text,
                (28, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                overlay_color,
                3,
                cv2.LINE_AA,
            )

        pix = bgr_to_qpixmap(display, self._preview_label.contentsRect().size())
        if not pix.isNull():
            self._preview_label.setText("")
            self._preview_label.setPixmap(pix)

    def _finish_capture(self) -> None:
        self._timer.stop()
        self._phase = "idle"
        self._start_btn.setEnabled(True)

        include_face = self._include_face_chk.isChecked()
        entry = self.build_entry_from_captured()
        if entry is None:
            if include_face and not self._captured_face_vectors:
                self._set_status(
                    "Capture failed: no facial expression/blendshapes detected. "
                    "Express clearly facing the camera and try again.",
                    "#f87171",
                )
            else:
                self._set_status(
                    "Capture failed: no valid pose detected. Ensure shoulders are visible and try again.",
                    "#f87171",
                )
            return

        self.recorded_entry = entry
        sample_count = max(len(self._captured_vectors), len(self._captured_face_vectors))
        self._set_status(
            f"Captured '{entry.name}' ({sample_count} samples)!",
            "#4ade80",
        )
        self.pose_recorded.emit(entry)
        self.accept()

    @staticmethod
    def _average_and_normalize(vectors: list[np.ndarray]) -> Optional[list[float]]:
        if not vectors:
            return None
        stacked = np.stack(vectors, axis=0)
        mean_vec = np.mean(stacked, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_vec))
        if norm <= 1e-6 or not np.isfinite(norm):
            return None
        normalized = mean_vec / norm
        return [float(x) for x in normalized]

    def build_entry_from_captured(self) -> Optional[CustomPoseEntry]:
        """Average and L2-normalize captured body/face vectors into a CustomPoseEntry."""
        include_face = self._include_face_chk.isChecked()
        body_list = self._average_and_normalize(self._captured_vectors)
        face_list = self._average_and_normalize(self._captured_face_vectors) if include_face else None

        if include_face:
            if face_list is None:
                return None
        elif body_list is None:
            return None

        name = self._name_edit.text().strip() or "Custom Pose"
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.upper()).strip("_") or "CUSTOM"
        pose_id = f"CUSTOM_{slug[:16]}_{uuid.uuid4().hex[:6].upper()}"

        meme_path = self._meme_path_edit.text().strip()
        audio_path = self._audio_path_edit.text().strip()
        if meme_path:
            meme_path = str(Path(meme_path))
        if audio_path:
            audio_path = str(Path(audio_path))

        return CustomPoseEntry(
            id=pose_id,
            name=name,
            vector=body_list if body_list is not None else [],
            meme_path=meme_path,
            audio_path=audio_path,
            threshold=float(self._threshold_spin.value()),
            include_face=include_face,
            face_vector=face_list,
        )

    def get_recorded_entry(self) -> Optional[CustomPoseEntry]:
        """Return the newly captured CustomPoseEntry if recording succeeded."""
        return self.recorded_entry

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().closeEvent(event)

    def reject(self) -> None:
        self._timer.stop()
        self._phase = "idle"
        super().reject()
