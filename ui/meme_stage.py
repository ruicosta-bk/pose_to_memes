"""Meme display stage widget supporting static images, animated GIFs, transitions, and PiP export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QEasingCurve, QIODevice, QPropertyAnimation, Qt
from PySide6.QtGui import QImage, QMovie, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def qimage_to_bgr(image: QImage) -> Optional[np.ndarray]:
    """Convert a QImage into a contiguous uint8 BGR numpy array."""
    if image.isNull():
        return None
    rgb_img = image.convertToFormat(QImage.Format.Format_RGB888)
    width = rgb_img.width()
    height = rgb_img.height()
    if width <= 0 or height <= 0:
        return None
    bytes_per_line = rgb_img.bytesPerLine()
    ptr = rgb_img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=bytes_per_line * height)
    arr = arr.reshape((height, bytes_per_line))[:, : width * 3]
    rgb = arr.reshape((height, width, 3))
    return rgb[:, :, ::-1].copy()


class MemeStageWidget(QWidget):
    """Displays active meme images or animated GIFs with smooth transition and title badge."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_pose_id: str = ""
        self._current_media_path: str = ""
        self._accent_color: str = "#22d3ee"

        self._source_pixmap: Optional[QPixmap] = None
        self._movie: Optional[QMovie] = None
        self._movie_buffer: Optional[QBuffer] = None
        self._movie_bytes: Optional[QByteArray] = None

        self._build_ui()
        self._apply_accent_style(self._accent_color)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self._card = QFrame(self)
        self._card.setObjectName("MemeStageCard")
        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        # Top badge header
        header = QHBoxLayout()
        header.setSpacing(8)

        self._title_badge = QLabel("WAITING FOR POSE...", self._card)
        self._title_badge.setObjectName("MemeTitleBadge")
        self._title_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._media_type_pill = QLabel("IDLE", self._card)
        self._media_type_pill.setObjectName("MediaTypePill")
        self._media_type_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header.addWidget(self._title_badge)
        header.addStretch(1)
        header.addWidget(self._media_type_pill)
        card_layout.addLayout(header)

        # Media canvas
        self._media_label = QLabel("No meme loaded", self._card)
        self._media_label.setObjectName("MemeCanvas")
        self._media_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._media_label.setMinimumSize(320, 260)
        self._media_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._media_label.setWordWrap(True)

        self._opacity_effect = QGraphicsOpacityEffect(self._media_label)
        self._opacity_effect.setOpacity(1.0)
        self._media_label.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setDuration(220)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        card_layout.addWidget(self._media_label, stretch=1)
        root_layout.addWidget(self._card)

    def _apply_accent_style(self, accent_color: str) -> None:
        color = accent_color or "#22d3ee"
        self._card.setStyleSheet(
            f"""
            QFrame#MemeStageCard {{
                background-color: #0f172a;
                border: 2px solid {color};
                border-radius: 14px;
            }}
            QLabel#MemeTitleBadge {{
                background-color: {color};
                color: #090d16;
                font-weight: 700;
                font-size: 13px;
                padding: 4px 12px;
                border-radius: 10px;
            }}
            QLabel#MediaTypePill {{
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #334155;
                font-weight: 600;
                font-size: 11px;
                padding: 3px 10px;
                border-radius: 9px;
            }}
            QLabel#MemeCanvas {{
                background-color: #090d16;
                border-radius: 10px;
                color: #94a3b8;
                font-size: 14px;
            }}
            """
        )

    def _stop_movie(self) -> None:
        if self._movie is not None:
            try:
                self._movie.frameChanged.disconnect(self._on_movie_frame)
            except (RuntimeError, TypeError):
                pass
            self._movie.stop()
            self._movie.setDevice(None)
            self._movie.deleteLater()
            self._movie = None
        if self._movie_buffer is not None:
            self._movie_buffer.close()
            self._movie_buffer.deleteLater()
            self._movie_buffer = None
        self._movie_bytes = None

    def _play_transition(self) -> None:
        self._fade_anim.stop()
        self._fade_anim.setStartValue(0.18)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    def _render_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            return
        target_size = self._media_label.contentsRect().size()
        if target_size.width() <= 10 or target_size.height() <= 10:
            self._media_label.setPixmap(pixmap)
            return
        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._media_label.setPixmap(scaled)

    def _on_movie_frame(self, _frame_number: int) -> None:
        if self._movie is None:
            return
        pix = self._movie.currentPixmap()
        if not pix.isNull():
            self._render_pixmap(pix)

    def set_meme(
        self,
        pose_id: str,
        title: str,
        media_path: Optional[str | Path],
        accent_color: str = "#22d3ee",
        force_animate: bool = False,
    ) -> None:
        """Load and display a static image or animated GIF for the active pose."""
        resolved_str = str(media_path) if media_path else ""
        changed = (
            pose_id != self._current_pose_id
            or resolved_str != self._current_media_path
            or force_animate
        )

        self._current_pose_id = pose_id
        self._current_media_path = resolved_str
        self._accent_color = accent_color or "#22d3ee"
        self._title_badge.setText(title or pose_id or "NEUTRAL")
        self._apply_accent_style(self._accent_color)

        if not changed:
            return

        self._stop_movie()
        self._source_pixmap = None

        path_obj = Path(resolved_str) if resolved_str else None
        if path_obj is None or not path_obj.is_file():
            self._media_label.setPixmap(QPixmap())
            self._media_label.setText(f"{title or pose_id}\n(Media file not found)")
            self._media_type_pill.setText("MISSING")
            self._play_transition()
            return

        suffix = path_obj.suffix.lower()
        if suffix == ".gif":
            try:
                raw_bytes = path_obj.read_bytes()
            except OSError:
                raw_bytes = b""
            qbytes = QByteArray(raw_bytes)
            buf = QBuffer(self)
            buf.setData(qbytes)
            buf.open(QIODevice.OpenModeFlag.ReadOnly)
            movie = QMovie(buf, b"gif", parent=self)
            if not movie.isValid():
                movie.setDevice(None)
                movie.deleteLater()
                buf.close()
                buf.deleteLater()
                self._media_label.setPixmap(QPixmap())
                self._media_label.setText(f"Failed to load GIF:\n{path_obj.name}")
                self._media_type_pill.setText("ERROR")
                return
            self._movie_bytes = qbytes
            self._movie_buffer = buf
            self._movie = movie
            self._media_type_pill.setText("GIF")
            self._media_label.setText("")
            movie.frameChanged.connect(self._on_movie_frame)
            movie.start()
            first_pix = movie.currentPixmap()
            if not first_pix.isNull():
                self._render_pixmap(first_pix)
            self._play_transition()
            return

        pixmap = QPixmap(str(path_obj))
        if pixmap.isNull():
            self._media_label.setPixmap(QPixmap())
            self._media_label.setText(f"Failed to load image:\n{path_obj.name}")
            self._media_type_pill.setText("ERROR")
            return

        self._source_pixmap = pixmap
        self._media_type_pill.setText(suffix.lstrip(".").upper() or "IMAGE")
        self._media_label.setText("")
        self._render_pixmap(pixmap)
        self._play_transition()

    def get_current_qimage(self) -> Optional[QImage]:
        """Return the current unscaled meme frame as a QImage, if any."""
        if self._movie is not None:
            img = self._movie.currentImage()
            return None if img.isNull() else img
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            return self._source_pixmap.toImage()
        return None

    def get_current_frame_bgr(self) -> Optional[np.ndarray]:
        """Return the current meme frame (including active GIF frame) as BGR ndarray."""
        img = self.get_current_qimage()
        if img is None:
            return None
        return qimage_to_bgr(img)

    def cleanup(self) -> None:
        """Stop any running QMovie and release media resources."""
        self._fade_anim.stop()
        self._stop_movie()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._movie is not None:
            pix = self._movie.currentPixmap()
            if not pix.isNull():
                self._render_pixmap(pix)
        elif self._source_pixmap is not None and not self._source_pixmap.isNull():
            self._render_pixmap(self._source_pixmap)
