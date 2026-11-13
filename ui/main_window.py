"""Main Studio window for Pose-to-Meme Studio."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from assets import (
    AudioPlayer,
    MemesConfig,
    ensure_default_assets,
    load_memes_config,
    resolve_pose_media,
)
from classifier import (
    BUILTIN_POSES,
    CustomPoseEntry,
    CustomPoseStore,
    PipelineResult,
    PosePipeline,
)
from vision import (
    CameraTracker,
    FrameDetection,
    composite_pip,
    draw_neon_hud,
    list_available_cameras,
)

from .meme_stage import MemeStageWidget
from .recorder_dialog import RecorderDialog, bgr_to_qpixmap

ROLE_POSE_ID = Qt.ItemDataRole.UserRole
ROLE_IS_CUSTOM = Qt.ItemDataRole.UserRole + 1

STUDIO_QSS = """
QMainWindow, QWidget#CentralRoot {
    background-color: #090d16;
    color: #e2e8f0;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QFrame#StudioCard {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
}
QLabel#BrandTitle {
    font-size: 17px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: 1px;
}
QLabel#SectionHeading {
    font-size: 12px;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.8px;
}
QLabel#FpsPill {
    background-color: #1e293b;
    color: #22d3ee;
    border: 1px solid #0e7490;
    border-radius: 10px;
    padding: 4px 10px;
    font-weight: 700;
    font-size: 12px;
}
QLabel#StatusPill {
    background-color: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 4px 12px;
    font-weight: 700;
    font-size: 12px;
}
QLabel#WebcamCanvas {
    background-color: #050811;
    border: 1px solid #1e293b;
    border-radius: 10px;
    color: #64748b;
    font-size: 14px;
}
QComboBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px 10px;
    color: #f8fafc;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #22d3ee;
}
QComboBox QAbstractItemView {
    background-color: #0f172a;
    color: #f8fafc;
    selection-background-color: #0e7490;
}
QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 7px 12px;
    color: #e2e8f0;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}
QPushButton:disabled {
    background-color: #111827;
    color: #475569;
    border-color: #1e293b;
}
QPushButton#CameraStartBtn {
    background-color: #0891b2;
    border: 1px solid #22d3ee;
    color: #041017;
    font-weight: 700;
}
QPushButton#CameraStartBtn:hover {
    background-color: #06b6d4;
}
QPushButton#CameraStopBtn {
    background-color: #be123c;
    border: 1px solid #f43f5e;
    color: #ffffff;
    font-weight: 700;
}
QPushButton#RecordCtaBtn {
    background-color: #7c3aed;
    border: 1px solid #a855f7;
    color: #ffffff;
    font-weight: 700;
    padding: 9px 14px;
}
QPushButton#RecordCtaBtn:hover {
    background-color: #6d28d9;
}
QPushButton:checked#MuteBtn {
    background-color: #be123c;
    border-color: #fb7185;
    color: #ffffff;
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
QProgressBar {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 7px;
    text-align: center;
    color: #f8fafc;
    font-weight: 700;
    min-height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #06b6d4,
        stop: 1 #a855f7
    );
    border-radius: 6px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #22d3ee;
    border: 2px solid #0f172a;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QListWidget {
    background-color: #090d16;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 7px 10px;
    border-radius: 6px;
    margin: 2px 0;
    color: #e2e8f0;
}
QListWidget::item:selected {
    background-color: #1e293b;
    border: 1px solid #22d3ee;
    color: #ffffff;
}
QStatusBar {
    background-color: #090d16;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
}
"""


class MainWindow(QMainWindow):
    """Dark-mode Pose-to-Meme Studio main window."""

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        config: Optional[MemesConfig] = None,
        custom_store: Optional[CustomPoseStore] = None,
        audio_player: Optional[AudioPlayer] = None,
        pipeline: Optional[PosePipeline] = None,
        tracker: Optional[CameraTracker] = None,
        probe_cameras: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parent.parent

        if config is None:
            ensure_default_assets(self.root_dir)
            self.config = load_memes_config(self.root_dir)
        else:
            self.config = config

        self.custom_store = (
            custom_store
            if custom_store is not None
            else CustomPoseStore(self.root_dir / "data" / "custom_poses.json")
        )
        self.audio_player = audio_player if audio_player is not None else AudioPlayer()
        if pipeline is not None:
            self.pipeline = pipeline
            if self.pipeline.custom_store is None:
                self.pipeline.custom_store = self.custom_store
        else:
            self.pipeline = PosePipeline(custom_store=self.custom_store)

        self.tracker = (
            tracker
            if tracker is not None
            else CameraTracker(
                camera_index=0,
                mirror=True,
                models_dir=self.root_dir / "models",
                parent=self,
            )
        )

        self._last_active_pose: Optional[str] = None
        self._current_stage_title: str = "WAITING FOR POSE..."
        self._active_recorder: Optional[RecorderDialog] = None
        self._vcam: Any = None
        self._vcam_size: Optional[tuple[int, int]] = None

        self.setWindowTitle("Pose-to-Meme Studio")
        self.setMinimumSize(1120, 720)
        self.setStyleSheet(STUDIO_QSS)

        self._build_ui()
        self._connect_signals()
        self._populate_cameras(probe=probe_cameras)
        self._refresh_pose_list()
        self._show_neutral_stage()

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("CentralRoot")
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(12)

        # Top studio bar
        topbar = QHBoxLayout()
        topbar.setSpacing(10)

        brand = QLabel("POSE-TO-MEME STUDIO", central)
        brand.setObjectName("BrandTitle")
        topbar.addWidget(brand)
        topbar.addStretch(1)

        self._fps_pill = QLabel("0.0 FPS", central)
        self._fps_pill.setObjectName("FpsPill")
        topbar.addWidget(self._fps_pill)

        self._status_pill = QLabel("IDLE", central)
        self._status_pill.setObjectName("StatusPill")
        topbar.addWidget(self._status_pill)

        outer.addLayout(topbar)

        # Main splitter (Left: Webcam + Telemetry | Right: Meme Stage + Audio + Poses)
        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.setHandleWidth(8)
        outer.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_pane(splitter))
        splitter.addWidget(self._build_right_pane(splitter))
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 5)

        self.statusBar().showMessage("Ready — start camera or test-trigger a meme pose.")

    def _build_left_pane(self, parent: QWidget) -> QWidget:
        pane = QWidget(parent)
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(12)

        # Webcam Feed Card
        cam_card = QFrame(pane)
        cam_card.setObjectName("StudioCard")
        cam_layout = QVBoxLayout(cam_card)
        cam_layout.setContentsMargins(12, 12, 12, 12)
        cam_layout.setSpacing(10)

        self._webcam_label = QLabel("Camera stopped.\nSelect a device and click Start Camera.", cam_card)
        self._webcam_label.setObjectName("WebcamCanvas")
        self._webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._webcam_label.setMinimumSize(480, 340)
        cam_layout.addWidget(self._webcam_label, stretch=1)

        # Camera device + Start/Stop row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self._camera_combo = QComboBox(cam_card)
        ctrl_row.addWidget(self._camera_combo, stretch=1)

        self._refresh_cam_btn = QPushButton("↻", cam_card)
        self._refresh_cam_btn.setToolTip("Refresh available cameras")
        self._refresh_cam_btn.setFixedWidth(36)
        ctrl_row.addWidget(self._refresh_cam_btn)

        self._cam_toggle_btn = QPushButton("Start Camera", cam_card)
        self._cam_toggle_btn.setObjectName("CameraStartBtn")
        ctrl_row.addWidget(self._cam_toggle_btn)

        cam_layout.addLayout(ctrl_row)

        # Toggles row
        toggles_row = QHBoxLayout()
        toggles_row.setSpacing(14)

        self._hud_toggle = QCheckBox("Neon Skeleton HUD", cam_card)
        self._hud_toggle.setChecked(True)

        self._mirror_toggle = QCheckBox("Mirror", cam_card)
        self._mirror_toggle.setChecked(True)

        self._pip_toggle = QCheckBox("Picture-in-Picture (PiP)", cam_card)
        self._pip_toggle.setChecked(False)

        self._vcam_toggle = QCheckBox("Virtual Cam (OBS)", cam_card)
        self._vcam_toggle.setChecked(False)

        toggles_row.addWidget(self._hud_toggle)
        toggles_row.addWidget(self._mirror_toggle)
        toggles_row.addWidget(self._pip_toggle)
        toggles_row.addWidget(self._vcam_toggle)
        toggles_row.addStretch(1)

        cam_layout.addLayout(toggles_row)
        layout.addWidget(cam_card, stretch=1)

        # Telemetry Card (Confidence Bar + Top Scores)
        telem_card = QFrame(pane)
        telem_card.setObjectName("StudioCard")
        telem_layout = QVBoxLayout(telem_card)
        telem_layout.setContentsMargins(14, 12, 14, 12)
        telem_layout.setSpacing(8)

        telem_hdr = QHBoxLayout()
        telem_title = QLabel("POSE CONFIDENCE", telem_card)
        telem_title.setObjectName("SectionHeading")
        self._candidate_label = QLabel("Candidate: —", telem_card)
        self._candidate_label.setStyleSheet("color: #94a3b8; font-weight: 600;")
        telem_hdr.addWidget(telem_title)
        telem_hdr.addStretch(1)
        telem_hdr.addWidget(self._candidate_label)
        telem_layout.addLayout(telem_hdr)

        self._confidence_bar = QProgressBar(telem_card)
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setValue(0)
        self._confidence_bar.setFormat("%p%")
        telem_layout.addWidget(self._confidence_bar)

        scores_title = QLabel("TOP POSE SCORES", telem_card)
        scores_title.setObjectName("SectionHeading")
        telem_layout.addWidget(scores_title)

        self._scores_label = QLabel("No detections yet.", telem_card)
        self._scores_label.setStyleSheet(
            "color: #cbd5e1; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;"
        )
        self._scores_label.setWordWrap(True)
        telem_layout.addWidget(self._scores_label)

        layout.addWidget(telem_card)
        return pane

    def _build_right_pane(self, parent: QWidget) -> QWidget:
        pane = QWidget(parent)
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(12)

        # Meme Stage Widget
        self.meme_stage = MemeStageWidget(pane)
        layout.addWidget(self.meme_stage, stretch=3)

        # Audio Controls Card
        audio_card = QFrame(pane)
        audio_card.setObjectName("StudioCard")
        audio_layout = QHBoxLayout(audio_card)
        audio_layout.setContentsMargins(14, 10, 14, 10)
        audio_layout.setSpacing(10)

        audio_lbl = QLabel("AUDIO SFX", audio_card)
        audio_lbl.setObjectName("SectionHeading")
        audio_layout.addWidget(audio_lbl)

        self._mute_btn = QPushButton("Mute", audio_card)
        self._mute_btn.setObjectName("MuteBtn")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setChecked(self.audio_player.is_muted())
        self._update_mute_button_text()
        audio_layout.addWidget(self._mute_btn)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal, audio_card)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(int(round(self.audio_player.get_volume() * 100)))
        audio_layout.addWidget(self._volume_slider, stretch=1)

        self._volume_pct_label = QLabel(f"{self._volume_slider.value()}%", audio_card)
        self._volume_pct_label.setFixedWidth(40)
        self._volume_pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        audio_layout.addWidget(self._volume_pct_label)

        layout.addWidget(audio_card)

        # Pose Library & Custom Poses Card
        library_card = QFrame(pane)
        library_card.setObjectName("StudioCard")
        lib_layout = QVBoxLayout(library_card)
        lib_layout.setContentsMargins(14, 12, 14, 12)
        lib_layout.setSpacing(8)

        lib_hdr = QHBoxLayout()
        lib_title = QLabel("POSE & MEME LIBRARY", library_card)
        lib_title.setObjectName("SectionHeading")
        lib_hdr.addWidget(lib_title)
        lib_hdr.addStretch(1)

        self._record_cta_btn = QPushButton("Record New Pose", library_card)
        self._record_cta_btn.setObjectName("RecordCtaBtn")
        lib_hdr.addWidget(self._record_cta_btn)
        lib_layout.addLayout(lib_hdr)

        self._pose_list = QListWidget(library_card)
        self._pose_list.setMinimumHeight(140)
        lib_layout.addWidget(self._pose_list, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._test_trigger_btn = QPushButton("Test Trigger", library_card)
        self._set_media_btn = QPushButton("Set Media", library_card)
        self._set_audio_btn = QPushButton("Set Audio", library_card)
        self._delete_pose_btn = QPushButton("Delete", library_card)
        self._delete_pose_btn.setEnabled(False)

        btn_row.addWidget(self._test_trigger_btn)
        btn_row.addWidget(self._set_media_btn)
        btn_row.addWidget(self._set_audio_btn)
        btn_row.addWidget(self._delete_pose_btn)
        lib_layout.addLayout(btn_row)

        layout.addWidget(library_card, stretch=2)
        return pane

    def _connect_signals(self) -> None:
        self._cam_toggle_btn.clicked.connect(self._toggle_camera)
        self._refresh_cam_btn.clicked.connect(lambda: self._populate_cameras(probe=True))
        self._camera_combo.currentIndexChanged.connect(self._on_camera_selected)
        self._mirror_toggle.toggled.connect(self.tracker.set_mirror)
        self._vcam_toggle.toggled.connect(self._on_vcam_toggled)

        self._mute_btn.toggled.connect(self._on_mute_toggled)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)

        self._pose_list.currentItemChanged.connect(self._on_pose_selection_changed)
        self._pose_list.itemDoubleClicked.connect(lambda _item: self._test_trigger_selected())
        self._test_trigger_btn.clicked.connect(self._test_trigger_selected)
        self._set_media_btn.clicked.connect(self._reassign_media_selected)
        self._set_audio_btn.clicked.connect(self._reassign_audio_selected)
        self._delete_pose_btn.clicked.connect(self._delete_selected_pose)
        self._record_cta_btn.clicked.connect(self._open_recorder_dialog)

        self.tracker.frame_ready.connect(self.handle_frame)
        self.tracker.status_changed.connect(self._on_tracker_status)
        self.tracker.error_occurred.connect(self._on_tracker_error)
        self.tracker.finished.connect(self._on_tracker_finished)

    def _populate_cameras(self, probe: bool = True) -> None:
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        indices = list_available_cameras(4) if probe else []
        if not indices:
            self._camera_combo.addItem("Camera 0 (default)", 0)
        else:
            for idx in indices:
                self._camera_combo.addItem(f"Camera {idx}", idx)
        self._camera_combo.blockSignals(False)

    def _on_camera_selected(self, combo_idx: int) -> None:
        if combo_idx < 0:
            return
        cam_idx = self._camera_combo.itemData(combo_idx)
        if cam_idx is not None:
            self.tracker.set_camera(int(cam_idx))

    def _toggle_camera(self) -> None:
        if self.tracker.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        """Start the background CameraTracker thread."""
        if self.tracker.isRunning():
            return
        cam_data = self._camera_combo.currentData()
        if cam_data is not None:
            self.tracker.set_camera(int(cam_data))
        self.tracker.set_mirror(self._mirror_toggle.isChecked())
        self._cam_toggle_btn.setText("Stop Camera")
        self._cam_toggle_btn.setObjectName("CameraStopBtn")
        self._cam_toggle_btn.style().unpolish(self._cam_toggle_btn)
        self._cam_toggle_btn.style().polish(self._cam_toggle_btn)
        self.tracker.start()

    def stop_camera(self) -> None:
        """Stop the background CameraTracker thread safely."""
        if self.tracker.isRunning():
            self.tracker.stop()
        self._set_camera_stopped_ui()

    def _set_camera_stopped_ui(self) -> None:
        self._cam_toggle_btn.setText("Start Camera")
        self._cam_toggle_btn.setObjectName("CameraStartBtn")
        self._cam_toggle_btn.style().unpolish(self._cam_toggle_btn)
        self._cam_toggle_btn.style().polish(self._cam_toggle_btn)
        self._fps_pill.setText("0.0 FPS")

    @Slot(str)
    def _on_tracker_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 4000)

    @Slot(str)
    def _on_tracker_error(self, message: str) -> None:
        self._status_pill.setText("CAMERA ERROR")
        self.statusBar().showMessage(message, 6000)
        self._set_camera_stopped_ui()

    @Slot()
    def _on_tracker_finished(self) -> None:
        self._set_camera_stopped_ui()

    @Slot(bool)
    def _on_mute_toggled(self, muted: bool) -> None:
        self.audio_player.set_muted(muted)
        self._update_mute_button_text()

    def _update_mute_button_text(self) -> None:
        self._mute_btn.setText("Muted" if self._mute_btn.isChecked() else "Mute")

    @Slot(int)
    def _on_volume_changed(self, value: int) -> None:
        self._volume_pct_label.setText(f"{value}%")
        self.audio_player.set_volume(value / 100.0)

    @Slot(bool)
    def _on_vcam_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._close_vcam()
            self.statusBar().showMessage("Virtual Cam disabled.", 3000)
            return

        try:
            import pyvirtualcam  # type: ignore  # noqa: F401
        except ImportError:
            self._vcam_toggle.blockSignals(True)
            self._vcam_toggle.setChecked(False)
            self._vcam_toggle.blockSignals(False)
            self.statusBar().showMessage(
                "Virtual Cam unavailable: install 'pyvirtualcam' and OBS Virtual Camera.",
                6000,
            )
            return

        self.statusBar().showMessage("Virtual Cam armed (starts streaming on next frame).", 4000)

    def _send_to_vcam(self, frame_bgr: np.ndarray) -> None:
        if not self._vcam_toggle.isChecked():
            return
        h, w = frame_bgr.shape[:2]
        try:
            import pyvirtualcam  # type: ignore

            if self._vcam is None or self._vcam_size != (w, h):
                self._close_vcam()
                self._vcam = pyvirtualcam.Camera(width=w, height=h, fps=30)
                self._vcam_size = (w, h)
                self.statusBar().showMessage(f"Virtual Cam active: {self._vcam.device}", 4000)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._vcam.send(rgb)
        except Exception as exc:
            self._close_vcam()
            self._vcam_toggle.blockSignals(True)
            self._vcam_toggle.setChecked(False)
            self._vcam_toggle.blockSignals(False)
            self.statusBar().showMessage(f"Virtual Cam fallback (unavailable): {exc}", 6000)

    def _close_vcam(self) -> None:
        if self._vcam is not None:
            try:
                self._vcam.close()
            except Exception:
                pass
            self._vcam = None
            self._vcam_size = None

    def _pose_display_name(self, pose_id: str) -> str:
        entry = self.custom_store.get_pose(pose_id)
        if entry is not None:
            return entry.name
        cfg_entry = self.config.get(pose_id, {})
        if isinstance(cfg_entry, dict) and cfg_entry.get("title"):
            return str(cfg_entry["title"])
        return pose_id

    def _show_pose_on_stage(self, pose_id: str, play_audio: bool = True) -> None:
        media = resolve_pose_media(pose_id, self.config, self.custom_store, self.root_dir)
        title = media.get("title") or pose_id
        self._current_stage_title = title
        self.meme_stage.set_meme(
            pose_id=media.get("pose_id") or pose_id,
            title=title,
            media_path=media.get("meme_path"),
            accent_color=media.get("accent_color") or "#22d3ee",
        )
        if play_audio and media.get("audio_path"):
            self.audio_player.play(media["audio_path"])

    def _show_neutral_stage(self) -> None:
        self._show_pose_on_stage("NEUTRAL", play_audio=False)

    @Slot(np.ndarray, object, float)
    def handle_frame(
        self,
        frame_bgr: np.ndarray,
        detection: Optional[FrameDetection],
        fps: float = 30.0,
    ) -> Optional[PipelineResult]:
        """Process a frame + detection, update pipeline, stage, audio, telemetry, and video."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        if self._active_recorder is not None and self._active_recorder.isVisible():
            self._active_recorder.on_frame(frame_bgr, detection, fps)

        det = detection if detection is not None else FrameDetection()
        result = self.pipeline.process_detection(det)

        # Trigger or release meme stage
        if result.triggered_pose is not None:
            self._show_pose_on_stage(result.triggered_pose, play_audio=True)
        elif result.active_pose is None and self._last_active_pose is not None:
            self._show_neutral_stage()

        self._last_active_pose = result.active_pose

        # Update telemetry HUD
        self._fps_pill.setText(f"{fps:.1f} FPS")
        if result.active_pose is not None:
            active_name = self._pose_display_name(result.active_pose)
            self._status_pill.setText(f"ACTIVE: {active_name}")
            self._candidate_label.setText(f"Active: {active_name}")
        elif result.candidate_pose is not None and result.raw_confidence >= 0.35:
            cand_name = self._pose_display_name(result.candidate_pose)
            self._status_pill.setText(f"HOLDING: {cand_name}")
            self._candidate_label.setText(f"Candidate: {cand_name}")
        else:
            self._status_pill.setText("TRACKING")
            self._candidate_label.setText("Candidate: NEUTRAL")

        conf_pct = int(round(max(0.0, min(1.0, result.confidence)) * 100))
        self._confidence_bar.setValue(conf_pct)

        if result.scores:
            top_items = sorted(result.scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
            score_parts = [
                f"{self._pose_display_name(pid)}: {val * 100:4.0f}%"
                for pid, val in top_items
            ]
            self._scores_label.setText("  |  ".join(score_parts))

        # Render display frame (Neon HUD + optional PiP)
        display_frame = (
            draw_neon_hud(
                frame_bgr,
                det,
                active_pose=result.active_pose,
                confidence=result.confidence,
            )
            if self._hud_toggle.isChecked()
            else frame_bgr
        )

        if self._pip_toggle.isChecked():
            meme_frame = self.meme_stage.get_current_frame_bgr()
            display_frame = composite_pip(
                display_frame,
                meme_frame,
                label=self._current_stage_title,
            )

        self._send_to_vcam(display_frame)

        pix = bgr_to_qpixmap(display_frame, self._webcam_label.contentsRect().size())
        if not pix.isNull():
            self._webcam_label.setText("")
            self._webcam_label.setPixmap(pix)

        return result

    def _refresh_pose_list(self, select_pose_id: Optional[str] = None) -> None:
        self._pose_list.clear()

        for pose_id in BUILTIN_POSES:
            media = resolve_pose_media(pose_id, self.config, self.custom_store, self.root_dir)
            title = media.get("title") or pose_id
            has_audio = "SFX" if media.get("audio_path") else "No SFX"
            item = QListWidgetItem(f"[BUILT-IN]  {title}  ({has_audio})")
            item.setData(ROLE_POSE_ID, pose_id)
            item.setData(ROLE_IS_CUSTOM, False)
            self._pose_list.addItem(item)

        for entry in self.custom_store.list_poses():
            media_tag = Path(entry.meme_path).name if entry.meme_path else "No media"
            audio_tag = Path(entry.audio_path).name if entry.audio_path else "No SFX"
            item = QListWidgetItem(f"[CUSTOM]    {entry.name}  ({media_tag} | {audio_tag})")
            item.setData(ROLE_POSE_ID, entry.id)
            item.setData(ROLE_IS_CUSTOM, True)
            self._pose_list.addItem(item)

        if select_pose_id is not None:
            for row in range(self._pose_list.count()):
                it = self._pose_list.item(row)
                if it.data(ROLE_POSE_ID) == select_pose_id:
                    self._pose_list.setCurrentRow(row)
                    break
        elif self._pose_list.count() > 0:
            self._pose_list.setCurrentRow(0)

        self._on_pose_selection_changed(self._pose_list.currentItem(), None)

    def _on_pose_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        has_sel = current is not None
        is_custom = bool(current.data(ROLE_IS_CUSTOM)) if current is not None else False
        self._test_trigger_btn.setEnabled(has_sel)
        self._set_media_btn.setEnabled(has_sel)
        self._set_audio_btn.setEnabled(has_sel)
        self._delete_pose_btn.setEnabled(is_custom)

    def _selected_pose_info(self) -> tuple[Optional[str], bool]:
        item = self._pose_list.currentItem()
        if item is None:
            return None, False
        return str(item.data(ROLE_POSE_ID)), bool(item.data(ROLE_IS_CUSTOM))

    def _test_trigger_selected(self) -> None:
        pose_id, _ = self._selected_pose_info()
        if not pose_id:
            return
        self._show_pose_on_stage(pose_id, play_audio=True)
        self.statusBar().showMessage(f"Test triggered: {self._pose_display_name(pose_id)}", 3000)

    def _save_memes_config(self) -> None:
        cfg_path = self.root_dir / "data" / "memes_config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(self.config), indent=2)
        fd, tmp_name = tempfile.mkstemp(dir=str(cfg_path.parent), prefix=".memes_cfg_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, cfg_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def _reassign_media_selected(self) -> None:
        pose_id, is_custom = self._selected_pose_info()
        if not pose_id:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Meme Image or GIF",
            "",
            "Media Files (*.png *.jpg *.jpeg *.webp *.gif *.bmp);;All Files (*)",
        )
        if not file_path:
            return

        if is_custom:
            self.custom_store.update_pose(pose_id, meme_path=file_path)
        else:
            entry = dict(self.config.get(pose_id, {}))
            entry["meme_path"] = file_path
            self.config[pose_id] = entry
            self._save_memes_config()

        self._refresh_pose_list(select_pose_id=pose_id)
        self._show_pose_on_stage(pose_id, play_audio=False)
        self.statusBar().showMessage(f"Updated meme media for {self._pose_display_name(pose_id)}", 3000)

    def _reassign_audio_selected(self) -> None:
        pose_id, is_custom = self._selected_pose_info()
        if not pose_id:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio SFX",
            "",
            "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)",
        )
        if not file_path:
            return

        if is_custom:
            self.custom_store.update_pose(pose_id, audio_path=file_path)
        else:
            entry = dict(self.config.get(pose_id, {}))
            entry["audio_path"] = file_path
            self.config[pose_id] = entry
            self._save_memes_config()

        self._refresh_pose_list(select_pose_id=pose_id)
        self.statusBar().showMessage(f"Updated audio SFX for {self._pose_display_name(pose_id)}", 3000)

    def _delete_selected_pose(self) -> None:
        pose_id, is_custom = self._selected_pose_info()
        if not pose_id or not is_custom:
            return
        if self.custom_store.remove_pose(pose_id):
            self._refresh_pose_list()
            if self._last_active_pose == pose_id:
                self._last_active_pose = None
                self._show_neutral_stage()
            self.statusBar().showMessage("Custom pose deleted.", 3000)

    def _open_recorder_dialog(self) -> None:
        dialog = RecorderDialog(self)
        self._active_recorder = dialog
        dialog.pose_recorded.connect(self._on_custom_pose_recorded)
        try:
            dialog.exec()
        finally:
            self._active_recorder = None

    @Slot(object)
    def _on_custom_pose_recorded(self, entry: CustomPoseEntry) -> None:
        saved = self.custom_store.add_pose(entry)
        self._refresh_pose_list(select_pose_id=saved.id)
        self._show_pose_on_stage(saved.id, play_audio=True)
        self.statusBar().showMessage(f"Saved custom pose '{saved.name}'!", 4000)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_camera()
        self._close_vcam()
        self.meme_stage.cleanup()
        self.audio_player.close()
        super().closeEvent(event)
