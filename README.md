# Pose-to-Meme Studio

Real-time webcam pose, hand, and facial-expression detection desktop studio built with **Python 3.12**, **PySide6**, **MediaPipe Tasks**, **OpenCV**, **Pillow**, and **pygame**.

Strike classic meme poses—or record your own custom poses in 3 seconds—to trigger stylized meme cards, animated GIFs, synthesized/custom audio SFX, neon cyberpunk skeleton overlays, Picture-in-Picture (PiP) compositing, and optional OBS virtual camera streaming.

## Features

- **Threaded Vision Pipeline (`vision/`)**:
  - `CameraTracker` (`QThread`) keeps webcam capture and MediaPipe inference off the UI thread.
  - Supports **MediaPipe Tasks API** (`mediapipe >= 0.10.30`, verified on `mediapipe 1.2.3`) with automatic atomic download of `pose_landmarker_lite.task`, `hand_landmarker.task`, and `face_landmarker.task` into `models/` on first camera start, plus legacy `mp.solutions` fallback if present.
  - `draw_neon_hud` renders a neon skeleton overlay across body, hands, and facial contours.
  - `composite_pip` overlays the active meme (including live animated GIF frames and BGRA alpha images) onto the webcam feed.
- **Two-Tier Pose Classification & Stabilization (`classifier/`)**:
  - **Tier 1 (Rule-Based Geometry + Face Blendshapes)**: 7 built-in poses — `ABSOLUTE_CINEMA`, `ERM_ACTUALLY`, `THINKING`, `SHOCKED`, `DOMAIN_EXPANSION`, `PEACE_SIGN`, and `SIGMA_RIZZ` (using MediaPipe ARKit face blendshapes with FaceMesh landmark fallback).
  - **Tier 2 (Custom Pose & Face Expression Recorder)**: Scale- and translation-invariant 150-D body/hand vector (`extract_pose_vector`) and optional 51-D ARKit face blendshape vector (`extract_face_blendshape_vector`, supporting body-only, face-only, or combined body+face matching) matched via Cosine Similarity (`CustomPoseStore`, persisted in `data/custom_poses.json`).
  - **Temporal Stabilizer (`TemporalStabilizer` / `PosePipeline`)**: EMA-smoothed confidence bar, 250 ms hold debounce, trigger/release hysteresis, and cooldown anti-flicker protection.
- **Procedural Assets & Audio (`assets/`)**:
  - `ensure_default_assets` idempotently generates 800×600 PNG meme cards and 44.1 kHz mono `.wav` SFX into `assets/default/` and `data/memes_config.json`.
  - Drop user overrides into `assets/memes/` (e.g., `ABSOLUTE_CINEMA.gif`, `SIGMA_RIZZ.jpg`, `SHOCKED.png`, `PEACE_SIGN.wav`) to override built-in media without editing code.
- **Dark-Mode Studio UI (`ui/`, `main.py`)**:
  - Live webcam view, camera selector, Start/Stop, Neon HUD, Mirror, PiP, and optional `pyvirtualcam` (OBS Virtual Camera) toggles.
  - `MemeStageWidget` with static image + animated GIF playback and fade transitions.
  - `RecorderDialog` with 3-second countdown, ~0.55 s vector averaging, and an **Include Face Expression (Blendshapes)** toggle for recording custom facial expressions or body+face combos.

## Setup (Windows / Python 3.12)

From the project root (`Pose-to-Meme/`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Optional: OBS Virtual Camera Output

Install OBS Studio (with OBS Virtual Camera) and optionally install `pyvirtualcam`:

```powershell
.\.venv\Scripts\python.exe -m pip install pyvirtualcam
```

If `pyvirtualcam` or OBS Virtual Camera is unavailable, the studio remains fully functional and reports a graceful status notice when toggled.

## Running the Studio

```powershell
.\.venv\Scripts\python.exe main.py
```

> **First run note:** When starting the camera with MediaPipe Tasks (`mediapipe >= 0.10.30`), `pose_landmarker_lite.task`, `hand_landmarker.task`, and `face_landmarker.task` are downloaded automatically into `models/`.

## Running Tests

Run the full test suite in `.venv` (uses `QT_QPA_PLATFORM=offscreen` inside `tests/test_pipeline.py`, or you can set it explicitly):

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -v
```

## Project Layout

```text
Pose-to-Meme/
├── main.py                  # Application entry point (create_app, build_studio, main)
├── requirements.txt         # Runtime and test dependencies
├── assets/
│   ├── audio.py             # Procedural WAV SFX synthesizer & AudioPlayer
│   ├── generator.py         # Pillow meme card generator & media override resolver
│   ├── default/             # Auto-generated default PNG cards & WAV SFX
│   └── memes/               # User media overrides (e.g. ABSOLUTE_CINEMA.gif)
├── classifier/
│   ├── features.py          # Translation/scale-invariant 150-D body + 51-D face vectors
│   ├── rules.py             # Tier 1 geometric + blendshape rules for 7 built-in meme poses
│   └── custom_store.py      # Tier 2 CustomPoseStore & TemporalStabilizer / PosePipeline
├── data/
│   └── memes_config.json    # Pose-to-media/audio/accent mapping
├── ui/
│   ├── main_window.py       # MainWindow studio layout, controls, and pose library
│   ├── meme_stage.py        # MemeStageWidget (PNG/JPG/WEBP/GIF + fade transitions + PiP export)
│   └── recorder_dialog.py   # RecorderDialog (3s countdown custom pose capture)
├── vision/
│   ├── tracker.py           # FrameDetection, MediaPipeLandmarker, CameraTracker
│   └── overlay.py           # draw_neon_hud and composite_pip
└── tests/
    ├── test_classifier.py   # Synthetic landmark, invariance, store, and stabilizer tests
    └── test_pipeline.py     # Asset, overlay, tracker compatibility, and headless UI smoke tests
```
