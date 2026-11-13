"""Procedural SFX synthesizer and cross-backend audio player."""

from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE = 44100

DEFAULT_AUDIO_FILES: tuple[str, ...] = (
    "vine_boom.wav",
    "dramatic_hit.wav",
    "anime_chime.wav",
    "erm_actually.wav",
    "domain_pulse.wav",
    "peace_chime.wav",
)


def _default_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _envelope_adsr(
    n_samples: int,
    attack_s: float = 0.005,
    release_s: float = 0.03,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    env = np.ones(n_samples, dtype=np.float64)
    atk = min(n_samples, max(1, int(attack_s * sample_rate)))
    rel = min(n_samples, max(1, int(release_s * sample_rate)))
    env[:atk] = np.linspace(0.0, 1.0, atk, endpoint=True)
    env[-rel:] *= np.linspace(1.0, 0.0, rel, endpoint=True)
    return env


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Normalize safely and write mono 16-bit PCM WAV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(int(0.1 * sample_rate), dtype=np.float64)

    arr = arr * _envelope_adsr(arr.size, attack_s=0.003, release_s=0.015, sample_rate=sample_rate)
    peak = float(np.max(np.abs(arr)))
    if peak > 1e-6:
        arr = arr * (0.88 / max(peak, 1.0)) if peak > 0.88 else arr
    pcm = np.clip(arr * 32767.0, -32768.0, 32767.0).astype("<i2")

    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".wav", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _synth_vine_boom(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Deep sub-bass drop (~110Hz -> 32Hz) with soft saturation & decay."""
    dur = 0.95
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    freq = 32.0 + 78.0 * np.exp(-11.0 * t)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    sub = np.sin(phase) + 0.35 * np.sin(2.0 * phase)
    thump = 0.45 * np.sin(2.0 * np.pi * 155.0 * t) * np.exp(-38.0 * t)
    raw = (sub + thump) * np.exp(-3.3 * t)
    return np.tanh(raw * 2.1) * 0.88


def _synth_dramatic_hit(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cinematic orchestral-style impact + noise transient + low boom."""
    dur = 1.25
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    boom_freq = 36.0 + 54.0 * np.exp(-9.0 * t)
    boom_phase = 2.0 * np.pi * np.cumsum(boom_freq) / sr
    boom = 0.75 * np.sin(boom_phase) * np.exp(-2.8 * t)

    chord = np.zeros_like(t)
    for f, amp in ((110.0, 0.32), (164.81, 0.25), (220.0, 0.26), (329.63, 0.18), (440.0, 0.12)):
        chord += amp * (np.sin(2.0 * np.pi * f * t) + 0.35 * np.sin(4.0 * np.pi * f * t))
    chord *= np.exp(-2.6 * t)

    rng = np.random.default_rng(1337)
    noise = rng.standard_normal(t.size)
    # Simple one-pole lowpass for punchy snare/cymbal burst
    lp = np.empty_like(noise)
    alpha = 0.28
    acc = 0.0
    for idx in range(noise.size):
        acc += alpha * (noise[idx] - acc)
        lp[idx] = acc
    transient = lp * np.exp(-18.0 * t) * 0.65

    return np.tanh((boom + chord * 0.55 + transient) * 1.45) * 0.85


def _synth_anime_chime(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Bright harmonic arpeggio/bell chime."""
    dur = 0.85
    n_total = int(sr * dur)
    out = np.zeros(n_total, dtype=np.float64)
    notes = (
        (0.00, 659.25),   # E5
        (0.07, 830.61),   # G#5
        (0.14, 987.77),   # B5
        (0.21, 1318.51),  # E6
    )
    for start_s, freq in notes:
        i0 = int(start_s * sr)
        t = np.arange(n_total - i0, dtype=np.float64) / sr
        bell = (
            0.65 * np.sin(2.0 * np.pi * freq * t)
            + 0.25 * np.sin(4.0 * np.pi * freq * t) * np.exp(-8.0 * t)
            + 0.12 * np.sin(6.0 * np.pi * freq * t) * np.exp(-12.0 * t)
        )
        env = (1.0 - np.exp(-120.0 * t)) * np.exp(-5.5 * t)
        out[i0:] += bell * env * 0.42
    return np.tanh(out * 1.2) * 0.82


def _synth_erm_actually(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Quirky nerd-alert dual-tone chirp."""
    dur = 0.44
    n_total = int(sr * dur)
    out = np.zeros(n_total, dtype=np.float64)

    # First staccato chirp: 0.00s - 0.11s
    n1 = int(0.11 * sr)
    t1 = np.arange(n1, dtype=np.float64) / sr
    s1 = 0.65 * np.sin(2.0 * np.pi * 622.25 * t1) + 0.22 * np.sin(6.0 * np.pi * 622.25 * t1)
    out[:n1] += s1 * _envelope_adsr(n1, attack_s=0.005, release_s=0.025, sample_rate=sr) * 0.65

    # Second upward-inflected chirp: 0.14s - 0.41s
    i2 = int(0.14 * sr)
    n2 = int(0.27 * sr)
    t2 = np.arange(n2, dtype=np.float64) / sr
    f2 = 783.99 + 196.0 * (1.0 - np.exp(-10.0 * t2))
    ph2 = 2.0 * np.pi * np.cumsum(f2) / sr
    s2 = 0.68 * np.sin(ph2) + 0.20 * np.sin(3.0 * ph2)
    out[i2 : i2 + n2] += s2 * _envelope_adsr(n2, attack_s=0.006, release_s=0.05, sample_rate=sr) * 0.72

    return np.tanh(out * 1.25) * 0.82


def _synth_domain_pulse(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Ominous resonant bass swell + cursed shimmer."""
    dur = 1.35
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    swell = np.sin(np.pi * np.clip(t / dur, 0.0, 1.0)) ** 1.4

    bass = (
        0.55 * np.sin(2.0 * np.pi * 55.0 * t)
        + 0.45 * np.sin(2.0 * np.pi * 55.7 * t)
        + 0.30 * np.sin(2.0 * np.pi * 82.41 * t)
    )
    mid_freq = 165.0 + 140.0 * np.sin(np.pi * t / dur)
    mid_phase = 2.0 * np.pi * np.cumsum(mid_freq) / sr
    mid = 0.28 * np.sin(mid_phase)

    lfo = 0.55 + 0.45 * np.sin(2.0 * np.pi * 6.5 * t)
    shimmer = (
        0.14 * np.sin(2.0 * np.pi * 1318.5 * t)
        + 0.10 * np.sin(2.0 * np.pi * 1975.5 * t)
    ) * lfo * (t / dur)

    return np.tanh((bass + mid + shimmer) * swell * 1.55) * 0.85


def _synth_peace_chime(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Upbeat two-note sparkle."""
    dur = 0.62
    n_total = int(sr * dur)
    out = np.zeros(n_total, dtype=np.float64)
    for start_s, freq in ((0.0, 1046.50), (0.095, 1567.98)):
        i0 = int(start_s * sr)
        t = np.arange(n_total - i0, dtype=np.float64) / sr
        tone = 0.72 * np.sin(2.0 * np.pi * freq * t) + 0.20 * np.sin(4.0 * np.pi * freq * t)
        env = (1.0 - np.exp(-140.0 * t)) * np.exp(-6.8 * t)
        out[i0:] += tone * env * 0.52
    return np.tanh(out * 1.15) * 0.82


_SYNTH_MAP: dict[str, Callable[[int], np.ndarray]] = {
    "vine_boom.wav": _synth_vine_boom,
    "dramatic_hit.wav": _synth_dramatic_hit,
    "anime_chime.wav": _synth_anime_chime,
    "erm_actually.wav": _synth_erm_actually,
    "domain_pulse.wav": _synth_domain_pulse,
    "peace_chime.wav": _synth_peace_chime,
}


def ensure_default_audio(root_dir: Optional[str | Path] = None) -> list[Path]:
    """Generate missing default SFX WAV files idempotently under assets/default/."""
    base = Path(root_dir) if root_dir is not None else _default_root_dir()
    default_dir = base / "assets" / "default"
    default_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for filename in DEFAULT_AUDIO_FILES:
        out_path = default_dir / filename
        if not out_path.is_file() or out_path.stat().st_size == 0:
            samples = _SYNTH_MAP[filename](SAMPLE_RATE)
            _write_wav(out_path, samples, SAMPLE_RATE)
        paths.append(out_path)
    return paths


class AudioPlayer:
    """Audio SFX player with pygame.mixer primary and safe headless fallbacks."""

    def __init__(self, volume: float = 0.8, muted: bool = False) -> None:
        self._volume = float(min(1.0, max(0.0, volume)))
        self._muted = bool(muted)
        self._backend = "noop"
        self._pygame = None
        self._active_channel = None
        self._sound_cache: dict[str, object] = {}
        self._qt_effect = None
        self._winsound = None
        self._init_backend()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def muted(self) -> bool:
        return self._muted

    def _init_backend(self) -> None:
        # 1. Primary: pygame.mixer
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            self._pygame = pygame
            self._backend = "pygame"
            return
        except Exception:
            self._pygame = None

        # 2. Fallback: PySide6.QtMultimedia.QSoundEffect (requires QCoreApplication)
        try:
            from PySide6.QtCore import QCoreApplication
            from PySide6.QtMultimedia import QSoundEffect

            if QCoreApplication.instance() is not None:
                effect = QSoundEffect()
                effect.setVolume(0.0 if self._muted else self._volume)
                self._qt_effect = effect
                self._backend = "qt"
                return
        except Exception:
            self._qt_effect = None

        # 3. Fallback: Windows stdlib winsound (WAV only)
        if sys.platform.startswith("win"):
            try:
                import winsound

                self._winsound = winsound
                self._backend = "winsound"
                return
            except Exception:
                self._winsound = None

        self._backend = "noop"

    def set_volume(self, volume: float) -> None:
        self._volume = float(min(1.0, max(0.0, volume)))
        effective = 0.0 if self._muted else self._volume
        if self._backend == "pygame" and self._pygame is not None:
            try:
                if self._active_channel is not None:
                    self._active_channel.set_volume(effective)
                self._pygame.mixer.music.set_volume(effective)
            except Exception:
                pass
        elif self._backend == "qt" and self._qt_effect is not None:
            try:
                self._qt_effect.setVolume(effective)
            except Exception:
                pass

    def get_volume(self) -> float:
        return self._volume

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if self._muted:
            self.stop()
        self.set_volume(self._volume)

    def is_muted(self) -> bool:
        return self._muted

    def play(self, path: Optional[str | Path]) -> bool:
        if self._muted or not path:
            return False
        file_path = Path(path)
        if not file_path.is_file():
            return False

        ext = file_path.suffix.lower()
        if ext not in {".wav", ".mp3", ".ogg"}:
            return False

        if self._backend == "pygame" and self._pygame is not None:
            return self._play_pygame(file_path, ext)
        if self._backend == "qt" and self._qt_effect is not None and ext == ".wav":
            return self._play_qt(file_path)
        if self._backend == "winsound" and self._winsound is not None and ext == ".wav":
            return self._play_winsound(file_path)
        return False

    def _play_pygame(self, file_path: Path, ext: str) -> bool:
        try:
            self.stop()
            if ext in {".wav", ".ogg"}:
                key = str(file_path.resolve())
                sound = self._sound_cache.get(key)
                if sound is None:
                    sound = self._pygame.mixer.Sound(key)
                    if len(self._sound_cache) < 32:
                        self._sound_cache[key] = sound
                sound.set_volume(self._volume)
                self._active_channel = sound.play()
                return True

            self._pygame.mixer.music.load(str(file_path))
            self._pygame.mixer.music.set_volume(self._volume)
            self._pygame.mixer.music.play()
            return True
        except Exception:
            return False

    def _play_qt(self, file_path: Path) -> bool:
        try:
            from PySide6.QtCore import QUrl

            self._qt_effect.stop()
            self._qt_effect.setSource(QUrl.fromLocalFile(str(file_path.resolve())))
            self._qt_effect.setVolume(self._volume)
            self._qt_effect.play()
            return True
        except Exception:
            return False

    def _play_winsound(self, file_path: Path) -> bool:
        try:
            flags = (
                self._winsound.SND_FILENAME
                | self._winsound.SND_ASYNC
                | self._winsound.SND_NODEFAULT
            )
            self._winsound.PlaySound(str(file_path), flags)
            return True
        except Exception:
            return False

    def stop(self) -> None:
        if self._backend == "pygame" and self._pygame is not None:
            try:
                if self._active_channel is not None:
                    self._active_channel.stop()
                    self._active_channel = None
                self._pygame.mixer.music.stop()
            except Exception:
                pass
        elif self._backend == "qt" and self._qt_effect is not None:
            try:
                self._qt_effect.stop()
            except Exception:
                pass
        elif self._backend == "winsound" and self._winsound is not None:
            try:
                self._winsound.PlaySound(None, self._winsound.SND_PURGE)
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        self._sound_cache.clear()
