"""Tier 2 custom pose store and temporal stabilizer / classifier pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .features import (
    cosine_similarity,
    extract_face_blendshape_vector,
    extract_pose_vector,
)
from .rules import classify_builtin


@dataclass
class CustomPoseEntry:
    """Serialized custom pose/expression template stored in ``data/custom_poses.json``."""

    id: str
    name: str
    vector: list[float]
    meme_path: str = ""
    audio_path: str = ""
    threshold: float = 0.88
    include_face: bool = False
    face_vector: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CustomPoseEntry:
        raw_fv = raw.get("face_vector")
        face_vec = [float(x) for x in raw_fv] if raw_fv is not None else None
        return cls(
            id=str(raw["id"]),
            name=str(raw["name"]),
            vector=[float(x) for x in raw.get("vector", [])],
            meme_path=str(raw.get("meme_path", "")),
            audio_path=str(raw.get("audio_path", "")),
            threshold=float(raw.get("threshold", 0.88)),
            include_face=bool(raw.get("include_face", False)),
            face_vector=face_vec,
        )


class CustomPoseStore:
    """JSON-backed persistence and Cosine Similarity matching for custom poses."""

    def __init__(self, storage_path: Union[str, Path] = "data/custom_poses.json") -> None:
        self.storage_path = Path(storage_path)
        self._entries: list[CustomPoseEntry] = []
        self.load()

    def load(self) -> None:
        if not self.storage_path.exists():
            self._entries = []
            return
        try:
            text = self.storage_path.read_text(encoding="utf-8").strip()
            if not text:
                self._entries = []
                return
            data = json.loads(text)
            items = data.get("poses", data) if isinstance(data, dict) else data
            self._entries = [CustomPoseEntry.from_dict(item) for item in items]
        except (OSError, ValueError, KeyError, TypeError):
            self._entries = []

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"poses": [entry.to_dict() for entry in self._entries]},
            indent=2,
        )
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.storage_path.parent),
            prefix=".custom_poses_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, self.storage_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def add_pose(
        self,
        name_or_entry: Union[str, CustomPoseEntry],
        vector: Optional[Union[np.ndarray, list[float]]] = None,
        meme_path: str = "",
        audio_path: str = "",
        threshold: float = 0.88,
        pose_id: Optional[str] = None,
        include_face: bool = False,
        face_vector: Optional[Union[np.ndarray, list[float]]] = None,
    ) -> CustomPoseEntry:
        """Add a custom pose (either from a ``CustomPoseEntry`` or raw fields) and persist."""
        if isinstance(name_or_entry, CustomPoseEntry):
            raw_vec = name_or_entry.vector
            raw_fv = name_or_entry.face_vector
            inc_face = bool(name_or_entry.include_face)
            vec_list = self._normalize_vector_list(raw_vec) if raw_vec else []
            fv_list = self._normalize_vector_list(raw_fv) if raw_fv is not None else None
            if not vec_list and not (inc_face and fv_list):
                raise ValueError("CustomPoseEntry must provide body vector or face_vector")
            entry = CustomPoseEntry(
                id=name_or_entry.id or f"CUSTOM_{uuid.uuid4().hex[:8].upper()}",
                name=name_or_entry.name,
                vector=vec_list,
                meme_path=name_or_entry.meme_path,
                audio_path=name_or_entry.audio_path,
                threshold=float(name_or_entry.threshold),
                include_face=inc_face,
                face_vector=fv_list,
            )
        else:
            fv_list = self._normalize_vector_list(face_vector) if face_vector is not None else None
            if vector is None:
                if include_face and fv_list:
                    vec_list = []
                else:
                    raise ValueError(
                        "vector must be provided when adding a pose by name "
                        "(unless include_face=True with face_vector)"
                    )
            else:
                vec_list = self._normalize_vector_list(vector)

            entry = CustomPoseEntry(
                id=pose_id or f"CUSTOM_{uuid.uuid4().hex[:8].upper()}",
                name=str(name_or_entry),
                vector=vec_list,
                meme_path=str(meme_path),
                audio_path=str(audio_path),
                threshold=float(threshold),
                include_face=bool(include_face),
                face_vector=fv_list,
            )

        self._entries = [e for e in self._entries if e.id != entry.id]
        self._entries.append(entry)
        self.save()
        return entry

    def remove_pose(self, pose_id: str) -> bool:
        """Remove a custom pose by ID and persist if found."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != pose_id]
        if len(self._entries) == before:
            return False
        self.save()
        return True

    def get_pose(self, pose_id: str) -> Optional[CustomPoseEntry]:
        for entry in self._entries:
            if entry.id == pose_id:
                return entry
        return None

    def update_pose(
        self,
        pose_id: str,
        *,
        name: Optional[str] = None,
        meme_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        threshold: Optional[float] = None,
        include_face: Optional[bool] = None,
        face_vector: Optional[Union[np.ndarray, list[float]]] = None,
    ) -> Optional[CustomPoseEntry]:
        entry = self.get_pose(pose_id)
        if entry is None:
            return None
        if name is not None:
            entry.name = str(name)
        if meme_path is not None:
            entry.meme_path = str(meme_path)
        if audio_path is not None:
            entry.audio_path = str(audio_path)
        if threshold is not None:
            entry.threshold = float(threshold)
        if include_face is not None:
            entry.include_face = bool(include_face)
        if face_vector is not None:
            entry.face_vector = self._normalize_vector_list(face_vector)
        self.save()
        return entry

    def list_poses(self) -> list[CustomPoseEntry]:
        return list(self._entries)

    @staticmethod
    def _score_entry(
        entry: CustomPoseEntry,
        vector: Optional[np.ndarray],
        face_vector: Optional[np.ndarray],
    ) -> float:
        has_body = len(entry.vector) > 0
        has_face = bool(entry.include_face and entry.face_vector)

        if has_face:
            if face_vector is None:
                return 0.0
            face_sim = cosine_similarity(
                face_vector, np.asarray(entry.face_vector, dtype=np.float32)
            )
            if has_body:
                body_sim = (
                    cosine_similarity(vector, np.asarray(entry.vector, dtype=np.float32))
                    if vector is not None
                    else 0.0
                )
                return float(np.clip(0.35 * body_sim + 0.65 * face_sim, 0.0, 1.0))
            return float(np.clip(face_sim, 0.0, 1.0))

        if not has_body or vector is None:
            return 0.0
        return float(
            np.clip(
                cosine_similarity(vector, np.asarray(entry.vector, dtype=np.float32)),
                0.0,
                1.0,
            )
        )

    def score_all(
        self,
        vector: Optional[np.ndarray],
        face_vector: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Return similarity scores keyed by custom pose ID."""
        if (vector is None and face_vector is None) or not self._entries:
            return {}
        return {
            entry.id: self._score_entry(entry, vector, face_vector)
            for entry in self._entries
        }

    def match(
        self,
        vector: Optional[np.ndarray],
        face_vector: Optional[np.ndarray] = None,
    ) -> tuple[Optional[CustomPoseEntry], float]:
        """Return ``(best_matching_entry, similarity)`` if above threshold, else ``(None, best_sim)``."""
        if (vector is None and face_vector is None) or not self._entries:
            return None, 0.0

        best_qualified: Optional[CustomPoseEntry] = None
        best_qualified_sim = -1.0
        best_overall_sim = -1.0

        for entry in self._entries:
            sim = self._score_entry(entry, vector, face_vector)
            if sim > best_overall_sim:
                best_overall_sim = sim
            if sim >= entry.threshold and sim > best_qualified_sim:
                best_qualified = entry
                best_qualified_sim = sim

        if best_qualified is not None:
            return best_qualified, best_qualified_sim
        return None, max(best_overall_sim, 0.0)

    @staticmethod
    def _normalize_vector_list(vector: Union[np.ndarray, list[float]]) -> list[float]:
        arr = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(arr))
        if norm > 1e-8:
            arr = arr / norm
        return [float(x) for x in arr]


@dataclass
class PipelineResult:
    """Output of a single stabilized pose classification step."""

    active_pose: Optional[str]
    triggered_pose: Optional[str]
    candidate_pose: Optional[str]
    confidence: float
    raw_confidence: float
    scores: dict[str, float]
    is_custom: bool = False
    custom_entry: Optional[CustomPoseEntry] = None


class TemporalStabilizer:
    """Combined Tier 1 + Tier 2 pose pipeline with hold debounce, hysteresis, EMA, and cooldown."""

    def __init__(
        self,
        custom_store: Optional[CustomPoseStore] = None,
        hold_ms: int = 250,
        trigger_threshold: float = 0.72,
        release_threshold: float = 0.50,
        cooldown_ms: int = 900,
        ema_alpha: float = 0.40,
    ) -> None:
        self.custom_store = custom_store
        self.hold_ms = int(hold_ms)
        self.trigger_threshold = float(trigger_threshold)
        self.release_threshold = float(release_threshold)
        self.cooldown_ms = int(cooldown_ms)
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))

        self._active_pose: Optional[str] = None
        self._active_is_custom: bool = False
        self._active_custom_entry: Optional[CustomPoseEntry] = None

        self._hold_pose: Optional[str] = None
        self._hold_start_ms: Optional[int] = None
        self._last_trigger_ms: Optional[int] = None
        self._smoothed_confidence: float = 0.0

    def reset(self) -> None:
        """Clear active state, hold timer, cooldown, and EMA buffer."""
        self._active_pose = None
        self._active_is_custom = False
        self._active_custom_entry = None
        self._hold_pose = None
        self._hold_start_ms = None
        self._last_trigger_ms = None
        self._smoothed_confidence = 0.0

    def process_detection(self, detection: Any) -> PipelineResult:
        """Convenience wrapper accepting a ``FrameDetection``-like object."""
        return self.update(
            pose=getattr(detection, "pose", None),
            hands=getattr(detection, "hands", None),
            face=getattr(detection, "face", None),
            timestamp_ms=getattr(detection, "timestamp_ms", None),
            face_blendshapes=getattr(detection, "face_blendshapes", None),
        )

    def update(
        self,
        pose: Optional[np.ndarray],
        hands: Optional[list[tuple[str, np.ndarray]]] = None,
        face: Optional[np.ndarray] = None,
        timestamp_ms: Optional[int] = None,
        face_blendshapes: Optional[dict[str, float]] = None,
    ) -> PipelineResult:
        """Classify current frame and apply temporal stabilization."""
        now_ms = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)

        builtin_pose, builtin_conf, builtin_scores = classify_builtin(
            pose, hands, face, face_blendshapes=face_blendshapes
        )
        all_scores = dict(builtin_scores)

        custom_entry: Optional[CustomPoseEntry] = None
        custom_sim = 0.0
        if self.custom_store is not None:
            vec = extract_pose_vector(pose, hands)
            face_vec = extract_face_blendshape_vector(face_blendshapes)
            custom_scores = self.custom_store.score_all(vec, face_vector=face_vec)
            all_scores.update(custom_scores)
            custom_entry, custom_sim = self.custom_store.match(vec, face_vector=face_vec)

        if custom_entry is not None:
            cand_pose: Optional[str] = custom_entry.id
            raw_conf = custom_sim
            cand_is_custom = True
            cand_entry: Optional[CustomPoseEntry] = custom_entry
        else:
            cand_pose = builtin_pose
            raw_conf = builtin_conf
            cand_is_custom = False
            cand_entry = None

        triggered_pose: Optional[str] = None

        # Release active pose if its confidence drops below hysteresis release_threshold.
        if self._active_pose is not None:
            active_raw = all_scores.get(self._active_pose, 0.0)
            if active_raw < self.release_threshold:
                self._active_pose = None
                self._active_is_custom = False
                self._active_custom_entry = None
                self._hold_pose = None
                self._hold_start_ms = None

        # Evaluate candidate pose for activation when idle.
        if self._active_pose is None:
            if cand_pose is not None and raw_conf >= self.trigger_threshold:
                if self._hold_pose != cand_pose:
                    self._hold_pose = cand_pose
                    self._hold_start_ms = now_ms

                held_ms = now_ms - (self._hold_start_ms if self._hold_start_ms is not None else now_ms)
                cooldown_ok = (
                    self._last_trigger_ms is None
                    or (now_ms - self._last_trigger_ms) >= self.cooldown_ms
                )
                if held_ms >= self.hold_ms and cooldown_ok:
                    self._active_pose = cand_pose
                    self._active_is_custom = cand_is_custom
                    self._active_custom_entry = cand_entry
                    self._last_trigger_ms = now_ms
                    triggered_pose = cand_pose
            else:
                self._hold_pose = None
                self._hold_start_ms = None

        target_conf = (
            all_scores.get(self._active_pose, raw_conf)
            if self._active_pose is not None
            else raw_conf
        )
        self._smoothed_confidence = (
            self.ema_alpha * target_conf
            + (1.0 - self.ema_alpha) * self._smoothed_confidence
        )
        smoothed = float(np.clip(self._smoothed_confidence, 0.0, 1.0))

        return PipelineResult(
            active_pose=self._active_pose,
            triggered_pose=triggered_pose,
            candidate_pose=cand_pose,
            confidence=smoothed,
            raw_confidence=float(np.clip(raw_conf, 0.0, 1.0)),
            scores=all_scores,
            is_custom=self._active_is_custom if self._active_pose is not None else cand_is_custom,
            custom_entry=(
                self._active_custom_entry if self._active_pose is not None else cand_entry
            ),
        )


PosePipeline = TemporalStabilizer
