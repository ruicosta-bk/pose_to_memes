"""Pose-to-Meme Studio classifier package."""

from .custom_store import (
    CustomPoseEntry,
    CustomPoseStore,
    PipelineResult,
    PosePipeline,
    TemporalStabilizer,
)
from .features import POSE_VECTOR_DIM, cosine_similarity, extract_pose_vector
from .rules import BUILTIN_POSES, classify_builtin

__all__ = [
    "BUILTIN_POSES",
    "CustomPoseEntry",
    "CustomPoseStore",
    "POSE_VECTOR_DIM",
    "PipelineResult",
    "PosePipeline",
    "TemporalStabilizer",
    "classify_builtin",
    "cosine_similarity",
    "extract_pose_vector",
]
