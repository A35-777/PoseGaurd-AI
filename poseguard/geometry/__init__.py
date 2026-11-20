"""Deterministic geometric defect detection module."""

from .kinematics import (
    compute_velocities,
    compute_accelerations,
    compute_jerks,
    compute_bone_lengths,
    compute_foot_contact_state
)
from .foot_sliding import detect_foot_sliding
from .ground_penetration import detect_ground_penetration
from .pose_transition import detect_unnatural_transitions
from .contact_inconsistency import detect_contact_inconsistency
from .bone_stretch import detect_bone_stretch
from .detector import GeometricDefectDetector

__all__ = [
    "compute_velocities",
    "compute_accelerations",
    "compute_jerks",
    "compute_bone_lengths",
    "compute_foot_contact_state",
    "detect_foot_sliding",
    "detect_ground_penetration",
    "detect_unnatural_transitions",
    "detect_contact_inconsistency",
    "detect_bone_stretch",
    "GeometricDefectDetector"
]

