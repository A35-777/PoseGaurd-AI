"""Skeleton and AnimationClip data structures for 3D skeletal motion."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

@dataclass
class JointNode:
    """Represents a joint in the skeletal hierarchy."""
    name: str
    parent_index: int  # -1 for root
    offset: np.ndarray  # [3] shape local offset from parent
    channels: List[str] = field(default_factory=list)
    channel_indices: List[int] = field(default_factory=list)
    is_end_site: bool = False

@dataclass
class Skeleton:
    """Represents the skeletal hierarchy and joint definitions."""
    joint_names: List[str]
    parents: List[int]  # parent index for each joint, -1 for root
    offsets: np.ndarray  # [J, 3] float32
    nodes: Dict[str, JointNode] = field(default_factory=dict)
    
    @property
    def num_joints(self) -> int:
        return len(self.joint_names)
        
    def get_joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError:
            return -1

@dataclass
class AnimationClip:
    """Represents a loaded 3D skeletal animation sequence."""
    skeleton: Skeleton
    fps: float
    frame_time: float
    num_frames: int
    joint_positions: np.ndarray  # [T, J, 3] world coordinates in meters
    joint_rotations: Optional[np.ndarray] = None  # [T, J, 3] Euler or [T, J, 3, 3] matrices
    root_motion: Optional[np.ndarray] = None     # [T, 3] root displacement
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0
        
    def copy(self) -> "AnimationClip":
        return AnimationClip(
            skeleton=self.skeleton,
            fps=self.fps,
            frame_time=self.frame_time,
            num_frames=self.num_frames,
            joint_positions=np.copy(self.joint_positions),
            joint_rotations=np.copy(self.joint_rotations) if self.joint_rotations is not None else None,
            root_motion=np.copy(self.root_motion) if self.root_motion is not None else None,
            metadata=dict(self.metadata)
        )

