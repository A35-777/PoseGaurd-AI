"""Skeleton normalization, canonical joint remapping, and framerate resampling."""

import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
from ..config import CANONICAL_JOINTS, CANONICAL_PARENTS, JOINT_ALIASES, PoseGuardConfig
from .skeleton import Skeleton, JointNode, AnimationClip

class SkeletonNormalizer:
    """Normalizes animation clips to a canonical skeleton, standard framerate, and standard height."""
    
    def __init__(self, config: Optional[PoseGuardConfig] = None):
        self.config = config or PoseGuardConfig()
        
    def find_canonical_match(self, raw_names: List[str]) -> Dict[str, int]:
        """Find best index in raw_names for each canonical joint."""
        mapping: Dict[str, int] = {}
        lower_raw = [name.lower().replace(":", "_").replace(" ", "_") for name in raw_names]
        
        for can_name in CANONICAL_JOINTS:
            aliases = JOINT_ALIASES.get(can_name, [can_name.lower()])
            matched_idx = -1
            
            # Exact match first
            for alias in aliases:
                for idx, r_name in enumerate(lower_raw):
                    if r_name == alias:
                        matched_idx = idx
                        break
                if matched_idx != -1:
                    break
                    
            # Substring match if exact match failed
            if matched_idx == -1:
                for alias in aliases:
                    for idx, r_name in enumerate(lower_raw):
                        if alias in r_name:
                            matched_idx = idx
                            break
                    if matched_idx != -1:
                        break
                        
            if matched_idx != -1:
                mapping[can_name] = matched_idx
                
        return mapping

    def remap_to_canonical(self, clip: AnimationClip) -> AnimationClip:
        """Remap arbitrary skeleton joint positions to the canonical 21-joint skeleton."""
        raw_names = clip.skeleton.joint_names
        match_map = self.find_canonical_match(raw_names)
        
        num_frames = clip.num_frames
        can_positions = np.zeros((num_frames, len(CANONICAL_JOINTS), 3), dtype=np.float32)
        
        # Fill mapped joints
        for i, can_name in enumerate(CANONICAL_JOINTS):
            if can_name in match_map:
                raw_idx = match_map[can_name]
                can_positions[:, i, :] = clip.joint_positions[:, raw_idx, :]
            else:
                # Fallback / interpolation for missing joint
                parent_name = CANONICAL_PARENTS.get(can_name, "")
                if parent_name and parent_name in match_map:
                    parent_raw_idx = match_map[parent_name]
                    # Estimate offset slightly in Y or Z
                    can_positions[:, i, :] = clip.joint_positions[:, parent_raw_idx, :] + np.array([0.0, -0.08, 0.05], dtype=np.float32)
                else:
                    can_positions[:, i, :] = clip.joint_positions[:, 0, :]
                    
        can_parents = [CANONICAL_JOINTS.index(CANONICAL_PARENTS[name]) if CANONICAL_PARENTS[name] else -1 
                       for name in CANONICAL_JOINTS]
        
        can_skeleton = Skeleton(
            joint_names=list(CANONICAL_JOINTS),
            parents=can_parents,
            offsets=np.zeros((len(CANONICAL_JOINTS), 3), dtype=np.float32)
        )
        
        meta = dict(clip.metadata)
        meta["mapped_canonical_joints"] = len(match_map)
        meta["total_canonical_joints"] = len(CANONICAL_JOINTS)
        
        return AnimationClip(
            skeleton=can_skeleton,
            fps=clip.fps,
            frame_time=clip.frame_time,
            num_frames=num_frames,
            joint_positions=can_positions,
            root_motion=can_positions[:, 0, :],
            metadata=meta
        )

    def resample_framerate(self, clip: AnimationClip, target_fps: Optional[float] = None) -> AnimationClip:
        """Resample animation sequence to target framerate (default 30 fps)."""
        target_fps = target_fps or self.config.target_fps
        if abs(clip.fps - target_fps) < 0.5:
            return clip
            
        old_frames = clip.num_frames
        duration = clip.duration_seconds
        new_frames = max(2, int(round(duration * target_fps)))
        
        old_t = np.linspace(0.0, duration, old_frames)
        new_t = np.linspace(0.0, duration, new_frames)
        
        # Vectorized linear interpolation along time axis
        interpolator = interp1d(old_t, clip.joint_positions, axis=0, kind='linear', fill_value="extrapolate")
        new_positions = interpolator(new_t).astype(np.float32)
        
        new_root = None
        if clip.root_motion is not None:
            root_interp = interp1d(old_t, clip.root_motion, axis=0, kind='linear', fill_value="extrapolate")
            new_root = root_interp(new_t).astype(np.float32)
            
        new_clip = clip.copy()
        new_clip.fps = target_fps
        new_clip.frame_time = 1.0 / target_fps
        new_clip.num_frames = new_frames
        new_clip.joint_positions = new_positions
        new_clip.root_motion = new_root
        new_clip.metadata["resampled_from_fps"] = clip.fps
        return new_clip

    def normalize_height(self, clip: AnimationClip, target_height: float = 1.75) -> Tuple[AnimationClip, float]:
        """Normalize skeleton height to standard 1.75m reference height."""
        # Estimate standing height from min Y of feet to max Y of head
        l_foot_idx = clip.skeleton.get_joint_index("LeftFoot")
        r_foot_idx = clip.skeleton.get_joint_index("RightFoot")
        head_idx = clip.skeleton.get_joint_index("Head")
        
        if head_idx != -1 and (l_foot_idx != -1 or r_foot_idx != -1):
            foot_indices = [idx for idx in [l_foot_idx, r_foot_idx] if idx != -1]
            min_foot_y = np.percentile(clip.joint_positions[:, foot_indices, 1], 5)
            max_head_y = np.percentile(clip.joint_positions[:, head_idx, 1], 95)
            measured_height = max_head_y - min_foot_y
        else:
            measured_height = np.max(clip.joint_positions[:, :, 1]) - np.min(clip.joint_positions[:, :, 1])
            
        if measured_height <= 0.1:
            scale_factor = 1.0
        else:
            scale_factor = target_height / measured_height
            
        new_clip = clip.copy()
        new_clip.joint_positions = new_clip.joint_positions * scale_factor
        if new_clip.root_motion is not None:
            new_clip.root_motion = new_clip.root_motion * scale_factor
        new_clip.metadata["height_scale_factor"] = float(scale_factor)
        new_clip.metadata["original_measured_height"] = float(measured_height)
        return new_clip, scale_factor

    def estimate_ground_plane(self, clip: AnimationClip) -> float:
        """Estimate the floor plane level from lowest foot contact heights."""
        l_foot_idx = clip.skeleton.get_joint_index("LeftFoot")
        r_foot_idx = clip.skeleton.get_joint_index("RightFoot")
        l_toe_idx = clip.skeleton.get_joint_index("LeftToeBase")
        r_toe_idx = clip.skeleton.get_joint_index("RightToeBase")
        
        foot_indices = [idx for idx in [l_foot_idx, r_foot_idx, l_toe_idx, r_toe_idx] if idx != -1]
        if not foot_indices:
            # Fallback to absolute lowest joint
            return float(np.percentile(clip.joint_positions[:, :, 1], 2))
            
        foot_y = clip.joint_positions[:, foot_indices, 1]
        # Ground plane is estimated as the 5th percentile of foot Y values
        ground_y = float(np.percentile(foot_y, 5))
        return ground_y

    def normalize(self, clip: AnimationClip) -> Tuple[AnimationClip, float]:
        """Full normalization pipeline: remap -> resample fps -> normalize height -> estimate ground."""
        remapped = self.remap_to_canonical(clip)
        resampled = self.resample_framerate(remapped)
        scaled, scale_factor = self.normalize_height(resampled)
        ground_y = self.estimate_ground_plane(scaled)
        scaled.metadata["estimated_ground_y"] = ground_y
        return scaled, ground_y

