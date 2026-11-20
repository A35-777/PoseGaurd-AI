"""Standardized NumPy (.npz / .npy) animation import and export."""

import json
import numpy as np
from typing import Dict, Any, Optional
from .skeleton import Skeleton, JointNode, AnimationClip

class NumpyAnimationIO:
    """Read and write standardized skeletal motion data in NumPy .npz / .npy format."""
    
    @staticmethod
    def load(file_path: str) -> AnimationClip:
        """Load an animation clip from an .npz or .npy file."""
        if file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            positions = data['joint_positions'].astype(np.float32)
            
            # Extract joint names
            if 'joint_names' in data:
                joint_names = [str(name) for name in data['joint_names']]
            else:
                joint_names = [f"Joint_{i}" for i in range(positions.shape[1])]
                
            # Extract parents
            if 'parents' in data:
                parents = data['parents'].tolist()
            else:
                parents = [-1] + [0] * (len(joint_names) - 1)
                
            fps = float(data['fps']) if 'fps' in data else 30.0
            frame_time = 1.0 / fps
            
            root_motion = data['root_motion'].astype(np.float32) if 'root_motion' in data else positions[:, 0, :]
            
            metadata = {}
            if 'metadata' in data:
                try:
                    meta_raw = data['metadata'].item() if hasattr(data['metadata'], 'item') else str(data['metadata'])
                    metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
                except Exception:
                    metadata = {}
                    
            offsets = np.zeros((len(joint_names), 3), dtype=np.float32)
            if 'offsets' in data:
                offsets = data['offsets'].astype(np.float32)
                
            skeleton = Skeleton(
                joint_names=joint_names,
                parents=parents,
                offsets=offsets
            )
            
            return AnimationClip(
                skeleton=skeleton,
                fps=fps,
                frame_time=frame_time,
                num_frames=positions.shape[0],
                joint_positions=positions,
                root_motion=root_motion,
                metadata=metadata
            )
        elif file_path.endswith('.npy'):
            positions = np.load(file_path).astype(np.float32)
            if positions.ndim != 3 or positions.shape[2] != 3:
                raise ValueError(f"Invalid shape for .npy animation: expected [T, J, 3], got {positions.shape}")
                
            num_joints = positions.shape[1]
            joint_names = [f"Joint_{i}" for i in range(num_joints)]
            parents = [-1] + [0] * (num_joints - 1)
            
            skeleton = Skeleton(
                joint_names=joint_names,
                parents=parents,
                offsets=np.zeros((num_joints, 3), dtype=np.float32)
            )
            
            return AnimationClip(
                skeleton=skeleton,
                fps=30.0,
                frame_time=1.0 / 30.0,
                num_frames=positions.shape[0],
                joint_positions=positions,
                root_motion=positions[:, 0, :]
            )
        else:
            raise ValueError(f"Unsupported NumPy format for file: {file_path}")

    @staticmethod
    def save(clip: AnimationClip, output_path: str) -> None:
        """Save an AnimationClip to standardized .npz format."""
        meta_json = json.dumps(clip.metadata)
        np.savez_compressed(
            output_path,
            joint_positions=clip.joint_positions,
            joint_names=np.array(clip.skeleton.joint_names),
            parents=np.array(clip.skeleton.parents),
            offsets=clip.skeleton.offsets,
            fps=clip.fps,
            root_motion=clip.root_motion if clip.root_motion is not None else clip.joint_positions[:, 0, :],
            metadata=meta_json
        )

