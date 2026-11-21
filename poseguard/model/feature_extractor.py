"""Feature extraction module for converting 3D skeletal motion into kinematic window tensors."""

import numpy as np
from typing import Tuple, List, Optional
from ..parser.skeleton import AnimationClip
from ..geometry.kinematics import (
    compute_velocities,
    compute_accelerations,
    compute_jerks,
    compute_bone_lengths,
    compute_foot_contact_state
)

FEATURE_CHANNELS = 48

class KinematicFeatureExtractor:
    """Extracts 48 continuous kinematic feature channels per frame and slices into temporal windows."""
    
    def __init__(self, window_size: int = 32, window_hop: int = 8):
        self.window_size = window_size
        self.window_hop = window_hop
        
    def extract_frame_features(self, clip: AnimationClip, ground_y: Optional[float] = None) -> np.ndarray:
        """
        Extract [T, 48] continuous kinematic feature matrix.
        """
        T = clip.num_frames
        dt = clip.frame_time if clip.frame_time > 0 else 1.0 / 30.0
        gy = ground_y if ground_y is not None else clip.metadata.get("estimated_ground_y", 0.0)
        
        pos = clip.joint_positions  # [T, J, 3]
        vel = compute_velocities(pos, dt)
        acc = compute_accelerations(vel, dt)
        jrk = compute_jerks(acc, dt)
        
        # Joint lookups
        def j_idx(name: str) -> int:
            idx = clip.skeleton.get_joint_index(name)
            return idx if idx != -1 else 0
            
        idx_hips = j_idx("Hips")
        idx_l_foot = j_idx("LeftFoot")
        idx_r_foot = j_idx("RightFoot")
        idx_l_toe = j_idx("LeftToeBase")
        idx_r_toe = j_idx("RightToeBase")
        idx_head = j_idx("Head")
        idx_spine = j_idx("Spine")
        idx_l_hand = j_idx("LeftHand")
        idx_r_hand = j_idx("RightHand")
        
        features = np.zeros((T, FEATURE_CHANNELS), dtype=np.float32)
        
        # 0..2: Left foot relative pos to Hips (X - X_hip, Y - gy, Z - Z_hip)
        features[:, 0] = pos[:, idx_l_foot, 0] - pos[:, idx_hips, 0]
        features[:, 1] = pos[:, idx_l_foot, 1] - gy
        features[:, 2] = pos[:, idx_l_foot, 2] - pos[:, idx_hips, 2]
        
        # 3..5: Right foot relative pos to Hips (X - X_hip, Y - gy, Z - Z_hip)
        features[:, 3] = pos[:, idx_r_foot, 0] - pos[:, idx_hips, 0]
        features[:, 4] = pos[:, idx_r_foot, 1] - gy
        features[:, 5] = pos[:, idx_r_foot, 2] - pos[:, idx_hips, 2]
        
        # 6..8: Left toe relative to Left foot
        features[:, 6] = pos[:, idx_l_toe, 0] - pos[:, idx_l_foot, 0]
        features[:, 7] = pos[:, idx_l_toe, 1] - pos[:, idx_l_foot, 1]
        features[:, 8] = pos[:, idx_l_toe, 2] - pos[:, idx_l_foot, 2]
        
        # 9..11: Right toe relative to Right foot
        features[:, 9] = pos[:, idx_r_toe, 0] - pos[:, idx_r_foot, 0]
        features[:, 10] = pos[:, idx_r_toe, 1] - pos[:, idx_r_foot, 1]
        features[:, 11] = pos[:, idx_r_toe, 2] - pos[:, idx_r_foot, 2]
        
        # 12..15: Hips height and 3D velocity
        features[:, 12] = pos[:, idx_hips, 1] - gy
        features[:, 13:16] = vel[:, idx_hips, :]
        
        # 16..19: Left foot velocity and speed
        features[:, 16:19] = vel[:, idx_l_foot, :]
        features[:, 19] = np.linalg.norm(vel[:, idx_l_foot, :], axis=-1)
        
        # 20..23: Right foot velocity and speed
        features[:, 20:23] = vel[:, idx_r_foot, :]
        features[:, 23] = np.linalg.norm(vel[:, idx_r_foot, :], axis=-1)
        
        # 24..25: Left foot acceleration & jerk magnitude
        features[:, 24] = np.linalg.norm(acc[:, idx_l_foot, :], axis=-1)
        features[:, 25] = np.linalg.norm(jrk[:, idx_l_foot, :], axis=-1)
        
        # 26..27: Right foot acceleration & jerk magnitude
        features[:, 26] = np.linalg.norm(acc[:, idx_r_foot, :], axis=-1)
        features[:, 27] = np.linalg.norm(jrk[:, idx_r_foot, :], axis=-1)
        
        # 28..31: Head, Spine, LeftHand, RightHand acceleration magnitudes
        features[:, 28] = np.linalg.norm(acc[:, idx_head, :], axis=-1)
        features[:, 29] = np.linalg.norm(acc[:, idx_spine, :], axis=-1)
        features[:, 30] = np.linalg.norm(acc[:, idx_l_hand, :], axis=-1)
        features[:, 31] = np.linalg.norm(acc[:, idx_r_hand, :], axis=-1)
        
        # 32..33: Upper body max acceleration & max velocity
        all_acc_mag = np.linalg.norm(acc, axis=-1)
        all_vel_mag = np.linalg.norm(vel, axis=-1)
        features[:, 32] = np.max(all_acc_mag, axis=-1)
        features[:, 33] = np.max(all_vel_mag, axis=-1)
        
        # 34..37: Leg and full-skeleton bone length deviations
        bone_lens = compute_bone_lengths(pos, clip.skeleton.parents)
        all_bone_diffs = np.zeros(T, dtype=np.float32)
        for j in range(1, clip.skeleton.num_joints):
            if clip.skeleton.parents[j] != -1:
                med_j = np.median(bone_lens[:, j]) if bone_lens.shape[0] > 0 else 0.3
                if med_j > 0.02:
                    diff_j = np.abs(bone_lens[:, j] - med_j) / med_j
                    all_bone_diffs = np.maximum(all_bone_diffs, diff_j)
                    
        for i, j_target in enumerate([idx_l_foot, idx_r_foot]):
            median_b = np.median(bone_lens[:, j_target]) if bone_lens.shape[0] > 0 else 0.4
            diff_b = np.abs(bone_lens[:, j_target] - median_b) / max(0.01, median_b)
            features[:, 34 + i * 2] = bone_lens[:, j_target]
            # Max deviation combining local leg and skeleton-wide anomaly
            features[:, 35 + i * 2] = np.maximum(diff_b, all_bone_diffs)
            
        # 38..39: Left & right foot contact state indicators
        l_contact = compute_foot_contact_state(pos[:, idx_l_foot, 1], vel[:, idx_l_foot, 1], gy)
        r_contact = compute_foot_contact_state(pos[:, idx_r_foot, 1], vel[:, idx_r_foot, 1], gy)
        features[:, 38] = l_contact.astype(np.float32)
        features[:, 39] = r_contact.astype(np.float32)
        
        # 40..41: Horizontal foot speed strictly during contact phase
        l_horiz = np.sqrt(vel[:, idx_l_foot, 0]**2 + vel[:, idx_l_foot, 2]**2)
        r_horiz = np.sqrt(vel[:, idx_r_foot, 0]**2 + vel[:, idx_r_foot, 2]**2)
        features[:, 40] = l_horiz * features[:, 38]
        features[:, 41] = r_horiz * features[:, 39]
        
        # 42..44: Ground penetration depths for feet (ankle & toe) and hips
        l_lowest_y = np.minimum(pos[:, idx_l_foot, 1], pos[:, idx_l_toe, 1])
        r_lowest_y = np.minimum(pos[:, idx_r_foot, 1], pos[:, idx_r_toe, 1])
        features[:, 42] = np.maximum(0.0, gy - l_lowest_y)
        features[:, 43] = np.maximum(0.0, gy - r_lowest_y)
        features[:, 44] = np.maximum(0.0, gy - pos[:, idx_hips, 1])
        
        # 45..47: Temporal differences of velocity (jerk indicators)
        features[:, 45] = np.abs(acc[:, idx_l_foot, 1])
        features[:, 46] = np.abs(acc[:, idx_r_foot, 1])
        features[:, 47] = np.linalg.norm(vel[:, idx_hips, :], axis=-1)
        
        # Robust clipping of extreme outliers
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        return features

    def extract_windows(
        self,
        clip: AnimationClip,
        ground_y: Optional[float] = None,
        hop: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract sliding window tensor [N, 48, 32] and list of (start_frame, end_frame).
        """
        hop_step = hop or self.window_hop
        W = self.window_size
        feat = self.extract_frame_features(clip, ground_y)  # [T, 48]
        T = feat.shape[0]
        
        if T < W:
            # Pad sequence if shorter than window
            pad_len = W - T
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='edge')
            T = W
            
        windows = []
        frame_ranges = []
        
        for start in range(0, T - W + 1, hop_step):
            end = start + W
            window = feat[start:end, :].T  # [48, 32]
            windows.append(window)
            frame_ranges.append((start, end - 1))
            
        if not windows:
            # Fallback for boundary
            windows.append(feat[:W, :].T)
            frame_ranges.append((0, W - 1))
            
        return np.array(windows, dtype=np.float32), frame_ranges

