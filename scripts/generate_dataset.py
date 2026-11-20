"""
Dataset generation and synthetic defect injection pipeline for PoseGuard AI.
Synthesizes pristine biomechanical base animations and injects controlled, labeled defects.
Saves data partitioned strictly by subject ID to prevent data leakage.
"""

import os
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Any

# Ensure local imports work
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from poseguard.config import CANONICAL_JOINTS, CANONICAL_PARENTS
from poseguard.parser.skeleton import Skeleton, AnimationClip
from poseguard.parser.numpy_io import NumpyAnimationIO
from poseguard.parser.bvh_parser import BVHParser
from poseguard.model.feature_extractor import KinematicFeatureExtractor

def create_canonical_skeleton() -> Skeleton:
    """Build canonical 21-joint skeleton hierarchy."""
    parents = [CANONICAL_JOINTS.index(CANONICAL_PARENTS[name]) if CANONICAL_PARENTS[name] else -1 
               for name in CANONICAL_JOINTS]
    offsets = np.zeros((len(CANONICAL_JOINTS), 3), dtype=np.float32)
    
    # Standard anatomical proportions in meters (total height ~1.75m)
    offsets[CANONICAL_JOINTS.index("Hips")] = [0.0, 0.83, 0.0]
    offsets[CANONICAL_JOINTS.index("Spine")] = [0.0, 0.15, 0.0]
    offsets[CANONICAL_JOINTS.index("Spine1")] = [0.0, 0.20, 0.0]
    offsets[CANONICAL_JOINTS.index("Neck")] = [0.0, 0.18, 0.0]
    offsets[CANONICAL_JOINTS.index("Head")] = [0.0, 0.15, 0.0]
    
    offsets[CANONICAL_JOINTS.index("LeftShoulder")] = [0.18, 0.05, 0.0]
    offsets[CANONICAL_JOINTS.index("LeftArm")] = [0.15, -0.26, 0.0]
    offsets[CANONICAL_JOINTS.index("LeftForeArm")] = [0.0, -0.24, 0.0]
    offsets[CANONICAL_JOINTS.index("LeftHand")] = [0.0, -0.16, 0.0]
    
    offsets[CANONICAL_JOINTS.index("RightShoulder")] = [-0.18, 0.05, 0.0]
    offsets[CANONICAL_JOINTS.index("RightArm")] = [-0.15, -0.26, 0.0]
    offsets[CANONICAL_JOINTS.index("RightForeArm")] = [0.0, -0.24, 0.0]
    offsets[CANONICAL_JOINTS.index("RightHand")] = [0.0, -0.16, 0.0]
    
    offsets[CANONICAL_JOINTS.index("LeftUpLeg")] = [0.11, -0.05, 0.0]
    offsets[CANONICAL_JOINTS.index("LeftLeg")] = [0.0, -0.42, 0.0]
    offsets[CANONICAL_JOINTS.index("LeftFoot")] = [0.0, -0.40, 0.0]
    offsets[CANONICAL_JOINTS.index("LeftToeBase")] = [0.0, -0.05, 0.12]
    
    offsets[CANONICAL_JOINTS.index("RightUpLeg")] = [-0.11, -0.05, 0.0]
    offsets[CANONICAL_JOINTS.index("RightLeg")] = [0.0, -0.42, 0.0]
    offsets[CANONICAL_JOINTS.index("RightFoot")] = [0.0, -0.40, 0.0]
    offsets[CANONICAL_JOINTS.index("RightToeBase")] = [0.0, -0.05, 0.12]
    
    return Skeleton(
        joint_names=list(CANONICAL_JOINTS),
        parents=parents,
        offsets=offsets
    )

def synthesize_base_motion(
    motion_type: str,
    subject_id: str,
    num_frames: int = 120,
    fps: float = 30.0
) -> AnimationClip:
    """
    Generate mathematically pristine, biomechanically valid 3D skeletal motion.
    Zero defects, rigid bones, perfect foot contact without sliding or penetration.
    """
    skeleton = create_canonical_skeleton()
    J = skeleton.num_joints
    positions = np.zeros((num_frames, J, 3), dtype=np.float32)
    dt = 1.0 / fps
    
    time_arr = np.linspace(0.0, num_frames * dt, num_frames, endpoint=False)
    
    # Biomechanical parameters
    if motion_type == "walk":
        stride_freq = 1.0  # Hz (steps/sec)
        forward_speed = 1.25 # m/s
        step_len = forward_speed / (stride_freq * 2.0)
        hip_bounce_amp = 0.03
        sway_amp = 0.025
    elif motion_type == "brisk_walk":
        stride_freq = 1.35
        forward_speed = 1.65
        step_len = forward_speed / (stride_freq * 2.0)
        hip_bounce_amp = 0.04
        sway_amp = 0.02
    elif motion_type == "jog":
        stride_freq = 1.6
        forward_speed = 2.4
        step_len = forward_speed / (stride_freq * 2.0)
        hip_bounce_amp = 0.06
        sway_amp = 0.015
    elif motion_type == "sprint":
        stride_freq = 2.0
        forward_speed = 4.2
        step_len = forward_speed / (stride_freq * 2.0)
        hip_bounce_amp = 0.08
        sway_amp = 0.01
    elif motion_type == "jump":
        stride_freq = 0.5
        forward_speed = 0.4
        step_len = 0.2
        hip_bounce_amp = 0.35
        sway_amp = 0.0
    elif motion_type == "pivot_turn":
        stride_freq = 0.8
        forward_speed = 0.6
        step_len = 0.3
        hip_bounce_amp = 0.03
        sway_amp = 0.04
    elif motion_type == "lateral_shuffle":
        stride_freq = 1.2
        forward_speed = 0.8
        step_len = 0.35
        hip_bounce_amp = 0.03
        sway_amp = 0.1
    elif motion_type == "idle":
        stride_freq = 0.2
        forward_speed = 0.0
        step_len = 0.0
        hip_bounce_amp = 0.005
        sway_amp = 0.01
    elif motion_type == "crouch_walk":
        stride_freq = 0.9
        forward_speed = 0.8
        step_len = 0.35
        hip_bounce_amp = 0.02
        sway_amp = 0.025
    else: # step_climb
        stride_freq = 0.7
        forward_speed = 0.7
        step_len = 0.4
        hip_bounce_amp = 0.08
        sway_amp = 0.02

    # Synthesize joint trajectories frame by frame
    omega = 2.0 * math.pi * stride_freq
    
    for t in range(num_frames):
        cur_t = time_arr[t]
        phase = omega * cur_t
        
        # Root (Hips)
        if motion_type == "idle":
            root_x = sway_amp * math.sin(phase * 0.5)
            root_y = 0.83 + hip_bounce_amp * math.cos(phase)
            root_z = 0.0
        elif motion_type == "jump":
            jump_cycle = (cur_t % 2.0) / 2.0
            root_x = forward_speed * cur_t * 0.3
            root_z = 0.0
            if jump_cycle < 0.25: # crouch
                root_y = 0.83 - 0.2 * math.sin(jump_cycle * 4.0 * math.pi * 0.5)
            elif jump_cycle < 0.65: # airborne
                air_phase = (jump_cycle - 0.25) / 0.40
                root_y = 0.83 + hip_bounce_amp * math.sin(air_phase * math.pi)
            else: # land & recovery
                land_phase = (jump_cycle - 0.65) / 0.35
                root_y = 0.83 - 0.15 * math.sin(land_phase * math.pi)
        elif motion_type == "lateral_shuffle":
            root_x = forward_speed * cur_t
            root_y = 0.81 + hip_bounce_amp * math.cos(phase * 2.0)
            root_z = 0.0
        elif motion_type == "crouch_walk":
            root_x = sway_amp * math.sin(phase)
            root_y = 0.65 + hip_bounce_amp * math.cos(phase * 2.0)
            root_z = forward_speed * cur_t
        elif motion_type == "pivot_turn":
            angle = (cur_t / (num_frames * dt)) * (math.pi / 2.0)
            root_x = 1.0 * math.sin(angle)
            root_y = 0.83 + hip_bounce_amp * math.cos(phase * 2.0)
            root_z = 1.0 * (1.0 - math.cos(angle))
        else:
            root_x = sway_amp * math.sin(phase)
            root_y = 0.83 + hip_bounce_amp * math.cos(phase * 2.0)
            root_z = forward_speed * cur_t
            
        hip_pos = np.array([root_x, root_y, root_z], dtype=np.float32)
        positions[t, CANONICAL_JOINTS.index("Hips")] = hip_pos
        
        # Upper body (Spine, Neck, Head, Shoulders, Arms)
        spine_tilt = 0.03 * math.sin(phase)
        positions[t, CANONICAL_JOINTS.index("Spine")] = hip_pos + np.array([0.0, 0.15, spine_tilt])
        positions[t, CANONICAL_JOINTS.index("Spine1")] = hip_pos + np.array([0.0, 0.35, spine_tilt * 1.5])
        positions[t, CANONICAL_JOINTS.index("Neck")] = hip_pos + np.array([0.0, 0.53, spine_tilt * 1.8])
        positions[t, CANONICAL_JOINTS.index("Head")] = hip_pos + np.array([0.0, 0.68, spine_tilt * 2.0])
        
        # Arms swing opposite to legs with exact spherical rotation (constant bone lengths)
        arm_swing = math.sin(phase) * 0.25 if forward_speed > 0 else 0.05 * math.sin(phase * 0.5)
        theta_l = arm_swing * 0.7
        theta_r = -arm_swing * 0.7
        
        pos_l_sh = positions[t, CANONICAL_JOINTS.index("Spine1")] + np.array([0.18, 0.05, 0.0], dtype=np.float32)
        pos_r_sh = positions[t, CANONICAL_JOINTS.index("Spine1")] + np.array([-0.18, 0.05, 0.0], dtype=np.float32)
        positions[t, CANONICAL_JOINTS.index("LeftShoulder")] = pos_l_sh
        positions[t, CANONICAL_JOINTS.index("RightShoulder")] = pos_r_sh
        
        positions[t, CANONICAL_JOINTS.index("LeftArm")] = pos_l_sh + np.array([0.08, -0.26 * math.cos(theta_l), -0.26 * math.sin(theta_l)], dtype=np.float32)
        positions[t, CANONICAL_JOINTS.index("LeftForeArm")] = positions[t, CANONICAL_JOINTS.index("LeftArm")] + np.array([0.02, -0.24 * math.cos(theta_l * 1.2), -0.24 * math.sin(theta_l * 1.2)], dtype=np.float32)
        positions[t, CANONICAL_JOINTS.index("LeftHand")] = positions[t, CANONICAL_JOINTS.index("LeftForeArm")] + np.array([0.0, -0.16 * math.cos(theta_l * 1.4), -0.16 * math.sin(theta_l * 1.4)], dtype=np.float32)
        
        positions[t, CANONICAL_JOINTS.index("RightArm")] = pos_r_sh + np.array([-0.08, -0.26 * math.cos(theta_r), -0.26 * math.sin(theta_r)], dtype=np.float32)
        positions[t, CANONICAL_JOINTS.index("RightForeArm")] = positions[t, CANONICAL_JOINTS.index("RightArm")] + np.array([-0.02, -0.24 * math.cos(theta_r * 1.2), -0.24 * math.sin(theta_r * 1.2)], dtype=np.float32)
        positions[t, CANONICAL_JOINTS.index("RightHand")] = positions[t, CANONICAL_JOINTS.index("RightForeArm")] + np.array([0.0, -0.16 * math.cos(theta_r * 1.4), -0.16 * math.sin(theta_r * 1.4)], dtype=np.float32)
        
        # Legs kinematics: Left and Right Leg (with antiphase pi)
        positions[t, CANONICAL_JOINTS.index("LeftUpLeg")] = hip_pos + np.array([0.11, -0.05, 0.0])
        positions[t, CANONICAL_JOINTS.index("RightUpLeg")] = hip_pos + np.array([-0.11, -0.05, 0.0])
        
        # Left Foot & Right Foot Trajectories
        # In natural locomotion: stance phase (ground contact, 0 velocity relative to world ground)
        # swing phase (foot lifts, swings forward, touches down)
        for is_right, prefix, leg_phase in [(False, "Left", phase), (True, "Right", phase + math.pi)]:
            up_idx = CANONICAL_JOINTS.index(f"{prefix}UpLeg")
            leg_idx = CANONICAL_JOINTS.index(f"{prefix}Leg")
            foot_idx = CANONICAL_JOINTS.index(f"{prefix}Foot")
            toe_idx = CANONICAL_JOINTS.index(f"{prefix}ToeBase")
            
            x_offset = 0.11 if not is_right else -0.11
            
            if forward_speed == 0.0: # idle
                foot_y = 0.03
                foot_z = 0.0
                knee_pos = positions[t, up_idx] + np.array([0.0, -0.42, 0.05])
                positions[t, leg_idx] = knee_pos
                positions[t, foot_idx] = np.array([hip_pos[0] + x_offset, foot_y, foot_z])
                positions[t, toe_idx] = positions[t, foot_idx] + np.array([0.0, -0.01, 0.12])
                continue
                
            norm_cycle = (leg_phase / (2.0 * math.pi)) % 1.0
            
            if norm_cycle < 0.55: # Stance Phase (Planted on ground!)
                # Ground contact: foot Y = 0.025 (floor level), zero horizontal movement relative to ground!
                # Calculate fixed contact point in world coordinates
                step_idx = math.floor(leg_phase / (2.0 * math.pi))
                contact_world_z = (step_idx + 0.5) * (step_len * 2.0)
                if motion_type == "lateral_shuffle":
                    contact_world_x = (step_idx + 0.5) * (step_len * 2.0) + x_offset
                    foot_pos = np.array([contact_world_x, 0.025, 0.0])
                elif motion_type == "pivot_turn":
                    step_angle = min(math.pi / 2.0, step_idx * 0.35)
                    contact_world_x = 1.0 * math.sin(step_angle) + x_offset * math.cos(step_angle)
                    contact_world_z = 1.0 * (1.0 - math.cos(step_angle)) - x_offset * math.sin(step_angle)
                    foot_pos = np.array([contact_world_x, 0.025, contact_world_z], dtype=np.float32)
                else:
                    foot_pos = np.array([root_x + x_offset, 0.025, contact_world_z], dtype=np.float32)
            else: # Swing Phase (Foot in air swinging forward)
                swing_prog = (norm_cycle - 0.55) / 0.45
                step_idx = math.floor(leg_phase / (2.0 * math.pi))
                swing_y = 0.025 + 0.12 * math.sin(swing_prog * math.pi) # clear arc above ground
                
                if motion_type == "lateral_shuffle":
                    start_x = (step_idx + 0.5) * (step_len * 2.0) + x_offset
                    end_x = (step_idx + 1.5) * (step_len * 2.0) + x_offset
                    swing_x = start_x + (end_x - start_x) * swing_prog
                    foot_pos = np.array([swing_x, swing_y, 0.0], dtype=np.float32)
                elif motion_type == "pivot_turn":
                    start_angle = min(math.pi / 2.0, step_idx * 0.35)
                    end_angle = min(math.pi / 2.0, (step_idx + 1) * 0.35)
                    s_x = 1.0 * math.sin(start_angle) + x_offset * math.cos(start_angle)
                    s_z = 1.0 * (1.0 - math.cos(start_angle)) - x_offset * math.sin(start_angle)
                    e_x = 1.0 * math.sin(end_angle) + x_offset * math.cos(end_angle)
                    e_z = 1.0 * (1.0 - math.cos(end_angle)) - x_offset * math.sin(end_angle)
                    interp_x = s_x + (e_x - s_x) * swing_prog
                    interp_z = s_z + (e_z - s_z) * swing_prog
                    foot_pos = np.array([interp_x, swing_y, interp_z], dtype=np.float32)
                else:
                    start_z = (step_idx + 0.5) * (step_len * 2.0)
                    end_z = (step_idx + 1.5) * (step_len * 2.0)
                    swing_z = start_z + (end_z - start_z) * (3.0 * swing_prog**2 - 2.0 * swing_prog**3)
                    foot_pos = np.array([root_x + x_offset, swing_y, swing_z], dtype=np.float32)
                    
            # Analytical 2-bone IK for thigh (l1=0.42m) and shin (l2=0.40m)
            hip_j = positions[t, up_idx]
            l1, l2 = 0.42, 0.40
            d_vec = foot_pos - hip_j
            d_len = float(np.linalg.norm(d_vec))
            d_clamped = min(l1 + l2 - 0.005, max(abs(l1 - l2) + 0.02, d_len))
            d_dir = d_vec / max(1e-5, d_len)
            
            cos_alpha = np.clip((l1**2 + d_clamped**2 - l2**2) / (2.0 * l1 * d_clamped), -1.0, 1.0)
            sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha**2))
            
            fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            perp = fwd - np.dot(fwd, d_dir) * d_dir
            perp_len = np.linalg.norm(perp)
            if perp_len < 1e-4:
                perp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                perp = perp - np.dot(perp, d_dir) * d_dir
                perp_len = np.linalg.norm(perp)
            perp_dir = perp / max(1e-5, perp_len)
            
            knee_pos = hip_j + d_dir * (l1 * cos_alpha) + perp_dir * (l1 * sin_alpha)
            positions[t, leg_idx] = knee_pos
            # Exact shin constraint
            final_foot = knee_pos + (d_dir * (d_clamped - l1 * cos_alpha) - perp_dir * (l1 * sin_alpha))
            positions[t, foot_idx] = final_foot
            positions[t, toe_idx] = final_foot + np.array([0.0, -0.01, 0.12], dtype=np.float32)
            
    return AnimationClip(
        skeleton=skeleton,
        fps=fps,
        frame_time=dt,
        num_frames=num_frames,
        joint_positions=positions,
        root_motion=positions[:, 0, :],
        metadata={
            "source": "PoseGuard Procedural Biomechanical Generator",
            "motion_type": motion_type,
            "subject_id": subject_id,
            "is_clean_ground_truth": True
        }
    )

def inject_synthetic_defects(
    base_clip: AnimationClip,
    defect_type: str,
    severity: str = "medium"
) -> Tuple[AnimationClip, Dict[str, Any]]:
    """
    Inject controlled, mathematically labeled synthetic defects into a pristine animation clip.
    Strictly recorded in metadata.
    """
    mod_clip = base_clip.copy()
    pos = mod_clip.joint_positions # [T, J, 3]
    T = mod_clip.num_frames
    dt = mod_clip.frame_time
    
    label_vec = [0, 0, 0, 0, 0] # [sliding, penetration, transition, contact, stretch]
    defect_meta = {"type": defect_type, "severity": severity, "is_synthetic": True}
    
    # Target interval for defect injection (middle section of animation)
    start_f = int(T * 0.3)
    end_f = int(T * 0.65)
    defect_meta["start_frame"] = start_f
    defect_meta["end_frame"] = end_f
    
    l_foot_idx = mod_clip.skeleton.get_joint_index("LeftFoot")
    l_toe_idx = mod_clip.skeleton.get_joint_index("LeftToeBase")
    r_foot_idx = mod_clip.skeleton.get_joint_index("RightFoot")
    r_toe_idx = mod_clip.skeleton.get_joint_index("RightToeBase")
    
    if defect_type == "foot_sliding":
        label_vec[0] = 1
        # Add horizontal drift during planted stance phase
        slide_speed = 0.35 if severity == "medium" else 0.60 # m/s
        for f in range(start_f, end_f):
            # Drift along X and Z
            drift_x = slide_speed * (f - start_f) * dt * 0.5
            drift_z = slide_speed * (f - start_f) * dt * 0.8
            pos[f, l_foot_idx, 0] += drift_x
            pos[f, l_foot_idx, 2] += drift_z
            pos[f, l_toe_idx, 0] += drift_x
            pos[f, l_toe_idx, 2] += drift_z
        defect_meta["slide_speed_mps"] = slide_speed
        defect_meta["target_joint"] = "LeftFoot"
        
    elif defect_type == "ground_penetration":
        label_vec[1] = 1
        # Push feet below ground level (gy = 0.0)
        penetration_depth = 0.12 if severity == "medium" else 0.20 # meters
        for f in range(start_f, end_f):
            pos[f, l_foot_idx, 1] -= penetration_depth
            pos[f, l_toe_idx, 1] -= penetration_depth
            pos[f, r_foot_idx, 1] -= penetration_depth
            pos[f, r_toe_idx, 1] -= penetration_depth
        defect_meta["penetration_depth_m"] = penetration_depth
        defect_meta["target_joints"] = ["LeftFoot", "RightFoot"]
        
    elif defect_type == "unnatural_pose_transition":
        label_vec[2] = 1
        # Sudden 1-frame position teleportation / acceleration jerk spike
        pop_frame = int((start_f + end_f) / 2)
        displacement = 0.45 if severity == "medium" else 0.85 # meters instant pop
        head_idx = mod_clip.skeleton.get_joint_index("Head")
        hand_idx = mod_clip.skeleton.get_joint_index("RightHand")
        pos[pop_frame, head_idx, :] += np.array([displacement * 0.5, displacement, -displacement * 0.3], dtype=np.float32)
        pos[pop_frame + 1, hand_idx, :] += np.array([-displacement, displacement * 0.4, displacement], dtype=np.float32)
        defect_meta["pop_frame"] = pop_frame
        defect_meta["displacement_m"] = displacement
        defect_meta["target_joints"] = ["Head", "RightHand"]
        
    elif defect_type == "contact_inconsistency":
        label_vec[3] = 1
        # Rapidly modulate foot height right at floor boundary causing high-frequency chatter
        for f in range(start_f, end_f):
            oscillate_y = 0.06 * math.sin(f * 2.5)
            pos[f, l_foot_idx, 1] = max(-0.01, pos[f, l_foot_idx, 1] + oscillate_y)
        defect_meta["oscillation_freq"] = 2.5
        defect_meta["target_joint"] = "LeftFoot"
        
    elif defect_type == "bone_stretch_anomaly":
        label_vec[4] = 1
        # Stretch shin bone length by 25%
        stretch_factor = 1.25 if severity == "medium" else 1.40
        l_leg_idx = mod_clip.skeleton.get_joint_index("LeftLeg")
        for f in range(start_f, end_f):
            knee_p = pos[f, l_leg_idx]
            foot_p = pos[f, l_foot_idx]
            vec = foot_p - knee_p
            pos[f, l_foot_idx] = knee_p + vec * stretch_factor
        defect_meta["stretch_factor"] = stretch_factor
        defect_meta["target_bone"] = "LeftLeg->LeftFoot"
        
    elif defect_type == "compound_defect":
        # Multi-label: Foot sliding + Ground penetration
        label_vec[0] = 1
        label_vec[1] = 1
        slide_speed = 0.40
        pen_depth = 0.09
        for f in range(start_f, end_f):
            pos[f, l_foot_idx, 0] += slide_speed * (f - start_f) * dt
            pos[f, l_foot_idx, 1] -= pen_depth
        defect_meta["slide_speed_mps"] = slide_speed
        defect_meta["penetration_depth_m"] = pen_depth
        defect_meta["target_joint"] = "LeftFoot"
        
    mod_clip.joint_positions = pos
    mod_clip.metadata["is_synthetic_defect"] = True
    mod_clip.metadata["defect_labels"] = label_vec
    mod_clip.metadata["defect_metadata"] = defect_meta
    
    return mod_clip, defect_meta

def build_full_dataset() -> Dict[str, Any]:
    """Generate all raw base motions, synthetic defects, and partition into train/val/test splits."""
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    raw_dir = os.path.join(out_dir, "raw")
    syn_dir = os.path.join(out_dir, "synthetic")
    proc_dir = os.path.join(out_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(syn_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    subjects_config = [
        {"subject_id": "subj_01", "motion": "walk", "split": "train"},
        {"subject_id": "subj_02", "motion": "brisk_walk", "split": "train"},
        {"subject_id": "subj_03", "motion": "jog", "split": "train"},
        {"subject_id": "subj_04", "motion": "sprint", "split": "train"},
        {"subject_id": "subj_05", "motion": "jump", "split": "val"},
        {"subject_id": "subj_06", "motion": "pivot_turn", "split": "test"}, # Strictly unseen test sequence/subject!
        {"subject_id": "subj_07", "motion": "lateral_shuffle", "split": "train"},
        {"subject_id": "subj_08", "motion": "idle", "split": "train"},
        {"subject_id": "subj_09", "motion": "crouch_walk", "split": "train"},
        {"subject_id": "subj_10", "motion": "step_climb", "split": "val"},
    ]
    
    defect_variants = [
        "clean",
        "foot_sliding",
        "ground_penetration",
        "unnatural_pose_transition",
        "contact_inconsistency",
        "bone_stretch_anomaly",
        "compound_defect"
    ]
    
    feature_extractor_train = KinematicFeatureExtractor(window_size=32, window_hop=4) # Dense sampling for training
    feature_extractor_eval = KinematicFeatureExtractor(window_size=32, window_hop=8)  # Regular hop for val/test
    
    splits_data = {
        "train": {"windows": [], "labels": [], "seq_ids": []},
        "val": {"windows": [], "labels": [], "seq_ids": []},
        "test": {"windows": [], "labels": [], "seq_ids": []}
    }
    
    dataset_manifest = []
    
    print("Generating base motion sequences and synthetic defect variants...")
    
    for s_info in subjects_config:
        subj = s_info["subject_id"]
        motion = s_info["motion"]
        split = s_info["split"]
        
        # 1. Base clean motion
        base_clip = synthesize_base_motion(motion, subj, num_frames=120, fps=30.0)
        base_filename = f"{subj}_{motion}_clean.npz"
        base_path = os.path.join(raw_dir, base_filename)
        NumpyAnimationIO.save(base_clip, base_path)
        
        # Also export BVH for inspection & testing
        bvh_path = os.path.join(raw_dir, f"{subj}_{motion}_clean.bvh")
        BVHParser.export(base_clip, bvh_path)
        
        for def_var in defect_variants:
            seq_id = f"{subj}_{motion}_{def_var}"
            if def_var == "clean":
                clip = base_clip.copy()
                labels = [0, 0, 0, 0, 0]
                meta = {"type": "clean", "is_synthetic": False}
                clip_path = base_path
            else:
                clip, meta = inject_synthetic_defects(base_clip, def_var, severity="medium")
                labels = clip.metadata["defect_labels"]
                clip_path = os.path.join(syn_dir, f"{seq_id}.npz")
                NumpyAnimationIO.save(clip, clip_path)
                
            manifest_entry = {
                "sequence_id": seq_id,
                "subject_id": subj,
                "motion_type": motion,
                "defect_variant": def_var,
                "split": split,
                "labels": labels,
                "num_frames": clip.num_frames,
                "file_path": clip_path
            }
            dataset_manifest.append(manifest_entry)
            
            # Extract sliding temporal windows
            extractor = feature_extractor_train if split == "train" else feature_extractor_eval
            windows, frame_ranges = extractor.extract_windows(clip, ground_y=0.0)
            
            # Assign window-level multi-label ground truth based on temporal overlap
            for w_idx, (w_start, w_end) in enumerate(frame_ranges):
                win_label = [0, 0, 0, 0, 0]
                if def_var != "clean":
                    d_start = meta.get("start_frame", 0)
                    d_end = meta.get("end_frame", clip.num_frames)
                    # Check temporal overlap
                    overlap_start = max(w_start, d_start)
                    overlap_end = min(w_end, d_end)
                    if overlap_end >= overlap_start + 6: # At least 6 frames overlap
                        win_label = list(labels)
                        
                splits_data[split]["windows"].append(windows[w_idx])
                splits_data[split]["labels"].append(win_label)
                splits_data[split]["seq_ids"].append(seq_id)
                
    # Save processed numpy arrays
    summary = {}
    for split_name, s_data in splits_data.items():
        win_arr = np.array(s_data["windows"], dtype=np.float32)
        lab_arr = np.array(s_data["labels"], dtype=np.float32)
        seq_arr = np.array(s_data["seq_ids"])
        
        save_path = os.path.join(proc_dir, f"{split_name}_data.npz")
        np.savez_compressed(
            save_path,
            windows=win_arr,
            labels=lab_arr,
            sequence_ids=seq_arr
        )
        
        summary[split_name] = {
            "total_windows": int(win_arr.shape[0]),
            "feature_shape": list(win_arr.shape[1:]),
            "label_distribution": {
                "foot_sliding": int(np.sum(lab_arr[:, 0])),
                "ground_penetration": int(np.sum(lab_arr[:, 1])),
                "unnatural_pose_transition": int(np.sum(lab_arr[:, 2])),
                "contact_inconsistency": int(np.sum(lab_arr[:, 3])),
                "bone_stretch_anomaly": int(np.sum(lab_arr[:, 4])),
                "clean_windows": int(np.sum(np.all(lab_arr == 0, axis=-1)))
            }
        }
        print(f"[{split_name.upper()} SPLIT] {win_arr.shape[0]} windows, shape: {win_arr.shape}")
        
    # Save manifest
    with open(os.path.join(proc_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "manifest": dataset_manifest}, f, indent=2)
        
    print("Dataset generation complete. Summary:", json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    build_full_dataset()

