"""Configuration, constants, and default thresholds for PoseGuard AI."""

from dataclasses import dataclass, field
from typing import List, Dict
from .version import MODEL_VERSION, __version__


DEFECT_TYPES = [
    "foot_sliding",
    "ground_penetration",
    "unnatural_pose_transition",
    "contact_inconsistency",
    "bone_stretch_anomaly"
]

CANONICAL_JOINTS = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase"
]

# Standard joint naming aliases for mapping diverse BVH / FBX naming schemes
JOINT_ALIASES: Dict[str, List[str]] = {
    "Hips": ["hips", "pelvis", "root", "hip", "waist", "bip01_pelvis", "b_pelvis"],
    "Spine": ["spine", "spine_01", "lowerback", "bip01_spine", "b_spine"],
    "Spine1": ["spine1", "spine_02", "chest", "torso", "bip01_spine1", "bip01_spine2", "b_spine1"],
    "Neck": ["neck", "neck_01", "bip01_neck", "b_neck"],
    "Head": ["head", "head_top", "bip01_head", "b_head"],
    "LeftShoulder": ["leftshoulder", "l_shoulder", "lshoulder", "shoulder_l", "leftcollar", "bip01_l_clavicle"],
    "LeftArm": ["leftarm", "l_arm", "larm", "upperarm_l", "leftupperarm", "bip01_l_upperarm"],
    "LeftForeArm": ["leftforearm", "l_forearm", "lforearm", "lowerarm_l", "leftlowerarm", "bip01_l_forearm"],
    "LeftHand": ["lefthand", "l_hand", "lhand", "hand_l", "bip01_l_hand"],
    "RightShoulder": ["rightshoulder", "r_shoulder", "rshoulder", "shoulder_r", "rightcollar", "bip01_r_clavicle"],
    "RightArm": ["rightarm", "r_arm", "rarm", "upperarm_r", "rightupperarm", "bip01_r_upperarm"],
    "RightForeArm": ["rightforearm", "r_forearm", "rforearm", "lowerarm_r", "rightlowerarm", "bip01_r_forearm"],
    "RightHand": ["righthand", "r_hand", "rhand", "hand_r", "bip01_r_hand"],
    "LeftUpLeg": ["leftupleg", "l_upleg", "lupleg", "thigh_l", "leftthigh", "bip01_l_thigh"],
    "LeftLeg": ["leftleg", "l_leg", "lleg", "calf_l", "leftshin", "bip01_l_calf"],
    "LeftFoot": ["leftfoot", "l_foot", "lfoot", "foot_l", "leftankle", "bip01_l_foot"],
    "LeftToeBase": ["lefttoebase", "l_toebase", "ltoe", "toe_l", "lefttoe", "bip01_l_toe0"],
    "RightUpLeg": ["rightupleg", "r_upleg", "rupleg", "thigh_r", "rightthigh", "bip01_r_thigh"],
    "RightLeg": ["rightleg", "r_leg", "rleg", "calf_r", "rightshin", "bip01_r_calf"],
    "RightFoot": ["rightfoot", "r_foot", "rfoot", "foot_r", "rightankle", "bip01_r_foot"],
    "RightToeBase": ["righttoebase", "r_toebase", "rtoe", "toe_r", "righttoe", "bip01_r_toe0"]
}

# Standard kinematic parents for canonical skeleton
CANONICAL_PARENTS: Dict[str, str] = {
    "Hips": "",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Neck": "Spine1",
    "Head": "Neck",
    "LeftShoulder": "Spine1",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine1",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot"
}

@dataclass
class PoseGuardConfig:
    """Configurable thresholds and hyperparameters for PoseGuard AI."""
    # Temporal & Dimensional settings
    target_fps: float = 30.0
    window_size: int = 32
    window_hop: int = 8
    
    # Ground reference
    ground_y: float = 0.0
    ground_tolerance: float = 0.03  # 3 cm margin
    
    # Foot sliding thresholds
    foot_contact_height_threshold: float = 0.08  # meters above ground for potential contact
    foot_contact_vel_threshold: float = 0.12     # vertical velocity m/s
    foot_slide_speed_threshold: float = 0.15     # horizontal velocity m/s during contact
    foot_slide_min_duration_frames: int = 4      # consecutive frames to confirm sliding
    
    # Ground penetration thresholds
    ground_penetration_threshold: float = -0.02  # -2 cm below ground plane
    penetration_min_duration_frames: int = 2
    
    # Unnatural pose transitions & acceleration spikes
    joint_accel_threshold: float = 90.0          # m/s^2 acceleration spike limit
    joint_jerk_threshold: float = 400.0          # m/s^3 jerk limit
    angular_velocity_threshold: float = 25.0     # rad/s sudden angular flip
    
    # Contact inconsistency
    contact_inconsistency_oscillation_limit: int = 3  # flip-flops within 10 frames
    
    # Bone stretch / rigidity
    bone_stretch_relative_tolerance: float = 0.06  # 6% bone length variation allowed
    
    # Decision Engine Weights
    neural_weight: float = 0.45
    geometric_weight: float = 0.55
    min_detection_confidence: float = 0.40
    high_confidence_threshold: float = 0.75

