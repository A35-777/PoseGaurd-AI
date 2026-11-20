"""Deterministic foot sliding defect detector."""

import numpy as np
from typing import List, Dict, Any, Optional
from ..config import PoseGuardConfig
from ..parser.skeleton import AnimationClip
from .kinematics import compute_velocities, compute_foot_contact_state

def detect_foot_sliding(
    clip: AnimationClip,
    config: Optional[PoseGuardConfig] = None,
    ground_y: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Detect foot sliding defects where planted foot moves horizontally."""
    cfg = config or PoseGuardConfig()
    gy = ground_y if ground_y is not None else cfg.ground_y
    dt = clip.frame_time if clip.frame_time > 0 else 1.0 / 30.0
    
    positions = clip.joint_positions  # [T, J, 3]
    velocities = compute_velocities(positions, dt)
    
    foot_joint_pairs = [
        ("LeftFoot", ["LeftFoot", "LeftToeBase"]),
        ("RightFoot", ["RightFoot", "RightToeBase"])
    ]
    
    findings = []
    
    for side_name, joint_names in foot_joint_pairs:
        # Find indices
        indices = [clip.skeleton.get_joint_index(name) for name in joint_names]
        indices = [idx for idx in indices if idx != -1]
        if not indices:
            continue
            
        # Use primary ankle/foot joint
        primary_idx = indices[0]
        pos_y = positions[:, primary_idx, 1]
        vel_y = velocities[:, primary_idx, 1]
        
        # Horizontal velocity
        vel_x = velocities[:, primary_idx, 0]
        vel_z = velocities[:, primary_idx, 2]
        horiz_speed = np.sqrt(vel_x**2 + vel_z**2)
        
        contact_state = compute_foot_contact_state(
            pos_y, vel_y, gy,
            cfg.foot_contact_height_threshold,
            cfg.foot_contact_vel_threshold
        )
        
        sliding_mask = contact_state & (horiz_speed > cfg.foot_slide_speed_threshold)
        
        # Cluster consecutive frames
        T = clip.num_frames
        t = 0
        while t < T:
            if sliding_mask[t]:
                start_f = t
                while t < T and sliding_mask[t]:
                    t += 1
                end_f = t - 1
                duration = end_f - start_f + 1
                
                if duration >= cfg.foot_slide_min_duration_frames:
                    window_speeds = horiz_speed[start_f:end_f + 1]
                    max_speed = float(np.max(window_speeds))
                    mean_speed = float(np.mean(window_speeds))
                    total_displacement = float(np.sum(window_speeds * dt))
                    
                    if max_speed > 0.60:
                        severity = "critical"
                    elif max_speed > 0.40:
                        severity = "high"
                    elif max_speed > 0.22:
                        severity = "medium"
                    else:
                        severity = "low"
                        
                    findings.append({
                        "type": "foot_sliding",
                        "joint": side_name,
                        "start_frame": int(start_f),
                        "end_frame": int(end_f),
                        "duration_frames": int(duration),
                        "duration_sec": round(duration * dt, 3),
                        "severity": severity,
                        "confidence": min(0.99, max(0.65, 0.5 + (mean_speed / 0.5) * 0.45)),
                        "geometric_evidence": {
                            "max_slide_speed_mps": round(max_speed, 4),
                            "mean_slide_speed_mps": round(mean_speed, 4),
                            "total_slide_distance_m": round(total_displacement, 4),
                            "foot_height_above_ground_m": round(float(np.mean(pos_y[start_f:end_f + 1] - gy)), 4),
                            "ground_y": round(float(gy), 4)
                        },
                        "suggested_correction": f"Lock {side_name} world transform or apply Inverse Kinematics (IK) pinning between frames {start_f} and {end_f}."
                    })
            else:
                t += 1
                
    return findings

