"""Deterministic ground penetration defect detector."""

import numpy as np
from typing import List, Dict, Any, Optional
from ..config import PoseGuardConfig
from ..parser.skeleton import AnimationClip

def detect_ground_penetration(
    clip: AnimationClip,
    config: Optional[PoseGuardConfig] = None,
    ground_y: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Detect ground penetration where skeletal joints sink below floor level."""
    cfg = config or PoseGuardConfig()
    gy = ground_y if ground_y is not None else cfg.ground_y
    dt = clip.frame_time if clip.frame_time > 0 else 1.0 / 30.0
    
    positions = clip.joint_positions  # [T, J, 3]
    pen_threshold = cfg.ground_penetration_threshold  # e.g. -0.02 m
    
    # Check all lower body joints
    target_joints = ["LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "LeftLeg", "RightLeg", "Hips"]
    findings = []
    
    for j_name in target_joints:
        j_idx = clip.skeleton.get_joint_index(j_name)
        if j_idx == -1:
            continue
            
        pos_y = positions[:, j_idx, 1]
        relative_y = pos_y - gy
        penetrating = relative_y < pen_threshold
        
        T = clip.num_frames
        t = 0
        while t < T:
            if penetrating[t]:
                start_f = t
                while t < T and penetrating[t]:
                    t += 1
                end_f = t - 1
                duration = end_f - start_f + 1
                
                if duration >= cfg.penetration_min_duration_frames:
                    window_depths = np.abs(relative_y[start_f:end_f + 1])
                    max_depth = float(np.max(window_depths))
                    mean_depth = float(np.mean(window_depths))
                    
                    if max_depth > 0.15:
                        severity = "critical"
                    elif max_depth > 0.08:
                        severity = "high"
                    elif max_depth > 0.04:
                        severity = "medium"
                    else:
                        severity = "low"
                        
                    findings.append({
                        "type": "ground_penetration",
                        "joint": j_name,
                        "start_frame": int(start_f),
                        "end_frame": int(end_f),
                        "duration_frames": int(duration),
                        "duration_sec": round(duration * dt, 3),
                        "severity": severity,
                        "confidence": min(0.99, max(0.70, 0.6 + (max_depth / 0.15) * 0.38)),
                        "geometric_evidence": {
                            "max_penetration_depth_m": round(max_depth, 4),
                            "mean_penetration_depth_m": round(mean_depth, 4),
                            "lowest_y_world_m": round(float(np.min(pos_y[start_f:end_f + 1])), 4),
                            "ground_y": round(float(gy), 4)
                        },
                        "suggested_correction": f"Adjust ground contact constraint or offset root trajectory Y by +{round(max_depth * 100, 1)} cm between frames {start_f} and {end_f}."
                    })
            else:
                t += 1
                
    return findings

