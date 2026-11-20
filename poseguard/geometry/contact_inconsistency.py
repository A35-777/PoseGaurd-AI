"""Deterministic contact inconsistency defect detector."""

import numpy as np
from typing import List, Dict, Any, Optional
from ..config import PoseGuardConfig
from ..parser.skeleton import AnimationClip
from .kinematics import compute_velocities, compute_foot_contact_state

def detect_contact_inconsistency(
    clip: AnimationClip,
    config: Optional[PoseGuardConfig] = None,
    ground_y: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Detect contact inconsistencies: contact oscillation/chattering and irregular liftoffs."""
    cfg = config or PoseGuardConfig()
    gy = ground_y if ground_y is not None else cfg.ground_y
    dt = clip.frame_time if clip.frame_time > 0 else 1.0 / 30.0
    
    positions = clip.joint_positions
    velocities = compute_velocities(positions, dt)
    
    foot_joints = ["LeftFoot", "RightFoot"]
    findings = []
    T = clip.num_frames
    
    for side_name in foot_joints:
        j_idx = clip.skeleton.get_joint_index(side_name)
        if j_idx == -1:
            continue
            
        pos_y = positions[:, j_idx, 1]
        vel_y = velocities[:, j_idx, 1]
        
        contact_state = compute_foot_contact_state(
            pos_y, vel_y, gy,
            cfg.foot_contact_height_threshold,
            cfg.foot_contact_vel_threshold
        )
        
        # Detect rapid contact toggles / chatter: transitions within small rolling window (e.g. 8 frames)
        window_size = 10
        contact_diff = np.abs(np.diff(contact_state.astype(int)))  # 1 when toggles
        
        if len(contact_diff) < window_size:
            continue
            
        # Rolling sum of state toggles
        toggles = np.convolve(contact_diff, np.ones(window_size, dtype=int), mode='valid')
        
        t = 0
        while t < len(toggles):
            if toggles[t] >= cfg.contact_inconsistency_oscillation_limit:
                start_f = t
                while t < len(toggles) and toggles[t] >= cfg.contact_inconsistency_oscillation_limit:
                    t += 1
                end_f = min(T - 1, t + window_size)
                duration = end_f - start_f + 1
                toggle_count = int(np.sum(contact_diff[start_f:end_f]))
                
                severity = "high" if toggle_count > 5 else "medium"
                
                findings.append({
                    "type": "contact_inconsistency",
                    "joint": side_name,
                    "start_frame": int(start_f),
                    "end_frame": int(end_f),
                    "duration_frames": int(duration),
                    "duration_sec": round(duration * dt, 3),
                    "severity": severity,
                    "confidence": min(0.95, 0.55 + (toggle_count / 8.0) * 0.35),
                    "geometric_evidence": {
                        "contact_state_toggles": toggle_count,
                        "window_frames": duration,
                        "mean_foot_height_m": round(float(np.mean(pos_y[start_f:end_f + 1] - gy)), 4),
                        "ground_y": round(float(gy), 4)
                    },
                    "suggested_correction": f"Filter foot contact flags and enforce minimum contact hold duration (hysteresis) on {side_name} between frames {start_f} and {end_f}."
                })
                t = end_f + 2
            else:
                t += 1
                
    return findings

