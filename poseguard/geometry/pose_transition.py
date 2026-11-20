"""Deterministic unnatural pose transition and acceleration spike detector."""

import numpy as np
from typing import List, Dict, Any, Optional
from ..config import PoseGuardConfig
from ..parser.skeleton import AnimationClip
from .kinematics import compute_velocities, compute_accelerations, compute_jerks

def detect_unnatural_transitions(
    clip: AnimationClip,
    config: Optional[PoseGuardConfig] = None
) -> List[Dict[str, Any]]:
    """Detect unnatural pose discontinuities, acceleration spikes, and popping joints."""
    cfg = config or PoseGuardConfig()
    dt = clip.frame_time if clip.frame_time > 0 else 1.0 / 30.0
    
    positions = clip.joint_positions  # [T, J, 3]
    velocities = compute_velocities(positions, dt)
    accelerations = compute_accelerations(velocities, dt)
    jerks = compute_jerks(accelerations, dt)
    
    accel_magnitudes = np.linalg.norm(accelerations, axis=-1)  # [T, J]
    jerk_magnitudes = np.linalg.norm(jerks, axis=-1)           # [T, J]
    
    T, J, _ = positions.shape
    findings = []
    
    for j_idx in range(J):
        j_name = clip.skeleton.joint_names[j_idx]
        j_accels = accel_magnitudes[:, j_idx]
        j_jerks = jerk_magnitudes[:, j_idx]
        
        # Check acceleration spike above threshold
        spike_mask = j_accels > cfg.joint_accel_threshold
        
        t = 0
        while t < T:
            if spike_mask[t]:
                start_f = max(0, t - 1)
                while t < T and spike_mask[t]:
                    t += 1
                end_f = min(T - 1, t)
                duration = end_f - start_f + 1
                
                window_accels = j_accels[start_f:end_f + 1]
                window_jerks = j_jerks[start_f:end_f + 1]
                max_accel = float(np.max(window_accels))
                max_jerk = float(np.max(window_jerks))
                
                # Check if it's isolated pop vs smooth continuous athletic movement
                # In true pops/teleports, jerk is massive and neighboring frames have huge gradient
                if max_accel > 250.0 or max_jerk > 1200.0:
                    severity = "critical"
                elif max_accel > 160.0 or max_jerk > 700.0:
                    severity = "high"
                elif max_accel > 100.0:
                    severity = "medium"
                else:
                    severity = "low"
                    
                confidence = min(0.98, max(0.60, 0.5 + (max_accel / 200.0) * 0.45))
                
                findings.append({
                    "type": "unnatural_pose_transition",
                    "joint": j_name,
                    "start_frame": int(start_f),
                    "end_frame": int(end_f),
                    "duration_frames": int(duration),
                    "duration_sec": round(duration * dt, 3),
                    "severity": severity,
                    "confidence": round(confidence, 3),
                    "geometric_evidence": {
                        "peak_acceleration_mps2": round(max_accel, 2),
                        "peak_jerk_mps3": round(max_jerk, 2),
                        "threshold_mps2": round(cfg.joint_accel_threshold, 2)
                    },
                    "suggested_correction": f"Apply trajectory smoothing / spherical linear interpolation (Slerp) on {j_name} between frames {start_f} and {end_f}."
                })
                # Skip forward slightly to prevent overlapping clusters
                t = end_f + 2
            else:
                t += 1
                
    return findings

