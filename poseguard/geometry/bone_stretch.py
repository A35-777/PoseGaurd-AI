"""Deterministic bone stretch and kinematic rigidity violation detector."""

import numpy as np
from typing import List, Dict, Any, Optional
from ..config import PoseGuardConfig
from ..parser.skeleton import AnimationClip
from .kinematics import compute_bone_lengths

def detect_bone_stretch(
    clip: AnimationClip,
    config: Optional[PoseGuardConfig] = None
) -> List[Dict[str, Any]]:
    """Detect non-rigid bone length variations exceeding anatomical tolerances."""
    cfg = config or PoseGuardConfig()
    dt = clip.frame_time if clip.frame_time > 0 else 1.0 / 30.0
    
    positions = clip.joint_positions
    parents = clip.skeleton.parents
    T, J, _ = positions.shape
    
    bone_lengths = compute_bone_lengths(positions, parents)  # [T, J]
    findings = []
    
    for j in range(1, J):  # skip root
        p = parents[j]
        if p == -1:
            continue
            
        j_name = clip.skeleton.joint_names[j]
        p_name = clip.skeleton.joint_names[p]
        bone_name = f"{p_name}->{j_name}"
        
        lens = bone_lengths[:, j]
        median_len = float(np.median(lens))
        if median_len < 0.02:  # very short or zero offset
            continue
            
        rel_diff = np.abs(lens - median_len) / median_len
        stretch_mask = rel_diff > cfg.bone_stretch_relative_tolerance
        
        t = 0
        while t < T:
            if stretch_mask[t]:
                start_f = t
                while t < T and stretch_mask[t]:
                    t += 1
                end_f = t - 1
                duration = end_f - start_f + 1
                
                if duration >= 2:
                    window_diffs = rel_diff[start_f:end_f + 1]
                    max_diff_pct = float(np.max(window_diffs)) * 100.0
                    mean_diff_pct = float(np.mean(window_diffs)) * 100.0
                    
                    if max_diff_pct > 25.0:
                        severity = "critical"
                    elif max_diff_pct > 15.0:
                        severity = "high"
                    elif max_diff_pct > 8.0:
                        severity = "medium"
                    else:
                        severity = "low"
                        
                    findings.append({
                        "type": "bone_stretch_anomaly",
                        "joint": bone_name,
                        "start_frame": int(start_f),
                        "end_frame": int(end_f),
                        "duration_frames": int(duration),
                        "duration_sec": round(duration * dt, 3),
                        "severity": severity,
                        "confidence": min(0.99, 0.65 + (max_diff_pct / 30.0) * 0.32),
                        "geometric_evidence": {
                            "max_stretch_percent": round(max_diff_pct, 2),
                            "mean_stretch_percent": round(mean_diff_pct, 2),
                            "nominal_bone_length_m": round(median_len, 4),
                            "max_measured_length_m": round(float(np.max(lens[start_f:end_f + 1])), 4)
                        },
                        "suggested_correction": f"Enforce rigid length constraint on bone segment '{bone_name}' to maintain constant {round(median_len * 100, 1)} cm length."
                    })
                t = end_f + 1
            else:
                t += 1
                
    return findings

