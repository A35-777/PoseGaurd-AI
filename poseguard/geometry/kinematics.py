"""Kinematic analysis: velocity, acceleration, jerk, and bone distances."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..parser.skeleton import AnimationClip

def compute_velocities(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute central difference velocities [T, J, 3] in m/s."""
    T, J, _ = positions.shape
    vel = np.zeros_like(positions)
    if T < 2:
        return vel
        
    # Forward difference at start
    vel[0] = (positions[1] - positions[0]) / dt
    # Backward difference at end
    vel[-1] = (positions[-1] - positions[-2]) / dt
    # Central difference in interior
    if T > 2:
        vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    return vel

def compute_accelerations(velocities: np.ndarray, dt: float) -> np.ndarray:
    """Compute central difference accelerations [T, J, 3] in m/s^2."""
    return compute_velocities(velocities, dt)

def compute_jerks(accelerations: np.ndarray, dt: float) -> np.ndarray:
    """Compute central difference jerks [T, J, 3] in m/s^3."""
    return compute_velocities(accelerations, dt)

def compute_bone_lengths(positions: np.ndarray, parents: List[int]) -> np.ndarray:
    """Compute bone lengths [T, J] for each joint relative to its parent."""
    T, J, _ = positions.shape
    bone_lengths = np.zeros((T, J), dtype=np.float32)
    for j in range(J):
        p = parents[j]
        if p != -1:
            diff = positions[:, j, :] - positions[:, p, :]
            bone_lengths[:, j] = np.linalg.norm(diff, axis=-1)
    return bone_lengths

def compute_foot_contact_state(
    foot_pos_y: np.ndarray,
    foot_vel_y: np.ndarray,
    ground_y: float,
    height_threshold: float = 0.08,
    vel_threshold: float = 0.12
) -> np.ndarray:
    """Classify frame-by-frame foot contact state: True if foot is planted/touching ground."""
    height_above_ground = foot_pos_y - ground_y
    is_near_ground = (height_above_ground >= -0.05) & (height_above_ground <= height_threshold)
    is_low_vert_speed = np.abs(foot_vel_y) <= vel_threshold
    return is_near_ground & is_low_vert_speed

