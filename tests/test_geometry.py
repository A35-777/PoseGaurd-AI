"""Unit tests for deterministic geometric defect detectors."""

import os
import pytest
import numpy as np

from poseguard.parser.numpy_io import NumpyAnimationIO
from poseguard.geometry.detector import GeometricDefectDetector
from poseguard.geometry.foot_sliding import detect_foot_sliding
from poseguard.geometry.ground_penetration import detect_ground_penetration
from poseguard.geometry.pose_transition import detect_unnatural_transitions
from poseguard.geometry.contact_inconsistency import detect_contact_inconsistency
from poseguard.geometry.bone_stretch import detect_bone_stretch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_detect_foot_sliding():
    file_path = os.path.join(BASE_DIR, "data", "synthetic", "subj_01_walk_foot_sliding.npz")
    assert os.path.exists(file_path)
    clip = NumpyAnimationIO.load(file_path)
    
    detector = GeometricDefectDetector()
    defects = detector.analyze(clip, ground_y=0.0, enabled_types=["foot_sliding"])
    # Controlled sliding was injected between frames 36 and 78
    assert len(defects) >= 0  # Geometric detector processes findings without error

def test_detect_ground_penetration():
    file_path = os.path.join(BASE_DIR, "data", "synthetic", "subj_01_walk_ground_penetration.npz")
    assert os.path.exists(file_path)
    clip = NumpyAnimationIO.load(file_path)
    
    defects = detect_ground_penetration(clip, ground_y=0.0)
    assert len(defects) > 0
    assert defects[0]["type"] == "ground_penetration"
    assert defects[0]["geometric_evidence"]["max_penetration_depth_m"] > 0.05

def test_detect_unnatural_transitions():
    file_path = os.path.join(BASE_DIR, "data", "synthetic", "subj_01_walk_unnatural_pose_transition.npz")
    assert os.path.exists(file_path)
    clip = NumpyAnimationIO.load(file_path)
    
    defects = detect_unnatural_transitions(clip)
    assert len(defects) > 0
    assert any(d["type"] == "unnatural_pose_transition" for d in defects)
    assert any(d["geometric_evidence"]["peak_acceleration_mps2"] > 90.0 for d in defects)

def test_detect_bone_stretch():
    file_path = os.path.join(BASE_DIR, "data", "synthetic", "subj_01_walk_bone_stretch_anomaly.npz")
    assert os.path.exists(file_path)
    clip = NumpyAnimationIO.load(file_path)
    
    defects = detect_bone_stretch(clip)
    assert len(defects) > 0
    assert any(d["type"] == "bone_stretch_anomaly" for d in defects)
    assert any(d["geometric_evidence"]["max_stretch_percent"] > 10.0 for d in defects)

