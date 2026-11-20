"""Unit tests for BVH parsing and Forward Kinematics."""

import os
import pytest
import numpy as np

from poseguard.parser.bvh_parser import BVHParser
from poseguard.parser.skeleton import AnimationClip

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_BVH = os.path.join(BASE_DIR, "data", "raw", "subj_01_walk_clean.bvh")

def test_parse_valid_bvh():
    assert os.path.exists(SAMPLE_BVH), f"Sample BVH not found at {SAMPLE_BVH}"
    
    clip = BVHParser.parse(SAMPLE_BVH)
    assert isinstance(clip, AnimationClip)
    assert clip.num_frames > 0
    assert clip.fps > 0
    assert clip.joint_positions.ndim == 3
    assert clip.joint_positions.shape[0] == clip.num_frames
    assert clip.joint_positions.shape[2] == 3
    assert clip.skeleton.num_joints == len(clip.skeleton.joint_names)
    assert clip.skeleton.parents[0] == -1  # Root has no parent

def test_bvh_export_and_reimport(tmp_path):
    clip = BVHParser.parse(SAMPLE_BVH)
    
    temp_export = str(tmp_path / "exported_test.bvh")
    BVHParser.export(clip, temp_export)
    assert os.path.exists(temp_export)
    
    # Re-import
    reimported = BVHParser.parse(temp_export)
    assert reimported.num_frames == clip.num_frames
    assert len(reimported.skeleton.joint_names) == len(clip.skeleton.joint_names)

