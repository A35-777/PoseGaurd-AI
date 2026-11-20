"""Tests for error handling, malformed inputs, and edge cases."""

import os
import pytest
import numpy as np

from poseguard import PoseGuard
from poseguard.parser.bvh_parser import BVHParser
from poseguard.parser.numpy_io import NumpyAnimationIO
from poseguard.parser.skeleton import Skeleton, AnimationClip

def test_nonexistent_file_handling():
    detector = PoseGuard()
    with pytest.raises(FileNotFoundError):
        detector.analyze("nonexistent_path_file.bvh")

def test_unsupported_file_extension():
    detector = PoseGuard()
    # Create dummy text file
    dummy_path = "dummy_file.txt"
    try:
        with open(dummy_path, "w") as f:
            f.write("dummy content")
        with pytest.raises(ValueError):
            detector.analyze(dummy_path)
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

def test_malformed_bvh_header(tmp_path):
    bad_bvh = tmp_path / "bad_header.bvh"
    bad_bvh.write_text("INVALID_HEADER\nROOT Hips\n{\nOFFSET 0 0 0\n}\n")
    with pytest.raises(ValueError, match="Invalid BVH format"):
        BVHParser.parse(str(bad_bvh))

def test_bvh_missing_motion(tmp_path):
    bad_bvh = tmp_path / "no_motion.bvh"
    bad_bvh.write_text("HIERARCHY\nROOT Hips\n{\nOFFSET 0 0 0\nCHANNELS 3 Xposition Yposition Zposition\n}\n")
    with pytest.raises(ValueError, match="No MOTION section found"):
        BVHParser.parse(str(bad_bvh))

def test_single_frame_animation():
    detector = PoseGuard()
    # Create single frame animation clip
    skeleton = Skeleton(
        joint_names=["Hips", "LeftFoot", "RightFoot"],
        parents=[-1, 0, 0],
        offsets=np.zeros((3, 3), dtype=np.float32)
    )
    positions = np.zeros((1, 3, 3), dtype=np.float32)
    clip = AnimationClip(
        skeleton=skeleton,
        fps=30.0,
        frame_time=1.0/30.0,
        num_frames=1,
        joint_positions=positions
    )
    # Analysis should run gracefully without crashing on single-frame edge cases
    result = detector.analyze(clip)
    assert result.num_frames == 1
    assert result.status == "clean"

