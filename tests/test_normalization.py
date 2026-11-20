"""Unit tests for skeleton normalization, joint remapping, and framerate resampling."""

import os
import pytest
import numpy as np

from poseguard.parser.bvh_parser import BVHParser
from poseguard.parser.normalizer import SkeletonNormalizer
from poseguard.config import CANONICAL_JOINTS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_BVH = os.path.join(BASE_DIR, "data", "raw", "subj_01_walk_clean.bvh")

def test_canonical_remapping_and_normalization():
    clip = BVHParser.parse(SAMPLE_BVH)
    
    normalizer = SkeletonNormalizer()
    norm_clip, ground_y = normalizer.normalize(clip)
    
    # Verify canonical joint count
    assert norm_clip.skeleton.num_joints == len(CANONICAL_JOINTS)
    assert norm_clip.skeleton.joint_names == list(CANONICAL_JOINTS)
    assert norm_clip.joint_positions.shape[1] == len(CANONICAL_JOINTS)
    assert isinstance(ground_y, float)

def test_framerate_resampling():
    clip = BVHParser.parse(SAMPLE_BVH)
    
    normalizer = SkeletonNormalizer()
    # Resample to 60 fps
    resampled_60 = normalizer.resample_framerate(clip, target_fps=60.0)
    assert abs(resampled_60.fps - 60.0) < 0.1
    assert resampled_60.num_frames > clip.num_frames
    
    # Resample back to 30 fps
    resampled_30 = normalizer.resample_framerate(resampled_60, target_fps=30.0)
    assert abs(resampled_30.fps - 30.0) < 0.1
    assert abs(resampled_30.num_frames - clip.num_frames) <= 2

