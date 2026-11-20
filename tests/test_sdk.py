"""Unit tests for the PoseGuard developer SDK and schemas."""

import os
import pytest
from poseguard import PoseGuard, AnalysisResult, PoseGuardConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_BVH = os.path.join(BASE_DIR, "data", "raw", "subj_01_walk_clean.bvh")
SAMPLE_NPZ = os.path.join(BASE_DIR, "data", "synthetic", "subj_01_walk_foot_sliding.npz")

def test_sdk_basic_analysis():
    detector = PoseGuard()
    assert os.path.exists(SAMPLE_BVH)
    
    result = detector.analyze(SAMPLE_BVH)
    assert isinstance(result, AnalysisResult)
    assert result.num_frames > 0
    assert result.fps > 0
    assert hasattr(result, "quality_summary")
    assert hasattr(result, "defects")
    assert isinstance(result.defects, list)
    assert 0.0 <= result.quality_summary.defect_free_frame_percentage <= 100.0

def test_sdk_batch_analysis():
    detector = PoseGuard()
    files = [SAMPLE_BVH]
    if os.path.exists(SAMPLE_NPZ):
        files.append(SAMPLE_NPZ)
    else:
        files.append(os.path.join(BASE_DIR, "data", "raw", "subj_01_walk_clean.npz"))
    results = detector.analyze_batch(files)
    assert len(results) == 2
    assert all(isinstance(r, AnalysisResult) for r in results)

def test_sdk_schema_serialization():
    detector = PoseGuard()
    result = detector.analyze(SAMPLE_BVH)
    
    dumped = result.model_dump()
    assert "quality_summary" in dumped
    assert "defects" in dumped
    assert "ground_y_used" in dumped
    json_str = result.model_dump_json()
    assert isinstance(json_str, str)

