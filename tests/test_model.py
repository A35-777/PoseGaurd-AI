"""Unit tests for PoseGuardNet model architecture and CPU inference."""

import os
import torch
import numpy as np
import pytest

from poseguard.model.architecture import PoseGuardNet
from poseguard.model.feature_extractor import KinematicFeatureExtractor
from poseguard.model.inference import ModelInferenceEngine
from poseguard.parser.numpy_io import NumpyAnimationIO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_model_instantiation_and_forward():
    model = PoseGuardNet(in_channels=48, num_classes=5)
    params = model.count_parameters()
    assert 150_000 <= params <= 350_000, f"Unexpected parameter count: {params}"
    
    # Dummy batch of 4 windows of shape [4, 48, 32]
    x = torch.randn(4, 48, 32, dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (4, 5)
    
    probs = model.predict_proba(x)
    assert probs.shape == (4, 5)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

def test_saved_model_inference():
    model_path = os.path.join(BASE_DIR, "models", "poseguard_v1.pt")
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"
    
    engine = ModelInferenceEngine(model_path=model_path)
    file_path = os.path.join(BASE_DIR, "data", "raw", "subj_01_walk_clean.npz")
    clip = NumpyAnimationIO.load(file_path)
    
    results = engine.predict_clip(clip)
    assert "window_probabilities" in results
    assert "frame_probabilities" in results
    assert results["frame_probabilities"].shape == (clip.num_frames, 5)

