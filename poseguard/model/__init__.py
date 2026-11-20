"""Independently trained Tiny AI Model module."""

from .architecture import PoseGuardNet, TemporalResidualBlock
from .feature_extractor import KinematicFeatureExtractor, FEATURE_CHANNELS
from .dataset import PoseGuardDataset
from .trainer import ModelTrainer
from .inference import ModelInferenceEngine

__all__ = [
    "PoseGuardNet",
    "TemporalResidualBlock",
    "KinematicFeatureExtractor",
    "FEATURE_CHANNELS",
    "PoseGuardDataset",
    "ModelTrainer",
    "ModelInferenceEngine",
]

