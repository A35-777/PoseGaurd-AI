"""
PoseGuard AI — Autonomous 3D Skeletal Animation Quality Analysis System.
Identifies foot sliding, ground penetration, unnatural pose transitions,
contact inconsistencies, and bone stretch anomalies.
"""

import os
from typing import List, Dict, Any, Optional, Union

from .version import __version__, MODEL_VERSION
from .config import PoseGuardConfig, DEFECT_TYPES, CANONICAL_JOINTS
from .parser.skeleton import Skeleton, JointNode, AnimationClip
from .parser.bvh_parser import BVHParser
from .parser.numpy_io import NumpyAnimationIO
from .parser.normalizer import SkeletonNormalizer
from .diagnosis.schema import DefectItem, QualitySummary, AnalysisResult
from .diagnosis.engine import HybridDiagnosisEngine
from .diagnosis.reporter import AnalysisReporter

class PoseGuard:
    """
    Main developer SDK facade for analyzing 3D skeletal animations.
    Works 100% locally and offline without external dependencies or cloud APIs.
    """
    
    def __init__(
        self,
        config: Optional[PoseGuardConfig] = None,
        model_path: Optional[str] = None
    ):
        self.config = config or PoseGuardConfig()
        
        # Resolve default model path
        if model_path is None:
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            default_loc = os.path.join(pkg_dir, "..", "models", "poseguard_v1.pt")
            if os.path.exists(default_loc):
                model_path = os.path.abspath(default_loc)
            else:
                model_path = "models/poseguard_v1.pt"
                
        self.model_path = model_path
        self.normalizer = SkeletonNormalizer(self.config)
        self.engine = HybridDiagnosisEngine(self.config, model_path=self.model_path)
        
    def load(self, file_path: str) -> AnimationClip:
        """Load and parse an animation file (.bvh, .npz, or .npy)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Animation file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".bvh":
            return BVHParser.parse(file_path)
        elif ext in [".npz", ".npy"]:
            return NumpyAnimationIO.load(file_path)
        else:
            raise ValueError(f"Unsupported animation file extension '{ext}'. Supported: .bvh, .npz, .npy")

    def analyze(
        self,
        target: Union[str, AnimationClip],
        ground_y: Optional[float] = None,
        enabled_types: Optional[List[str]] = None,
        auto_normalize: bool = True
    ) -> AnalysisResult:
        """
        Analyze an animation file or AnimationClip instance for motion defects.
        """
        file_path = None
        if isinstance(target, str):
            file_path = target
            clip = self.load(target)
        elif isinstance(target, AnimationClip):
            clip = target.copy()
        else:
            raise TypeError(f"Expected file path string or AnimationClip, got {type(target)}")
            
        if auto_normalize:
            clip, estimated_gy = self.normalizer.normalize(clip)
            if ground_y is None:
                ground_y = estimated_gy
                
        return self.engine.analyze(
            clip=clip,
            file_path=file_path,
            ground_y=ground_y,
            enabled_types=enabled_types
        )

    def analyze_batch(
        self,
        file_paths: List[str],
        ground_y: Optional[float] = None,
        enabled_types: Optional[List[str]] = None
    ) -> List[AnalysisResult]:
        """Analyze a list of animation files in batch."""
        results = []
        for fp in file_paths:
            try:
                res = self.analyze(fp, ground_y=ground_y, enabled_types=enabled_types)
                results.append(res)
            except Exception as e:
                print(f"[PoseGuard Batch Warning] Error analyzing {fp}: {e}")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Retrieve loaded model version, parameters, and metadata."""
        if self.engine.model_engine and self.engine.model_engine.metadata:
            return self.engine.model_engine.metadata
        return {
            "model_version": MODEL_VERSION,
            "model_path": self.model_path,
            "status": "loaded" if self.engine.model_engine else "geometry_fallback",
            "supported_defects": DEFECT_TYPES
        }

__all__ = [
    "PoseGuard",
    "PoseGuardConfig",
    "AnimationClip",
    "Skeleton",
    "JointNode",
    "BVHParser",
    "NumpyAnimationIO",
    "SkeletonNormalizer",
    "AnalysisResult",
    "DefectItem",
    "QualitySummary",
    "AnalysisReporter",
    "__version__",
    "MODEL_VERSION",
    "DEFECT_TYPES",
    "CANONICAL_JOINTS"
]

