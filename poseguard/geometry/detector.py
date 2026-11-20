"""Master geometric defect detection engine."""

from typing import List, Dict, Any, Optional
from ..config import PoseGuardConfig
from ..parser.skeleton import AnimationClip
from .foot_sliding import detect_foot_sliding
from .ground_penetration import detect_ground_penetration
from .pose_transition import detect_unnatural_transitions
from .contact_inconsistency import detect_contact_inconsistency
from .bone_stretch import detect_bone_stretch

class GeometricDefectDetector:
    """Orchestrates deterministic geometric defect detection across all animation rules."""
    
    def __init__(self, config: Optional[PoseGuardConfig] = None):
        self.config = config or PoseGuardConfig()
        
    def analyze(
        self,
        clip: AnimationClip,
        ground_y: Optional[float] = None,
        enabled_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run all enabled deterministic geometric defect checks on the animation clip."""
        gy = ground_y if ground_y is not None else clip.metadata.get("estimated_ground_y", self.config.ground_y)
        types = enabled_types or [
            "foot_sliding",
            "ground_penetration",
            "unnatural_pose_transition",
            "contact_inconsistency",
            "bone_stretch_anomaly"
        ]
        
        all_defects = []
        
        if "foot_sliding" in types:
            all_defects.extend(detect_foot_sliding(clip, self.config, gy))
            
        if "ground_penetration" in types:
            all_defects.extend(detect_ground_penetration(clip, self.config, gy))
            
        if "unnatural_pose_transition" in types:
            all_defects.extend(detect_unnatural_transitions(clip, self.config))
            
        if "contact_inconsistency" in types:
            all_defects.extend(detect_contact_inconsistency(clip, self.config, gy))
            
        if "bone_stretch_anomaly" in types:
            all_defects.extend(detect_bone_stretch(clip, self.config))
            
        # Sort findings by start frame and severity
        severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_defects.sort(key=lambda d: (d["start_frame"], severity_rank.get(d.get("severity", "medium"), 2)))
        
        return all_defects

