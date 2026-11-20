"""Pydantic data models for structured PoseGuard AI diagnosis outputs."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DefectItem(BaseModel):
    """Represents an identified skeletal motion defect with multi-modal evidence."""
    type: str = Field(..., description="Defect classification (e.g. foot_sliding, ground_penetration)")
    joint: str = Field(..., description="Affected skeletal joint or bone segment")
    start_frame: int = Field(..., description="Initial frame of defect occurrence")
    end_frame: int = Field(..., description="Final frame of defect occurrence")
    duration_frames: int = Field(..., description="Duration in frames")
    duration_sec: float = Field(..., description="Duration in seconds")
    severity: str = Field(..., description="Severity classification: low, medium, high, critical")
    confidence: float = Field(..., description="Calibrated confidence score between 0.0 and 1.0")
    geometric_evidence: Dict[str, Any] = Field(default_factory=dict, description="Deterministic measured kinematics")
    model_prediction: Dict[str, Any] = Field(default_factory=dict, description="Neural temporal model confidence and support")
    suggested_correction: str = Field(..., description="Actionable animation correction guideline")
    verification_status: str = Field("verified_by_both", description="Provenance: verified_by_both, confirmed_by_geometry, flagged_by_model")
    is_uncertain: bool = Field(False, description="Flag for borderline/uncertain findings")

class QualitySummary(BaseModel):
    """High-level summary of animation clip quality."""
    overall_status: str = Field(..., description="clean, minor_issues, defects_detected, or critical_defects")
    total_defects: int = Field(0, description="Total confirmed defect intervals")
    defect_counts_by_type: Dict[str, int] = Field(default_factory=dict)
    defect_counts_by_severity: Dict[str, int] = Field(default_factory=dict)
    defect_free_frame_percentage: float = Field(100.0, description="Percentage of frames without detected defects")

class AnalysisResult(BaseModel):
    """Complete structured output schema for PoseGuard AI animation analysis."""
    status: str = Field(..., description="Overall analysis status (clean, defects_detected)")
    file_path: Optional[str] = Field(None, description="Analyzed file path or identifier")
    num_frames: int = Field(..., description="Total frames in animation clip")
    fps: float = Field(..., description="Frame rate of animation")
    duration_seconds: float = Field(..., description="Total clip duration in seconds")
    num_joints: int = Field(..., description="Total joints in skeleton")
    ground_y_used: float = Field(0.0, description="Ground plane height (Y) in meters used for checks")
    quality_summary: QualitySummary
    defects: List[DefectItem] = Field(default_factory=list, description="Confirmed defects")
    uncertain_findings: List[DefectItem] = Field(default_factory=list, description="Borderline or uncertain findings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata and model version")

