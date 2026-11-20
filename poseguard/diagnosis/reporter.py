"""Reporting and export utilities for AnalysisResult (JSON, CSV, and summary)."""

import csv
import io
import json
from typing import Dict, Any
from .schema import AnalysisResult

class AnalysisReporter:
    """Formats and exports PoseGuard AI analysis findings to JSON, CSV, and terminal reports."""
    
    @staticmethod
    def to_json(result: AnalysisResult, indent: int = 2) -> str:
        """Export result to formatted JSON string."""
        return result.model_dump_json(indent=indent)

    @staticmethod
    def to_csv(result: AnalysisResult) -> str:
        """Export defect findings to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "defect_id",
            "type",
            "joint",
            "start_frame",
            "end_frame",
            "duration_frames",
            "duration_sec",
            "severity",
            "confidence",
            "verification_status",
            "is_uncertain",
            "suggested_correction"
        ])
        
        all_items = result.defects + result.uncertain_findings
        for idx, d in enumerate(all_items, start=1):
            writer.writerow([
                idx,
                d.type,
                d.joint,
                d.start_frame,
                d.end_frame,
                d.duration_frames,
                d.duration_sec,
                d.severity,
                d.confidence,
                d.verification_status,
                d.is_uncertain,
                d.suggested_correction
            ])
            
        return output.getvalue()

    @staticmethod
    def format_text_summary(result: AnalysisResult) -> str:
        """Generate human-readable summary text."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  POSEGUARD AI — ANIMATION ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"File: {result.file_path or 'In-Memory Stream'}")
        lines.append(f"Frames: {result.num_frames} | FPS: {result.fps:.1f} | Duration: {result.duration_seconds:.2f}s | Joints: {result.num_joints}")
        lines.append(f"Ground Reference (Y): {result.ground_y_used:.3f}m")
        lines.append("-" * 60)
        lines.append(f"Status: {result.quality_summary.overall_status.upper()}")
        lines.append(f"Defect-free Frame Ratio: {result.quality_summary.defect_free_frame_percentage:.1f}%")
        lines.append(f"Confirmed Defects: {len(result.defects)} | Uncertain/Borderline: {len(result.uncertain_findings)}")
        lines.append("-" * 60)
        
        if result.defects:
            lines.append("CONFIRMED DEFECTS:")
            for i, d in enumerate(result.defects, 1):
                lines.append(f"  [{i}] {d.type.upper()} ({d.severity.upper()}) on joint '{d.joint}'")
                lines.append(f"      Frames: {d.start_frame} - {d.end_frame} ({d.duration_sec:.2f}s) | Confidence: {d.confidence:.2f} [{d.verification_status}]")
                lines.append(f"      Action: {d.suggested_correction}")
        else:
            lines.append("No motion defects detected. Animation meets physical kinematic standards.")
            
        if result.uncertain_findings:
            lines.append("-" * 60)
            lines.append("UNCERTAIN / BORDERLINE OBSERVATIONS:")
            for i, u in enumerate(result.uncertain_findings, 1):
                lines.append(f"  [?] {u.type.upper()} on frames {u.start_frame}-{u.end_frame} (Confidence: {u.confidence:.2f})")
                lines.append(f"      Note: {u.suggested_correction}")
                
        lines.append("=" * 60)
        return "\n".join(lines)

