"""Calibrated hybrid diagnosis engine: fuses geometric kinematics and neural predictions."""

import os
import numpy as np
from typing import List, Dict, Any, Optional

from ..config import PoseGuardConfig, DEFECT_TYPES, MODEL_VERSION
from ..parser.skeleton import AnimationClip
from ..geometry.detector import GeometricDefectDetector
from ..model.inference import ModelInferenceEngine
from .schema import DefectItem, QualitySummary, AnalysisResult

def _merge_adjacent_defects(defects: List[DefectItem], max_gap: int = 3) -> List[DefectItem]:
    """Merge closely adjacent defect intervals of the same type and joint."""
    if not defects:
        return []
    defects_sorted = sorted(defects, key=lambda d: (d.type, d.joint, d.start_frame))
    merged = []
    curr = defects_sorted[0]
    
    sev_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    
    for nxt in defects_sorted[1:]:
        if curr.type == nxt.type and curr.joint == nxt.joint and nxt.start_frame <= curr.end_frame + max_gap + 1:
            new_end = max(curr.end_frame, nxt.end_frame)
            new_dur = new_end - curr.start_frame + 1
            new_conf = round(max(curr.confidence, nxt.confidence), 3)
            new_sev = curr.severity if sev_rank.get(curr.severity, 1) >= sev_rank.get(nxt.severity, 1) else nxt.severity
            verif = "verified_by_both" if "verified_by_both" in (curr.verification_status, nxt.verification_status) else curr.verification_status
            
            curr = DefectItem(
                type=curr.type,
                joint=curr.joint,
                start_frame=curr.start_frame,
                end_frame=new_end,
                duration_frames=new_dur,
                duration_sec=round(new_dur * (curr.duration_sec / max(1, curr.duration_frames)), 3),
                severity=new_sev,
                confidence=new_conf,
                geometric_evidence=curr.geometric_evidence if curr.geometric_evidence else nxt.geometric_evidence,
                model_prediction=curr.model_prediction if curr.model_prediction else nxt.model_prediction,
                suggested_correction=curr.suggested_correction,
                verification_status=verif,
                is_uncertain=curr.is_uncertain and nxt.is_uncertain
            )
        else:
            merged.append(curr)
            curr = nxt
    merged.append(curr)
    return merged

class HybridDiagnosisEngine:
    """Fuses deterministic kinematic geometry with compact neural temporal model predictions."""
    
    def __init__(
        self,
        config: Optional[PoseGuardConfig] = None,
        model_path: str = "models/poseguard_v1.pt"
    ):
        self.config = config or PoseGuardConfig()
        self.geometric_detector = GeometricDefectDetector(self.config)
        self.model_path = model_path
        self.model_engine: Optional[ModelInferenceEngine] = None
        
        if os.path.exists(model_path):
            try:
                self.model_engine = ModelInferenceEngine(model_path=model_path)
            except Exception as e:
                print(f"[PoseGuard Warning] Could not load neural model at {model_path}: {e}")
                self.model_engine = None

    def analyze(
        self,
        clip: AnimationClip,
        file_path: Optional[str] = None,
        ground_y: Optional[float] = None,
        enabled_types: Optional[List[str]] = None
    ) -> AnalysisResult:
        """Execute complete hybrid analysis on AnimationClip."""
        gy = ground_y if ground_y is not None else clip.metadata.get("estimated_ground_y", self.config.ground_y)
        types = enabled_types or list(DEFECT_TYPES)
        
        # 1. Deterministic Geometric Checks
        geo_findings = self.geometric_detector.analyze(clip, ground_y=gy, enabled_types=types)
        
        # 2. Neural Temporal Inference
        neural_results = None
        if self.model_engine is not None:
            try:
                neural_results = self.model_engine.predict_clip(clip, ground_y=gy)
            except Exception as e:
                print(f"[PoseGuard Warning] Neural inference failed: {e}")
                neural_results = None
                
        # 3. Fuse Evidence
        confirmed_defects: List[DefectItem] = []
        uncertain_findings: List[DefectItem] = []
        defective_frames_mask = np.zeros(clip.num_frames, dtype=bool)
        
        type_to_idx = {name: i for i, name in enumerate(DEFECT_TYPES)}
        
        for g_defect in geo_findings:
            d_type = g_defect["type"]
            start_f = g_defect["start_frame"]
            end_f = g_defect["end_frame"]
            c_geo = g_defect["confidence"]
            
            p_neural = 0.5 # Default prior if model unavailable
            if neural_results is not None and d_type in type_to_idx:
                c_idx = type_to_idx[d_type]
                prob_curve = neural_results["frame_probabilities"][:, c_idx]
                p_neural = float(np.mean(prob_curve[start_f:end_f + 1]))
                
            # Calibrated confidence fusion
            if neural_results is not None:
                calibrated_conf = float(np.clip(
                    self.config.geometric_weight * c_geo + self.config.neural_weight * p_neural,
                    0.0, 0.99
                ))
                if p_neural >= 0.38:
                    verif_status = "verified_by_both"
                    is_borderline = False
                elif p_neural < 0.15 and c_geo < 0.65:
                    verif_status = "flagged_by_geometry"
                    is_borderline = True
                else:
                    verif_status = "confirmed_by_geometry"
                    is_borderline = False
            else:
                calibrated_conf = c_geo
                verif_status = "confirmed_by_geometry"
                is_borderline = False
                
            model_pred_info = {
                "neural_support_probability": round(p_neural, 3),
                "model_version": MODEL_VERSION if self.model_engine else "none"
            }
            
            item = DefectItem(
                type=d_type,
                joint=g_defect.get("joint", "Unknown"),
                start_frame=start_f,
                end_frame=end_f,
                duration_frames=g_defect["duration_frames"],
                duration_sec=g_defect["duration_sec"],
                severity=g_defect["severity"],
                confidence=round(calibrated_conf, 3),
                geometric_evidence=g_defect.get("geometric_evidence", {}),
                model_prediction=model_pred_info,
                suggested_correction=g_defect["suggested_correction"],
                verification_status=verif_status,
                is_uncertain=is_borderline
            )
            if is_borderline:
                uncertain_findings.append(item)
            else:
                confirmed_defects.append(item)
                defective_frames_mask[start_f:end_f + 1] = True
            
        # 4. Check for Neural-Only Candidates (Borderline / Uncertain findings)
        if neural_results is not None:
            for n_cand in neural_results.get("neural_candidates", []):
                n_type = n_cand["type"]
                if n_type not in types:
                    continue
                n_start = n_cand["start_frame"]
                n_end = n_cand["end_frame"]
                
                # Check if this interval was already confirmed by geometry
                overlap = np.any(defective_frames_mask[n_start:n_end + 1])
                if not overlap and n_cand["confidence"] >= 0.72:
                    uncertain_item = DefectItem(
                        type=n_type,
                        joint="Skeletal Kinematic Chain",
                        start_frame=n_start,
                        end_frame=n_end,
                        duration_frames=n_cand["duration_frames"],
                        duration_sec=round(n_cand["duration_frames"] * clip.frame_time, 3),
                        severity="low",
                        confidence=round(n_cand["confidence"] * 0.75, 3), # Dampened due to lack of hard geometric breach
                        geometric_evidence={"note": "Kinematic violation is near boundary threshold"},
                        model_prediction={
                            "neural_support_probability": round(n_cand["confidence"], 3),
                            "model_version": MODEL_VERSION
                        },
                        suggested_correction=f"Inspect frames {n_start}-{n_end} for subtle {n_type.replace('_', ' ')} (flagged by neural temporal detector).",
                        verification_status="flagged_by_model",
                        is_uncertain=True
                    )
                    uncertain_findings.append(uncertain_item)

        # 5. Temporal Cluster Merge & Quality Summary
        confirmed_defects = _merge_adjacent_defects(confirmed_defects, max_gap=3)
        uncertain_findings = _merge_adjacent_defects(uncertain_findings, max_gap=3)

        total_defective_frames = int(np.sum(defective_frames_mask))
        defect_free_pct = round(100.0 * (1.0 - (total_defective_frames / max(1, clip.num_frames))), 1)
        
        counts_by_type = {}
        for d in confirmed_defects:
            counts_by_type[d.type] = counts_by_type.get(d.type, 0) + 1
            
        counts_by_sev = {}
        for d in confirmed_defects:
            counts_by_sev[d.severity] = counts_by_sev.get(d.severity, 0) + 1
            
        if len(confirmed_defects) == 0:
            overall_status = "clean"
        elif any(d.severity == "critical" for d in confirmed_defects):
            overall_status = "critical_defects"
        elif any(d.severity == "high" for d in confirmed_defects) or len(confirmed_defects) > 3:
            overall_status = "defects_detected"
        else:
            overall_status = "minor_issues"
            
        summary = QualitySummary(
            overall_status=overall_status,
            total_defects=len(confirmed_defects),
            defect_counts_by_type=counts_by_type,
            defect_counts_by_severity=counts_by_sev,
            defect_free_frame_percentage=defect_free_pct
        )
        
        return AnalysisResult(
            status="clean" if len(confirmed_defects) == 0 else "defects_detected",
            file_path=file_path,
            num_frames=clip.num_frames,
            fps=clip.fps,
            duration_seconds=round(clip.duration_seconds, 3),
            num_joints=clip.skeleton.num_joints,
            ground_y_used=round(float(gy), 4),
            quality_summary=summary,
            defects=confirmed_defects,
            uncertain_findings=uncertain_findings,
            metadata={
                "model_version": MODEL_VERSION,
                "model_available": self.model_engine is not None,
                "clip_metadata": clip.metadata
            }
        )

