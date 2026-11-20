"""FastAPI application server providing REST endpoints and static UI for PoseGuard AI."""

import os
import shutil
import tempfile
import uuid
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psutil

from .. import PoseGuard, PoseGuardConfig, AnimationClip
from ..version import __version__, MODEL_VERSION
from ..config import DEFECT_TYPES
from ..diagnosis.reporter import AnalysisReporter

# Storage for active sessions and analysis history in memory
SESSIONS: Dict[str, Dict[str, Any]] = {}
HISTORY: List[Dict[str, Any]] = []

def create_app() -> FastAPI:
    app = FastAPI(
        title="PoseGuard AI",
        description="3D Skeletal Animation Motion Defect Analysis System",
        version=__version__
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    detector = PoseGuard()
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    
    @app.get("/api/info")
    async def get_system_info():
        """Retrieve model metadata, CPU stats, and supported defect categories."""
        mem = psutil.virtual_memory()
        return {
            "app_version": __version__,
            "model_version": MODEL_VERSION,
            "offline_mode": True,
            "supported_defects": DEFECT_TYPES,
            "device": "CPU",
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_available_gb": round(mem.available / (1024**3), 2),
            "ram_percent": mem.percent,
            "model_info": detector.get_model_info()
        }

    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB limit for safety

    @app.post("/api/upload")
    async def upload_animation(file: UploadFile = File(...)):
        """Upload and parse an animation file (.bvh, .npz, .npy) with size limits and path sanitization."""
        raw_name = file.filename or "upload.bvh"
        filename = os.path.basename(raw_name)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".bvh", ".npz", ".npy"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Must be .bvh, .npz, or .npy")
            
        session_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"poseguard_{session_id}_{filename}")
        
        try:
            total_bytes = 0
            with open(temp_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_SIZE:
                        raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed upload size of 50 MB.")
                    buffer.write(chunk)
                
            clip = detector.load(temp_path)
            norm_clip, estimated_gy = detector.normalizer.normalize(clip)
            
            # Subsample frames for fast 3D web preview if animation is very long (>600 frames)
            display_step = max(1, norm_clip.num_frames // 400)
            preview_positions = norm_clip.joint_positions[::display_step].tolist()
            
            SESSIONS[session_id] = {
                "session_id": session_id,
                "filename": filename,
                "temp_path": temp_path,
                "clip": norm_clip,
                "estimated_ground_y": estimated_gy,
                "result": None
            }
            
            return {
                "session_id": session_id,
                "filename": filename,
                "num_frames": norm_clip.num_frames,
                "fps": norm_clip.fps,
                "duration_seconds": round(norm_clip.duration_seconds, 3),
                "num_joints": norm_clip.skeleton.num_joints,
                "joint_names": norm_clip.skeleton.joint_names,
                "parents": norm_clip.skeleton.parents,
                "estimated_ground_y": round(float(estimated_gy), 4),
                "joint_positions": preview_positions,
                "frame_step": display_step
            }
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=422, detail=f"Failed to parse animation file: {str(e)}")

    @app.get("/api/demo/{name}")
    async def load_demo_animation(name: str):
        """Load a built-in demo animation for one-click testing."""
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        demo_map = {
            "walk_clean": os.path.join(root_dir, "data", "raw", "subj_01_walk_clean.bvh"),
            "foot_sliding": os.path.join(root_dir, "data", "synthetic", "subj_01_walk_foot_sliding.npz"),
            "ground_penetration": os.path.join(root_dir, "data", "synthetic", "subj_01_walk_ground_penetration.npz"),
            "pose_transition": os.path.join(root_dir, "data", "synthetic", "subj_01_walk_unnatural_pose_transition.npz"),
            "contact_inconsistency": os.path.join(root_dir, "data", "synthetic", "subj_01_walk_contact_inconsistency.npz"),
            "bone_stretch": os.path.join(root_dir, "data", "synthetic", "subj_01_walk_bone_stretch_anomaly.npz"),
            "pivot_turn_clean": os.path.join(root_dir, "data", "raw", "subj_06_pivot_turn_clean.bvh")
        }
        
        if name not in demo_map or not os.path.exists(demo_map[name]):
            raise HTTPException(status_code=404, detail=f"Demo animation '{name}' not found")
            
        file_path = demo_map[name]
        filename = os.path.basename(file_path)
        session_id = f"demo_{name}_{str(uuid.uuid4())[:4]}"
        
        clip = detector.load(file_path)
        norm_clip, estimated_gy = detector.normalizer.normalize(clip)
        
        display_step = max(1, norm_clip.num_frames // 400)
        preview_positions = norm_clip.joint_positions[::display_step].tolist()
        
        SESSIONS[session_id] = {
            "session_id": session_id,
            "filename": filename,
            "temp_path": file_path,
            "clip": norm_clip,
            "estimated_ground_y": estimated_gy,
            "result": None
        }
        
        return {
            "session_id": session_id,
            "filename": filename,
            "num_frames": norm_clip.num_frames,
            "fps": norm_clip.fps,
            "duration_seconds": round(norm_clip.duration_seconds, 3),
            "num_joints": norm_clip.skeleton.num_joints,
            "joint_names": norm_clip.skeleton.joint_names,
            "parents": norm_clip.skeleton.parents,
            "estimated_ground_y": round(float(estimated_gy), 4),
            "joint_positions": preview_positions,
            "frame_step": display_step
        }

    class AnalyzeRequest(BaseModel):
        session_id: str
        ground_y: Optional[float] = None
        enabled_types: Optional[List[str]] = None

    @app.post("/api/analyze")
    async def run_analysis(req: AnalyzeRequest):
        """Execute hybrid defect analysis on the loaded animation session."""
        if req.session_id not in SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found or expired. Please upload file again.")
            
        session = SESSIONS[req.session_id]
        clip: AnimationClip = session["clip"]
        
        gy = req.ground_y if req.ground_y is not None else session["estimated_ground_y"]
        result = detector.engine.analyze(
            clip=clip,
            file_path=session["filename"],
            ground_y=gy,
            enabled_types=req.enabled_types
        )
        session["result"] = result
        
        # Add to history
        history_entry = {
            "session_id": req.session_id,
            "filename": session["filename"],
            "status": result.quality_summary.overall_status,
            "total_defects": result.quality_summary.total_defects,
            "defect_free_pct": result.quality_summary.defect_free_frame_percentage,
            "duration": result.duration_seconds,
            "timestamp": AnalysisReporter.format_text_summary(result).split("\n")[2] if False else "Just now"
        }
        # Prepend to history
        HISTORY.insert(0, history_entry)
        if len(HISTORY) > 25:
            HISTORY.pop()
            
        return result.model_dump()

    @app.get("/api/history")
    async def get_history():
        """Retrieve recent analysis history."""
        return {"history": HISTORY}

    @app.delete("/api/history")
    async def clear_history():
        """Clear analysis history."""
        HISTORY.clear()
        return {"status": "cleared"}

    @app.get("/api/export/json/{session_id}")
    async def export_json(session_id: str):
        """Export analysis report as JSON."""
        if session_id not in SESSIONS or SESSIONS[session_id]["result"] is None:
            raise HTTPException(status_code=404, detail="No completed analysis found for this session.")
        result = SESSIONS[session_id]["result"]
        json_data = AnalysisReporter.to_json(result)
        return Response(
            content=json_data,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=poseguard_{session_id}_report.json"}
        )

    @app.get("/api/export/csv/{session_id}")
    async def export_csv(session_id: str):
        """Export analysis report as CSV."""
        if session_id not in SESSIONS or SESSIONS[session_id]["result"] is None:
            raise HTTPException(status_code=404, detail="No completed analysis found for this session.")
        result = SESSIONS[session_id]["result"]
        csv_data = AnalysisReporter.to_csv(result)
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=poseguard_{session_id}_defects.csv"}
        )

    # Mount static files at root
    if os.path.exists(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app

