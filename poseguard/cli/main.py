"""PoseGuard AI Command Line Interface (CLI)."""

import os
import sys
import argparse
import json
import numpy as np
import uvicorn
from typing import List

from ..version import __version__, MODEL_VERSION
from ..config import DEFECT_TYPES, PoseGuardConfig
from ..diagnosis.reporter import AnalysisReporter
from .. import PoseGuard

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poseguard",
        description="PoseGuard AI — 3D Skeletal Animation Motion Defect Analysis System"
    )
    parser.add_argument("--version", action="version", version=f"PoseGuard AI {__version__} (Model: {MODEL_VERSION})")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    p_analyze = subparsers.add_parser("analyze", help="Analyze an animation file for motion defects")
    p_analyze.add_argument("file", type=str, help="Path to .bvh, .npz, or .npy animation file")
    p_analyze.add_argument("--format", choices=["text", "json", "csv"], default="text", help="Output format (default: text)")
    p_analyze.add_argument("--ground-y", type=float, default=None, help="Explicit ground reference height in meters")
    p_analyze.add_argument("--out-file", type=str, default=None, help="Save report to file")
    p_analyze.add_argument("--filter", nargs="+", choices=DEFECT_TYPES, help="Analyze only selected defect types")
    
    # Inspect command
    p_inspect = subparsers.add_parser("inspect", help="Inspect animation file metadata, hierarchy, and kinematics")
    p_inspect.add_argument("file", type=str, help="Path to animation file")
    
    # Batch command
    p_batch = subparsers.add_parser("batch", help="Batch analyze all animation files in a folder")
    p_batch.add_argument("folder", type=str, help="Directory containing animation files")
    p_batch.add_argument("--format", choices=["text", "json", "csv"], default="text", help="Output format")
    p_batch.add_argument("--out-dir", type=str, default=None, help="Directory to save individual reports")
    
    # Info command
    subparsers.add_parser("info", help="Display PoseGuard AI model and system information")
    
    # Serve command
    p_serve = subparsers.add_parser("serve", help="Launch the local desktop-friendly web application")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    
    return parser

def main(args: List[str] = None):
    parser = build_parser()
    parsed = parser.parse_args(args)
    
    if not parsed.command:
        parser.print_help()
        sys.exit(0)
        
    detector = PoseGuard()
    
    if parsed.command == "info":
        info = detector.get_model_info()
        print("=" * 60)
        print("  POSEGUARD AI SYSTEM INFO")
        print("=" * 60)
        print(f"Software Version: {__version__}")
        print(f"Model Version: {MODEL_VERSION}")
        print(f"Model File: {detector.model_path}")
        print(f"Supported Defect Classes:")
        for d in DEFECT_TYPES:
            print(f"  • {d}")
        print("-" * 60)
        print("Model Metadata:")
        print(json.dumps(info, indent=2))
        print("=" * 60)
        
    elif parsed.command == "inspect":
        clip = detector.load(parsed.file)
        print("=" * 60)
        print(f"  ANIMATION FILE INSPECTION: {parsed.file}")
        print("=" * 60)
        print(f"Frames: {clip.num_frames}")
        print(f"Frame Rate: {clip.fps:.2f} FPS (Frame Time: {clip.frame_time * 1000:.2f} ms)")
        print(f"Duration: {clip.duration_seconds:.2f} seconds")
        print(f"Joint Count: {clip.skeleton.num_joints}")
        print(f"Root Position Range X: [{np.min(clip.joint_positions[:, 0, 0]):.2f}, {np.max(clip.joint_positions[:, 0, 0]):.2f}] m")
        print(f"Root Position Range Y: [{np.min(clip.joint_positions[:, 0, 1]):.2f}, {np.max(clip.joint_positions[:, 0, 1]):.2f}] m")
        print(f"Root Position Range Z: [{np.min(clip.joint_positions[:, 0, 2]):.2f}, {np.max(clip.joint_positions[:, 0, 2]):.2f}] m")
        est_gy = detector.normalizer.estimate_ground_plane(clip)
        print(f"Estimated Ground Plane Y: {est_gy:.4f} m")
        print("-" * 60)
        print("Joint Hierarchy:")
        for idx, (name, parent_idx) in enumerate(zip(clip.skeleton.joint_names, clip.skeleton.parents)):
            parent_name = clip.skeleton.joint_names[parent_idx] if parent_idx != -1 else "ROOT"
            print(f"  [{idx:02d}] {name:<20} -> Parent: {parent_name}")
        print("=" * 60)
        
    elif parsed.command == "analyze":
        result = detector.analyze(
            parsed.file,
            ground_y=parsed.ground_y,
            enabled_types=parsed.filter
        )
        
        if parsed.format == "json":
            out_str = AnalysisReporter.to_json(result)
        elif parsed.format == "csv":
            out_str = AnalysisReporter.to_csv(result)
        else:
            out_str = AnalysisReporter.format_text_summary(result)
            
        if parsed.out_file:
            with open(parsed.out_file, "w", encoding="utf-8") as f:
                f.write(out_str)
            print(f"Report saved to {parsed.out_file}")
        else:
            print(out_str)
            
    elif parsed.command == "batch":
        folder = parsed.folder
        if not os.path.exists(folder):
            print(f"Error: Directory not found: {folder}")
            sys.exit(1)
            
        supported_exts = [".bvh", ".npz", ".npy"]
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in supported_exts]
        
        if not files:
            print(f"No supported animation files found in {folder}")
            sys.exit(0)
            
        print(f"Analyzing {len(files)} animation files in batch...")
        results = detector.analyze_batch(files)
        
        summary_rows = []
        for res in results:
            summary_rows.append({
                "file": os.path.basename(res.file_path or "clip"),
                "status": res.quality_summary.overall_status,
                "defects": res.quality_summary.total_defects,
                "clean_frame_pct": f"{res.quality_summary.defect_free_frame_percentage:.1f}%",
                "duration": f"{res.duration_seconds:.2f}s"
            })
            
        print("\n" + "=" * 80)
        print(f"{'File':<35} | {'Status':<18} | {'Defects':<8} | {'Clean %':<8} | {'Duration'}")
        print("-" * 80)
        for r in summary_rows:
            print(f"{r['file']:<35} | {r['status']:<18} | {r['defects']:<8} | {r['clean_frame_pct']:<8} | {r['duration']}")
        print("=" * 80)
        
    elif parsed.command == "serve":
        from ..app.server import create_app
        print("=" * 60)
        print(f"  STARTING POSEGUARD AI WEB SERVER")
        print(f"  URL: http://{parsed.host}:{parsed.port}")
        print(f"  Offline mode: 100% Active (Zero external calls)")
        print("=" * 60)
        uvicorn.run(create_app(), host=parsed.host, port=parsed.port)

if __name__ == "__main__":
    main()

