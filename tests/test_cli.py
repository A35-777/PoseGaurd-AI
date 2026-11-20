"""Unit tests for the PoseGuard CLI commands."""

import os
import subprocess
import sys
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_BVH = os.path.join(BASE_DIR, "data", "raw", "subj_01_walk_clean.bvh")

def run_cli_command(args):
    cmd = [sys.executable, "-m", "poseguard.cli.main"] + args
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=BASE_DIR)
    return res

def test_cli_info():
    res = run_cli_command(["info"])
    assert res.returncode == 0
    assert "POSEGUARD AI SYSTEM INFO" in res.stdout
    assert "poseguard-tiny-v1" in res.stdout

def test_cli_inspect():
    res = run_cli_command(["inspect", SAMPLE_BVH])
    assert res.returncode == 0
    assert "ANIMATION FILE INSPECTION" in res.stdout
    assert "Joint Hierarchy" in res.stdout

def test_cli_analyze_json():
    res = run_cli_command(["analyze", SAMPLE_BVH, "--format", "json"])
    assert res.returncode == 0
    assert '"quality_summary"' in res.stdout
    assert '"overall_status"' in res.stdout

def test_cli_analyze_csv():
    res = run_cli_command(["analyze", SAMPLE_BVH, "--format", "csv"])
    assert res.returncode == 0
    assert "defect_id,type,joint" in res.stdout

