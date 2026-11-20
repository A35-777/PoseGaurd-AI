"""Integration tests for the PoseGuard AI FastAPI web backend."""

import os
import pytest
from fastapi.testclient import TestClient
from poseguard.app.server import create_app

@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c

def test_api_info(client):
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.json()
    assert "app_version" in data
    assert "model_version" in data
    assert data["offline_mode"] is True

def test_api_demo_load(client):
    res = client.get("/api/demo/walk_clean")
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    assert data["num_frames"] > 0
    assert len(data["joint_names"]) > 0

def test_api_analyze(client):
    # First load demo to get session_id
    demo_res = client.get("/api/demo/walk_clean")
    session_id = demo_res.json()["session_id"]
    
    # Run analysis
    res = client.post("/api/analyze", json={
        "session_id": session_id,
        "enabled_types": ["foot_sliding", "ground_penetration"]
    })
    assert res.status_code == 200
    result = res.json()
    assert "quality_summary" in result
    assert "defects" in result

def test_api_export_json_and_csv(client):
    demo_res = client.get("/api/demo/foot_sliding")
    session_id = demo_res.json()["session_id"]
    
    # Run analysis
    client.post("/api/analyze", json={"session_id": session_id})
    
    # Test JSON export
    res_json = client.get(f"/api/export/json/{session_id}")
    assert res_json.status_code == 200
    assert "quality_summary" in res_json.text
    
    # Test CSV export
    res_csv = client.get(f"/api/export/csv/{session_id}")
    assert res_csv.status_code == 200
    assert "defect_id,type,joint" in res_csv.text

