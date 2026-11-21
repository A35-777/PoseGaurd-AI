# PoseGuard AI — REST API Documentation

PoseGuard AI includes a local high-performance FastAPI web backend serving both REST endpoints for automation and the 3D interactive web studio.

- **Base URL**: `http://127.0.0.1:8000`
- **Authentication**: None (Localhost network bound).
- **Format**: JSON (`application/json`) / Multipart Form (`multipart/form-data`).

---

## Endpoints

### 1. System Information
`GET /api/info`

Returns runtime hardware statistics, loaded model metadata, and supported defect classes.

**Example Response**:
```json
{
  "app_version": "1.0.2",
  "model_version": "poseguard-tiny-v1",
  "offline_mode": true,
  "supported_defects": [
    "foot_sliding",
    "ground_penetration",
    "unnatural_pose_transition",
    "contact_inconsistency",
    "bone_stretch_anomaly"
  ],
  "device": "CPU",
  "cpu_count": 4,
  "ram_total_gb": 7.43,
  "ram_available_gb": 0.73,
  "ram_percent": 90.2,
  "model_info": {
    "model_version": "poseguard-tiny-v1",
    "parameter_count": 235221,
    "model_file_size_mb": 0.915
  }
}
```

---

### 2. Upload Animation File
`POST /api/upload`

Uploads and parses a 3D animation file (`.bvh`, `.npz`, `.npy`).

**Request**:
- Content-Type: `multipart/form-data`
- Body Parameter: `file` (Binary animation file, $< 50$ MB)

**Response (200 OK)**:
```json
{
  "session_id": "a1b2c3d4",
  "filename": "run_cycle.bvh",
  "num_frames": 120,
  "fps": 30.0,
  "duration_seconds": 4.0,
  "num_joints": 21,
  "joint_names": ["Hips", "Spine", "Spine1", "LeftFoot", "RightFoot"],
  "parents": [-1, 0, 1, 14, 18],
  "estimated_ground_y": 0.025,
  "frame_step": 1,
  "joint_positions": [...]
}
```

---

### 3. Load Built-In Demo Animation
`GET /api/demo/{name}`

Loads a bundled demo animation file without requiring manual upload.

**Parameters**:
- `name` (path parameter):
  - `walk_clean`
  - `foot_sliding`
  - `ground_penetration`
  - `pose_transition`
  - `contact_inconsistency`
  - `bone_stretch`
  - `pivot_turn_clean`

---

### 4. Run Defect Analysis
`POST /api/analyze`

Executes the hybrid defect detection engine on an active session.

**Request Body**:
```json
{
  "session_id": "a1b2c3d4",
  "ground_y": 0.0,
  "enabled_types": [
    "foot_sliding",
    "ground_penetration",
    "unnatural_pose_transition"
  ]
}
```

**Response (200 OK)**:
```json
{
  "status": "defects_detected",
  "file_path": "run_cycle.bvh",
  "num_frames": 120,
  "fps": 30.0,
  "duration_seconds": 4.0,
  "num_joints": 21,
  "ground_y_used": 0.0,
  "quality_summary": {
    "overall_status": "defects_detected",
    "total_defects": 1,
    "defect_counts_by_type": {
      "foot_sliding": 1
    },
    "defect_counts_by_severity": {
      "high": 1
    },
    "defect_free_frame_percentage": 75.0
  },
  "defects": [
    {
      "type": "foot_sliding",
      "joint": "LeftFoot",
      "start_frame": 36,
      "end_frame": 66,
      "duration_frames": 31,
      "duration_sec": 1.033,
      "severity": "high",
      "confidence": 0.89,
      "geometric_evidence": {
        "max_slide_speed_mps": 0.45,
        "mean_slide_speed_mps": 0.32,
        "total_slide_distance_m": 0.33
      },
      "model_prediction": {
        "neural_support_probability": 0.84,
        "model_version": "poseguard-tiny-v1"
      },
      "suggested_correction": "Lock LeftFoot world transform or apply Inverse Kinematics (IK) pinning between frames 36 and 66.",
      "verification_status": "verified_by_both",
      "is_uncertain": false
    }
  ],
  "uncertain_findings": []
}
```

---

### 5. Export Reports
- `GET /api/export/json/{session_id}`: Downloads structured analysis JSON report.
- `GET /api/export/csv/{session_id}`: Downloads CSV defect spreadsheet.
- `GET /api/history`: Retrieves list of recent analyses.
- `DELETE /api/history`: Clears history cache.

