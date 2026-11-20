# PoseGuard AI — Command Line Interface (CLI) Guide

PoseGuard AI provides a command line utility `poseguard` for inspecting, analyzing, and batch processing 3D skeletal animations.

---

## 1. Global Flags & Version

Display tool version and loaded model architecture:
```cmd
poseguard --version
```
Output:
```
PoseGuard AI 1.0.0 (Model: poseguard-tiny-v1)
```

Display help and command overview:
```cmd
poseguard --help
```

---

## 2. Commands Reference

### A. `poseguard info`
Displays loaded model parameters, device, memory, and supported defect classes.

```cmd
poseguard info
```

---

### B. `poseguard inspect <file>`
Inspects an animation file, printing hierarchy, frame count, frame rate, duration, bounding coordinates, and estimated ground level.

```cmd
poseguard inspect data\raw\subj_01_walk_clean.bvh
```

**Example Output**:
```
============================================================
  ANIMATION FILE INSPECTION: data\raw\subj_01_walk_clean.bvh
============================================================
Frames: 120
Frame Rate: 30.00 FPS (Frame Time: 33.33 ms)
Duration: 4.00 seconds
Joint Count: 21
Root Position Range X: [-0.02, 0.02] m
Root Position Range Y: [0.81, 0.85] m
Root Position Range Z: [0.00, 5.00] m
Estimated Ground Plane Y: 0.0250 m
------------------------------------------------------------
Joint Hierarchy:
  [00] Hips                 -> Parent: ROOT
  [01] Spine                -> Parent: Hips
  [02] Spine1               -> Parent: Spine
  [03] Neck                 -> Parent: Spine1
  [04] Head                 -> Parent: Neck
  [05] LeftShoulder         -> Parent: Spine1
  ...
============================================================
```

---

### C. `poseguard analyze <file>`
Analyzes an animation sequence for motion defects.

#### Basic Usage (Human-Readable Summary):
```cmd
poseguard analyze data\synthetic\subj_01_walk_foot_sliding.npz
```

#### Export Structured JSON:
```cmd
poseguard analyze data\synthetic\subj_01_walk_foot_sliding.npz --format json --out-file analysis.json
```

#### Export CSV Defect Table:
```cmd
poseguard analyze data\synthetic\subj_01_walk_foot_sliding.npz --format csv --out-file defects.csv
```

#### Specify Custom Ground Reference Height:
```cmd
poseguard analyze walk.bvh --ground-y 0.05
```

#### Filter Selected Defect Categories:
```cmd
poseguard analyze walk.bvh --filter foot_sliding ground_penetration
```

---

### D. `poseguard batch <folder>`
Batch analyzes all animation files (`.bvh`, `.npz`, `.npy`) in a folder.

```cmd
poseguard batch data\synthetic
```

**Example Output**:
```
Analyzing 6 animation files in batch...

================================================================================
File                                | Status             | Defects  | Clean %  | Duration
--------------------------------------------------------------------------------
subj_01_walk_bone_stretch.npz       | CRITICAL_DEFECTS   | 1        | 79.2%    | 4.00s
subj_01_walk_contact_inconsistency  | DEFECTS_DETECTED   | 2        | 65.0%    | 4.00s
subj_01_walk_foot_sliding.npz       | DEFECTS_DETECTED   | 1        | 75.0%    | 4.00s
subj_01_walk_ground_penetration.npz | DEFECTS_DETECTED   | 1        | 70.0%    | 4.00s
subj_01_walk_pose_transition.npz    | CRITICAL_DEFECTS   | 1        | 95.8%    | 4.00s
================================================================================
```

---

### E. `poseguard serve`
Launches the local desktop web application:

```cmd
poseguard serve --port 8000
```
Open browser to: `http://127.0.0.1:8000`.

