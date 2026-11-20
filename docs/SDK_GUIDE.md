# PoseGuard AI — Developer SDK Integration Guide

The `poseguard` Python package provides a clean, type-safe API for integrating motion defect analysis into automated asset pipelines, game engine import scripts, and continuous integration checks.

---

## 1. Quick Example

```python
from poseguard import PoseGuard

# Initialize detector (automatically loads models/poseguard_v1.pt onto CPU)
detector = PoseGuard()

# Analyze an animation file (.bvh, .npz, or .npy)
result = detector.analyze("animations/character_run.bvh")

print(f"Overall Quality Status: {result.quality_summary.overall_status.upper()}")
print(f"Defect-Free Frames: {result.quality_summary.defect_free_frame_percentage:.1f}%")

for defect in result.defects:
    print(f"Defect: {defect.type} on '{defect.joint}'")
    print(f"  Frames: {defect.start_frame} to {defect.end_frame} ({defect.duration_sec:.2f}s)")
    print(f"  Severity: {defect.severity.upper()} | Confidence: {defect.confidence:.2f}")
    print(f"  Verification: {defect.verification_status}")
    print(f"  Recommended Correction: {defect.suggested_correction}\n")
```

---

## 2. Customizing Thresholds with `PoseGuardConfig`

Animators can fine-tune physical detection thresholds using `PoseGuardConfig`:

```python
from poseguard import PoseGuard, PoseGuardConfig

# Define custom thresholds for high-speed martial arts or stylized movement
custom_config = PoseGuardConfig(
    # Allow faster horizontal foot drift before flagging sliding (e.g. 25 cm/s)
    foot_slide_speed_threshold=0.25,
    
    # Floor penetration threshold (-3 cm)
    ground_penetration_threshold=-0.03,
    
    # Increase joint acceleration threshold for acrobatic moves (140 m/s^2)
    joint_accel_threshold=140.0,
    
    # Enforce strict bone length rigidity (only 4% variation allowed)
    bone_stretch_relative_tolerance=0.04,
    
    # Weighting between deterministic geometry (0.60) and neural model (0.40)
    geometric_weight=0.60,
    neural_weight=0.40
)

detector = PoseGuard(config=custom_config)
result = detector.analyze("animations/acrobatic_flip.bvh")
```

---

## 3. Batch Analysis Pipeline

For continuous integration (CI/CD) testing across an entire animation library:

```python
import os
from poseguard import PoseGuard

detector = PoseGuard()

anim_folder = "game_assets/characters/paladin/animations"
files = [os.path.join(anim_folder, f) for f in os.listdir(anim_folder) if f.endswith(".bvh")]

# Batch analyze all clips
results = detector.analyze_batch(files)

# Assert all production clips meet minimum quality gate (e.g. > 90% clean frames)
failed_clips = []
for res in results:
    if res.quality_summary.defect_free_frame_percentage < 90.0:
        failed_clips.append((res.file_path, res.quality_summary.defect_free_frame_percentage))

if failed_clips:
    print(f"Quality Gate Failed for {len(failed_clips)} clips:")
    for path, pct in failed_clips:
        print(f"  - {path}: {pct:.1f}% clean frames")
```

---

## 4. Programmatic Report Generation (JSON & CSV)

```python
from poseguard import PoseGuard, AnalysisReporter

detector = PoseGuard()
result = detector.analyze("walk_cycle.bvh")

# Export structured JSON string
json_data = AnalysisReporter.to_json(result, indent=2)
with open("walk_quality_report.json", "w") as f:
    f.write(json_data)

# Export CSV defect spreadsheet
csv_data = AnalysisReporter.to_csv(result)
with open("walk_defects.csv", "w") as f:
    f.write(csv_data)
```

---

## 5. Direct Skeletal Data Access & Kinematics

You can directly load animation clips, inspect joint hierarchies, and access Cartesian 3D joint positions:

```python
from poseguard import PoseGuard

detector = PoseGuard()
clip = detector.load("walk_cycle.bvh")

print(f"Total Frames: {clip.num_frames}")
print(f"FPS: {clip.fps:.1f}")
print(f"Joints: {clip.skeleton.joint_names}")

# Access 3D Cartesian coordinates [Frames, Joints, XYZ] in meters
positions = clip.joint_positions # NumPy array [T, J, 3]

# Get world position of LeftFoot at frame 0
left_foot_idx = clip.skeleton.get_joint_index("LeftFoot")
print(f"LeftFoot at Frame 0: {positions[0, left_foot_idx]}")
```

