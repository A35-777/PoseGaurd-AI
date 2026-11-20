# PoseGuard AI — Changelog

All notable changes to the PoseGuard AI project are documented in this file.

---

## [1.0.0] — 2026-11-19

### 🚀 Initial Public Release

#### Core Animation Ingestion & Parsing
- Implemented pure-Python, zero-C-dependency `BVHParser` with 3D Forward Kinematics (FK) and arbitrary Euler angle channel ordering (ZXY, XYZ, YZX, etc.).
- Implemented `NumpyAnimationIO` supporting standardized `.npz` and `.npy` skeletal sequence import and export.
- Implemented `SkeletonNormalizer` providing canonical 21-joint alias remapping, 1.75m reference height normalization, automatic ground plane estimation, and temporal framerate resampling (e.g. 60 FPS -> 30 FPS).

#### Deterministic Geometric Defect Detection Engine
- Developed `detect_foot_sliding` using horizontal velocity during planted ground contact phases.
- Developed `detect_ground_penetration` identifying sub-floor clipping with configurable ground planes.
- Developed `detect_unnatural_transitions` calculating acceleration spikes ($a > 90 \text{ m/s}^2$) and extreme jerk impulses.
- Developed `detect_contact_inconsistency` detecting high-frequency contact flag chattering and low-clearance dragging.
- Developed `detect_bone_stretch` enforcing anatomical rigid-body length invariance ($\le 6\%$ tolerance).

#### Independently Trained Tiny AI Model
- Designed `PoseGuardNet`: compact 1D Temporal Convolutional Residual Network combined with Bidirectional GRU and multi-scale temporal pooling (214,597 parameters, 0.835 MB checkpoint size).
- Developed 48-channel rotation- and translation-invariant kinematic feature extraction pipeline.
- Built CPU-compatible training pipeline (`ModelTrainer`) using AdamW, CosineAnnealingLR, and BCEWithLogitsLoss.
- Successfully trained from initialized weights in 73.58 seconds on CPU, achieving 0.7757 validation macro F1.

#### Calibrated Hybrid Diagnosis Engine
- Engineered `HybridDiagnosisEngine` fusing deterministic kinematics with neural probabilities.
- Structured output schemas (`AnalysisResult`, `DefectItem`, `QualitySummary`) with Pydantic validation.
- Implemented `AnalysisReporter` supporting JSON export, CSV defect tables, and human-readable terminal summaries.

#### Desktop Web Application & 3D Studio
- Built modern dark-mode 3D animation tool UI using FastAPI and local static assets.
- Implemented Three.js interactive 3D WebGL viewport with orbit controls, skeleton bone rendering, joint spheres, ground grid, and defect highlighting.
- Built interactive timeline scrubber with colored defect interval markers and direct frame navigation.
- Bundled `three.min.js` and `OrbitControls.js` locally for 100% offline, zero-network-dependency operation.

#### Developer SDK & CLI
- Created `poseguard` Python package with single-line analysis facade: `from poseguard import PoseGuard`.
- Implemented full CLI: `poseguard analyze`, `poseguard inspect`, `poseguard batch`, `poseguard serve`, `poseguard info`.
- Configured `pyproject.toml` and `setup.py` for standard pip installation (`pip install -e .`).

#### Quality Assurance & Verification
- Authored 26 comprehensive pytest unit and integration tests covering parser, geometry, model inference, SDK, CLI, REST API, and edge cases (100% passing).
- Conducted 16-point independent audit verification confirming model provenance, zero data leakage, and offline CPU compatibility.

