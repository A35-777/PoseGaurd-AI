# Changelog

All notable changes to PoseGuard AI are documented here.
Format adheres to Keep a Changelog, and this project adheres to Semantic Versioning.

## [1.0.2-beta] - 2026-11-21

### Added
- **Squeeze-and-Excitation Channel Attention**: Added lightweight 1D SE attention (reduction ratio 8) to `TemporalResidualBlock`, enabling dynamic channel recalibration across kinematic signals.
- **Multi-Scale Temporal Feature Fusion**: Integrated convolution temporal pooling skip-connection into classification head alongside BiGRU dual pooling (235,221 parameters total, strictly $\le 250\text{k}$).
- **Continuous Soft Contact Kinematics**: Introduced smooth continuous contact probability features for foot sliding and ground penetration to eliminate discrete threshold truncation artifacts.
- **Stance-Aware Ground Plane Estimation**: Stabilized floor plane estimation by anchoring to zero-vertical-velocity stance frames.
- **Calibrated Multi-Tier Hybrid Gating**: Redesigned dual-gate evidence fusion between geometric kinematic bounds and temporal neural probabilities.
- **Temporal Defect Interval Merging**: Added non-maximum interval clustering (`_merge_adjacent_defects`) to coalesce neighboring borderline defect flickers.

### Changed
- **Evaluation Benchmark Leap**: On the held-out unseen Subject 6 test split, exact match accuracy rose from **75.00% to 84.52%** (+9.52%), Macro F1 rose from **0.7337 to 0.8462** (+11.25%), and recall jumped from **65.00% to 85.00%** (+20.00%) while maintaining a conservative **1.92% False Positive Rate**.
- **Model Checkpoint**: Re-trained from initialized weights strictly on CPU in 135.0s, yielding a self-contained **941 KB (0.92 MB)** `.pt` checkpoint ($\le 1.0\text{ MB}$ target).
- **Latency & Throughput**: Single-window CPU latency reduced to **0.74 ms** (neural) and **0.92 ms** (hybrid), retaining sub-millisecond execution.

## [1.0.1-beta] - 2026-11-21

### Added
- **Interactive Visual Assets**: Professional 1280x720 architecture pipeline diagram (`docs/images/architecture_diagram.png`), 5-class defect visual breakdown (`docs/images/defect_examples.png`), and high-res 3D studio preview graphics in `docs/images/`.
- **Pre-Packaged Runnable Examples**: Added `examples/` directory containing sample clean walk BVH, pivot turn BVH, defective foot-sliding NPZ, and corresponding expected JSON audit reports.
- **GitHub Actions CI Pipeline**: Added automated multi-OS and multi-version test workflow (`.github/workflows/ci.yml`) covering Python 3.10, 3.11, and 3.12.
- **Code Ownership & Community Governance**: Added `.github/CODEOWNERS`, expanded contributor workflow with test instructions, code style, and PR checklist in `CONTRIBUTING.md`.
- **Security & Vulnerability Reporting**: Enhanced `SECURITY.md` with structured disclosure pathways and response timelines.

### Changed
- **README Overhaul**: Added "Why PoseGuard?" value proposition, explicit Core Philosophy & Guiding Principles, 2-minute clone-to-result quickstart, live demo section with real CLI and Python SDK outputs, roadmap milestones, and status badges.
- **Model Card Provenance**: Corrected model card authorship to independent open-source research contributors.
- **Benchmark & Evaluation Protocol**: Expanded `docs/BENCHMARKS.md` with sampling window methodology, held-out Subject 6 zero-leakage protocol, and future multi-dataset evaluation roadmap.
- **Limitations Transparency**: Enhanced `docs/LIMITATIONS.md` visibility and boundary condition explanations for planar ground and stylized animations.

## [1.0.0-beta] - 2026-11-19

### Added
- **PoseGuardNet Architecture**: 214,597 parameter 1D convolutional residual network (855 KB checkpoint).
- **Geometric Detectors**: Analytical algorithms for foot sliding, ground penetration, unnatural pose transitions, contact inconsistency, and bone stretch.
- **Hybrid Diagnostic Engine**: Multi-tiered calibration combining neural temporal feature scoring with geometric boundary checks.
- **Biomechanical Normalizer**: Translation-invariant, hip-ground referenced normalization with automatic floor level estimation.
- **Animation Parser**: Pure-Python forward-kinematics BVH parser, along with NumPy (.npz, .npy) ingestion pipelines.
- **Interactive 3D Web Studio**: Three.js skeletal viewer with defect timeline scrub bar, real-time bounding overlays, and frame-accurate inspection.
- **REST API**: Localhost FastAPI backend (`127.0.0.1:8000`) for headless automated QA pipelines.
- **Command Line Interface (`poseguard`)**: Subcommands for `analyze`, `inspect`, `batch`, `info`, and `serve`.
- **Python SDK**: High-level programmatic API via `from poseguard import PoseGuard`.
- **Offline & CPU First**: Zero external telemetry, zero paid APIs, zero CUDA dependencies. Operates on standard multi-core CPUs.
- **Test Suite**: 26 automated unit and integration tests covering geometry, models, normalizer, CLI, and API endpoints.

### Changed
- Converted foot feature coordinates to root-relative space to guarantee rotational invariance during pivot turns.
- Hardened API endpoints with filename sanitization and local origin CORS policies.

