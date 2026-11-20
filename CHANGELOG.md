# Changelog

All notable changes to PoseGuard AI are documented here.
Format adheres to Keep a Changelog, and this project adheres to Semantic Versioning.

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

