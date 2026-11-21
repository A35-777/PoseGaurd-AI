# PoseGuard AI 🛡️
### Autonomous 3D Skeletal Animation Motion Defect Analysis System

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](docs/CHANGELOG.md)
[![Model](https://img.shields.io/badge/model-poseguard--tiny--v1%20(855%20KB)-emerald.svg)](docs/MODEL_CARD.md)
[![Device](https://img.shields.io/badge/hardware-CPU%20Only%20(No%20CUDA)-orange.svg)](docs/BENCHMARKS.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-26%20Passed-brightgreen.svg)](tests/)

PoseGuard AI is an independently engineered, lightweight, hybrid AI and geometric analysis system designed to autonomously inspect 3D skeletal character animations and detect subtle kinematic defects.

---

## 🌟 Key Capabilities

PoseGuard AI identifies five major classes of 3D skeletal animation defects:

1. **Foot Sliding**: Identifies instances where a character's foot appears planted or in ground contact but drifts or slides horizontally ($v_{xz} > \tau_{slide}$) relative to world ground.
2. **Ground Penetration**: Pinpoints frames where foot, ankle, or body joints clip or sink below the ground plane reference ($Y < Y_{ground} - \tau_{pen}$).
3. **Unnatural Pose Transitions**: Discovers sudden velocity/acceleration spikes ($a > 90 \text{ m/s}^2$), unnatural jerks, and popping joints indicative of frame drops, bad retargeting, or gimbal lock flips.
4. **Contact Inconsistency**: Flags rapid on/off contact state chatter, dragging without vertical clearance during liftoff, and conflicting foot flight states.
5. **Bone Stretch Anomalies**: Enforces anatomical rigid-body kinematic invariance, identifying non-rigid bone expansion or compression ($\Delta L / L_{nominal} > 6\%$).

---

## 🏗️ Hybrid Architecture

![PoseGuard AI Technical Architecture](images/architecture_diagram.png)

PoseGuard AI operates on a three-tiered hybrid architecture that combines deterministic geometric verification with a compact temporal neural classifier:
- **Neural Layer (`PoseGuardNet`)**: 214K parameter temporal model generating calibrated defect probability distributions.
- **Deterministic Geometric Engine**: Exact analytical kinematic verification for contact velocity, ground boundary clearance, joint jerk, and bone rigidity preservation.
- **Calibrated Hybrid Gating**: Multi-modal evidence fusion suppressing false positives down to 2.06%.

---

## ⚡ Hardware & Environment Requirements

PoseGuard AI is engineered to run with zero budget on entry-level, non-GPU hardware:

- **Host Processor**: AMD PRO A12-9800B R7 (4 physical CPU cores, 4 threads).
- **RAM**: 8 GB physical memory (peak runtime usage $< 280$ MB).
- **GPU**: Not required (100% CPU inference; no CUDA required).
- **Operating System**: Windows 10 Pro 64-bit (also compatible with Linux / macOS).
- **Python**: Python 3.12 (in dedicated project-local virtual environment `.venv`).
- **Network**: 100% Offline. Zero external cloud APIs, zero telemetry, zero tracking.

---

## 🚀 Quick Start (Windows CMD)

### 1. Installation
Open Command Prompt (`cmd.exe`) and navigate to the project directory:

```cmd
cd poseguard-ai
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
```

### 2. Run the Desktop Web Application
Launch the local web application:

```cmd
poseguard serve --port 8000
```
Open your web browser to:
👉 **`http://127.0.0.1:8000`**

### 3. Analyze an Animation via CLI
```cmd
poseguard analyze data\synthetic\subj_01_walk_foot_sliding.npz
```

Export structured JSON report:
```cmd
poseguard analyze data\raw\subj_01_walk_clean.bvh --format json --out-file report.json
```

### 4. Python SDK Usage
```python
from poseguard import PoseGuard

# Initialize detector (loads compact model on CPU)
detector = PoseGuard()

# Analyze BVH animation
result = detector.analyze("data/raw/subj_01_walk_clean.bvh")

print(f"Status: {result.quality_summary.overall_status}")
print(f"Defect-Free Frames: {result.quality_summary.defect_free_frame_percentage}%")

for defect in result.defects:
    print(f"[{defect.severity.upper()}] {defect.type} on {defect.joint}")
    print(f"  Frames: {defect.start_frame} - {defect.end_frame} ({defect.duration_sec}s)")
    print(f"  Confidence: {defect.confidence:.2f} ({defect.verification_status})")
    print(f"  Action: {defect.suggested_correction}")
```

---

## 📊 Benchmark Summary (Unseen Test Motion Split)

Evaluated on the held-out unseen sequence (`subj_06_pivot_turn`, 84 temporal windows):

| Model Architecture | Exact Accuracy | Macro F1 | Precision | Recall | CPU Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic Geometry Baseline** | 27.38% | 0.4351 | 0.3290 | 76.25% | **0.10 ms** | 0.0 MB |
| **Simple Statistical Baseline (Z-Score)** | 17.86% | 0.1231 | 0.1600 | 10.00% | 0.14 ms | 0.0 MB |
| **Compact Neural Model (PoseGuardNet)** | **82.14%** | **0.7962** | 91.67% | 72.50% | 0.53 ms | **855 KB** |
| **Calibrated Hybrid Model (PoseGuard AI)** | **77.38%** | **0.7447** | **92.18%** | 65.00% | 0.64 ms | **855 KB** |

*Hardware: AMD PRO A12-9800B CPU, Single-Threaded CPU Execution, PyTorch 2.15.0+cpu.*

---

## 📚 Complete Documentation Index

1. [INSTALLATION.md](INSTALLATION.md) — Step-by-step setup and verification for Windows CMD.
2. [MODEL_CARD.md](MODEL_CARD.md) — Neural architecture, weights, hyperparameters, and provenance.
3. [DATASET_CARD.md](DATASET_CARD.md) — Dataset statistics, classes, splits, and augmentation details.
4. [DATASET_SOURCES.md](DATASET_SOURCES.md) — Public source licenses (CMU, HDM05, SFU) and procedural gait models.
5. [TRAINING.md](TRAINING.md) — Reproducible CPU training pipeline, loss curves, and validation checkpoints.
6. [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) — Deep dive into Conv1D-Residual-BiGRU network design.
7. [API.md](API.md) — FastAPI REST endpoint reference and JSON schemas.
8. [SDK_GUIDE.md](SDK_GUIDE.md) — Python developer integration guide and code snippets.
9. [CLI_GUIDE.md](CLI_GUIDE.md) — Command line tool reference with all flags and examples.
10. [PRIVACY.md](PRIVACY.md) — Data privacy guarantees, local processing, and isolation.
11. [SECURITY.md](SECURITY.md) — Input validation, resource limits, and security architecture.
12. [LIMITATIONS.md](LIMITATIONS.md) — Explicit boundary conditions and non-supported edge cases.
13. [BENCHMARKS.md](BENCHMARKS.md) — Detailed 4-way benchmark comparison and latency metrics.
14. [CHANGELOG.md](CHANGELOG.md) — Version 1.0.0 release notes and feature changelog.
15. [FINAL_AUDIT.md](FINAL_AUDIT.md) — Autonomous 16-point independent audit report with empirical verification.
16. [Runnable Examples & Datasets](../examples/README.md) — Pre-packaged sample animations and expected JSON audit reports.

