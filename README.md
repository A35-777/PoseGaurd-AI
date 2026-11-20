# PoseGuard AI 🛡️
### Autonomous 3D Skeletal Animation Motion Defect Analysis System

[![Release](https://img.shields.io/badge/release-v1.0.0--beta-blue.svg)](CHANGELOG.md)
[![Model](https://img.shields.io/badge/model-PoseGuardNet%20(855%20KB)-emerald.svg)](docs/MODEL_CARD.md)
[![Execution](https://img.shields.io/badge/runtime-100%25%20CPU%20Offline-orange.svg)](docs/BENCHMARKS.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-26%20Passed-brightgreen.svg)](tests/)

> **⚠️ Public Beta Notice**: PoseGuard AI is currently in Public Beta. The system generates advisory kinematic recommendations and automated defect suggestions to accelerate animator review and technical QA pipelines. It is not an automated replacement for human aesthetic direction. Benchmarks reflect held-out test sequences; broader field performance may vary across non-standard character rigs.

---

## 📸 Interactive 3D Studio Preview

```
+-------------------------------------------------------------------------------+
|  PoseGuard AI 3D Studio                                              [v1.0.0] |
|  [ Load File ] [ Demo Walk ] [ Demo Pivot Turn ]                              |
| +-----------------------------------------------+ +-------------------------+ |
| |                                               | | Defect Breakdown        | |
| |           [ 3D Skeleton Viewport ]            | | • Foot Sliding: Frame 42| |
| |                                               | | • Ground Pen: Frame 78  | |
| |                                               | | Overall: NEEDS_REVIEW   | |
| +-----------------------------------------------+ +-------------------------+ |
| |  Timeline: [====|========================] 42/120                           | |
+-------------------------------------------------------------------------------+
```
*(Screenshot placeholder: To display a screenshot or GIF in your fork, place a 1280x720 image at `docs/images/studio_preview.png` and link it here: `![PoseGuard AI 3D Studio](docs/images/studio_preview.png)`)*

---

## 🌟 What PoseGuard AI Does

PoseGuard AI is a lightweight, hybrid AI and geometric analysis system designed to autonomously inspect 3D character animation sequences (.bvh, .npz, .npy) and detect five critical motion defect classes:

1. **Foot Sliding**: Detects frames where feet are grounded but drift horizontally ($v_{xz} > \tau_{slide}$).
2. **Ground Penetration**: Pinpoints frames where joints sink beneath the estimated floor ($Y < Y_{ground} - \tau_{pen}$).
3. **Unnatural Pose Transitions**: Flags sudden velocity or acceleration spikes ($a > 90 \text{ m/s}^2$) indicative of frame drops, bad retargeting, or joint popping.
4. **Contact Inconsistency**: Identifies rapid on/off contact state chatter and dragging without vertical clearance.
5. **Bone Stretch Anomalies**: Enforces anatomical rigid-body kinematic invariance, identifying non-rigid bone expansion or compression ($\Delta L / L_{nominal} > 6\%$).

---

## 🏗️ Technical Architecture

- **Tiny Neural Classifier (PoseGuardNet)**: 214,597 parameter Conv1D-Residual-BiGRU network requiring only **855 KB** of storage. Operates on 48-channel translation-invariant kinematic features.
- **Deterministic Geometric Engine**: Exact analytical kinematic verification for contact velocity, ground boundary clearance, and bone length preservation.
- **Calibrated Hybrid Gating**: Cross-validates neural temporal defect scores against physical geometric bounds to suppress false positives (FPR 2.06%).
- **100% Offline & CPU-Native**: Inference runs in $< 0.65\text{ ms}$ per temporal window on standard 4-core consumer CPUs without requiring CUDA, GPUs, or external cloud APIs.

---

## ⚡ System Requirements

- **Operating System**: Windows 10/11 (64-bit) or Linux (Ubuntu 20.04+, Debian 11+, Fedora 38+).
- **Python**: Python 3.10, 3.11, or 3.12.
- **Hardware**: Standard multi-core CPU (tested on AMD PRO A12-9800B @ 2.7 GHz).
- **Memory**: 4 GB RAM minimum (peak process footprint $< 280\text{ MB}$).
- **Storage**: Under 25 MB disk space total (including model checkpoint and dependencies).

---

## 🚀 Installation

### Option A: Windows (CMD)
```cmd
git clone https://github.com/A35-777/PoseGaurd-AI.git
cd poseguard-ai
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Option B: Linux (Bash)
```bash
git clone https://github.com/A35-777/PoseGaurd-AI.git
cd poseguard-ai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 💻 Quick Start & Usage

### 1. Launch Interactive 3D Web Studio
```cmd
poseguard serve --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in any modern web browser to load files, inspect skeletons in 3D, and scrub the defect timeline.

### 2. CLI Analysis
```cmd
poseguard analyze data\raw\subj_01_walk_clean.bvh
```
Export structured reports in JSON or CSV format:
```cmd
poseguard analyze data\raw\subj_01_walk_clean.bvh --format json --out-file report.json
poseguard analyze data\raw\subj_01_walk_clean.bvh --format csv --out-file defects.csv
```

Batch analyze an entire directory of animations:
```cmd
poseguard batch data\raw --format text
```

### 3. Python SDK
```python
from poseguard import PoseGuard

# Initialize detector (loads 855 KB checkpoint on CPU)
detector = PoseGuard()

# Analyze animation clip
result = detector.analyze("data/raw/subj_01_walk_clean.bvh")

print(f"Overall Status: {result.quality_summary.overall_status}")
print(f"Defect-Free Frames: {result.quality_summary.defect_free_frame_percentage:.1f}%")

for defect in result.defects:
    print(f"[{defect.severity.upper()}] {defect.type} on joint '{defect.joint}'")
    print(f"  Frames: {defect.start_frame} - {defect.end_frame} ({defect.duration_sec:.2f}s)")
    print(f"  Confidence: {defect.confidence:.2f} ({defect.verification_status})")
    print(f"  Correction: {defect.suggested_correction}")
```

---

## 📊 Benchmark Evaluation

The benchmark below compares the 4 diagnostic approaches on an unseen, held-out test motion sequence (`subj_06_pivot_turn`, 84 temporal windows):

| Model Architecture | Exact Accuracy | Macro F1 | Precision | Recall | CPU Latency | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic Geometry Baseline** | 27.38% | 0.4351 | 32.90% | **76.25%** | **0.10 ms** | 0.0 MB |
| **Simple Statistical Baseline (Z-Score)** | 17.86% | 0.1231 | 16.00% | 10.00% | 0.14 ms | 0.0 MB |
| **Compact Neural Model (PoseGuardNet)** | **82.14%** | **0.7962** | 91.67% | 72.50% | 0.53 ms | **855 KB** |
| **Calibrated Hybrid Model (PoseGuard AI)** | **77.38%** | **0.7447** | **92.18%** | 65.00% | 0.64 ms | **855 KB** |

*Note on Evaluation: The reported benchmark strictly represents performance on the held-out Subject 6 (Pivot Turn) motion family with zero sequence leakage. It demonstrates temporal generalization across sharp turning kinematics rather than universal validation across all commercial game rigs.*

---

## 🛡️ Privacy & Security

- **100% Offline**: No network calls, telemetry, or remote analytics.
- **Localhost Only**: Web server binds by default to `127.0.0.1:8000`.
- **Zero Cloud Dependence**: Everything from parsing to neural inference executes in-process on CPU.

---

## 📖 Documentation & Community

- [Changelog](CHANGELOG.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Privacy Notice](PRIVACY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Citation Guide](CITATION.cff)
- [Full Documentation Index](docs/README.md)
  - [Installation Guide](docs/INSTALLATION.md)
  - [Model Card](docs/MODEL_CARD.md)
  - [Dataset Card](docs/DATASET_CARD.md)
  - [CLI Reference](docs/CLI_GUIDE.md)
  - [SDK Reference](docs/SDK_GUIDE.md)
  - [API Reference](docs/API.md)
  - [Known Limitations](docs/LIMITATIONS.md)

---

## 📄 License

PoseGuard AI is open source software licensed under the [Apache License 2.0](LICENSE).
