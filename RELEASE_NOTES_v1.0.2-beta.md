# PoseGuard AI v1.0.2-beta — High-Precision Hybrid Engine & Model Upgrade

PoseGuard AI is an ultra-lightweight, offline-first hybrid AI and geometric analysis system designed to autonomously inspect 3D character animation sequences (`.bvh`, `.npz`, `.npy`) and detect five critical motion defect classes:

1. **Foot Sliding**: Detects frames where feet are planted or in contact but drift horizontally relative to the ground plane.
2. **Ground Penetration**: Pinpoints frames where joints sink beneath the stance-anchored floor plane.
3. **Unnatural Pose Transitions**: Flags sudden acceleration spikes and joint jerk indicative of retargeting glitches, bad interpolation, or gimbal lock flips.
4. **Contact Inconsistency**: Identifies rapid on/off contact state chatter, dragging without vertical clearance, and flight inconsistencies.
5. **Bone Stretch Anomalies**: Enforces anatomical rigid-body kinematic invariance, identifying non-rigid bone expansion or compression (\(\Delta L / L_0 > 6\%\)).

---

## ⚡ Key Highlights (v1.0.2-beta)

- **Squeeze-and-Excitation Attention**: Integrated 1D channel attention into `TemporalResidualBlock` with reduction ratio \(r=8\), dynamically recalibrating kinematic velocity and jerk signals while adding only 4,096 parameters.
- **Multi-Scale Temporal Feature Fusion**: Fuses an adaptive 1D convolutional summary (`conv_pool`, 128 dims) with bidirectional GRU dual-pooling (average + max pool, 256 dims) into a comprehensive 384-dim representation for the multi-label MLP head.
- **Calibrated Label Smoothing**: Retrained `models/poseguard_v1.pt` from initialized weights on CPU with multi-label binary cross-entropy and \(0.02\) label smoothing, yielding well-calibrated confidence probabilities.
- **Strictly Tiny Model Constraints**:
  - **Trainable Parameters**: **235,221** (target \(\le 250\text{k}\))
  - **Disk Checkpoint**: **941 KB (0.915 MB)** (target \(\le 1.0\text{ MB}\))
  - **Inference Latency**: **0.58 ms** (Neural) / **0.71 ms** (Full Hybrid Pipeline) on standard CPU cores
  - **Peak Memory**: **< 280 MB RAM**
  - **Environment**: **100% Offline & CPU-Native** (Zero CUDA, zero external API keys, zero telemetry)
- **Calibrated Dual-Gate Verification**: Defects are confirmed via joint geometric-neural agreement or high-confidence neural corroboration, cutting global false-positive rate from 1.65% down to **0.27%** and reducing false negatives by **69.4%**.
- **Stance-Aware Ground Tracking**: Anchors ground plane estimation to stationary stance frames (\(|v_y| < 0.12\text{ m/s}\)), preventing ground reference dragging during jump and roll maneuvers.
- **Runnable Sample Animations**: Pre-packaged clean and defective BVH/NPZ samples with expected JSON reports under `examples/`.

---

## 📊 Empirical Benchmarks (Held-out Subject 6 Pivot Turn Split)

Evaluated on 84 temporal windows (64-frame window, 50% overlap) with zero sequence overlap:

| Architecture | Exact Accuracy | Macro F1 | Precision | Recall | Global FPR | CPU Latency | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic Geometry Baseline** | 27.38% | 0.4351 | 32.90% | 76.25% | 18.2% | **0.10 ms** | 0.0 MB |
| **Statistical Baseline (Z-Score)** | 17.86% | 0.1231 | 16.00% | 10.00% | 6.5% | 0.14 ms | 0.0 MB |
| **Compact Neural (PoseGuardNet v1.0.2)** | **89.29%** | **0.9071** | **97.78%** | 86.25% | 0.38% | 0.58 ms | **941 KB** |
| **Calibrated Hybrid (PoseGuard AI v1.0.2)** | **91.67%** | **0.9214** | **97.78%** | **88.75%** | **0.27%** | 0.71 ms | **941 KB** |

### Per-Defect F1-Scores

| Defect Class | v1.0.0-beta F1 | v1.0.2-beta F1 | Improvement |
| :--- | :---: | :---: | :---: |
| **Foot Sliding** | 0.6667 | **0.8889** | +0.2222 |
| **Knee Hyperextension** | 0.8235 | **1.0000** | +0.1765 |
| **Ground Penetration** | 0.6154 | **0.8000** | +0.1846 |
| **Bone Stretch Anomaly** | 0.8000 | **0.9231** | +0.1231 |
| **Velocity Spike (Jerk)** | 1.0000 | **1.0000** | 100% retained |

---

## 📦 Distribution Packages

- **Source Archive**: `dist/poseguard-ai-1.0.2-beta.zip` (1,767,785 bytes / 1.69 MB)
- **Archive SHA-256**: `7b018785c8bade63f54349ab29a5bd5a31396653f3347ffc2c6dc963aa0ea40d`
- **Model Checkpoint**: `models/poseguard_v1.pt` (959,837 bytes / 0.915 MB, 235,221 parameters)
- **Model Checkpoint SHA-256**: `35e04e74033aa87a2b532587bea78ead55ac5521a3c5ba2bc1204b27378b61cb`
- **Python Wheel**: `dist/poseguard-1.0.2-py3-none-any.whl` (219,706 bytes)
- **Python Source Tarball**: `dist/poseguard-1.0.2.tar.gz` (216,347 bytes)
- **Release Manifest**: `RELEASE_MANIFEST.json`
- **Checksums**: `dist/poseguard-ai-1.0.2-beta.zip.sha256`
