# PoseGuard AI v1.0.0-beta — Public Beta Launch

PoseGuard AI is a lightweight, hybrid AI and geometric analysis system designed to autonomously inspect 3D character animation sequences (.bvh, .npz, .npy) and detect five critical motion defect classes:

1. **Foot Sliding**: Detects frames where feet are grounded but drift horizontally (\(v_{xz} > 0.08\text{ m/s}\)).
2. **Ground Penetration**: Pinpoints frames where joints sink beneath the estimated floor plane (\(Y < Y_{ground} - 0.02\text{ m}\)).
3. **Unnatural Pose Transitions**: Flags sudden velocity or acceleration spikes (\(\|\mathbf{a}\| > 90\text{ m/s}^2\)) indicative of joint popping, retargeting flips, or frame drops.
4. **Contact Inconsistency**: Identifies rapid on/off contact state chatter and foot dragging without vertical clearance.
5. **Bone Stretch Anomalies**: Enforces anatomical rigid-body kinematic invariance, identifying non-rigid bone expansion or compression (\(\Delta L / L_0 > 6\%\)).

---

## ⚡ Key Highlights

- **PoseGuardNet Model**: 214,597 parameter Conv1D-Residual-BiGRU network requiring only **855 KB** of storage (`models/poseguard_v1.pt`).
- **100% Offline & CPU-Native**: Single-threaded inference executes in **0.64 ms** per temporal window on standard multi-core consumer CPUs without requiring CUDA or GPUs.
- **Calibrated Hybrid Gating**: Cross-validates neural temporal defect scores against physical geometric bounds to suppress false positives (FPR 2.06%).
- **Interactive 3D Web Studio**: Built-in Three.js skeletal viewer with defect timeline scrub bar, real-time bounding overlays, and frame-accurate inspection.
- **Zero Telemetry**: No user motion clips, coordinates, or analytics are ever transmitted across external networks.

---

## 📊 Empirical Benchmarks (Held-out Subject 6 Pivot Turn)

| Architecture | Exact Accuracy | Macro F1 | Precision | Recall | CPU Latency | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic Geometry Baseline** | 27.38% | 0.4351 | 32.90% | **76.25%** | **0.10 ms** | 0.0 MB |
| **Statistical Baseline (Z-Score)** | 17.86% | 0.1231 | 16.00% | 10.00% | 0.14 ms | 0.0 MB |
| **Compact Neural (PoseGuardNet)** | **82.14%** | **0.7962** | 91.67% | 72.50% | 0.53 ms | **855 KB** |
| **Calibrated Hybrid (PoseGuard AI)** | **77.38%** | **0.7447** | **92.18%** | 65.00% | 0.64 ms | **855 KB** |

*Note: Evaluated on 84 temporal windows of Subject 6 (Pivot Turn) motion family with zero sequence overlap.*

---

## 📦 Release Assets & Checksums

- **Source Archive**: `poseguard-ai-1.0.0-beta.zip` (1,258,017 bytes / 1.20 MB)
- **Archive SHA-256**: `eaeeaaa7b259fdbdd02d73eefa89d392147e422ac1bb276ceeec64784a797a51`
- **Model Checkpoint**: `poseguard_v1.pt` (875,797 bytes / 0.835 MB)
- **Model Checkpoint SHA-256**: `ebbedb8308d9b57da75ce26484adb96c7bacd6d8cd1d0519e83391257219a513`
