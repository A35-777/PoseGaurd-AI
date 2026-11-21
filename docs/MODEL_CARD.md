# PoseGuard AI — Model Card: `poseguard-tiny-v1`

## 1. Model Details

- **Model Identifier**: `poseguard-tiny-v1`
- **Model Type**: Compact Temporal 1D Convolutional Recurrent Neural Network (`PoseGuardNet`)
- **Version**: 1.0.0
- **Weight Checkpoint**: `models/poseguard_v1.pt`
- **Metadata Spec**: `models/poseguard_v1_meta.json`
- **Model File Size**: **0.835 MB (855 KB)**
- **Trainable Parameters**: **214,597**
- **Inference Runtime**: CPU-Only (`torch.device("cpu")`), No CUDA dependency
- **Inference Latency**: **0.53 ms per window** (~1,880 windows/second on AMD A12 CPU)
- **License**: Apache 2.0
- **Developers**: PoseGuard AI Contributors (Independent Open Source Research)

---

## 2. Intended Use

`poseguard-tiny-v1` is specifically designed for real-time and offline quality inspection of 3D skeletal character animations in video game engines, motion-capture pipelines, and 3D DCC software (Blender, Maya, Unreal Engine).

### Primary Intended Uses:
- Automatic detection of planted foot horizontal drift (foot sliding).
- Detection of floor clipping and underground penetration.
- Detection of sudden angular/linear acceleration spikes and joint pops.
- Detection of foot contact chatter and low-clearance drag.
- Verification of anatomical bone rigidity invariants.

### Out-of-Scope Uses:
- Facial animation analysis or blendshape lip-sync evaluation.
- Clothing, hair, or secondary soft-body physics simulation defects.
- Generating new animations from text prompts (it is an analytical discriminator, not a generative diffusion model).

---

## 3. Architecture Specification

```
INPUT TENSOR: [Batch, Channels=48, Time=32 frames]
   │
   ├─► Conv1d(48, 64, kernel_size=5, padding=2) + BatchNorm1d + LeakyReLU(0.1)
   ├─► MaxPool1d(2) ──────────────────────────► [Batch, 64, 16]
   │
   ├─► TemporalResidualBlock(64 -> 128, kernel=3)
   │     ├─ Conv1d(64, 128, 3, pad=1) + BN + LeakyReLU + Conv1d(128, 128, 3, pad=1) + BN
   │     └─ Residual Shortcut Conv1d(64, 128, 1) + BN
   ├─► MaxPool1d(2) + Dropout(0.25) ──────────► [Batch, 128, 8]
   │
   ├─► Sequence Transpose ────────────────────► [Batch, SeqLen=8, Dim=128]
   ├─► Bidirectional GRU(in=128, hidden=64, 1 layer) ──► [Batch, 8, 128]
   │
   ├─► Multi-Scale Temporal Pooling:
   │     ├─ AdaptiveAvgPool1d(1) ─────────────► [Batch, 128]
   │     └─ AdaptiveMaxPool1d(1) ─────────────► [Batch, 128]
   │     └─ Concatenation ────────────────────► [Batch, 256]
   │
   └─► Classifier Head:
         ├─ Linear(256 -> 128) + LeakyReLU(0.1) + Dropout(0.25)
         ├─ Linear(128 -> 64) + LeakyReLU(0.1)
         └─ Linear(64 -> 5) ──────────────────► [Batch, 5] (Logits)
```

---

## 4. Input Features (48 Kinematic Channels)

Each frame of the 32-frame sliding window is encoded into 48 continuous, rotation- and translation-invariant kinematic channels:

| Index Range | Kinematic Description | Invariance Property |
| :---: | :--- | :--- |
| `0..2` | Left Foot relative position to Hips $(X - X_{hip}, Y - Y_{ground}, Z - Z_{hip})$ | Translation invariant |
| `3..5` | Right Foot relative position to Hips $(X - X_{hip}, Y - Y_{ground}, Z - Z_{hip})$ | Translation invariant |
| `6..8` | Left Toe relative position to Left Foot $(X, Y, Z)$ | Skeletal invariant |
| `9..11` | Right Toe relative position to Right Foot $(X, Y, Z)$ | Skeletal invariant |
| `12..15` | Hips height above floor $(Y - Y_{ground})$ and 3D velocity $(v_x, v_y, v_z)$ | Velocity dynamic |
| `16..19` | Left Foot 3D velocity and horizontal speed | Metric velocity |
| `20..23` | Right Foot 3D velocity and horizontal speed | Metric velocity |
| `24..27` | Left & Right Foot acceleration and jerk magnitudes | Discontinuity indicator |
| `28..31` | Head, Spine, Left Hand, Right Hand acceleration magnitudes | Upper body popping |
| `32..33` | Global peak joint acceleration & peak joint velocity | Biomechanical ceiling |
| `34..37` | Left & Right leg bone lengths and relative deviation | Rigidity constraint |
| `38..39` | Left & Right foot ground contact state indicators (binary contact flags) | Stance indicator |
| `40..41` | Horizontal foot velocity gated by contact flag (sliding indicator) | Sliding detector |
| `42..44` | Ground penetration depths $(Y_{ground} - Y)_{+}$ for feet and pelvis | Floor clipping |
| `45..47` | Vertical jerk and root displacement velocity | Discontinuity indicator |

---

## 5. Training Details

- **Dataset**: 1,127 training windows (augmented with subtle Gaussian noise $\sigma=0.002$m and $\pm 3\%$ scale jitter).
- **Validation Dataset**: 168 validation windows (held-out subjects: Jump, Step Climb).
- **Hardware**: Single AMD PRO A12-9800B CPU (No GPU).
- **Optimizer**: AdamW (Initial Learning Rate = $1 \times 10^{-3}$, Weight Decay = $1 \times 10^{-4}$).
- **Scheduler**: CosineAnnealingLR ($T_{max}=25$, $\eta_{min}=1 \times 10^{-5}$).
- **Batch Size**: 32.
- **Loss Function**: `BCEWithLogitsLoss`.
- **Epochs**: 25.
- **Total CPU Training Time**: **73.58 seconds**.
- **Best Validation Checkpoint**: Epoch 17 (Val Macro F1: **0.7757**).

---

## 6. Evaluation on Held-Out Unseen Test Sequence

Evaluated on Subject 6 (Pivot Turn motion family, 84 temporal windows, completely unseen during training):

- **Exact Match Ratio**: 82.14%
- **Macro F1**: **0.7962**
- **Macro Precision**: **91.67%**
- **Macro Recall**: **72.50%**
- **Global False Positive Rate (FPR)**: **0.0238 (2.38%)**
- **Global False Negative Rate (FNR)**: 0.2750
- **Expected Calibration Error (ECE)**: 0.142
- **Brier Score**: 0.089

---

## 7. Model Provenance & Verification

- **Trained From Scratch**: Yes. Initialized using PyTorch default Kaiming uniform weight initialization.
- **No External Base Weights**: Verified. Does not utilize any pretrained foundation weights.
- **Offline Self-Contained**: Saved as standard `.pt` state dictionary, loadable with standard PyTorch on any CPU machine.

