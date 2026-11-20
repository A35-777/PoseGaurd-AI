# PoseGuard AI — Neural Network Architecture Deep-Dive

This document provides a comprehensive technical breakdown of `PoseGuardNet`, the compact temporal neural network powering PoseGuard AI.

---

## 1. Architectural Philosophy: The Tiny AI Paradigm

In character animation systems, inference latency and memory footprint are paramount. A quality analysis tool must not disrupt the animator's workflow or monopolize GPU resources reserved for rendering.

`PoseGuardNet` was designed to achieve high multi-label classification accuracy with:
- **Zero GPU dependency** (Runs on basic CPU cores).
- **Sub-megabyte model size** (0.835 MB).
- **Sub-millisecond inference latency** (0.53 ms per 32-frame window).
- **Total Parameters**: 214,597 (0.21M).

---

## 2. Layer-by-Layer Architectural Breakdown

```
Layer (type:depth-idx)                   Output Shape              Param #
========================================================================
PoseGuardNet                             [Batch, 5]                --
├─ Sequential: stem                      [Batch, 64, 16]           --
│    └─ Conv1d (in=48, out=64, k=5, p=2) [Batch, 64, 32]           15,424
│    └─ BatchNorm1d (64)                 [Batch, 64, 32]           128
│    └─ LeakyReLU (0.1)                  [Batch, 64, 32]           --
│    └─ MaxPool1d (k=2)                  [Batch, 64, 16]           --
├─ TemporalResidualBlock: res1           [Batch, 128, 16]          --
│    ├─ Conv1d (64 -> 128, k=3, p=1)     [Batch, 128, 16]          24,704
│    ├─ BatchNorm1d (128)                [Batch, 128, 16]          256
│    ├─ LeakyReLU (0.1)                  [Batch, 128, 16]          --
│    ├─ Conv1d (128 -> 128, k=3, p=1)   [Batch, 128, 16]          49,280
│    ├─ BatchNorm1d (128)                [Batch, 128, 16]          256
│    └─ Sequential: shortcut             [Batch, 128, 16]          --
│         ├─ Conv1d (64 -> 128, k=1)     [Batch, 128, 16]          8,320
│         └─ BatchNorm1d (128)           [Batch, 128, 16]          256
├─ MaxPool1d (k=2)                       [Batch, 128, 8]           --
├─ Dropout (p=0.25)                      [Batch, 128, 8]           --
├─ GRU: gru (in=128, hidden=64, BiGRU)   [Batch, 8, 128]           74,496
├─ AdaptiveAvgPool1d (1)                 [Batch, 128]              --
├─ AdaptiveMaxPool1d (1)                 [Batch, 128]              --
└─ Sequential: classifier                [Batch, 5]                --
     ├─ Linear (256 -> 128)              [Batch, 128]              32,896
     ├─ LeakyReLU (0.1)                  [Batch, 128]              --
     ├─ Dropout (p=0.25)                 [Batch, 128]              --
     ├─ Linear (128 -> 64)               [Batch, 64]               8,256
     ├─ LeakyReLU (0.1)                  [Batch, 64]               --
     └─ Linear (64 -> 5)                 [Batch, 5]                325
========================================================================
Total Trainable Parameters: 214,597
Checkpoint Size: 0.835 MB (855,142 bytes)
```

---

## 3. Key Design Choices

### 1. 1D Temporal Convolutions vs. 2D / 3D Convolutions
Human skeletal kinematics do not have a planar grid spatial structure like image pixels. Applying 2D convolutions over adjacent joint indices creates artificial spatial adjacency artifacts. Instead, PoseGuard treats each kinematic measurement as a continuous channel along the 1D temporal dimension ($T=32$ frames). This models motion patterns naturally as multivariate time-series waveforms.

### 2. Multi-Scale Temporal Pooling: AvgPool + MaxPool
Different animation defects exhibit distinct temporal signatures:
- **Foot Sliding** is a *persistent condition*: the foot drifts across 10-30 frames. Average pooling captures this sustained drift.
- **Unnatural Pose Transitions (Pops)** are *transient impulses*: a 1-2 frame spike in acceleration. Average pooling dilutes a 1-frame spike over 32 frames; Max pooling preserves the extreme peak value!
Concatenating Average Pooling ($128 \text{ dims}$) and Max Pooling ($128 \text{ dims}$) provides a comprehensive $256 \text{ dim}$ representation sensitive to both persistent and impulsive defects.

### 3. Bidirectional GRU Sequence Context
While causal networks only look backward, animation quality analysis operates on pre-recorded or buffered clips. The Bidirectional GRU processes the temporal trajectory both forward and backward, allowing the network to evaluate how an animation enters and exits a defect.

