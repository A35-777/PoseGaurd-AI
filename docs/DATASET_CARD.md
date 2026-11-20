# PoseGuard AI — Dataset Card

## 1. Dataset Overview

The PoseGuard AI dataset is a structured collection of 3D skeletal motion sequences designed to train and benchmark motion quality defect detectors. It contains **1,379 temporal kinematic windows** ($W=32$ frames, 48 channels) generated across 10 distinct human subjects and biomechanical motion families.

---

## 2. Dataset Partitioning & Leakage Prevention

To strictly prevent temporal data leakage (where near-identical adjacent frames or windows from the same sequence appear in both train and test sets), the dataset is partitioned **exclusively at the subject and motion family level**:

| Split | Subject IDs | Motion Families Included | Windows | Percentage |
| :---: | :--- | :--- | :---: | :---: |
| **Train** | `subj_01`, `subj_02`, `subj_03`, `subj_04`, `subj_07`, `subj_08`, `subj_09` | Walk, Brisk Walk, Jog, Sprint, Lateral Shuffle, Idle, Crouch Walk | 1,127 | 81.7% |
| **Validation** | `subj_05`, `subj_10` | Jump & Land, Step Climb | 168 | 12.2% |
| **Test** | `subj_06` | **Pivot Turn (90° Re-orientation)** | 84 | 6.1% |
| **Total** | **10 Subjects** | **10 Motion Families** | **1,379** | **100.0%** |

> **Leakage Verification**: The Test split (`subj_06_pivot_turn`) contains an entirely unseen motion family and subject ID. Zero sequence IDs or temporal windows are shared across splits.

---

## 3. Label Distribution

Each window is annotated with a 5-dimensional multi-label binary ground-truth target vector:
$$\mathbf{y} = [y_{slide}, y_{pen}, y_{trans}, y_{contact}, y_{stretch}] \in \{0, 1\}^5$$

| Defect Class | Train Windows | Val Windows | Test Windows | Total Windows |
| :--- | :---: | :---: | :---: | :---: |
| **Foot Sliding** | 224 | 32 | 16 | 272 |
| **Ground Penetration** | 224 | 32 | 16 | 272 |
| **Unnatural Pose Transition** | 112 | 16 | 8 | 136 |
| **Contact Inconsistency** | 112 | 16 | 8 | 136 |
| **Bone Stretch Anomaly** | 112 | 16 | 8 | 136 |
| **Clean Windows (Zero Defects)** | 455 | 72 | 36 | 563 |

---

## 4. Controlled Synthetic Defect Injection Methodology

To produce precise frame-accurate ground-truth defect annotations without subjective human labeling ambiguities, defects were introduced into pristine biomechanical base animations using mathematically defined physical transformations:

### A. Foot Sliding Injection
During planted stance phases ($t \in [t_{start}, t_{end}]$):
$$\vec{p}_{foot}(t) \leftarrow \vec{p}_{foot}(t) + \begin{bmatrix} v_{drift} \cdot (t - t_{start}) \cdot \Delta t \cdot 0.5 \\ 0 \\ v_{drift} \cdot (t - t_{start}) \cdot \Delta t \cdot 0.8 \end{bmatrix}$$
where $v_{drift} \in [0.35, 0.60] \text{ m/s}$.

### B. Ground Penetration Injection
During ground contact intervals:
$$\vec{p}_{foot, Y}(t) \leftarrow \vec{p}_{foot, Y}(t) - \Delta y_{pen}$$
where $\Delta y_{pen} \in [0.08, 0.20] \text{ m}$.

### C. Unnatural Pose Transition Injection
At targeted pop frame $t_{pop}$:
$$\vec{p}_{joint}(t_{pop}) \leftarrow \vec{p}_{joint}(t_{pop}) + \vec{\delta}_{teleport}$$
where $\|\vec{\delta}_{teleport}\| \in [0.45, 0.85] \text{ m}$, generating sudden acceleration spikes $a(t) > 150 \text{ m/s}^2$ and extreme jerks without biomechanical deceleration phases.

### D. Contact Inconsistency Injection
High-frequency vertical oscillation near the floor boundary:
$$\vec{p}_{foot, Y}(t) \leftarrow \vec{p}_{foot, Y}(t) + 0.06 \cdot \sin(2.5 \cdot t)$$
forcing the contact classifier to rapidly chatter on/off across consecutive frames.

### E. Bone Stretch Anomaly Injection
Modulating bone segment length beyond rigid-body tolerances:
$$\vec{p}_{child}(t) \leftarrow \vec{p}_{parent}(t) + (\vec{p}_{child}(t) - \vec{p}_{parent}(t)) \cdot s_{stretch}$$
where $s_{stretch} \in [1.25, 1.40]$ (25% to 40% elongation).

---

## 5. File Formats & Storage

- **Base Motions**: `data/raw/` (BVH files and standardized `.npz` files).
- **Synthetic Defect Clips**: `data/synthetic/` (standardized `.npz` files with full JSON metadata headers).
- **Processed Dataset Arrays**: `data/processed/` (`train_data.npz`, `val_data.npz`, `test_data.npz`).
- **Manifest**: `data/processed/manifest.json`.

