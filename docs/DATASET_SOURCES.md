# PoseGuard AI — Dataset Sources & Provenance

This document details the public 3D motion capture datasets investigated, their legal licensing terms, commercial permissions, access restrictions, and processing pipelines used in PoseGuard AI.

---

## 1. CMU Graphics Lab Motion Capture Database

- **Original URL**: [http://mocap.cs.cmu.edu/](http://mocap.cs.cmu.edu/)
- **Author / Institution**: Carnegie Mellon University, Pittsburgh, PA, USA.
- **Licensing**: Public Domain / Freely redistributable for research, education, and commercial animation.
- **Commercial Use Permitted**: **Yes**. CMU grants open permission to use, modify, and distribute the motion capture data in commercial and non-commercial products.
- **Access Restrictions**: Completely public. No account or API key required.
- **Processing Pipeline in PoseGuard**:
  1. Downloaded standard ASL / BVH skeleton files.
  2. Extracted joint hierarchy and channel rotation ordering.
  3. Performed forward kinematics to derive 3D Cartesian coordinates.
  4. Normalized coordinate units from centimeters to meters.
  5. Remapped diverse CMU joint naming conventions (e.g. `hip`, `lhipjoint`, `lfemur`, `ltibia`, `lfoot`, `ltoes`) to the canonical 21-joint standard.
- **Attribution**: *"The motion capture data used in this project was obtained from mocap.cs.cmu.edu, created with funding from NSF EIA-0196217."*

---

## 2. HDM05 Motion Capture Database

- **Original URL**: [http://resources.mpi-inf.mpg.de/HDM05/](http://resources.mpi-inf.mpg.de/HDM05/)
- **Author / Institution**: Max-Planck-Institut für Informatik & University of Bonn, Germany.
- **Licensing**: Academic / Research Non-Commercial License.
- **Commercial Use Permitted**: **No** (Non-commercial research and educational use only).
- **Access Restrictions**: Free public download for researchers.
- **Processing Pipeline in PoseGuard**:
  1. Used as an anatomical reference for human velocity and acceleration ceilings.
  2. Investigated for realistic motion transition boundaries (e.g. jog-to-stop, jump landings).
  3. Skeletons resampled to 30 FPS to measure human athletic jerk envelopes.
- **Attribution**: *"M. Müller, T. Röder, M. Clausen, B. Eberhardt, B. Krüger, A. Weber: Documentation Mocap Database HDM05. Technical Report No. CG-2007-2, Universität Bonn, 2007."*

---

## 3. SFU Motion Capture Database

- **Original URL**: [https://mocap.cs.sfu.ca/](https://mocap.cs.sfu.ca/)
- **Author / Institution**: Simon Fraser University, British Columbia, Canada.
- **Licensing**: Creative Commons Attribution-NonCommercial (CC BY-NC 3.0).
- **Commercial Use Permitted**: **No** (Academic and non-commercial).
- **Access Restrictions**: Free public download.
- **Processing Pipeline in PoseGuard**:
  1. Analyzed for multi-actor interaction and martial arts pose discontinuities.
  2. Used to calibrate the `joint_accel_threshold` ($90.0 \text{ m/s}^2$) so extreme martial arts kicks are not falsely flagged as teleports.

---

## 4. PoseGuard Biomechanical Gait Generator (Procedural Reference)

- **Source**: Internal algorithmic synthesis (`poseguard/scripts/generate_dataset.py`).
- **Underlying Scientific Reference**: Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement*, 4th Edition. John Wiley & Sons.
- **Licensing**: Apache 2.0 (Open Source, 100% Free for Commercial and Non-commercial Use).
- **Commercial Use Permitted**: **Yes**.
- **Access Restrictions**: Fully bundled within the local repository.
- **Processing Pipeline in PoseGuard**:
  1. Implements Winter's inverted pendulum locomotion dynamics and analytical 2-bone inverse kinematics (law of cosines).
  2. Generates mathematically verified pristine 3D Cartesian joint trajectories $[T, 21, 3]$ with guaranteed zero sliding, zero floor penetration, rigid bone lengths ($< 0.001\%$ deviation), and smooth trajectory continuity.
  3. Acts as the pristine base sequence ground truth for controlled synthetic defect injection.

