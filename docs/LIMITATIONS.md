# PoseGuard AI — Limitations & Boundary Conditions

PoseGuard AI is an automated motion defect detection system. To maintain technical honesty and avoid misleading animators or engineers, this document details known limitations and boundary conditions.

---

## 1. Supported Skeletal Representations

- **Supported Formats**: Biovision Hierarchy (`.bvh`), Standardized NumPy Arrays (`.npz`, `.npy`).
- **FBX Format Note**: Direct binary FBX parsing is not natively supported without Autodesk's proprietary C++ FBX SDK or Blender. For FBX character animations, animators should export to BVH via Blender (`File -> Export -> BioVision Hierarchy (.bvh)`).
- **Skeleton Types**: Bipedal humanoids. Quadruped or multi-limbed creature skeletons (e.g. horses, dragons, octopuses) are not supported by the canonical 21-joint remapper without custom joint alias mapping.

---

## 2. Terrain & Ground Assumptions

- **Planar Ground Reference**: Foot sliding and ground penetration algorithms evaluate contact against a horizontal reference plane ($Y = Y_{ground}$).
- **Uneven Terrain & Stairs**: On sloped hills, stairs, or dynamically deformable terrain, feet will naturally cross different elevations. For these sequences, automatic flat ground estimation must be overridden with specific local ground heights, or analyzed per-interval.

---

## 3. Stylized vs. Realistic Animation

- **Cartoony Squash-and-Stretch**: Traditional cartoon animations (e.g. Looney Tunes, Spider-Verse) deliberately utilize non-rigid bone elongation, extreme squash-and-stretch, and smear frames. PoseGuard AI's rigid bone check will flag these as `bone_stretch_anomaly` unless `bone_stretch_relative_tolerance` is widened in `PoseGuardConfig`.
- **Fast Athletic Moves**: Extreme martial arts spinning kicks or explosions may approach the `joint_accel_threshold` ($90 \text{ m/s}^2$). The dual-gating hybrid model mitigates false positives, but highly stylized keyframed anime motions may require increasing threshold parameters.

---

## 4. Secondary Physics & Meshes

- PoseGuard AI analyzes **skeletal joint transforms and kinematic trajectories**.
- It does **not** evaluate skinning weight vertex clipping, clothing collisions, soft-body muscle simulation, hair dynamics, or facial blendshape phoneme sync.
- It does **not** generate new motion clips from natural language (it is a quality validator, not a generative diffusion model).

