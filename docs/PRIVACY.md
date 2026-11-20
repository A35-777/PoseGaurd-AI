# PoseGuard AI — Privacy & Data Protection Policy

## 1. Core Privacy Philosophy: Absolute Local Isolation

Game development studios and animation houses frequently handle confidential, unreleased intellectual property, proprietary motion-capture data, and character rigs under strict Non-Disclosure Agreements (NDAs).

PoseGuard AI is engineered with an uncompromising privacy architecture:
- **Zero External Network Calls**: The system never initiates outbound connections to external servers, cloud providers, analytics vendors, or telemetry endpoints.
- **100% Offline Capability**: All neural model inferences, forward kinematics computations, geometric defect inspections, and 3D web visualizations execute strictly on the local workstation.
- **No Paid APIs or Cloud Dependencies**: Inference runs natively on local CPU hardware without routing animation files through third-party services.

---

## 2. Animation File Handling

- **Local Storage Only**: Animation files uploaded through the web application or processed via CLI/SDK remain on your local disk or inside temporary operating system directories.
- **No Embedded Code Execution**: Animation files (`.bvh`, `.npz`, `.npy`) are parsed strictly as numerical kinematic trajectories and hierarchy definitions. No executable scripts, macros, or arbitrary payloads are ever evaluated.
- **Path Sanitization**: API responses sanitize absolute file paths to protect workstation directory structures.

---

## 3. Telemetry, Tracking, and Logging

- **No Usage Tracking**: PoseGuard AI does not collect usage statistics, project identifiers, prompt history, user metadata, or feature frequencies.
- **No Crash Reporting Over Network**: Any exception tracebacks remain local to your console session and local logs.

---

## 4. Verification

To verify network isolation on Windows:
1. Disconnect your internet connection or enable Airplane Mode.
2. Run `poseguard serve --port 8000`.
3. Open `http://127.0.0.1:8000` and analyze any animation clip.
4. All features, 3D WebGL rendering (via bundled local Three.js), and neural diagnosis will continue functioning without degradation.

