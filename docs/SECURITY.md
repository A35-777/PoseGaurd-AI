# PoseGuard AI — Security Architecture & Threat Model

This document outlines the security controls, validation policies, and threat mitigations implemented across PoseGuard AI.

---

## 1. Threat Model & Untrusted Inputs

In animation pipelines, 3D asset files may originate from external contractors, public repositories, or third-party marketplaces. PoseGuard AI treats all ingested animation files as untrusted.

### Potential Threat Vectors:
1. **Malicious Serialization Payloads**: Arbitrary Python code execution embedded in serialized formats (e.g. unconstrained `pickle`).
2. **Buffer / Resource Exhaustion (DoS)**: Abnormally massive animation files (millions of frames or billions of joints) designed to exhaust workstation RAM.
3. **Path Traversal Attacks**: Malformed filenames attempting to write outside designated temporary directories.
4. **Local Network Exposure**: Unintended network listeners accessible across local area networks (LANs).

---

## 2. Mitigations Implemented

### A. Pure-Text & Safe NumPy Parsing
- **BVH Parsing**: The built-in `BVHParser` is a pure-Python string parser that strictly tokenizes hierarchical joints (`ROOT`, `JOINT`, `OFFSET`, `CHANNELS`, `MOTION`). It never evaluates Python code, executes shell commands, or invokes dynamic reflection (`eval`, `exec`).
- **NumPy Files**: When loading `.npz` files, array dimensions are strictly verified to ensure $[T, J, 3]$ bounds.

### B. Memory & File Size Guardrails
- **Upload Limit**: The web server enforces a 50 MB maximum payload limit for uploaded animation files.
- **Frame & Joint Sanity Checks**: Skeletons exceeding 128 joints or animations exceeding 20,000 frames are rejected before allocation to prevent memory exhaustion on 8 GB RAM machines.
- **Streaming & Chunking**: Processing operates over rolling temporal windows ($W=32$ frames) rather than allocating massive high-order global tensors.

### C. Localhost Binding
- By default, `poseguard serve` binds exclusively to the loopback interface (`127.0.0.1`). It does not bind to `0.0.0.0`, ensuring external devices on the same Wi-Fi or LAN cannot access the workstation server without explicit configuration.

### D. Schema Validation
- All REST requests and analysis responses are validated against Pydantic models with strict typing and boundary constraints.

