# Security Policy

PoseGuard AI takes the security of its open-source software and users seriously. As a lightweight, offline-first tool for 3D animation inspection, our security architecture is deliberately designed with minimal attack surface.

---

## Supported Versions

We provide security updates for the following active release branches:

| Version | Supported          | Status |
| :--- | :---: | :--- |
| **1.0.x (Public Beta)** | :white_check_mark: | Currently Supported |
| < 1.0 (Pre-release)     | :x:                | Unsupported |

---

## Security Model & Guarantees

- **100% Offline Execution**: PoseGuard AI performs all analytical operations locally on the host CPU. The system does not transmit animation clips, telemetry, joint coordinates, or environment metadata across network interfaces.
- **Strict Localhost Binding**: The embedded FastAPI server for the 3D Web Studio binds exclusively to loopback (`127.0.0.1:8000`) by default.
- **Path Traversal Protection**: All uploaded animation filenames and local file arguments are sanitized using strict base-name isolation (`os.path.basename`) to prevent directory traversal attacks.
- **Denial of Service (DoS) Guard**: Multipart file uploads are bounded by default to 50 MB to prevent memory exhaustion on consumer workstations.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly so we can resolve it before public disclosure:

1. **GitHub Security Advisory (Recommended)**:
   Submit a private report via the [PoseGuard AI Security Advisories Tab](https://github.com/A35-777/PoseGaurd-AI/security/advisories/new).
2. **Direct Contact**:
   Alternatively, email the maintainer directly at **`toaringoel@gmail.com`** with the subject `[SECURITY] PoseGuard AI Vulnerability`.

### Please Include:
- A clear description of the vulnerability and its potential impact.
- Step-by-step reproduction instructions or a minimal proof-of-concept `.bvh` / script.
- Any suggested remediations or mitigations.

### Response Timeline:
- **Initial Acknowledgment**: Within 48 hours.
- **Assessment & Triage**: Within 5 business days.
- **Patch Release & Advisory**: Disclosed in coordination with the reporter after a patch has been merged and published.

*Please do not open public GitHub issues for undisclosed security vulnerabilities.*
