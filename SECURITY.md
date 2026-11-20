# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security Model & Principles

PoseGuard AI is engineered for local, offline execution:
- **Localhost Binding**: By default, the embedded server binds strictly to `127.0.0.1:8000`. It is never exposed to public interfaces unless explicitly requested by the user.
- **Zero Telemetry**: No user motion clips, joint coordinates, or usage statistics are transmitted across network connections.
- **Path Sanitization**: Uploaded filenames are sanitized using `os.path.basename` to prevent directory traversal vulnerabilities.
- **File Limits**: Upload file sizes are bounded to 50 MB to prevent resource exhaustion attacks.

## Reporting a Vulnerability

If you discover a security vulnerability in PoseGuard AI, please report it responsibly:
- Email: `security@poseguard-ai.local` (or open a confidential security advisory on GitHub).
- Please do not disclose vulnerabilities in public GitHub issues.
- Provide a minimal reproduction script or steps to trigger the issue.
- Maintainers will review and issue a patched release within a reasonable timeframe.

