# Contributing to PoseGuard AI 🛡️

Thank you for your interest in contributing to PoseGuard AI! We welcome community contributions ranging from bug fixes, documentation clarity, and benchmark evaluations to novel biomechanical defect detection algorithms and game engine integrations.

---

## Code of Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all repository discussions, issue tickets, and pull request reviews.

---

## Quickstart Development Setup

1. **Fork the repository** on GitHub.
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/<your-username>/PoseGaurd-AI.git
   cd PoseGaurd-AI
   ```
3. **Set up a dedicated Python virtual environment** (Python 3.10, 3.11, or 3.12 recommended):
   ```bash
   # Windows
   py -3.12 -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. **Install development dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

---

## How to Run Tests

PoseGuard AI maintains strict automated testing across parsers, geometry rules, neural models, CLI, and API endpoints.

Run the complete test suite locally:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
# Test analytical geometric rules
pytest tests/test_geometry.py -v

# Test CLI commands and JSON/CSV outputs
pytest tests/test_cli.py -v

# Test BVH parser and normalization
pytest tests/test_bvh_parser.py tests/test_normalization.py -v

# Test Python SDK
pytest tests/test_sdk.py -v
```

---

## Code Style & Standards

We enforce high code quality standards to keep the runtime lightweight, robust, and readable:

- **Formatting & Style**: Follow PEP 8 guidelines. Format code using `black` (line length 100) and check with `ruff` or `flake8`.
- **Type Annotations**: Provide Python type hints on all public functions, classes, and SDK methods.
- **CPU-First & Low Memory**: Core runtime modules must never introduce mandatory CUDA or heavy cloud dependencies. Peak process memory must remain under 280 MB.
- **Deterministic Biomechanics**: Any new geometric detection logic must be grounded in physical kinematic rules (velocity, floor clearances, joint angle limits) with clearly documented mathematical rationale.
- **No Weight Leaks**: Do not alter default inference thresholds or retrain model checkpoints without providing comparative validation metrics against the benchmark suite.

---

## Pull Request Checklist

Before submitting your pull request, please verify the following:

- [ ] **Tests Pass**: `pytest tests/ -v` completes with 100% passes and zero regressions.
- [ ] **CLI & SDK Verified**: Tested `poseguard analyze <file>` and `from poseguard import PoseGuard` locally.
- [ ] **Documentation Updated**: If adding or altering a feature, relevant docs (`docs/CLI_GUIDE.md`, `docs/SDK_GUIDE.md`, `README.md`) are updated.
- [ ] **Commit Messages**: Commits follow [Conventional Commits](https://www.conventionalcommits.org/) format (e.g., `feat:`, `fix:`, `docs:`, `perf:`, `test:`, `chore:`).
- [ ] **Atomic Changes**: The PR focuses on one logical feature or fix rather than bundling unrelated changes.

---

## Community & Questions

Have questions or ideas before writing code? Open a [GitHub Discussion](https://github.com/A35-777/PoseGaurd-AI/discussions) or create a feature proposal issue.
